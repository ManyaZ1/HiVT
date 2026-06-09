"""
hivt_kd.py

HiVTKD — a LightningModule that wraps a HiVT student with the
permutation-invariant mixture-NLL Knowledge Distillation loss against
pre-saved HiVT teacher soft targets (see kd_loss.py for the method).

Drop-in replacement for HiVT in train_student_kd.py.

CHANGES vs the previous version:
  * The KD call site no longer asserts student.shape == teacher.shape.
    The new loss scores F teacher mode-means under the student's K-mode
    mixture, so K (student modes) and F (teacher modes) may differ; only
    B, H and the coord dim must agree.
  * KD log keys are now "kd/mix_nll" and "kd/total" (previously
    "kd/kl_laplace" and "kd/pi_ce"). Any wandb panels referencing the old
    keys will be empty.
"""

import argparse
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hivt import HiVT          # repo module (absolute; repo root on sys.path)
from KD.kd_loss import HiVTKDLoss     # sibling KD module (absolute from repo root)


def _count_parameters(model: nn.Module) -> Dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class HiVTKD(pl.LightningModule):
    """
    Student training with KD.

    The class deliberately does NOT subclass HiVT so that the interface
    stays clean and we can swap any future architecture as the student.
    All training / validation logic delegates to the inner HiVT instance.

    New hyper-parameters added on top of HiVT's own args:
        --lambda_task   weight for the GT WTA task loss     (default 1.0)
        --lambda_kl     weight for the KD mixture-NLL term  (default 0.5)
        --lambda_pi     DEPRECATED / ignored (see kd_loss)  (default 0.0)
        --teacher_dir   directory with saved teacher targets
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Pull KD-specific kwargs before forwarding the rest to HiVT
        self.lambda_task  = kwargs.pop("lambda_task",  1.0)
        self.lambda_kl    = kwargs.pop("lambda_kl",    0.5)
        self.lambda_pi    = kwargs.pop("lambda_pi",    0.0)
        self.teacher_dir  = kwargs.pop("teacher_dir",  None)
        # Sanity-test knobs: strip HiVT's cosine LR schedule (which decays to ~0
        # within a few dozen overfit steps) and optionally pin a constant LR.
        self.sanity_constant_lr = kwargs.pop("sanity_constant_lr", False)
        self.sanity_lr          = kwargs.pop("sanity_lr",          None)

        # Save ALL hparams (including the ones we popped) for wandb / ckpt
        self.save_hyperparameters()

        # Inner student model — embed_dim controls student width (e.g. 32)
        self.student = HiVT(**kwargs)

        # KD loss module (permutation-invariant mixture NLL)
        self.kd_loss_fn = HiVTKDLoss(
            lambda_kl=self.lambda_kl,
            lambda_pi=self.lambda_pi,
        )

    # ---------------------------------------------------------------------- #
    # Forward — thin pass-through to student decoder
    # ---------------------------------------------------------------------- #
    def forward(self, data):
        return self.student(data)

    # ---------------------------------------------------------------------- #
    # Training step
    # ---------------------------------------------------------------------- #
    def training_step(self, data, batch_idx):
        # ---- Student forward ----
        pred_s, pi_s = self.student(data)          # [K,N,H,4], [N,K]

        # ---- GT task loss (same computation as base HiVT.training_step) ----
        reg_mask = ~data['padding_mask'][:, self.student.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(pred_s[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [K, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = pred_s[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.student.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = torch.nn.functional.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.student.cls_loss(pi_s[cls_mask], soft_target)
        task_loss = reg_loss + cls_loss

        # ---- KD loss (only when teacher targets are present) ----
        has_teacher = getattr(data, "has_teacher", False)
        if isinstance(has_teacher, torch.Tensor):
            has_teacher = bool(has_teacher.all().item())
        else:
            has_teacher = bool(has_teacher)

        has_teacher = has_teacher and all(
            hasattr(data, attr) for attr in ("teacher_loc", "teacher_scale", "teacher_pi")
        )
        kd_logs = {}
        if has_teacher:
            agent_index = data.agent_index                          # [B]

            # Student focal-agent slice — [K, B, H, *]
            student_loc   = pred_s[:, agent_index, :, :2]          # [K, B, H, 2]
            student_scale = pred_s[:, agent_index, :, 2:]          # [K, B, H, 2]
            student_pi    = pi_s[agent_index]                      # [B, K]

            # Teacher targets — PyG stacked [B,F,H,2] -> [F,B,H,2]
            loc_t   = data.teacher_loc.permute(1, 0, 2, 3)         # [F, B, H, 2]
            scale_t = data.teacher_scale.permute(1, 0, 2, 3)       # [F, B, H, 2]
            pi_t    = data.teacher_pi                              # [B, F]

            # NOTE: K (student modes) and F (teacher modes) need NOT match.
            # The permutation-invariant mixture-NLL loss scores the F teacher
            # mean-trajectories under the student's K-mode mixture, so the only
            # dimensions that must agree are batch (B), horizon (H) and coords.
            assert student_loc.shape[1:] == loc_t.shape[1:], \
                f"B/H/coord mismatch: student {tuple(student_loc.shape)} vs teacher {tuple(loc_t.shape)}"

            kd_loss, kd_logs = self.kd_loss_fn(
                student_loc, student_scale, student_pi,
                loc_t.detach(), scale_t.detach(), pi_t.detach(),
            )
            total_loss = self.lambda_task * task_loss + kd_loss
        else:
            total_loss = self.lambda_task * task_loss

        # ---- Trajectory metrics on the TRAIN batch (memorization signal) ----
        # Mirrors validation_step's agent-level min-over-modes computation, but
        # computed directly (not via the stateful self.student.min* torchmetrics,
        # which accumulate val state). In a true single-batch overfit these
        # should collapse toward ~0 — unlike val_minADE, which measures the full
        # val set and stays pinned near the untrained baseline.
        with torch.no_grad():
            y_hat_agent = pred_s[:, data['agent_index'], :, :2]          # [K, B, H, 2]
            y_agent = data.y[data['agent_index']]                       # [B, H, 2]
            fde_modes = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)  # [K, B]
            best_mode_agent = fde_modes.argmin(dim=0)                   # [B]
            y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(y_agent.size(0))]  # [B, H, 2]
            fde_best = torch.norm(y_hat_best_agent[:, -1] - y_agent[:, -1], p=2, dim=-1)    # [B]
            train_minADE = torch.norm(y_hat_best_agent - y_agent, p=2, dim=-1).mean()
            train_minFDE = fde_best.mean()
            train_minMR  = (fde_best > 2.0).float().mean()
        bs = y_agent.size(0)
        self.log("train/minADE", train_minADE, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/minFDE", train_minFDE, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/minMR",  train_minMR,  on_step=True, on_epoch=True, batch_size=bs)

        # Scale-inflation diagnostic. pred_s[..., 2:] is the post-activation
        # Laplace scale b (decoder applies ELU+1+min_scale, see models/decoder.py),
        # used directly in the NLL as log(2b)+|y-loc|/b. If scale_mean climbs while
        # train/minADE stays ~14, the loss is dropping by inflating b rather than
        # moving the means — the Laplace-NLL scale-inflation pathology.
        self.log("train/scale_mean", pred_s[:, data['agent_index'], :, 2:].mean(),
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        # Monotonic peak GPU allocation since run start — once it plateaus you've
        # seen the densest batch and know your true VRAM headroom (see SMOKE_TESTS).
        if torch.cuda.is_available():
            self.log("train/gpu_peak_gb", torch.cuda.max_memory_allocated() / 1e9,
                     on_step=True, on_epoch=False, prog_bar=True, batch_size=bs)

        # ---- Logging ----
        self.log("train/reg_loss", reg_loss, on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        self.log("train/cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        self.log("train/loss_task",  task_loss,  on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        for k, v in kd_logs.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=self._batch_size(data))

        def _to_float(val):
            if isinstance(val, torch.Tensor):
                return float(val.detach().cpu().item())
            return float(val) if val is not None else 0.0
        self._last_metrics = {
            "train/loss_task":  _to_float(task_loss),
            "train/kd/mix_nll": _to_float(kd_logs.get("kd/mix_nll")),
        }
        return total_loss

    # ---------------------------------------------------------------------- #
    # Validation step — identical to base HiVT (no KD at val time)
    # ---------------------------------------------------------------------- #
    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.student.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.student.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log("val_reg_loss", reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat[:, data['agent_index'], :, :2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.student.minADE.update(y_hat_best_agent, y_agent)
        self.student.minFDE.update(y_hat_best_agent, y_agent)
        self.student.minMR.update(y_hat_best_agent, y_agent)
        self.log("val_minADE", self.student.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log("val_minFDE", self.student.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log("val_minMR", self.student.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    # ---------------------------------------------------------------------- #
    # Optimiser / scheduler — reuse student's configuration
    # ---------------------------------------------------------------------- #
    def configure_optimizers(self):
        # Default: delegate to HiVT (AdamW + CosineAnnealingLR over T_max epochs).
        if not self.sanity_constant_lr:
            return self.student.configure_optimizers()

        # Sanity-test mode: reuse the student's optimizer (keeps its weight-decay
        # param grouping) but drop the scheduler so the LR stays constant. Under
        # --overfit_batches one epoch == one optimizer step, so the cosine schedule
        # (T_max in *epochs*) anneals the LR to near-zero within a few dozen steps
        # and starves the decoder means of gradient. A constant LR lets the means
        # actually move so the memorization check is meaningful.
        cfg = self.student.configure_optimizers()
        optimizer = cfg["optimizer"] if isinstance(cfg, dict) else cfg
        if self.sanity_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = self.sanity_lr
        return optimizer

    # ---------------------------------------------------------------------- #
    # Argparse — extend HiVT's own args with KD-specific ones
    # ---------------------------------------------------------------------- #
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        # First add all of HiVT's args
        parser = HiVT.add_model_specific_args(parent_parser)
        # Then add KD-specific args
        parser.add_argument("--lambda_task",  type=float, default=1.0,
                            help="Weight for the GT WTA task loss")
        parser.add_argument("--lambda_kl",    type=float, default=0.5,
                            help="Weight for the KD mixture-NLL distillation term")
        parser.add_argument("--lambda_pi",    type=float, default=0.0,
                            help="DEPRECATED / ignored by the permutation-invariant "
                                 "loss. Kept for CLI backward-compatibility.")
        parser.add_argument("--teacher_dir",  type=str,   required=True,
                            help="Directory / cache containing teacher soft targets "
                                 "(output of save_teacher_outputs.py)")
        # Sanity-test only: bypass the cosine LR schedule for single-batch overfit.
        parser.add_argument("--sanity_constant_lr", action="store_true",
                            help="Drop HiVT's CosineAnnealingLR and train at a "
                                 "constant LR. For --overfit_batches memorization "
                                 "checks, where the schedule otherwise decays the LR "
                                 "to ~0 within a few dozen steps.")
        parser.add_argument("--sanity_lr", type=float, default=None,
                            help="Constant LR to pin when --sanity_constant_lr is set "
                                 "(default: keep HiVT's base --lr).")
        return parser

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    def _batch_size(self, data) -> int:
        """Return number of focal agents in this batch for correct logging."""
        try:
            return data.num_graphs
        except AttributeError:
            return 1

    def log_parameter_counts(self):
        """Call once after init to push param counts to wandb config."""
        counts = _count_parameters(self.student)
        self.hparams["student_params_total"]     = counts["total"]
        self.hparams["student_params_trainable"] = counts["trainable"]
        return counts

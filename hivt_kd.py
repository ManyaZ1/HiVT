"""
hivt_kd.py

HiVTKD — a LightningModule that wraps HiVT-64 (student) with the
output-distribution Knowledge Distillation loss against HiVT-128 teacher
soft targets that were pre-saved by save_teacher_outputs.py.

Drop-in replacement for HiVT in train_student_kd.py.
"""

import argparse
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hivt import HiVT          # your existing HiVT LightningModule
from kd_loss import HiVTKDLoss        # the file we wrote above


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
        --lambda_task   weight for the GT NLL loss          (default 1.0)
        --lambda_kl     weight for the Laplace KL term      (default 0.5)
        --lambda_pi     weight for the mode CE term         (default 0.5)
        --teacher_dir   directory with saved teacher .pt files
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Pull KD-specific kwargs before forwarding the rest to HiVT
        self.lambda_task  = kwargs.pop("lambda_task",  1.0)
        self.lambda_kl    = kwargs.pop("lambda_kl",    0.5)
        self.lambda_pi    = kwargs.pop("lambda_pi",    0.5)
        self.teacher_dir  = kwargs.pop("teacher_dir",  None)

        # Save ALL hparams (including the ones we popped) for wandb / ckpt
        self.save_hyperparameters()

        # Inner student model — embed_dim should be 64
        self.student = HiVT(**kwargs)

        # KD loss module
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
        pred_s, pi_s = self.student(data)          # [F,N,H,4], [N,F]

        # ---- GT task loss (same computation as base HiVT.training_step) ----
        reg_mask = ~data['padding_mask'][:, self.student.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(pred_s[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
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
        kd_logs     = {}

        if has_teacher:
            agent_index = data.agent_index
            if isinstance(agent_index, torch.Tensor) and agent_index.dim() == 0:
                agent_index = agent_index.unsqueeze(0)

            # Teacher cache stores one distribution per scene's focal agent.
            # Match that by selecting the student focal-agent predictions.
            student_loc = pred_s[:, agent_index, :, :2]
            student_scale = pred_s[:, agent_index, :, 2:]
            student_pi = pi_s[agent_index]
            if student_loc.dim() == 3:
                student_loc = student_loc.unsqueeze(0)
                student_scale = student_scale.unsqueeze(0)
                student_pi = student_pi.unsqueeze(0)
            else:
                student_loc = student_loc.permute(1, 0, 2, 3).contiguous()
                student_scale = student_scale.permute(1, 0, 2, 3).contiguous()

            # Teacher targets — detach: no gradient flows back to teacher files
            loc_t = data.teacher_loc.to(self.device)
            scale_t = data.teacher_scale.to(self.device)
            pi_t = data.teacher_pi.to(self.device)

            # Align teacher tensors to student focal-agent layout [B, F, H, 2] and [B, F].
            target_b, target_f, target_h, target_xy = student_loc.shape

            if loc_t.dim() == 3:
                if loc_t.size(0) == target_b * target_f and loc_t.size(1) == target_h and loc_t.size(2) == target_xy:
                    loc_t = loc_t.view(target_b, target_f, target_h, target_xy)
                    scale_t = scale_t.view(target_b, target_f, target_h, target_xy)
                elif target_b == 1 and loc_t.size(0) == target_f:
                    loc_t = loc_t.unsqueeze(0)
                    scale_t = scale_t.unsqueeze(0)
                else:
                    raise RuntimeError(
                        f"Unexpected teacher tensor shape: loc={tuple(loc_t.shape)}; "
                        f"expected [B,F,H,2] with B={target_b}, F={target_f}, H={target_h}."
                    )
            elif loc_t.dim() == 4 and loc_t.size(0) == target_f and loc_t.size(1) == target_b:
                loc_t = loc_t.permute(1, 0, 2, 3).contiguous()
                scale_t = scale_t.permute(1, 0, 2, 3).contiguous()

            if pi_t.dim() == 1:
                if pi_t.numel() == target_b * target_f:
                    pi_t = pi_t.view(target_b, target_f)
                elif target_b == 1 and pi_t.numel() == target_f:
                    pi_t = pi_t.unsqueeze(0)
                else:
                    raise RuntimeError(
                        f"Unexpected teacher pi shape: pi={tuple(pi_t.shape)}; "
                        f"expected [B,F] with B={target_b}, F={target_f}."
                    )
            elif pi_t.dim() == 2 and pi_t.size(0) == target_f and pi_t.size(1) == target_b:
                pi_t = pi_t.t().contiguous()

            kd_loss, kd_logs = self.kd_loss_fn(
                student_loc, student_scale, student_pi,
                loc_t.detach(), scale_t.detach(), pi_t.detach(),
            )
            total_loss = self.lambda_task * task_loss + kd_loss
        else:
            total_loss = self.lambda_task * task_loss

        # ---- Logging ----
        self.log("train/reg_loss", reg_loss, on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        self.log("train/cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        self.log("train/loss_task",  task_loss,  on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        for k, v in kd_logs.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=self._batch_size(data))

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
        return self.student.configure_optimizers()

    # ---------------------------------------------------------------------- #
    # Argparse — extend HiVT's own args with KD-specific ones
    # ---------------------------------------------------------------------- #
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        # First add all of HiVT's args
        parser = HiVT.add_model_specific_args(parent_parser)
        # Then add KD-specific args
        parser.add_argument("--lambda_task",  type=float, default=1.0,
                            help="Weight for the GT NLL task loss")
        parser.add_argument("--lambda_kl",    type=float, default=0.5,
                            help="Weight for the per-mode Laplace KL term")
        parser.add_argument("--lambda_pi",    type=float, default=0.5,
                            help="Weight for the mode probability CE term")
        parser.add_argument("--teacher_dir",  type=str,   required=True,
                            help="Directory containing teacher .pt files "
                                 "(output of save_teacher_outputs.py)")
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
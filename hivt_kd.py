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
        loc_s   = pred_s[..., :2]                  # [F, N, H, 2]
        scale_s = pred_s[..., 2:]                  # [F, N, H, 2]

        # ---- GT task loss (same NLL the base HiVT uses) ----
        task_loss = self.student.training_step_loss(pred_s, pi_s, data)

        # ---- KD loss (only when teacher targets are present) ----
        has_teacher = getattr(data, "has_teacher", False)
        kd_logs     = {}

        if has_teacher:
            # Teacher targets — detach: no gradient flows back to teacher files
            loc_t   = data.teacher_loc.to(self.device)    # [F, N, H, 2]  (after collation)
            scale_t = data.teacher_scale.to(self.device)  # [F, N, H, 2]
            pi_t    = data.teacher_pi.to(self.device)     # [N, F]

            kd_loss, kd_logs = self.kd_loss_fn(
                loc_s, scale_s, pi_s,
                loc_t.detach(), scale_t.detach(), pi_t.detach(),
            )
            total_loss = self.lambda_task * task_loss + kd_loss
        else:
            total_loss = self.lambda_task * task_loss

        # ---- Logging ----
        self.log("train/loss_task",  task_loss,  on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, batch_size=self._batch_size(data))
        for k, v in kd_logs.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=self._batch_size(data))

        return total_loss

    # ---------------------------------------------------------------------- #
    # Validation step — identical to base HiVT (no KD at val time)
    # ---------------------------------------------------------------------- #
    def validation_step(self, data, batch_idx):
        # Delegate fully to the student's own validation logic
        return self.student.validation_step(data, batch_idx)

    def validation_epoch_end(self, outputs):
        # Aggregate and log the same metrics as base HiVT
        return self.student.validation_epoch_end(outputs)

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
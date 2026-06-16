#!/usr/bin/env bash
# Phase A — LR calibration triage for the emb32 student.
#
# Goal: find the right peak LR for batch_size=128 (HiVT's default lr=5e-4 was
# tuned for bs=32, so it is ~4x too small here). We do NOT need full runs to
# RANK learning rates — cheap proxy runs establish the ordering.
#
# Proxy setting (≈6% of a full run → ~1h each instead of ~20h):
#   - 25% of train + 25% of val  (deterministic, seeded → identical data per LR)
#   - 15 epochs, T_max=15        (cosine fully anneals within the short horizon)
#   - lambda_kl=0, no teacher    (isolate optimization; skips cache validation)
#   - bf16                       (matches the baseline + 3x faster)
#
# Read the result as a RANKING, not a final number. Eliminate clearly-bad LRs
# (too-low = val loss still steeply descending at epoch 15 = undertrained;
#  too-high = early instability / high plateau). Confirm the winner on full
# data later. Runs are serial (single GPU).
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

for LR in 5e-4 1e-3 2e-3; do
  RUN_NAME="triage-emb32-lr${LR}-kl0"
  echo "=================================================================="
  echo "[triage] launching ${RUN_NAME}  (lr=${LR})"
  echo "=================================================================="
  python -m KD.train_student_kd \
    --root "$ROOT" \
    --embed_dim 32 \
    --batch_size 128 \
    --lr "$LR" \
    --lambda_kl 0.0 \
    --train_subsample 0.25 \
    --val_subsample 0.25 \
    --max_epochs 15 \
    --T_max 15 \
    --precision bf16 \
    --gpus 1 \
    --num_workers 8 \
    --checkpoint_every_n_epochs 50 \
    --wandb_project hivt-kd \
    --wandb_group triage-lr-emb32 \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[triage] all LR runs done. Compare val_minFDE in wandb group 'triage-lr-emb32'."

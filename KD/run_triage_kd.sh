#!/usr/bin/env bash
# Phase B — does KD help? Fixed LR (winner from Phase A = 3e-3), sweep lambda_kl.
#
# The no-KD baseline already exists from Phase A: triage-emb32-lr3e-3-kl0
# (lambda_kl=0 => teacher contributes zero gradient, so it is a valid control).
# Here we only run the KD-ON arms and compare against that baseline.
#
# Same proxy as Phase A so it stays cheap and the comparison is controlled:
#   25% train + 25% val (same seeds), 15 epochs, T_max=15, bf16, lr=3e-3.
# The ONLY thing that changes vs the kl=0 baseline is the teacher + lambda_kl.
#
# Read as a dose-response: kl=0 (baseline) vs 0.5 vs 1.0. If KD lowers
# val_minFDE/minADE, promote the best kl to the full-data run. Serial (1 GPU).
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
TEACHER=/home/manya/HiVT/teacher_outputs/train_fix.h5   # verified-good cache
LR=3e-3
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

for KL in 0.0 0.5 1.0; do
  RUN_NAME="triage-emb32-lr${LR}-kl${KL}"
  echo "=================================================================="
  echo "[triage-kd] launching ${RUN_NAME}  (lr=${LR}, lambda_kl=${KL})"
  echo "=================================================================="
  python -m KD.train_student_kd \
    --root "$ROOT" \
    --teacher_dir "$TEACHER" \
    --embed_dim 32 \
    --batch_size 128 \
    --lr "$LR" \
    --lambda_kl "$KL" \
    --train_subsample 0.25 \
    --val_subsample 0.25 \
    --max_epochs 15 \
    --T_max 15 \
    --precision bf16 \
    --gpus 1 \
    --num_workers 8 \
    --checkpoint_every_n_epochs 50 \
    --wandb_project hivt-kd \
    --wandb_group triage-kd-emb32-lr3e-3 \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[triage-kd] done. Compare these vs triage-emb32-lr3e-3-kl0 (the kl=0 baseline)."

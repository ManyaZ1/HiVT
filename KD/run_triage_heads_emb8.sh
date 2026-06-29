#!/usr/bin/env bash
# Phase A.2 — ATTENTION-HEAD triage for the emb8 student.
#
# Why: emb16 confirmed head_dim (= embed_dim / num_heads), not head COUNT, sets
# the representational floor. emb16 winner = nh2 -> head_dim 8; emb16 nh8 ->
# head_dim 2 was degenerate (stuck ~3.28). At emb8 the head_dim map shifts:
#   num_heads=1 -> head_dim 8  (preserves the emb16 winning head_dim -> prime)
#   num_heads=2 -> head_dim 4  (untested middle ground; "reuse 2 heads")
#   num_heads=4 -> head_dim 2  (known-degenerate regime -> SKIPPED)
#
# Question this answers: does head_dim 8 (nh1) still clear the floor at half the
# width, and is head_dim 4 (nh2) still viable or already sliding degenerate?
#
# Proxy (identical to run_triage_heads_emb16.sh so results are comparable):
#   - 25% train + 25% val, 15 epochs, T_max=15, lambda_kl=0, bf16
#
# Read as a RANKING. Confirm the winner on FULL data (64 ep) before the KD leg.
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

LR=${LR:-1e-3}                 # fixed at the emb16 lr-triage winner
HEADS=${HEADS:-"1 2"}          # head_dim 8 / 4  (nh4 -> head_dim 2 skipped: degenerate)

for NH in $HEADS; do
  RUN_NAME="triage-emb8-nh${NH}-lr${LR}-kl0"
  echo "=================================================================="
  echo "[triage] launching ${RUN_NAME}  (num_heads=${NH}, head_dim=$((8/NH)))"
  echo "=================================================================="
  python -m KD.train_student_kd \
    --root "$ROOT" \
    --embed_dim 8 \
    --num_heads "$NH" \
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
    --wandb_group triage-heads-emb8 \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[triage] all head runs done. Compare val_minFDE in wandb group 'triage-heads-emb8'."

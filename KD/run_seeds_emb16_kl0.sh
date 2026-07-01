#!/usr/bin/env bash
# Variance / robustness study — emb16 student, NO-KD (kl0.0) baseline.
#
# Why: emb16 no-KD is seed-unstable. Two seeds so far split bimodally:
#   default seed -> 1.61 (good basin)
#   seed 1       -> 3.24 (collapsed into the degenerate head_dim floor; confirmed
#                   wedged — flat 3.24 from epoch 26 through 34)
# Meanwhile all THREE KD (kl1.0) seeds converged tightly: 1.24 / 1.21 / 1.22.
# This run adds no-KD seeds 2,3,4 so we can quote a COLLAPSE RATE for no-KD
# (vs 0/3 for KD), turning "one unlucky seed" into a robustness result.
#
# Config is IDENTICAL to the existing no-KD runs (embed_dim 16, num_heads 2,
# lr 1e-3, lambda_kl 0.0, bs 128, full data, 64 epochs) — only --seed varies.
# Seeds 1 & 3 are shared with the KD runs for a paired comparison.
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

SEEDS=${SEEDS:-"2 3 4"}

for S in $SEEDS; do
  RUN_NAME="emb16-bs128-nh2-lr1e-3-kl0.0-full-s${S}"
  echo "=================================================================="
  echo "[seeds] launching ${RUN_NAME}  (seed=${S})"
  echo "=================================================================="
  python -m KD.train_student_kd \
    --root "$ROOT" \
    --embed_dim 16 \
    --num_heads 2 \
    --batch_size 128 \
    --lr 1e-3 \
    --lambda_kl 0.0 \
    --max_epochs 64 \
    --T_max 64 \
    --precision bf16 \
    --gpus 1 \
    --num_workers 4 \
    --checkpoint_every_n_epochs 20 \
    --seed "$S" \
    --wandb_project hivt-kd \
    --wandb_group emb16-seeds \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[seeds] all no-KD seed runs done. Compare best val_minFDE in wandb group 'emb16-seeds'."

#!/usr/bin/env bash
# Phase A.2 — ATTENTION-HEAD triage for the emb16 student.
#
# Why: the full-data emb16 kl=0 baseline plateaus at val_minFDE ~3.28 and is
# INSENSITIVE to lr (1e-3 -> 3.289, 3e-3 -> 3.280: identical, both flat from
# ~few-k steps). emb32 sits at ~1.37. A floor that is hit early and unchanged by
# lr is a REPRESENTATIONAL ceiling, not an optimization problem. The remaining
# structural difference vs emb32 is head_dim: emb16 with num_heads=8 -> head_dim
# = 2 (near-degenerate 2-D attention). This sweep tests whether wider heads break
# the floor.
#
# Variable: num_heads only (lr fixed at the lr-triage outcome 1e-3).
#   num_heads=8 -> head_dim 2  (current floor; anchor)
#   num_heads=4 -> head_dim 4  (matches the emb32 student)
#   num_heads=2 -> head_dim 8  (matches HiVT-64)
#
# Proxy (identical to run_triage_lr*.sh so results are comparable):
#   - 25% train + 25% val, 15 epochs, T_max=15, lambda_kl=0, bf16
#   - reaches ~6k steps, where full-data emb16 is ALREADY flat at 3.3, so the
#     proxy is in the plateau regime and reveals whether head_dim lifts the floor.
#
# Read as a RANKING. If head_dim=4/8 drop well below ~3.3 here, confirm the
# winner on FULL data (64 ep) before the KD leg. If NONE beat 3.3, the floor is
# not the heads and the next suspect is width itself / numerics.
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

LR=${LR:-1e-3}                 # fixed at the lr-triage winner
HEADS=${HEADS:-"8 4 2"}        # head_dim 2 / 4 / 8

for NH in $HEADS; do
  RUN_NAME="triage-emb16-nh${NH}-lr${LR}-kl0"
  echo "=================================================================="
  echo "[triage] launching ${RUN_NAME}  (num_heads=${NH}, head_dim=$((16/NH)))"
  echo "=================================================================="
  python -m KD.train_student_kd \
    --root "$ROOT" \
    --embed_dim 16 \
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
    --wandb_group triage-heads-emb16 \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[triage] all head runs done. Compare val_minFDE in wandb group 'triage-heads-emb16'."

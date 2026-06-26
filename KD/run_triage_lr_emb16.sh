#!/usr/bin/env bash
# Phase A — LR calibration triage for the emb16 student.
#
# Why: the full-data emb16 kl=0 baseline trained with the emb32-tuned lr=3e-3
# converged to a BAD basin — val_minFDE plateaued at 3.22 (vs 1.157 at emb32:
# 2.8x worse; minMR 0.45 vs 0.12 = 3.7x worse). That is not capacity scaling
# (32->16 should cost ~10-25%), it is a wrong-LR signature. lr=3e-3 was triaged
# for emb32 and there is no reason that optimum transfers to a 4x smaller model.
#
# Goal: RANK learning rates for emb16, sweeping LOWER than 3e-3 (too-hot suspect)
# with 3e-3 included as the known-bad anchor. num_heads=8 (head_dim=2) kept to
# match the failed baseline, so LR is the ONLY variable.
#
# Proxy setting (identical to run_triage_lr.sh so results are comparable):
#   - 25% train + 25% val  (deterministic, seeded -> identical data per LR)
#   - 15 epochs, T_max=15  (cosine fully anneals within the short horizon)
#   - lambda_kl=0, no teacher; bf16
#
# Read as a RANKING, not a final number. Eliminate clearly-bad LRs (too-low =
# val loss still steeply descending at ep15 = undertrained; too-high = early
# instability / high plateau). Confirm the winner on FULL data before the KD leg.
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

# Sweep lower; 3e-3 anchors the known-bad full-data result in proxy form.
LRS=${LRS:-"5e-4 1e-3 2e-3 3e-3"}

for LR in $LRS; do
  RUN_NAME="triage-emb16-lr${LR}-kl0"
  echo "=================================================================="
  echo "[triage] launching ${RUN_NAME}  (lr=${LR})"
  echo "=================================================================="
  python -m KD.train_student_kd \
    --root "$ROOT" \
    --embed_dim 16 \
    --num_heads 8 \
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
    --wandb_group triage-lr-emb16 \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[triage] all LR runs done. Compare val_minFDE in wandb group 'triage-lr-emb16'."

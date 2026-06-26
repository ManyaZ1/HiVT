#!/usr/bin/env bash
# Phase D — KD lambda_kl sweep for the HiVT-16 student (v2 distribution-matching).
#
# Adapts run_kd_sweep.sh to the emb16 recipe established by the lr + heads
# triages: embed_dim=16, num_heads=2 (head_dim=8 — escapes the head_dim=2 floor),
# lr=1e-3. Teacher = HiVT-128 (train_fix.h5, verified cache). kd_mode=dist (v2),
# because at emb32 v1 collapsed calibration (mixNLL +41%, calib_err 5x) and v2 is
# what fixed it — so a calibration-aware sweep MUST use dist.
#
# Goal: RANK lambda_kl for emb16. We probe UPWARD past emb32's 0.5 because smaller
# students often absorb more distillation before saturating — watch whether
# geometry (minFDE/minMR) keeps improving while calibration (calib_err, mixNLL,
# b_scale) stays healthy, or starts to crack.
#
# Proxy (same as every triage so far): 25% data, 15 epochs, T_max=15, bf16.
# kl=0.0 is included as the in-batch anchor (full-data baseline = 1.608 minFDE).
#
# Read as a RANKING. Confirm the winning kl on FULL data (64 ep) vs the existing
# kl=0.0 full baseline before trusting absolute numbers — proxy ranks, it does
# not give finals (lesson from the lr round).
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
TEACHER=/home/manya/HiVT/teacher_outputs/train_fix.h5   # verified-good cache
EMBED=16
NUM_HEADS=2                                              # head_dim=8 (triage winner)
LR=${LR:-1e-3}                                           # emb16 lr-triage winner
KD_MODE=${KD_MODE:-dist}                                 # v2 distribution-matching
KD_NSAMPLES=${KD_NSAMPLES:-8}
KLS=${KLS:-"0.0 0.5 1.0 2.0"}                            # probe upward past emb32's 0.5
SUFFIX=${SUFFIX:--distv2}
GROUP=${GROUP:-triage-kd-emb16-lr${LR}${SUFFIX}}
EPOCHS=${EPOCHS:-15}                                     # also T_max
SUB=${SUB:-0.25}
NUM_WORKERS=${NUM_WORKERS:-8}
CKPT_EVERY=${CKPT_EVERY:-50}
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

SUB_ARGS=()
if [[ -n "$SUB" && "$SUB" != "1" && "$SUB" != "full" ]]; then
  SUB_ARGS=(--train_subsample "$SUB" --val_subsample "$SUB")
fi

echo "[kd-sweep-emb16] LR=$LR KLS=$KLS kd_mode=$KD_MODE nh=$NUM_HEADS epochs=$EPOCHS sub=${SUB:-full} group=$GROUP"
for KL in $KLS; do
  RUN_NAME="triage-emb16-nh${NUM_HEADS}-lr${LR}-kl${KL}${SUFFIX}"
  echo "=================================================================="
  echo "[kd-sweep-emb16] launching ${RUN_NAME}  (lambda_kl=${KL})"
  echo "=================================================================="
  python -m KD.train_student_kd \
    --root "$ROOT" \
    --teacher_dir "$TEACHER" \
    --embed_dim "$EMBED" \
    --num_heads "$NUM_HEADS" \
    --batch_size 128 \
    --lr "$LR" \
    --lambda_kl "$KL" \
    --kd_mode "$KD_MODE" \
    --kd_n_samples "$KD_NSAMPLES" \
    "${SUB_ARGS[@]}" \
    --max_epochs "$EPOCHS" \
    --T_max "$EPOCHS" \
    --precision bf16 \
    --gpus 1 \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_every_n_epochs "$CKPT_EVERY" \
    --wandb_project hivt-kd \
    --wandb_group "$GROUP" \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[kd-sweep-emb16] done. Group '$GROUP' — rank by val_minFDE, watch calib_err/mixNLL."

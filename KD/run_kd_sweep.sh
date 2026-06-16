#!/usr/bin/env bash
# Parameterized KD lambda_kl sweep (supersedes run_triage_kd.sh).
#
# Defaults reproduce the Phase-B proxy (25% data, 15 epochs, lr=3e-3, bf16) but
# in a SEPARATE wandb group with a run-name SUFFIX, so re-running never collides
# with — or resumes — existing runs. Your previous oracle-only graphs stay intact
# in their original group; this sweep lands cleanly in its own group with the new
# calibration metrics (val_brier_minFDE/ADE, val_mixNLL).
#
# Override any knob from the env, e.g.:
#   SUFFIX=-cal2 GROUP=my-group LR=2e-3 KLS="0.0 0.5" bash KD/run_kd_sweep.sh
set -euo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
TEACHER=/home/manya/HiVT/teacher_outputs/train_fix.h5   # verified-good cache
LR=${LR:-3e-3}
KLS=${KLS:-"0.0 0.5 1.0"}
SUFFIX=${SUFFIX:--cal}                                   # appended to every run_name
GROUP=${GROUP:-triage-kd-emb32-lr${LR}${SUFFIX}}
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

echo "[kd-sweep] LR=$LR  KLS=$KLS  group=$GROUP  suffix=$SUFFIX"
for KL in $KLS; do
  RUN_NAME="triage-emb32-lr${LR}-kl${KL}${SUFFIX}"
  echo "=================================================================="
  echo "[kd-sweep] launching ${RUN_NAME}  (lr=${LR}, lambda_kl=${KL})"
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
    --wandb_group "$GROUP" \
    --run_name "$RUN_NAME" \
    2>&1 | tee "${LOGDIR}/${RUN_NAME}.log"
done

echo "[kd-sweep] done. Group '$GROUP' has kl in {$KLS} with brier + mixNLL metrics."

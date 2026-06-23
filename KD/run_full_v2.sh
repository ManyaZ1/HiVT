#!/usr/bin/env bash
# Full-data v2 (distribution-matching KD) training launcher.
#
# Companion to the EXISTING full-data v1 run kept at:
#     kd_ckpt/emb32-bs128-lkl0.5/         (v1 mean-target, lambda_kl=0.5)
#     kd_ckpt/emb32-bs128-lkl0.0/         (lambda_kl=0 baseline)
# This produces the matching v2 leg so the full-data baseline/v1/v2 triplet can
# be compared at the same nominal lambda=0.5. (Re-score the two existing ckpts
# with the new calibration metrics via KD.eval_lambda_sweep — no retrain needed.)
#
# Run names are emb32-bs128-lkl<KL>-distv2-full so they NEVER collide with the
# triage runs (triage-emb32-...-distv2) or the v1 full run (emb32-bs128-lkl0.5).
#
# Same safety harness as run_unattended.sh: memory watchdog + GPU-glitch
# auto-restart (resumes from last.ckpt) with a retry cap.
#
#   bash KD/run_full_v2.sh                 # v2 dist full-data, kl=0.5, 64 epochs
#   KLS="0.5 1.0" bash KD/run_full_v2.sh   # also do kl=1.0 (sequential)
set -uo pipefail

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
TEACHER=teacher_outputs/train_fix.h5      # verified-good cache (train_fix.h5)
LR=${LR:-3e-3}
KD_MODE=${KD_MODE:-dist}                   # dist = v2 distribution-matching
KD_NSAMPLES=${KD_NSAMPLES:-8}
KLS=${KLS:-"0.5"}
EPOCHS=${EPOCHS:-64}                        # matches the v1 full run horizon
NUM_WORKERS=${NUM_WORKERS:-4}              # memory-tight host
MEM_ABORT=${MEM_ABORT:-9.0}
MAX_RETRY=${MAX_RETRY:-6}
CKPT_EVERY=${CKPT_EVERY:-20}
GROUP=${GROUP:-full-emb32-distv2}
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

run_once() {
  local kl="$1" run="$2" log="$3"
  python -m KD.train_student_kd \
    --root "$ROOT" --teacher_dir "$TEACHER" \
    --embed_dim 32 --batch_size 128 --lr "$LR" \
    --kd_mode "$KD_MODE" --kd_n_samples "$KD_NSAMPLES" --lambda_kl "$kl" \
    --max_epochs "$EPOCHS" --T_max "$EPOCHS" \
    --precision bf16 --gpus 1 --num_workers "$NUM_WORKERS" \
    --checkpoint_every_n_epochs "$CKPT_EVERY" \
    --wandb_project hivt-kd --wandb_group "$GROUP" --run_name "$run" \
    >> "$log" 2>&1 &
  local pid=$!
  bash KD/mem_watchdog.sh "$pid" "$MEM_ABORT" >> "$log" 2>&1 &
  local wd=$!
  wait "$pid"; local rc=$?
  kill "$wd" 2>/dev/null
  return "$rc"
}

echo "[full-v2] mode=$KD_MODE KLS=$KLS epochs=$EPOCHS workers=$NUM_WORKERS mem_abort=${MEM_ABORT}GB group=$GROUP"
for KL in $KLS; do
  RUN="emb32-bs128-lkl${KL}-distv2-full"
  LOG="${LOGDIR}/${RUN}.log"
  echo "================= $RUN ================="
  n=0
  until run_once "$KL" "$RUN" "$LOG"; do
    n=$((n+1))
    if [ "$n" -ge "$MAX_RETRY" ]; then
      echo "[full-v2] $RUN failed ${MAX_RETRY}x — giving up, next kl"; break
    fi
    echo "[full-v2] $RUN died (retry $n/$MAX_RETRY in 30s; resumes from last.ckpt)"
    sleep 30
  done
  echo "[full-v2] $RUN finished (or gave up). log: $LOG"
done
echo "[full-v2] ALL DONE — group $GROUP"
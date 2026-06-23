#!/usr/bin/env bash
# Unattended KD triage launcher — "start in tmux and walk away" safe.
#
# For each lambda it trains train_student_kd with:
#   * a MEMORY WATCHDOG attached (KD/mem_watchdog.sh) so a runaway can't freeze
#     the host (kills the run inside WSL instead),
#   * GPU-glitch AUTO-RESTART: if the run dies non-zero (e.g. the dxgk/PCIe
#     hiccup), it retries and train_student_kd auto-resumes from last.ckpt,
#   * a retry CAP so a genuinely broken run (or repeated mem-kill) can't loop
#     forever — it logs and moves to the next lambda.
#
# Defaults = v2 (distribution-matching) triage on the 25%/15ep proxy. Override
# any knob from the env. Examples:
#   bash KD/run_unattended.sh                         # v2 dist, KLS="0.5 1.0"
#   KD_MODE=mean KLS="0.25" SUB=1 EPOCHS=64 bash KD/run_unattended.sh   # v1 full-data 0.25
set -uo pipefail   # NOT -e: failures are handled per-iteration by the retry loop

source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT

ROOT=/home/manya/argoverse
TEACHER=teacher_outputs/train_fix.h5
LR=${LR:-3e-3}
KD_MODE=${KD_MODE:-dist}                 # dist = v2 (this script's purpose); mean = v1
KD_NSAMPLES=${KD_NSAMPLES:-8}
KLS=${KLS:-"0.5 1.0"}
EPOCHS=${EPOCHS:-15}
SUB=${SUB:-0.25}
NUM_WORKERS=${NUM_WORKERS:-4}            # memory-tight host: keep modest
MEM_ABORT=${MEM_ABORT:-9.0}             # watchdog: kill if WSL used RAM exceeds this (GB)
MAX_RETRY=${MAX_RETRY:-4}
_DEF_SUFFIX=-cal; [[ "$KD_MODE" == dist ]] && _DEF_SUFFIX=-distv2
SUFFIX=${SUFFIX:-$_DEF_SUFFIX}
GROUP=${GROUP:-triage-emb32-lr${LR}${SUFFIX}}
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

SUB_ARGS=()
if [[ -n "$SUB" && "$SUB" != "1" && "$SUB" != "full" ]]; then
  SUB_ARGS=(--train_subsample "$SUB" --val_subsample "$SUB")
fi

# Train one lambda once (bg) under the memory watchdog; returns the trainer's RC.
run_once() {
  local kl="$1" run="$2" log="$3"
  python -m KD.train_student_kd \
    --root "$ROOT" --teacher_dir "$TEACHER" \
    --embed_dim 32 --batch_size 128 --lr "$LR" \
    --kd_mode "$KD_MODE" --kd_n_samples "$KD_NSAMPLES" --lambda_kl "$kl" \
    "${SUB_ARGS[@]}" \
    --max_epochs "$EPOCHS" --T_max "$EPOCHS" \
    --precision bf16 --gpus 1 --num_workers "$NUM_WORKERS" \
    --checkpoint_every_n_epochs 50 \
    --wandb_project hivt-kd --wandb_group "$GROUP" --run_name "$run" \
    >> "$log" 2>&1 &
  local pid=$!
  bash KD/mem_watchdog.sh "$pid" "$MEM_ABORT" >> "$log" 2>&1 &
  local wd=$!
  wait "$pid"; local rc=$?
  kill "$wd" 2>/dev/null
  return "$rc"
}

echo "[unattended] mode=$KD_MODE KLS=$KLS sub=$SUB epochs=$EPOCHS workers=$NUM_WORKERS mem_abort=${MEM_ABORT}GB group=$GROUP"
for KL in $KLS; do
  RUN="triage-emb32-lr${LR}-kl${KL}${SUFFIX}"
  LOG="${LOGDIR}/${RUN}.log"
  echo "================= $RUN ================="
  n=0
  until run_once "$KL" "$RUN" "$LOG"; do
    n=$((n+1))
    if [ "$n" -ge "$MAX_RETRY" ]; then
      echo "[unattended] $RUN failed ${MAX_RETRY}x — giving up, next lambda"; break
    fi
    echo "[unattended] $RUN died (retry $n/$MAX_RETRY in 30s; resumes from last.ckpt)"
    sleep 30
  done
  echo "[unattended] $RUN finished (or gave up). log: $LOG"
done
echo "[unattended] ALL DONE — group $GROUP"
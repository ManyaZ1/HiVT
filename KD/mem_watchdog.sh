#!/usr/bin/env bash
# Memory watchdog — last line of defence against a host freeze on this
# memory-tight WSL2 box. Watches a target PID and the WSL guest's used memory;
# if used RAM crosses the abort threshold it SIGTERMs (then SIGKILLs) the target
# so the process dies *inside* WSL instead of pushing Windows into swap thrash.
#
# Usage:
#   KD/mem_watchdog.sh <pid> [abort_used_gb]      # default abort at 9.0 GB
#
# Typical pattern (launch eval in bg, hand its PID to the watchdog):
#   python -m KD.eval_lambda_sweep ... & PID=$!
#   bash KD/mem_watchdog.sh "$PID" 9.0 & WD=$!
#   wait "$PID"; kill "$WD" 2>/dev/null
set -u
PID=${1:?usage: mem_watchdog.sh <pid> [abort_used_gb]}
ABORT_GB=${2:-9.0}

echo "[watchdog] watching PID $PID; will abort if WSL used RAM > ${ABORT_GB} GB"
while kill -0 "$PID" 2>/dev/null; do
  # used = MemTotal - MemAvailable (matches what Windows sees pressure from)
  used_gb=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%.2f",(t-a)/1048576}' /proc/meminfo)
  if awk -v u="$used_gb" -v a="$ABORT_GB" 'BEGIN{exit !(u>a)}'; then
    echo "[watchdog] WSL used ${used_gb} GB > ${ABORT_GB} GB — aborting PID $PID"
    kill -TERM "$PID" 2>/dev/null; sleep 5
    kill -KILL "$PID" 2>/dev/null
    exit 1
  fi
  sleep 2
done
echo "[watchdog] PID $PID exited cleanly (last used ${used_gb:-?} GB); watchdog done"

#!/usr/bin/env bash
# Evaluate the 3 completed HiVT-16 KD (kl1.0) seed checkpoints to extract the
# full metric suite (minFDE/minADE/minMR + calibration: calib_err, mixNLL,
# cov@p90, brier) for a mean +/- spread report. CPU-only (--gpus 0) so it does
# NOT contend with the no-KD seed runs training on the GPU.
set -uo pipefail
source /home/manya/miniconda3/etc/profile.d/conda.sh
conda activate hivt_new
cd /home/manya/HiVT
ROOT=/home/manya/argoverse
LOGDIR=kd_ckpt/_triage_logs
mkdir -p "$LOGDIR"

CKPTS=(
  "kd_ckpt/emb16-bs128-nh2-lr1e-3-kl1.0-full/best/HiVTKD-epoch=49-val_minFDE=1.24.ckpt"
  "kd_ckpt/emb16-bs128-nh2-lr1e-3-kl1.0-full-s1/best/HiVTKD-epoch=56-val_minFDE=1.21.ckpt"
  "kd_ckpt/emb16-bs128-nh2-lr1e-3-kl1.0-full-s3/best/HiVTKD-epoch=58-val_minFDE=1.22.ckpt"
)
TAGS=(kl1.0-full-default kl1.0-full-s1 kl1.0-full-s3)

for i in "${!CKPTS[@]}"; do
  TAG=${TAGS[$i]}
  echo "=================================================================="
  echo "[eval] ${TAG}  ::  ${CKPTS[$i]}"
  echo "=================================================================="
  python eval.py \
    --root "$ROOT" \
    --batch_size 32 \
    --num_workers 4 \
    --gpus 0 \
    --ckpt_path "${CKPTS[$i]}" \
    2>&1 | tee "${LOGDIR}/eval-${TAG}.log"
done
echo "[eval] done. Metrics in ${LOGDIR}/eval-kl1.0-full-*.log"

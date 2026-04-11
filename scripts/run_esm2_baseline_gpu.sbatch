#!/bin/bash
#SBATCH -J esm2_flab
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/zichende/projects/generative-ai-project-share/results/logs/%x_%A.out
#SBATCH -e /home/zichende/projects/generative-ai-project-share/results/logs/%x_%A.err

set -euo pipefail

ROOT=${ROOT:-/home/zichende/projects/generative-ai-project-share}
PYBIN=${PYBIN:-/home/zichende/.conda/envs/afffood/bin/python}
BATCH_SIZE=${BATCH_SIZE:-32}
INCLUDE_WARSZAWSKI=${INCLUDE_WARSZAWSKI:-0}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}

mkdir -p "$ROOT/results/logs"

echo "[START] $(date)"
echo "ROOT=$ROOT"
echo "PYBIN=$PYBIN"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "INCLUDE_WARSZAWSKI=$INCLUDE_WARSZAWSKI"
echo "LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

cd "$ROOT"

cmd=(
  "$PYBIN" "$ROOT/scripts/run_esm2_baseline.py"
  --batch-size "$BATCH_SIZE"
)

if [[ "$INCLUDE_WARSZAWSKI" == "1" ]]; then
  cmd+=(--include-warszawski)
fi

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  cmd+=(--local-files-only)
fi

"${cmd[@]}"

echo "[DONE] $(date)"

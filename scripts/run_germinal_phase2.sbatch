#!/bin/bash
#SBATCH --job-name=germinal_p2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/zichende/projects/generative-ai-project-share/results/logs/germinal_phase2/%x-%j.out
#SBATCH --error=/home/zichende/projects/generative-ai-project-share/results/logs/germinal_phase2/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/zichende/projects/generative-ai-project-share"
STAGE="${STAGE:-validate}"
GERMINAL_RUNTIME="${GERMINAL_RUNTIME:-auto}"

cd "$REPO_ROOT"
echo "Stage: $STAGE"
echo "Runtime: $GERMINAL_RUNTIME"
echo "Host: $(hostname)"

bash scripts/run_germinal_phase2.sh "$STAGE"

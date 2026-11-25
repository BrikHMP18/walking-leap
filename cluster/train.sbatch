#!/bin/bash
# Train a walking-leap task on one H100 of an Alliance Canada cluster (Rorqual).
#
# Slurm needs a job script; this one only sets up the environment and calls
# mjlab's `train` unchanged. One-time setup: README, "Training on an Alliance
# Canada cluster". Submit from the repository root:
#   sbatch cluster/train.sbatch                              # v0, one run, seed 42
#   sbatch --array=0-2 cluster/train.sbatch                  # v0, seeds 42, 43, 44
#   sbatch --export=ALL,TASK=Walking-Leap-Flat-v1 --array=0-2 cluster/train.sbatch
#   sbatch cluster/train.sbatch --agent.max-iterations 10    # extra args go to train
# TASK defaults to Walking-Leap-Flat-v0. Each run is named <task>_s<seed>, in its
# log directory and in W&B. The Slurm account comes from SBATCH_ACCOUNT, the W&B
# entity from WANDB_USERNAME (both exported in ~/.bashrc). W&B runs offline into
# wandb/ (compute nodes have no internet); upload them afterwards from a login
# node with `wandb sync`.
#
#SBATCH --job-name=walking-leap
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export UV_OFFLINE=1 # compute nodes have no internet; uv sync ran on the login node
export WANDB_MODE=offline
# Write progress to the Slurm log as it happens. In the first array job, two of
# three tasks trained normally but their log stopped after wandb.init.
export PYTHONUNBUFFERED=1 WANDB_CONSOLE=off

task="${TASK:-Walking-Leap-Flat-v0}"
seed=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))
echo "commit $(git rev-parse --short HEAD), task $task, seed $seed, node $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run --frozen train "$task" --env.scene.num-envs 4096 \
  --agent.logger wandb --agent.seed "$seed" --agent.run-name "${task}_s${seed}" "$@"

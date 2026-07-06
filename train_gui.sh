#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/forrest/data/repo/sharpa-rl-lab"
CONDA_ENV="sharpa-env"
TASK="${TASK:-Isaac-Inhand-Rotate-Sharpa-Wave-v0}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-}"
CACHE="${CACHE:-}"

export CONDA_NO_PLUGINS=true

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  source /home/forrest/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

cmd=(
  python rl_isaaclab/scripts/train.py
  --task "${TASK}"
  --num_envs "${NUM_ENVS}"
)

if [[ -n "${MAX_AGENT_STEPS}" ]]; then
  cmd+=(--max_agent_steps "${MAX_AGENT_STEPS}")
fi

if [[ -n "${CACHE}" ]]; then
  cmd+=(--cache "${CACHE}")
fi

echo "Running headed training: ${cmd[*]}"
"${cmd[@]}"

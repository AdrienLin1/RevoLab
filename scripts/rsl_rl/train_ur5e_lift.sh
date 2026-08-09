#!/usr/bin/env bash
# Train the UR5e + Revo3 right-hand Dexsuite lift tasks (ported from tactile-revo3).
#
# Usage:
#   ./scripts/rsl_rl/train_ur5e_lift.sh base    [extra train.py args...]
#   ./scripts/rsl_rl/train_ur5e_lift.sh tactile [extra train.py args...]
#
# Examples:
#   ./scripts/rsl_rl/train_ur5e_lift.sh base
#   ./scripts/rsl_rl/train_ur5e_lift.sh tactile --num_envs 256
#   ./scripts/rsl_rl/train_ur5e_lift.sh base --num_envs 64 --max_iterations 2   # smoke test
#
# Run from the repository root with the brain_co conda environment activated.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VARIANT="${1:-base}"
shift || true

case "${VARIANT}" in
  base)
    TASK="BrainCo-Ur5e-Dexsuite-Revo3-Right-Lift-v0"
    DEFAULT_NUM_ENVS=4096
    ;;
  tactile)
    # TacSL force-field sensors are memory-heavy; the reference setup trains with 512 envs.
    TASK="BrainCo-Ur5e-Dexsuite-Revo3-Right-Lift-Tactile-v0"
    DEFAULT_NUM_ENVS=512
    ;;
  *)
    echo "Unknown variant '${VARIANT}'. Use: base | tactile" >&2
    exit 1
    ;;
esac

# Only apply the default env count when the caller did not pass --num_envs.
NUM_ENVS_ARGS=(--num_envs "${DEFAULT_NUM_ENVS}")
for arg in "$@"; do
  if [[ "${arg}" == --num_envs* ]]; then
    NUM_ENVS_ARGS=()
    break
  fi
done

exec python "${REPO_ROOT}/scripts/rsl_rl/train.py" \
  --task "${TASK}" \
  "${NUM_ENVS_ARGS[@]}" \
  --headless \
  "$@"

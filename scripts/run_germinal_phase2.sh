#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: run_germinal_phase2.sh <validate|sync|pdl1_smoke|her2_debug|her2_pilot>" >&2
  exit 1
fi

STAGE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GERMINAL_ROOT="${GERMINAL_ROOT:-$HOME/projects/external/germinal}"
PHASE2_VENV="${PHASE2_VENV:-$HOME/.venvs/germinal_phase2}"
PHASE2_SIF="${PHASE2_SIF:-$HOME/.local/share/germinal_phase2/germinal_latest.sif}"
GERMINAL_RUNTIME="${GERMINAL_RUNTIME:-auto}"
RESULTS_ROOT="$REPO_ROOT/results/germinal_phase2"
RUNS_ROOT="$RESULTS_ROOT/runs"
HANDOFF_ROOT="$RESULTS_ROOT/handoffs"

mkdir -p "$RUNS_ROOT" "$HANDOFF_ROOT" "$REPO_ROOT/results/logs/germinal_phase2"

sync_assets() {
  install -d "$GERMINAL_ROOT/configs/run" "$GERMINAL_ROOT/configs/target" "$GERMINAL_ROOT/pdbs"
  install -m 0644 "$REPO_ROOT/configs/germinal/run/phase2_scfv_pdl1_chai.yaml" \
    "$GERMINAL_ROOT/configs/run/phase2_scfv_pdl1_chai.yaml"
  install -m 0644 "$REPO_ROOT/configs/germinal/run/phase2_scfv_her2_chai.yaml" \
    "$GERMINAL_ROOT/configs/run/phase2_scfv_her2_chai.yaml"
  install -m 0644 "$REPO_ROOT/configs/germinal/run/phase2_scfv_her2_rescue_chai.yaml" \
    "$GERMINAL_ROOT/configs/run/phase2_scfv_her2_rescue_chai.yaml"
  install -m 0644 "$REPO_ROOT/configs/germinal/target/phase2_her2_1n8z.yaml" \
    "$GERMINAL_ROOT/configs/target/phase2_her2_1n8z.yaml"
  install -m 0644 \
    "$REPO_ROOT/data/raw/germinal_targets/her2_trastuzumab_1n8z/1n8z_her2_chainA.pdb" \
    "$GERMINAL_ROOT/pdbs/1n8z_her2_chainA.pdb"
}

venv_ready() {
  [[ -x "$PHASE2_VENV/bin/python" ]] || return 1
  "$PHASE2_VENV/bin/python" - <<'PY' >/dev/null 2>&1
import germinal  # noqa: F401
import jax  # noqa: F401
import pyrosetta  # noqa: F401
import torch  # noqa: F401
PY
}

container_ready() {
  [[ -f "$PHASE2_SIF" ]] || return 1
  [[ -f "$GERMINAL_ROOT/params/DAlphaBall.gcc" ]] || return 1
}

choose_runtime() {
  case "$GERMINAL_RUNTIME" in
    venv)
      venv_ready
      echo "venv"
      ;;
    container)
      container_ready
      echo "container"
      ;;
    auto)
      if venv_ready; then
        echo "venv"
      elif container_ready; then
        echo "container"
      else
        echo "No usable Germinal runtime found. Run scripts/setup_germinal_phase2_env.sh first." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported runtime: $GERMINAL_RUNTIME" >&2
      exit 1
      ;;
  esac
}

run_validate() {
  local runtime="$1"
  if [[ "$runtime" == "venv" ]]; then
    "$PHASE2_VENV/bin/python" "$GERMINAL_ROOT/validate_install.py"
  else
    singularity exec --nv --bind "$GERMINAL_ROOT:/workspace" "$PHASE2_SIF" \
      bash -lc "cd /workspace && python validate_install.py"
  fi
}

run_germinal() {
  local runtime="$1"
  shift
  local args=("$@")
  export HYDRA_FULL_ERROR=1
  export PYTHONUNBUFFERED=1
  export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.5}"
  if [[ "$runtime" == "venv" ]]; then
    (
      cd "$GERMINAL_ROOT"
      "$PHASE2_VENV/bin/python" "$GERMINAL_ROOT/run_germinal.py" "${args[@]}"
    )
  else
    singularity exec --nv --bind "$GERMINAL_ROOT:/workspace" "$PHASE2_SIF" \
      bash -lc "cd /workspace && python run_germinal.py $(printf '%q ' "${args[@]}")"
  fi
}

collect_stage_outputs() {
  local experiment_name="$1"
  local run_config="$2"
  local prefix="$3"
  local run_dir="$RUNS_ROOT/$experiment_name/$run_config"
  python3 "$REPO_ROOT/scripts/collect_germinal_candidates.py" \
    --run-dir "$run_dir" \
    --output-dir "$HANDOFF_ROOT" \
    --output-prefix "$prefix"
}

run_stage() {
  local runtime="$1"
  local run_cfg="$2"
  local target_cfg="$3"
  local filter_initial="$4"
  local filter_final="$5"
  local experiment_name="$6"
  local run_config="$7"
  local max_trajectories="$8"
  local max_hallucinated="$9"
  local max_passing="${10}"
  local prefix="${11}"

  run_germinal "$runtime" \
    run="$run_cfg" \
    target="$target_cfg" \
    "filter/initial=$filter_initial" \
    "filter/final=$filter_final" \
    "project_dir=$REPO_ROOT/results" \
    "results_dir=germinal_phase2/runs" \
    "experiment_name=$experiment_name" \
    "run_config=$run_config" \
    "max_trajectories=$max_trajectories" \
    "max_hallucinated_trajectories=$max_hallucinated" \
    "max_passing_designs=$max_passing"

  collect_stage_outputs "$experiment_name" "$run_config" "$prefix"
}

sync_assets

if [[ "$STAGE" == "sync" ]]; then
  echo "Synced Phase 2 assets into $GERMINAL_ROOT"
  exit 0
fi

RUNTIME="$(choose_runtime)"

case "$STAGE" in
  validate)
    run_validate "$RUNTIME"
    ;;
  pdl1_smoke)
    run_stage "$RUNTIME" \
      phase2_scfv_pdl1_chai \
      pdl1 \
      scfv_pdl1 \
      scfv_pdl1 \
      phase2_pdl1_smoke \
      pdl1_smoke \
      8 \
      8 \
      1 \
      pdl1_smoke
    ;;
  her2_debug)
    run_stage "$RUNTIME" \
      phase2_scfv_her2_chai \
      phase2_her2_1n8z \
      scfv \
      scfv \
      phase2_her2 \
      her2_debug \
      8 \
      8 \
      1 \
      her2_debug
    ;;
  her2_rescue)
    run_stage "$RUNTIME" \
      phase2_scfv_her2_rescue_chai \
      phase2_her2_1n8z \
      scfv \
      scfv \
      phase2_her2 \
      her2_rescue \
      1 \
      1 \
      1 \
      her2_rescue
    ;;
  her2_pilot)
    run_stage "$RUNTIME" \
      phase2_scfv_her2_chai \
      phase2_her2_1n8z \
      scfv \
      scfv \
      phase2_her2 \
      her2_pilot \
      32 \
      32 \
      4 \
      her2_pilot
    ;;
  *)
    echo "Unsupported stage: $STAGE" >&2
    exit 1
    ;;
esac

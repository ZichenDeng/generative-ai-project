#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GERMINAL_ROOT="${GERMINAL_ROOT:-$HOME/projects/external/germinal}"
PHASE2_VENV="${PHASE2_VENV:-$HOME/.venvs/germinal_phase2}"
PHASE2_SIF="${PHASE2_SIF:-$HOME/.local/share/germinal_phase2/germinal_latest.sif}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$HOME/.conda/envs/afffood/bin/python}"
MODE="auto"
SKIP_VALIDATE=0
SKIP_PULL=0

usage() {
  cat <<'EOF'
Usage: setup_germinal_phase2_env.sh [--mode auto|venv|container] [--skip-validate] [--skip-pull]

This helper provisions the Phase 2 Germinal runtime. It prefers a direct venv
install, but it can also pull and bootstrap the public Singularity image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --skip-validate)
      SKIP_VALIDATE=1
      shift
      ;;
    --skip-pull)
      SKIP_PULL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ensure_workspace() {
  if [[ ! -d "$GERMINAL_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$GERMINAL_ROOT")"
    git clone https://github.com/SantiagoMille/germinal.git "$GERMINAL_ROOT"
  fi
}

ensure_venv() {
  if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
    echo "Bootstrap Python not found: $BOOTSTRAP_PYTHON" >&2
    return 1
  fi
  if [[ ! -d "$PHASE2_VENV" ]]; then
    mkdir -p "$(dirname "$PHASE2_VENV")"
    "$BOOTSTRAP_PYTHON" -m venv "$PHASE2_VENV"
  fi
}

ensure_af_params() {
  local params_dir="$GERMINAL_ROOT/params"
  mkdir -p "$params_dir"
  if [[ ! -f "$params_dir/params_model_1_multimer_v3.npz" ]]; then
    (
      cd "$params_dir"
      wget -c https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
      tar -xf alphafold_params_2022-12-06.tar
    )
  fi
}

install_venv_runtime() {
  ensure_workspace
  ensure_venv

  "$PHASE2_VENV/bin/python" -m pip install --upgrade pip setuptools wheel uv
  "$PHASE2_VENV/bin/python" -m uv pip install \
    pandas matplotlib numpy biopython scipy seaborn tqdm ffmpeg py3dmol \
    chex dm-haiku dm-tree joblib ml-collections immutabledict optax cvxopt mdtraj \
    colabfold ipsae==1.0.1 filelock hydra-core omegaconf

  "$PHASE2_VENV/bin/python" -m uv pip install -e "$GERMINAL_ROOT/colabdesign"
  "$PHASE2_VENV/bin/python" -m uv pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  "$PHASE2_VENV/bin/python" -m uv pip install \
    torchtyping==0.1.5 \
    "torch_geometric==2.6.*" \
    -f "https://data.pyg.org/whl/torch-2.6.0%2Bcu124.html"
  "$PHASE2_VENV/bin/python" -m uv pip install iglm chai-lab==0.6.1
  "$PHASE2_VENV/bin/python" -m uv pip install jax==0.5.3 dm-haiku==0.0.13
  "$PHASE2_VENV/bin/python" -m uv pip install \
    "jax[cuda12_pip]==0.5.3" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  "$PHASE2_VENV/bin/python" -m uv pip install ablang2==0.2.1 --no-deps
  "$PHASE2_VENV/bin/python" -m uv pip install rotary_embedding_torch==0.8.9 --no-deps
  "$PHASE2_VENV/bin/python" -m uv pip install pyrosetta-installer
  PATH="$PHASE2_VENV/bin:$PATH" \
    "$PHASE2_VENV/bin/python" -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
  "$PHASE2_VENV/bin/python" -m uv pip install -e "$GERMINAL_ROOT"
  ensure_af_params
}

pull_container_runtime() {
  mkdir -p "$(dirname "$PHASE2_SIF")"
  if [[ ! -f "$PHASE2_SIF" && "$SKIP_PULL" -eq 0 ]]; then
    singularity pull "$PHASE2_SIF" docker://jwang003/germinal:latest
  fi
}

bootstrap_workspace_from_image() {
  ensure_workspace
  if [[ ! -f "$PHASE2_SIF" ]]; then
    echo "Singularity image not found: $PHASE2_SIF" >&2
    return 1
  fi

  if [[ ! -f "$GERMINAL_ROOT/params/DAlphaBall.gcc" ]]; then
    rm -rf "$GERMINAL_ROOT/params"
    singularity exec "$PHASE2_SIF" bash -lc "cd /workspace && tar -cf - params" | tar -C "$GERMINAL_ROOT" -xf -
  fi
}

validate_venv_runtime() {
  "$PHASE2_VENV/bin/python" -c "import pyrosetta, torch, jax, germinal; print('venv runtime validated')"
  "$PHASE2_VENV/bin/python" "$GERMINAL_ROOT/validate_install.py"
}

validate_container_runtime() {
  singularity exec --nv --bind "$GERMINAL_ROOT:/workspace" "$PHASE2_SIF" bash -lc \
    "cd /workspace && python -c 'import pyrosetta, torch, jax, germinal; print(\"container runtime validated\")' && python validate_install.py"
}

main() {
  case "$MODE" in
    venv)
      install_venv_runtime
      [[ "$SKIP_VALIDATE" -eq 1 ]] || validate_venv_runtime
      ;;
    container)
      pull_container_runtime
      bootstrap_workspace_from_image
      [[ "$SKIP_VALIDATE" -eq 1 ]] || validate_container_runtime
      ;;
    auto)
      if install_venv_runtime; then
        [[ "$SKIP_VALIDATE" -eq 1 ]] || validate_venv_runtime
      else
        echo "Direct venv setup failed; falling back to the container runtime." >&2
        pull_container_runtime
        bootstrap_workspace_from_image
        [[ "$SKIP_VALIDATE" -eq 1 ]] || validate_container_runtime
      fi
      ;;
    *)
      echo "Unsupported mode: $MODE" >&2
      exit 1
      ;;
  esac
}

main

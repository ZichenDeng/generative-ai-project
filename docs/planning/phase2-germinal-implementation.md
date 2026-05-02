# Phase 2 Germinal Implementation

This note records the project-owned Phase 2 setup for the antibody-focused generative pipeline.

## Defaults

- Main generator: `Germinal`
- Official target: `HER2`
- Smoke-test target: `PD-L1`
- Initial format: `scFv`
- First backend: `chai`
- External tool workspace: `/home/zichende/projects/external/germinal`

## Project-Owned Assets

- Run configs live in `configs/germinal/run/`
- HER2 target config lives in `configs/germinal/target/`
- HER2 structure bundle lives in `data/raw/germinal_targets/her2_trastuzumab_1n8z/`
- Structured handoff outputs live in `results/germinal_phase2/handoffs/`

## Runtime Paths

The implementation supports two runtime modes:

1. `venv`
   - isolated Python environment at `~/.venvs/germinal_phase2`
   - intended to follow Germinal's direct package-install route

2. `container`
   - Singularity image at `~/.local/share/germinal_phase2/germinal_latest.sif`
   - preferred fallback when PyRosetta or AlphaFold parameter setup blocks the direct environment

Use the setup helper first:

```bash
bash scripts/setup_germinal_phase2_env.sh --mode auto
```

Then validate or launch a bounded stage:

```bash
sbatch --export=STAGE=validate scripts/run_germinal_phase2.sbatch
sbatch --export=STAGE=pdl1_smoke scripts/run_germinal_phase2.sbatch
sbatch --export=STAGE=her2_debug scripts/run_germinal_phase2.sbatch
sbatch --export=STAGE=her2_pilot scripts/run_germinal_phase2.sbatch
```

## Expected Outputs

Each stage writes raw Germinal outputs under:

```text
results/germinal_phase2/runs/<experiment_name>/<run_config>/
```

Each completed stage also writes Phase 3 handoff artifacts:

- `<stage>_all_candidates.csv`
- `<stage>_top_candidates.csv`
- `<stage>_summary.md`
- `<stage>_failure_log.txt` when no accepted designs are produced

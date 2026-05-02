# Scripts

## Zero-Dependency Baselines

Run:

```bash
python3 scripts/run_sequence_baselines.py
```

This creates:

- `results/baseline-results.csv`
- `results/baseline-results.md`

The script uses only the Python standard library. It runs:

- train-fold mean prediction
- sequence-feature ridge regression

The sequence-feature model uses heavy/light chain length, amino-acid composition, and simple grouped residue fractions. It is the first reference point before ESM-2 and AbMAP embedding baselines are added.

## ESM-2 Ridge Baseline

Run in an environment with `torch`, `transformers`, `pandas`, `scikit-learn`, and `scipy`:

```bash
/home/zichende/.conda/envs/afffood/bin/python scripts/run_esm2_baseline.py
```

By default this runs:

- Koenig binding
- Koenig expression
- frozen `facebook/esm2_t6_8M_UR50D` embeddings
- ridge regression with the shared folds

Optional Warszawski task:

```bash
/home/zichende/.conda/envs/afffood/bin/python scripts/run_esm2_baseline.py --include-warszawski
```

Smoke test:

```bash
/home/zichende/.conda/envs/afffood/bin/python scripts/run_esm2_baseline.py --max-rows-per-task 100
```

The first full run downloads ESM-2 weights from Hugging Face unless the model is already cached.

GPU Slurm run:

```bash
sbatch scripts/run_esm2_baseline_gpu.sbatch
```

Include Warszawski:

```bash
INCLUDE_WARSZAWSKI=1 sbatch scripts/run_esm2_baseline_gpu.sbatch
```

Use only cached Hugging Face files:

```bash
LOCAL_FILES_ONLY=1 sbatch scripts/run_esm2_baseline_gpu.sbatch
```

## MLP Benchmarks

Run in an environment with `numpy`, `pandas`, `torch`, `scikit-learn`, and `scipy`:

```bash
python scripts/run_mlp_benchmarks.py
```

By default this writes detailed outputs to:

- `results/mlp_benchmarks/single_task_mlp_results.csv`
- `results/mlp_benchmarks/single_task_mlp_predictions.csv`
- `results/mlp_benchmarks/single_task_mlp_summary.csv`
- `results/mlp_benchmarks/shared_task_mlp_results.csv`
- `results/mlp_benchmarks/shared_task_mlp_predictions.csv`
- `results/mlp_benchmarks/shared_task_mlp_summary.csv`

The script auto-detects `cuda` when available and otherwise runs on CPU. Current committed summary snapshots are also available in `results/` for quick review.

## Germinal Phase 2 Pipeline

Project-owned Phase 2 generation assets are documented in:

- `docs/planning/phase2-germinal-implementation.md`
- `configs/germinal/run/`
- `configs/germinal/target/`
- `data/raw/germinal_targets/`

The external Germinal workspace is expected at:

```bash
/home/zichende/projects/external/germinal
```

Set up the runtime:

```bash
bash scripts/setup_germinal_phase2_env.sh --mode auto
```

Sync the project-owned configs and target assets into the external Germinal clone:

```bash
bash scripts/run_germinal_phase2.sh sync
```

Validate the runtime:

```bash
sbatch --export=STAGE=validate,GERMINAL_RUNTIME=venv scripts/run_germinal_phase2.sbatch
```

Launch the main bounded stages:

```bash
sbatch --export=STAGE=pdl1_smoke,GERMINAL_RUNTIME=venv scripts/run_germinal_phase2.sbatch
sbatch --export=STAGE=her2_debug,GERMINAL_RUNTIME=venv scripts/run_germinal_phase2.sbatch
sbatch --export=STAGE=her2_pilot,GERMINAL_RUNTIME=venv scripts/run_germinal_phase2.sbatch
```

HER2 rescue run:

```bash
sbatch -p mit_preemptable --gpus=h200:1 \
  --export=STAGE=her2_rescue,GERMINAL_RUNTIME=venv,XLA_CLIENT_MEM_FRACTION=0.9 \
  scripts/run_germinal_phase2.sbatch
```

Structured handoff outputs are written to:

- `results/germinal_phase2/handoffs/`

Current checked-in handoff artifacts include:

- `pdl1_smoke_*`
- `her2_rescue_*`

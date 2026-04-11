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

# Data Execution Plan

## First Experiments

The first experiments should use the two Koenig `G6.31 / VEGF` tasks from FLAb:

- `binding/koenig2017mutational_kd_g6.csv`
- `expression/koenig2017mutational_er_g6.csv`

This pair is the cleanest starting point because both tasks use the same antibody family and the same sample size, which makes the project story much tighter. We can ask whether the same representation family transfers consistently across a binding-related task and an expression-related task without introducing target drift or major schema differences.

## Secondary Benchmark

After the Koenig pair is working, the next benchmark should be:

- `binding/warszawski2019_d44_Kd.csv`

This gives us a second binding task with moderate size and a simple schema. It is a better second benchmark than jumping immediately into very large compressed datasets or highly heterogeneous task collections.

## What To Avoid Initially

We should not begin with:

- `AbRank_dataset.csv.zip`, because it is large and introduces extra task complexity too early;
- the very large SARS-CoV-2 binary datasets, because they may dominate the project with dataset-scale engineering rather than benchmark design;
- tiny target-specific HER2 datasets, because they are better suited to later qualitative analysis than primary benchmarking.

## Minimal Reproducible Pipeline

The minimal reproducible pipeline should be:

1. Download one task from the manifest.
2. Inspect schema and label distribution.
3. Create deterministic cross-validation folds.
4. Run a trivial baseline.
5. Replace the baseline with embedding-based predictors.

That pipeline is enough to start real experiments without locking us into any one model family too early.

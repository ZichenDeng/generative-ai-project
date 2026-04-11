# Milestone Execution Plan

## Project Scope

The milestone should be a property prediction benchmark on existing wet-lab labels. It should not implement diffusion or claim that generated antibodies have better wet-lab properties.

The milestone question is:

Can pretrained protein or antibody representations improve prediction of antibody binding and expression labels over simple sequence baselines?

## Shared Tasks

Use the same FLAb task files, folds, and metrics across all benchmark tracks.

Primary milestone tasks:

- Koenig binding: `data/processed/koenig2017mutational_kd_g6_folds.csv`
- Koenig expression: `data/processed/koenig2017mutational_er_g6_folds.csv`

Optional if stable:

- Warszawski d44 binding: `data/processed/warszawski2019_d44_Kd_folds.csv`

## Metrics

Use regression metrics:

- Pearson correlation
- Spearman correlation
- RMSE
- MAE
- R2 if useful

## Parallel Work Tracks

### Tina: Baseline Track

Deliverables:

- train-fold mean baseline
- simple sequence-feature baseline
- short note on shared data, folds, and metrics

The baseline track should give the team a reference table even if embedding models take longer.

### Chris: ESM-2 Track

Deliverables:

- ESM-2 embeddings for heavy/light antibody sequences
- ridge or linear regressor using the same folds as the baseline track
- ESM-2 rows for the shared benchmark table

Minimum target:

- run ESM-2 on Koenig binding and Koenig expression
- use `scripts/run_esm2_baseline.py` in the `afffood` environment
- submit `scripts/run_esm2_baseline_gpu.sbatch` if running on a GPU node

Optional:

- add Warszawski d44 if the first two tasks are stable

### Edward: AbMAP Track

Deliverables:

- check whether AbMAP can be installed and run cleanly
- extract AbMAP embeddings if feasible
- use the same ridge or linear regressor setup as ESM-2
- add AbMAP rows to the shared benchmark table

If blocked:

- document the blocker and keep AbMAP as stretch rather than blocking the milestone

### Peter: Extension Track

Deliverables:

- test a small MLP head on available embeddings if embeddings are ready
- compare the MLP head against ridge or linear probing if time allows
- write the post-milestone surrogate-scoring plan

The extension track should not block the milestone if the embedding files are not ready.

## Milestone Table

Rows:

- train-fold mean baseline
- sequence-feature ridge baseline
- ESM-2 + ridge or linear regressor
- AbMAP + ridge or linear regressor, if feasible
- embedding + small MLP head, if feasible

Columns:

- Koenig binding
- Koenig expression
- optional Warszawski d44
- Pearson
- Spearman
- RMSE
- MAE

## Time Plan

Use this as a T-minus plan if the exact deadline changes.

### T-5 to T-4 Days

- freeze the milestone scope
- confirm task files, fold files, label column, and metrics
- run the zero-dependency baseline script
- start ESM-2 and AbMAP setup checks

### T-3 Days

- produce first ESM-2 results if dependencies are ready
- decide whether AbMAP is feasible for the milestone
- decide whether Warszawski d44 is included or held for after the milestone

### T-2 Days

- freeze the milestone experiment table
- stop adding new model ideas unless a core row is broken
- draft methods, preliminary results, and discussion

### T-1 Day

- polish the two-page milestone document
- check all table values, captions, and claims
- keep diffusion and generation in future work only

## After Milestone

After the milestone, use the best validated property predictor as a surrogate scoring model.

Concrete follow-up:

1. strengthen the predictor with ESM-2, AbMAP, and possibly a small MLP head
2. add another FLAb task or GDPa1 if feasible
3. generate candidate sequences using simple random mutation or CDR-region mutation first
4. score candidates with the trained predictor
5. compare predictor-guided candidates against random/simple mutation baselines
6. consider RFdiffusion or structural generation only as a stretch goal

Claims should stay conservative:

- safe claim: computationally predicted improvement under a surrogate model
- unsafe claim without wet lab validation: generated antibodies truly have better experimental developability

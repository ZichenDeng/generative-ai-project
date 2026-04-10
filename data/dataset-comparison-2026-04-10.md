# Dataset Comparison

## Goal

This note compares the current FLAb-based project plan against two additional antibody dataset leads:

- `ginkgo-datapoints/GDPa1`
- Jain et al. 2017 clinical-stage antibody biophysical dataset

## Summary Recommendation

### Primary benchmark

Keep `FLAb` as the primary benchmark for the course project.

### Best developability extension

Use `GDPa1` as the strongest secondary dataset if the team wants a more explicit developability-focused branch.

### Small-sample reference dataset

Use the Jain clinical-stage antibody dataset as a reference, sanity-check, or external validation set rather than the main benchmark.

## FLAb

### Why it remains the best primary dataset

- already aligned with the current proposal and milestone plan
- task definitions are clean and benchmark-friendly
- the selected Koenig binding and expression tasks are paired and easy to explain
- sample sizes are more practical for a course project than very small therapeutic panels
- recent antibody-representation work already uses FLAb, which makes the benchmark defensible

### Recommended first tasks

- `binding/koenig2017mutational_kd_g6.csv`
- `expression/koenig2017mutational_er_g6.csv`
- later: `binding/warszawski2019_d44_Kd.csv`

## GDPa1

### Why it is attractive

- explicitly focused on antibody developability
- contains multiple physical or biophysical assay targets
- has paired antibody sequence information and predefined evaluation folds
- better fit than FLAb if the team wants the story to center on multi-property developability prediction

### Main limitations

- access is gated, so teammates cannot assume frictionless download
- sample size is still modest compared with large benchmark corpora
- multiple assays have missing labels, which complicates a clean first experiment
- assay outcomes may be strongly shaped by subclass or experimental context, so confounding needs care

### Recommended role

`GDPa1` should be the first secondary dataset to add after the initial FLAb benchmark is stable.

## Jain Clinical-Stage Antibody Dataset

### Why it is valuable

- classic developability reference dataset
- clinical-stage antibodies make the story biologically meaningful
- includes multiple biophysical assays and is widely cited in antibody developability discussions

### Main limitations

- much smaller than FLAb and smaller than GDPa1
- better suited to low-data analysis than to serving as the only benchmark
- less convenient as a main project dataset if the goal is to show robust model comparison

### Recommended role

Use it for:

- external validation
- sanity checks on developability-related trends
- discussion and comparison with prior literature

Do not use it as the sole primary benchmark unless the project intentionally shifts to a small-data study.

## Final Stack Recommendation

1. `FLAb` as the main benchmark
2. `GDPa1` as the main developability extension
3. `Jain` as a small-sample reference or validation dataset

## Source Links

- FLAb: `https://github.com/Graylab/FLAb`
- GDPa1: `https://huggingface.co/datasets/ginkgo-datapoints/GDPa1`
- Jain paper: `https://www.pnas.org/doi/full/10.1073/pnas.1616408114#sec-4`

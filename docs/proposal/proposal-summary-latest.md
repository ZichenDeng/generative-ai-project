# Latest Proposal Summary

## Title

Benchmarking antibody-aware protein representations for antibody-antigen interaction and developability prediction

## Project Framing

The project focuses on a narrow and defensible computational biology question instead of a full therapeutic discovery pipeline. The core goal is to test whether pretrained antibody-aware or interaction-aware protein representations improve prediction quality on biologically relevant antibody benchmarks compared with simpler sequence-based baselines.

## Why This Direction

Recent generative and representation-learning methods have improved computational antibody modeling, but reliable prioritization of useful candidates remains difficult. A benchmark-centered study is more realistic for the course timeline and better aligned with the evidence from recent papers than promising full de novo design or clinical-grade screening.

## Planned Benchmark

The current execution plan is to use FLAb benchmark tasks, starting with:

- Koenig binding: `koenig2017mutational_kd_g6.csv`
- Koenig expression: `koenig2017mutational_er_g6.csv`

If the first benchmark is stable, the next task is:

- Warszawski binding: `warszawski2019_d44_Kd.csv`

## Model Comparison

The intended comparison is:

- a simple baseline model
- a general protein language model embedding baseline
- one stronger antibody-aware or interaction-aware representation method if feasible

## Evaluation

Metrics will depend on the chosen task type, with regression-focused evaluation currently emphasized for the initial benchmark. The immediate milestone should show a concrete dataset choice, a reproducible pipeline, and at least one baseline result table.

## Practical Outcome

If the benchmark experiments go well, the team may add a limited target-specific ranking analysis as a stretch goal. The main deliverable remains a careful empirical comparison and a clear discussion of what current models can and cannot do on antibody-related prediction tasks.

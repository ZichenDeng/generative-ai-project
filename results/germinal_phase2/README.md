# Germinal Phase 2 Results

This directory contains the project-owned outputs for the antibody-focused generative pipeline.

## Layout

- `handoffs/`: teammate-facing CSV and Markdown summaries for downstream scoring, reranking, and report writing
- `runs/`: raw bounded Germinal run directories copied from the project launches

## Current Checked-In Results

### PD-L1 smoke run

- status: end-to-end success
- target: `PD-L1`
- format: `scFv`
- backend: `chai`
- outcome: `8` redesign candidates, `0` fully accepted designs

Main handoff files:

- `handoffs/pdl1_smoke_all_candidates.csv`
- `handoffs/pdl1_smoke_top_candidates.csv`
- `handoffs/pdl1_smoke_summary.md`
- `handoffs/pdl1_smoke_failure_log.txt`

### HER2 rescue run

- status: runtime success on larger GPU
- target: `HER2` from the `1N8Z` trastuzumab-bound epitope bundle
- format: `scFv`
- backend: `chai`
- outcome: run completed without OOM, but the single rescue trajectory failed quality screening

Main handoff files:

- `handoffs/her2_rescue_all_candidates.csv`
- `handoffs/her2_rescue_top_candidates.csv`
- `handoffs/her2_rescue_summary.md`
- `handoffs/her2_rescue_failure_log.txt`

## Interpretation

These results show that the Phase 2 pipeline is operational and reproducible. The remaining work is now model and target tuning rather than installation or launch debugging.

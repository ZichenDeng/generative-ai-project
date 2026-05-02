# Generative AI Project Share

This repository contains the minimum set of materials for sharing the current project direction with teammates.

## Included Materials

- `docs/proposal/project-proposal.pdf`: submitted or presentation-ready proposal copy
- `docs/proposal/proposal-summary-latest.md`: latest concise summary of the revised project direction
- `docs/proposal/proposal-draft-v2-2026-04-09.md`: latest fuller proposal draft
- `docs/proposal/data-execution-plan-2026-04-09.md`: current benchmark and experiment plan
- `docs/planning/team-roles-and-milestones.md`: team work-package split and milestone plan
- `docs/planning/milestone-execution-plan-2026-04-11.md`: concrete parallel milestone plan
- `docs/planning/project-state-recap-2026-04-28.md`: current one-page project recap and next-step plan
- `docs/planning/phase2-germinal-implementation.md`: current Phase 2 generation workflow and runtime notes
- `docs/planning/next-steps.md`: immediate action items for the team
- `data/`: benchmark manifests plus Germinal target assets, including the HER2 `1N8Z` target bundle
- `scripts/`: reproducible scripts for benchmark runs and the Germinal Phase 2 pipeline
- `results/`: benchmark outputs plus `results/germinal_phase2/` handoff artifacts for the generation workstream

## Current Project Direction

The project now has two connected pieces:

- a benchmark and scorer workstream for antibody fitness prediction
- a Phase 2 generative workstream built around `Germinal`

The current Phase 2 implementation keeps `HER2` as the official case-study target, uses `PD-L1` as the first working smoke-test target, and exports structured candidate tables so teammates can continue reranking and analysis without needing to inspect raw Germinal internals.

Current Phase 2 status:

- `PD-L1` smoke run completed end-to-end and produced rerankable candidate tables
- `HER2` target packaging and runtime setup are complete
- `HER2` rescue run proved the target is runnable on a larger GPU, though the current design did not pass quality filters

## Scope

This share repo intentionally excludes large raw data and internal draft history. It includes small processed fold files, benchmark manifests, and lightweight scripts needed for quick team alignment and milestone execution.

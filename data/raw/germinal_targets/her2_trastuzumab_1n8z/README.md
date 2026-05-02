# HER2 Germinal Target Bundle

This Phase 2 target bundle uses PDB `1N8Z`, the extracellular domain of human HER2 in complex with trastuzumab Fab, as the reference HER2 epitope definition.

## Files

- `1n8z_her2_chainA.pdb`: HER2 target-only structure extracted from chain `C` of `1N8Z` and renamed to chain `A` for Germinal compatibility

## Epitope Definition

The hotspot set is anchored to the trastuzumab-contacting region on HER2 domain IV:

- `A557`
- `A558`
- `A560`
- `A561`
- `A569`
- `A570`
- `A571`
- `A572`
- `A573`

The single-residue Chai distance constraint uses `F573`, which sits near the center of the trastuzumab peptide epitope.

## Rationale

This bundle is intentionally conservative:

- it uses one fixed HER2 structure for the full project,
- it keeps the numbering from the deposited HER2 chain in `1N8Z`,
- and it focuses the first HER2 Germinal runs on a known therapeutic epitope rather than a hand-picked de novo surface patch.

If a later run requires a different HER2 epitope, add a new target bundle instead of changing this one in place.

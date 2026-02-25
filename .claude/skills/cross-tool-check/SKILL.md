---
name: cross-tool-check
description: Verify anti-salami scope boundary and consistency with sister tools
---

Check that bact-trait-cluster stays within its defined scope.

## Steps

1. **Read scope boundary**: Parse `docs/anti_salami_checklist.json`

2. **Grep for scope violations** in `src/bacttraitcluster/`:
   - Association terms: `mutual_information`, `phi_coefficient`, `odds_ratio`, `transfer_entropy`, `simplicial`
   - Phylo terms: `pagel`, `fritz_purvis`, `ancestral_state`, `newick`, `phylo_logistic`
   - MDR terms: `mdr_class`, `xdr`, `pdr`, `pc_algorithm`, `evpi`, `hypergraph`

3. **Check shared interfaces** (if sister repos at `../bact-*`)

Report: PASS / WARN / FAIL with details.

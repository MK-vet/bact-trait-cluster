---
name: add-analysis
description: Guided workflow to add a new clustering/validation analysis step
---

## Pre-flight

1. **Scope check**: Does this analysis involve clustering, consensus, or multi-view fusion?
   - Pairwise associations/networks → bact-assoc-net
   - Phylogenetic methods → bact-phylo-trait
   - MDR classification/causal → bact-mdr-profiler

2. **Method validation**: Appropriate for binary/categorical data?
3. **Dependency check**: New optional deps needed? Add to pyproject.toml[optional-dependencies]

## Implementation checklist

4. Add parameters to `src/bacttraitcluster/config.py`
5. Add step to `src/bacttraitcluster/pipeline.py`
6. Add tests with known cluster ground truth (ARI check)
7. Run `/validate-config`, `/run-tests`, `/selfcheck`, `/reliability-check`
8. Run `/cross-tool-check`

## Post-implementation

9. Update README.md if user-facing
10. Verify provenance

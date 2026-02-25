---
name: review-stats
description: Review statistical methods for correctness — clustering and validation audit
---

Review clustering, consensus, and validation method implementations.

## Checklist

1. **K-Modes**:
   - Uses Hamming distance (NOT Euclidean)?
   - Categorical mode update rule?
   - Not k-means accidentally applied to binary data?

2. **Consensus clustering**:
   - Bootstrap resampling (Monti 2003)?
   - Consensus matrix: entry = P(same cluster across resamples)?
   - Consensus index computed correctly?

3. **NVI stability**:
   - NVI (NOT NMI) — NMI is biased toward more clusters?
   - Stability path across K range?
   - Minimum NVI = optimal K?

4. **CKA fusion**:
   - Centered kernel alignment?
   - >= 2 layers required?
   - >= 6 common samples?
   - Kernel matrix positive semi-definite?

5. **MDL model selection**:
   - Lower MDL = better model?
   - Correctly penalizes model complexity?

6. **SHAP importance**:
   - Bootstrap CI for stability?
   - Cliff's delta effect size?
   - Auto-skip if deps missing?

7. **TDA**:
   - Hamming-distance Vietoris-Rips?
   - beta_0 = number of connected components?
   - Auto-skip if ripser missing?

## Key files
- `src/bacttraitcluster/clustering/consensus.py`, `model_selection.py`, `multiview.py`
- `src/bacttraitcluster/profiling/importance.py`

Report: method | check | status (PASS/WARN/FAIL) | evidence (file:line)

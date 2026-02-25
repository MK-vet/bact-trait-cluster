---
name: validate-on-synth
description: Validate signal recovery on synthetic clustering data
---

Run pipeline on synthetic data and verify signal recovery.

## Validation per scenario

1. **K-Modes**: ARI > 0.9 for clear clusters
2. **NVI stability**: NVI minimum at true K
3. **CKA fusion**: Fused ARI > best single-layer ARI
4. **Consensus**: Consensus matrix shows clear block structure
5. **MDL**: MDL minimum at true K
6. **SHAP**: Top-3 important features match ground truth informative features
7. **TDA**: beta_0 = true K, beta_1 matches expected

## Multi-seed
For K-Modes: seeds {42,43,44,45,46}:
- Mean ARI and variance
- Consensus stability across seeds

Report: scenario | metric | expected | observed | PASS/FAIL

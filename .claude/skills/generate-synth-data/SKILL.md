---
name: generate-synth-data
description: Generate synthetic clustering data with known ground truth
---

Generate synthetic binary matrices with known cluster structure.

## Scenarios

1. **K-Modes recovery** (seed=42, n=150, p=20, K=3):
   - 3 clusters with distinct binary centroids, noise features
   - Ground truth: cluster labels, expected ARI > 0.9

2. **NVI stability path** (seed=43, n=200, p=15):
   - Clear K=3 vs ambiguous K=5 structure
   - Ground truth: optimal K, NVI minimum at true K

3. **CKA fusion benefit** (seed=44, n=100, 2 layers):
   - Layer 1: partial signal (3 informative features)
   - Layer 2: complementary signal (3 different informative features)
   - Ground truth: fusion ARI > single-layer ARI

4. **Consensus convergence** (seed=45, n=100, K=3):
   - Known cluster membership
   - Ground truth: consensus matrix diagonal blocks

5. **MDL model selection** (seed=46, n=200, K=4):
   - Known K=4 structure with varying separation
   - Ground truth: MDL minimum at K=4

6. **SHAP feature importance** (seed=47, n=150, 10 features):
   - 3 informative + 7 noise features
   - Ground truth: SHAP top-3 = informative features

7. **TDA persistent homology** (seed=48, n=120, K=3):
   - Known Betti numbers from cluster topology
   - Ground truth: beta_0 = K, known beta_1

## Implementation
Python script, deterministic seeds, output to `synth_data/`.

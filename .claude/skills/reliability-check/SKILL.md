---
name: reliability-check
description: Run reliability preflight QC and quality gate check
---

```bash
python -m bacttraitcluster.cli --config examples/config.yaml --reliability-only
```

## What to check:
1. Sample count vs requested K (need n > K)
2. NA proportion per feature (flag >50% missing)
3. Degenerate features (zero variance, all-NA)
4. Multi-layer overlap for CKA fusion (>= 6 common samples)
5. Optional dependency availability (kmodes, shap, ripser)
6. Quality gate status (PASS/WARN/FAIL)

Report quality gate result and warnings with remediation suggestions.

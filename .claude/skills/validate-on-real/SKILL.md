---
name: validate-on-real
description: Validate pipeline on real data from examples/
---

```bash
python -m bacttraitcluster.cli --config examples/config.yaml
```

## Check invariants:
- No crash (exit code 0)
- Output files exist
- `run_manifest.json` and `config_used.yaml` generated
- No NA→0 conversion
- Cluster labels are valid integers (1..K)
- NVI/consensus metrics are finite and reasonable
- SHAP values finite (or gracefully skipped if deps missing)
- TDA gracefully skipped if ripser not installed

Report: structured checklist PASS/FAIL.

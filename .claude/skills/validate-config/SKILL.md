---
name: validate-config
description: Validate a YAML pipeline configuration file against schema v1.1
---

Validate config against bact-trait-cluster schema. Use `$ARGUMENTS` or `examples/config.yaml`.

```bash
python -c "
from bacttraitcluster.config import Config
import sys
path = sys.argv[1] if len(sys.argv) > 1 else 'examples/config.yaml'
cfg = Config.from_yaml(path)
print(f'Config loaded OK: schema_version={cfg.schema_version}')
print(f'Output dir: {cfg.output_dir}')
" $ARGUMENTS
```

Report: parsed OK, schema version, clustering parameters, any warnings.

# bact-trait-cluster

[![CI](https://github.com/MK-vet/bact-trait-cluster/actions/workflows/ci.yaml/badge.svg)](https://github.com/MK-vet/bact-trait-cluster/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/bact-trait-cluster.svg)](https://pypi.org/project/bact-trait-cluster/)
[![Python](https://img.shields.io/pypi/pyversions/bact-trait-cluster.svg)](https://pypi.org/project/bact-trait-cluster/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Multi-algorithm consensus clustering with topological validation for binary genotype–phenotype matrices.

## Novel Contributions

1. **Multi-algorithm consensus for binary data** — K-Modes + Spectral (Jaccard kernel) + Agglomerative (Hamming) fused via bootstrap consensus matrix (Monti et al. 2003)
2. **Stability paths via Normalised Variation of Information** — per-*k* stability with 95% CI bands (Meilă 2007)
3. **Persistent homology** on Hamming-distance Vietoris–Rips complex — captures multi-scale topology that PCA/MCA miss
4. **SHAP feature importance** with bootstrap confidence intervals via TreeSHAP + LightGBM
5. **Cliff's delta effect size** — non-parametric, with bootstrap CI (replaces chi-square "everything significant at large *n*")

## Installation

```bash
pip install bact-trait-cluster              # core
pip install bact-trait-cluster[shap,tda]    # with SHAP + persistent homology
pip install bact-trait-cluster[gui]         # interactive dashboard (marimo)
pip install bact-trait-cluster[all]         # everything including dev tools
```

## Interactive dashboard (marimo)

```bash
bact-trait-cluster-dashboard
```

Edit mode (shows notebook code):

```bash
bact-trait-cluster-dashboard --edit
```

## Quick Start

```bash
# Generate template config
bact-trait-cluster --generate-config config.yaml

# Edit config.yaml (set paths to your CSV files)

# Run
bact-trait-cluster config.yaml -v
```

### Python API

```python
from bacttraitcluster.config import Config
from bacttraitcluster.pipeline import Pipeline

cfg = Config.from_yaml("config.yaml")
results = Pipeline(cfg).run()
```

## Configuration

All parameters are in a single YAML file — **no hardcoded species or file paths**. See `examples/config.yaml` for a complete template.

Key sections:
- `layers`: list of binary input CSVs (any number, any feature set)
- `consensus`: algorithm selection, k-range, bootstrap parameters
- `profiling`: SHAP, TDA, effect size toggles

## Outputs

| File | Description |
|------|-------------|
| `stability_path.csv` | NVI per k with CI bands |
| `clusters.csv` | Final cluster assignments |
| `consensus_matrix.npy` | Co-clustering probability matrix |
| `shap.csv` | Mean \|SHAP\| per feature × cluster |
| `cliff_delta.csv` | Effect sizes with CI |
| `enrichment_z.csv` | z-scores with FDR correction |
| `persistence_diagram.csv` | TDA birth–death pairs |
| `tda_summary.csv` | Betti numbers + persistence entropy |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT

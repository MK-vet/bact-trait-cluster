# bact-trait-cluster — Claude Code Context

## Project Identity

- **Tool**: bact-trait-cluster v1.0.0
- **Package**: `bacttraitcluster` (in `src/bacttraitcluster/`)
- **Author**: Maciej Kochanowski, National Veterinary Research Institute (PIWet-PIB), Pulawy, Poland
- **Domain**: Antimicrobial resistance (AMR) epidemiology, *Streptococcus suis* and related bacteria
- **Purpose**: Multi-algorithm consensus clustering with topological validation for binary genotype-phenotype matrices — K-Modes, Spectral, Agglomerative, CKA kernel fusion, TDA persistent homology, SHAP importance
- **Publication**: Individual SoftwareX paper — do NOT mix scope with sister tools

## Sister Tools (Anti-Salami Boundaries)

| Tool | Scope | THIS tool does NOT do this |
|------|-------|---------------------------|
| bact-assoc-net | MI, phi, OR, CMH, PID, TE, simplicial topology | NO pairwise association networks |
| bact-phylo-trait | Pagel lambda, Fritz-Purvis D, Fitch ASR, phylo-logistic | NO phylogenetic comparative methods |
| bact-mdr-profiler | PC-algorithm causal, hypergraph MDR, Shapley, EVPI | NO causal discovery, NO MDR classification |
| **bact-trait-cluster** (THIS) | K-Modes consensus, CKA fusion, NVI, TDA, SHAP | — |

Reference: `docs/anti_salami_checklist.json`

## Architecture

```
src/bacttraitcluster/
  config.py                — YAML config v1.1, dataclass schema, strict mode
  pipeline.py              — Main orchestrator: load → cluster → consensus → stability → manifest
  cli.py                   — CLI entry: --config, --self-check, --benchmark, --reliability-*
  io/loader.py             — CSV loading, wide/long auto-detect, NA-preserving Int8
  clustering/
    consensus.py           — Bootstrap consensus matrix (Monti 2003)
    model_selection.py     — MDL, NVI stability paths, gap statistic
    multiview.py           — CKA kernel fusion for multi-layer data
  profiling/
    importance.py          — SHAP feature importance with bootstrap CI, Cliff's delta
  reliability/core.py      — Preflight QC, consistency checks, marimo warnings
  selfcheck.py             — Internal validation
  benchmark.py             — Synthetic benchmark
  gui/                     — Marimo interactive app
  dashboards/              — Plotly dashboard
```

**Key patterns**: config-driven pipeline, NA-preserving (never NA→0), provenance via `run_manifest.json` + `config_used.yaml`, deterministic seeds, optional deps (kmodes, ripser, shap, lightgbm).

## Development Quickstart

```bash
pip install -e ".[dev]"
ruff check src/
ruff format --check src/
python -m pytest tests/ -v --tb=short
python -m bacttraitcluster.cli --self-check
python -m bacttraitcluster.cli --benchmark
python -m bacttraitcluster.cli --config examples/config.yaml
```

## CRITICAL: Scientific Correctness Rules

**These rules are ABSOLUTE. Violating any of them produces scientifically invalid results.**

1. **NEVER fabricate** p-values, test statistics, confidence intervals, or effect sizes. All must come from actual computation on actual data.
2. **NEVER generate fake citations**. Valid references for this tool:
   - Monti et al. (2003) — Consensus clustering
   - Meila (2007) — NVI (Variation of Information)
   - Huang (1998) — K-Modes for categorical data
   - Kornblith et al. (2019) — CKA (Centered Kernel Alignment)
   - Cliff (1993) — Cliff's delta effect size
   - Rissanen (1978) — MDL (Minimum Description Length)
3. **NEVER convert NA to 0**. NA means "not tested"; 0 means "tested and susceptible". These are biologically different.
4. **NEVER use k-means on binary data**. K-Modes is the categorical equivalent. Euclidean distance is inappropriate for binary features.
5. **NEVER skip stability validation** when reporting cluster solutions. Always check NVI stability or consensus matrix.
6. **NEVER assume independence** of samples with shared phylogenetic origin without explicit justification.
7. **NEVER add functionality belonging to a sister tool** (see anti-salami table above).

## Statistical Constraints — THIS TOOL ONLY

- **K-Modes**: Categorical clustering, NOT k-means; Hamming distance, NOT Euclidean
- **NVI (Normalized Variation of Information)**: For stability assessment, NOT NMI (NMI is biased toward more clusters)
- **CKA kernel fusion**: Requires >= 2 data layers and >= 6 common samples; centered kernel alignment
- **MDL (Minimum Description Length)**: Lower = better model; for model selection (K choice)
- **Consensus clustering**: Bootstrap resampling (Monti 2003); consensus matrix entry = P(same cluster)
- **SHAP importance**: Requires optional deps (shap, lightgbm); bootstrap CI for stability; auto-skip if deps missing
- **TDA persistent homology**: Requires optional ripser; Hamming-distance Vietoris-Rips complex; auto-skip if not installed

## Data Format Constraints

- Input: CSV with binary features (0 = absent, 1 = present, NA = not tested)
- Internal dtype: nullable Int8 (`pd.Int8Dtype()`)
- Formats: wide (samples x features) or long (ID, feature, value) — auto-detected
- Multi-layer: multiple CSV files for CKA fusion (genotype layer + phenotype layer)
- NA handling: preserve throughout; imputation strategy configurable (never silent NA→0)
- Output: CSV with cluster assignments + stability metrics + importance scores, provenance JSON

## Testing Conventions

- **Real data**: Run on `examples/` CSV files with example configs
- **Synthetic data**: Generated with known cluster structure (`ground_truth.json`)
- **Signal recovery**: Verify K-Modes recovers known clusters (ARI > 0.9)
- **Multi-seed**: Run with multiple seeds for consensus stability
- **Edge cases**: Single cluster, all-NA columns, fewer samples than K, ties in K-Modes
- **Reproducibility**: Same seed + config → identical output (bit-for-bit)

## Publication Context (SoftwareX)

- CITATION.cff, LICENSE, README.md exist — **DO NOT MODIFY** these files
- Scope boundary defined in `docs/anti_salami_checklist.json` — **DO NOT MODIFY**
- Reproducibility artifacts: `run_manifest.json` (input SHA-256), `config_used.yaml` (config snapshot)

## Code Style

- Python 3.10+ (type hints with `X | Y` union syntax)
- Linter: ruff (check + format)
- Logging: `logging.getLogger(__name__)`, never print()
- Tests: pytest in `tests/`, parametrized where possible
- Optional deps: graceful import with auto-skip (kmodes, shap, lightgbm, ripser)

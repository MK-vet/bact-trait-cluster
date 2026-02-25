# bact-trait-cluster — AI Coding Assistant Instructions

## Context
This is **bact-trait-cluster**, a Python tool for multi-algorithm consensus clustering with topological validation for binary genotype-phenotype matrices. It implements K-Modes, Spectral, Agglomerative clustering, CKA kernel fusion, NVI stability paths, TDA persistent homology, and SHAP feature importance. Published as an individual SoftwareX paper.

## CRITICAL: Scientific Correctness Rules (NEVER violate)
1. NEVER fabricate p-values, test statistics, CI, or effect sizes
2. NEVER generate fake citations (valid: Monti et al. 2003, Meila 2007, Huang 1998, Kornblith et al. 2019)
3. NEVER convert NA to 0 (NA = "not tested", 0 = "tested and susceptible")
4. NEVER use k-means on binary data — use K-Modes (Hamming distance, not Euclidean)
5. NEVER use NMI for stability — use NVI (NMI is biased toward more clusters)
6. NEVER assume sample independence without phylogenetic justification
7. NEVER add functionality from sister tools (see scope below)

## Scope Boundary (Anti-Salami)
- THIS tool: K-Modes consensus, CKA fusion, NVI, TDA, SHAP
- NOT this tool: pairwise association networks, phylogenetic methods, MDR classification/causal

## Code Conventions
- Python 3.10+, ruff, pytest, `logging.getLogger(__name__)`
- Data: binary 0/1/NA CSV, nullable Int8
- Optional deps: kmodes, shap, lightgbm, ripser (graceful auto-skip)

## Dev Commands
```bash
pip install -e ".[dev]"
ruff check src/ && ruff format --check src/
python -m pytest tests/ -v --tb=short
python -m bacttraitcluster.cli --self-check
```

## Read-Only Files (DO NOT MODIFY)
- CITATION.cff, LICENSE, docs/anti_salami_checklist.json, examples/*

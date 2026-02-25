"""YAML-driven configuration — no hardcoded species, layers, or thresholds.

v1.1 changes
----------
- Adds schema_version and config validation with explicit reporting of unknown keys.
- Unknown keys are reported (WARN by default) and ignored; enable strict mode to fail.
- Keeps legacy alias: qc -> feature_qc.

Strict mode triggers
--------------------
- YAML key: config_strict: true
- Env var: SSUIS_CONFIG_STRICT=1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
import dataclasses as _dc
import logging
import os
from typing import get_args, get_origin

import yaml

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1.1"}


def _is_dataclass_type(tp: Any) -> bool:
    try:
        return hasattr(tp, "__dataclass_fields__")
    except Exception:
        return False


def _dataclass_allowed_fields(dc_cls: Any) -> set[str]:
    return {f.name for f in _dc.fields(dc_cls)}


def _validate_and_filter(
    dc_cls: Any, raw: Any, prefix: str, unknown_paths: List[str]
) -> Any:
    if raw is None:
        raw = {}
    if isinstance(raw, dict):
        allowed = _dataclass_allowed_fields(dc_cls)
        filtered: dict = {}
        for k, v in raw.items():
            if k not in allowed:
                unknown_paths.append(f"{prefix}.{k}" if prefix else str(k))
                continue
            f = next((ff for ff in _dc.fields(dc_cls) if ff.name == k), None)
            if f is None:
                filtered[k] = v
                continue
            tp = f.type
            if _is_dataclass_type(tp) and isinstance(v, dict):
                filtered[k] = _validate_and_filter(
                    tp, v, f"{prefix}.{k}" if prefix else k, unknown_paths
                )
            else:
                origin = get_origin(tp)
                args = get_args(tp)
                if (
                    origin in (list, List)
                    and args
                    and _is_dataclass_type(args[0])
                    and isinstance(v, list)
                ):
                    out_list = []
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            out_list.append(
                                _validate_and_filter(
                                    args[0],
                                    item,
                                    f"{prefix}.{k}[{i}]" if prefix else f"{k}[{i}]",
                                    unknown_paths,
                                )
                            )
                        else:
                            out_list.append(item)
                    filtered[k] = out_list
                else:
                    filtered[k] = v
        return filtered
    return raw


@dataclass
class LayerSpec:
    """Specification for one input layer.

    Supports:
    - wide binary matrices: ID column + many feature columns (0/1 with possible NA)
    - long positive lists: ID column + feature column (+ optional value/count column)
    """

    name: str
    path: str
    id_column: str = "Strain_ID"
    format: str = "auto"  # auto | wide | long
    feature_column: str | None = None
    value_column: str | None = None


@dataclass
class InputSpec:
    """Input alignment & missingness policy."""

    align_mode: str = "union"  # union | intersection (intersection mimics legacy)
    drop_samples_with_missing: bool = True
    max_missing_sample: float = 0.0  # applied per-layer after QC
    max_missing_feature: float = 0.0  # applied per-layer after QC


@dataclass
class FeatureQCSpec:
    """QC for binary features before clustering."""

    min_prev: float = 0.01
    max_prev: float = 0.99
    max_missing_frac: float = 0.0


@dataclass
class PhyloValidationSpec:
    """Optional phylogeny-aware validation of cluster quality (no comparative modelling)."""

    enabled: bool = False
    tree_path: str = ""
    id_match_required: bool = True
    n_perm: int = 500


@dataclass
class ConsensusSpec:
    """Multi-algorithm consensus clustering parameters."""

    algorithms: List[str] = field(
        default_factory=lambda: [
            "kmodes",
            "spectral_jaccard",
            "agglomerative_hamming",
        ]
    )
    k_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8])
    n_consensus_runs: int = 200
    subsample_fraction: float = 0.8
    n_stability_splits: int = 30


@dataclass
class ProfilingSpec:
    """Cluster profiling parameters."""

    shap_enabled: bool = True
    shap_n_background: int = 100
    shap_bootstrap: int = 50
    tda_enabled: bool = True
    tda_max_dim: int = 1
    tda_n_subsample: int = 200
    effect_size_bootstrap: int = 500


@dataclass
class Config:
    """Top-level pipeline configuration."""

    # schema + validation
    schema_version: str = SCHEMA_VERSION
    config_strict: bool = False

    layers: List[LayerSpec] = field(default_factory=list)
    input: InputSpec = field(default_factory=InputSpec)
    feature_qc: FeatureQCSpec = field(default_factory=FeatureQCSpec)
    phylo_validation: PhyloValidationSpec = field(default_factory=PhyloValidationSpec)
    output_dir: str = "results"
    consensus: ConsensusSpec = field(default_factory=ConsensusSpec)
    profiling: ProfilingSpec = field(default_factory=ProfilingSpec)
    fdr_method: str = "fdr_bh"
    alpha: float = 0.05
    n_jobs: int = -1
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}

        # legacy alias
        if "qc" in raw and "feature_qc" not in raw:
            raw["feature_qc"] = raw.pop("qc")

        schema_in = str(raw.get("schema_version", "1.0"))
        strict = bool(raw.get("config_strict", False)) or (
            os.environ.get("SSUIS_CONFIG_STRICT", "0") == "1"
        )
        if schema_in not in SUPPORTED_SCHEMA_VERSIONS:
            msg = f"Unsupported schema_version={schema_in!r}. Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        unknown_paths: List[str] = []
        filtered = _validate_and_filter(
            cls, raw, prefix="", unknown_paths=unknown_paths
        )

        layers = [
            LayerSpec(**(lyr or {})) for lyr in (filtered.pop("layers", []) or [])
        ]
        inp = InputSpec(**(filtered.pop("input", {}) or {}))
        fqc = FeatureQCSpec(**(filtered.pop("feature_qc", {}) or {}))
        pv = PhyloValidationSpec(**(filtered.pop("phylo_validation", {}) or {}))
        consensus = ConsensusSpec(**(filtered.pop("consensus", {}) or {}))
        profiling = ProfilingSpec(**(filtered.pop("profiling", {}) or {}))

        cfg = cls(
            layers=layers,
            input=inp,
            feature_qc=fqc,
            phylo_validation=pv,
            consensus=consensus,
            profiling=profiling,
            **filtered,
        )

        cfg._config_validation = {
            "schema_version_in": schema_in,
            "schema_version_effective": cfg.schema_version,
            "supported_schema_versions": sorted(SUPPORTED_SCHEMA_VERSIONS),
            "unknown_keys": sorted(set(unknown_paths)),
            "strict": strict,
            "status": "PASS"
            if (schema_in in SUPPORTED_SCHEMA_VERSIONS and not unknown_paths)
            else ("WARN" if not strict else "FAIL"),
        }

        if unknown_paths:
            msg = (
                f"Unknown config keys ignored ({len(set(unknown_paths))}): {sorted(set(unknown_paths))[:20]}"
                + (" ..." if len(set(unknown_paths)) > 20 else "")
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        return cfg

    def to_yaml(self, path: str | Path) -> None:
        import dataclasses as dc

        def ser(o: Any) -> Any:
            if dc.is_dataclass(o):
                return {k: ser(v) for k, v in dc.asdict(o).items()}
            if isinstance(o, list):
                return [ser(v) for v in o]
            return o

        Path(path).write_text(
            yaml.dump(ser(self), default_flow_style=False, sort_keys=False)
        )

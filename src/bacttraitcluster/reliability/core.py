from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_ID_COL_CANDIDATES = (
    "strain",
    "sample",
    "isolate",
    "id",
    "ID",
    "Strain",
    "Sample",
    "Isolate",
)


def _ensure_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _json_default(o):
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_json(data, path: str | Path, indent: int = 2) -> Path:
    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    path = _ensure_path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_id_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    cols = list(df.columns)
    for c in DEFAULT_ID_COL_CANDIDATES:
        if c in cols:
            return c
    # Heuristic: first non-numeric column with unique-ish values
    for c in cols:
        ser = df[c]
        if ser.dtype == object or str(ser.dtype).startswith(("string", "category")):
            nunq = ser.nunique(dropna=True)
            if nunq >= max(3, int(0.7 * len(df))):
                return c
    return cols[0] if cols else None


def read_table(path: str | Path) -> pd.DataFrame:
    path = _ensure_path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else None
        if sep is None:
            # robust autodetect for comma/semicolon/tab
            try:
                return pd.read_csv(path)
            except Exception:
                for s in [";", "\t", ","]:
                    try:
                        return pd.read_csv(path, sep=s)
                    except Exception:
                        pass
                raise
        return pd.read_csv(path, sep=sep)
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format for {path}")


def normalize_binary_like(
    df: pd.DataFrame, id_col: Optional[str] = None
) -> pd.DataFrame:
    out = df.copy()
    if id_col and id_col in out.columns:
        feature_cols = [c for c in out.columns if c != id_col]
    else:
        feature_cols = list(out.columns)
    mapping = {
        "present": 1,
        "absent": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "t": 1,
        "f": 0,
        "+": 1,
        "-": 0,
    }
    for c in feature_cols:
        ser = out[c]
        if ser.dtype == bool:
            out[c] = ser.astype(int)
            continue
        if ser.dtype == object:
            low = ser.astype(str).str.strip().str.lower()
            # coerce common forms
            out[c] = pd.to_numeric(low.map(mapping).fillna(low), errors="coerce")
        else:
            out[c] = pd.to_numeric(ser, errors="coerce")
    return out


@dataclass
class DatasetPreflight:
    name: str
    path: str
    rows: int
    cols: int
    id_col: Optional[str]
    n_unique_ids: int
    missing_id_rows: int
    duplicate_id_rows: int
    sha256: str


def preflight_dataset(
    path: str | Path, name: Optional[str] = None, id_col: Optional[str] = None
) -> Tuple[DatasetPreflight, pd.DataFrame]:
    p = _ensure_path(path)
    df = read_table(p)
    chosen_id = detect_id_column(df, explicit=id_col)
    if chosen_id is None:
        ids = pd.Series([], dtype=object)
    else:
        ids = df[chosen_id]
    nonnull_ids = ids.dropna().astype(str) if len(ids) else pd.Series([], dtype=str)
    pf = DatasetPreflight(
        name=name or p.stem,
        path=str(p),
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        id_col=chosen_id,
        n_unique_ids=int(nonnull_ids.nunique()),
        missing_id_rows=int(ids.isna().sum()) if len(ids) else 0,
        duplicate_id_rows=int(nonnull_ids.duplicated().sum())
        if len(nonnull_ids)
        else 0,
        sha256=sha256_file(p),
    )
    return pf, df


def pairwise_id_overlap(
    datasets: Mapping[str, pd.DataFrame], id_cols: Mapping[str, Optional[str]]
) -> pd.DataFrame:
    names = list(datasets.keys())
    rows = []
    id_sets = {}
    for n in names:
        idc = id_cols.get(n)
        df = datasets[n]
        if idc and idc in df.columns:
            s = set(df[idc].dropna().astype(str))
        else:
            s = set()
        id_sets[n] = s
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            A, B = id_sets[a], id_sets[b]
            inter = len(A & B)
            union = len(A | B)
            j = (inter / union) if union else math.nan
            rec_a = (inter / len(A)) if A else math.nan
            rec_b = (inter / len(B)) if B else math.nan
            rows.append(
                {
                    "dataset_a": a,
                    "dataset_b": b,
                    "n_ids_a": len(A),
                    "n_ids_b": len(B),
                    "intersection": inter,
                    "union": union,
                    "jaccard": j,
                    "recall_a_in_b": rec_a,
                    "recall_b_in_a": rec_b,
                }
            )
    return pd.DataFrame(rows)


def layer_coverage_against_reference(
    datasets: Mapping[str, pd.DataFrame],
    id_cols: Mapping[str, Optional[str]],
    reference_name: Optional[str] = None,
) -> pd.DataFrame:
    names = list(datasets.keys())
    if not names:
        return pd.DataFrame(
            columns=[
                "dataset",
                "reference",
                "n_dataset",
                "n_reference",
                "intersection",
                "coverage_vs_reference",
            ]
        )
    if reference_name is None:
        # Prefer MIC if present, otherwise first
        lower_map = {n.lower(): n for n in names}
        reference_name = lower_map.get("mic") or next(
            (n for n in names if "mic" in n.lower()), names[0]
        )
    ref_df = datasets[reference_name]
    ref_id_col = id_cols.get(reference_name)
    ref_ids = (
        set(ref_df[ref_id_col].dropna().astype(str))
        if ref_id_col and ref_id_col in ref_df.columns
        else set()
    )
    rows = []
    for n in names:
        idc = id_cols.get(n)
        ids = (
            set(datasets[n][idc].dropna().astype(str))
            if idc and idc in datasets[n].columns
            else set()
        )
        inter = len(ids & ref_ids)
        cov = (inter / len(ref_ids)) if ref_ids else math.nan
        rows.append(
            {
                "dataset": n,
                "reference": reference_name,
                "n_dataset": len(ids),
                "n_reference": len(ref_ids),
                "intersection": inter,
                "coverage_vs_reference": cov,
            }
        )
    return pd.DataFrame(rows)


def feature_informativeness_index(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    prevalence_floor: float = 0.02,
    prevalence_ceiling: float = 0.98,
) -> pd.DataFrame:
    """
    Mixed-type feature informativeness:
    - binary-like columns: prevalence + binary entropy
    - numeric/categorical columns: Shannon entropy over observed categories/values
    """
    work = df.copy()
    if id_col and id_col in work.columns:
        work = work.drop(columns=[id_col])
    out_rows = []
    n = len(work)
    for c in work.columns:
        raw = work[c]
        valid_ser = raw.dropna()
        valid = int(valid_ser.shape[0])
        missing_frac = float(1 - (valid / n)) if n else math.nan

        prevalence = math.nan
        nz = 0
        no = 0
        entropy_bits = math.nan
        const = True
        feature_type = "unknown"
        n_unique = 0

        if valid > 0:
            # Try binary-like interpretation first
            mapped = raw.copy()
            if raw.dtype == bool:
                mapped = raw.astype(int)
            elif raw.dtype == object:
                low = raw.astype(str).str.strip().str.lower()
                mapping = {
                    "present": 1,
                    "absent": 0,
                    "yes": 1,
                    "no": 0,
                    "true": 1,
                    "false": 0,
                    "t": 1,
                    "f": 0,
                    "+": 1,
                    "-": 0,
                }
                mapped = pd.to_numeric(low.map(mapping).fillna(low), errors="coerce")
            else:
                mapped = pd.to_numeric(raw, errors="coerce")

            mv = mapped.dropna()
            if len(mv) > 0 and set(pd.unique(mv.astype(float).round(6))).issubset(
                {0.0, 1.0}
            ):
                # True binary feature
                feature_type = "binary"
                bx = mv.astype(int)
                n_unique = int(bx.nunique(dropna=True))
                const = bool(n_unique <= 1)
                prevalence = float(bx.mean())
                nz = int((bx == 1).sum())
                no = int((bx == 0).sum())
                if prevalence in (0.0, 1.0):
                    entropy_bits = 0.0
                else:
                    p_ = prevalence
                    entropy_bits = float(
                        -(p_ * math.log2(p_) + (1 - p_) * math.log2(1 - p_))
                    )
            else:
                # Non-binary: categorical or numeric discrete/continuous
                if pd.api.types.is_numeric_dtype(raw):
                    feature_type = "numeric"
                    vals = pd.to_numeric(raw, errors="coerce").dropna()
                    # For high-cardinality numeric, compute entropy on rounded string bins (lightweight)
                    if len(vals) > 0:
                        vals_for_counts = vals.round(6).astype(str)
                    else:
                        vals_for_counts = vals.astype(str)
                else:
                    feature_type = "categorical"
                    vals_for_counts = raw.dropna().astype(str)

                n_unique = int(vals_for_counts.nunique(dropna=True))
                const = bool(n_unique <= 1)
                if len(vals_for_counts) > 0:
                    probs = vals_for_counts.value_counts(
                        normalize=True, dropna=True
                    ).values.astype(float)
                    entropy_bits = float(-(probs * np.log2(probs)).sum())
                # prevalence_* not applicable for non-binary features

        low_info = const
        if feature_type == "binary" and not math.isnan(prevalence):
            low_info = low_info or (
                prevalence < prevalence_floor or prevalence > prevalence_ceiling
            )

        score = 0.0
        if not math.isnan(entropy_bits):
            score += entropy_bits
        if not math.isnan(missing_frac):
            score *= max(0.0, 1.0 - missing_frac)

        out_rows.append(
            {
                "feature": c,
                "feature_type": feature_type,
                "n_total": int(n),
                "n_valid": int(valid),
                "n_unique": int(n_unique),
                "missing_fraction": missing_frac,
                "n_present": int(nz),
                "n_absent": int(no),
                "prevalence_present": prevalence,
                "entropy_bits": entropy_bits,
                "constant_feature": const,
                "low_informativeness_flag": bool(low_info),
                "informativeness_score": float(score),
            }
        )
    out = pd.DataFrame(out_rows).sort_values(
        ["low_informativeness_flag", "informativeness_score"], ascending=[True, False]
    )
    return out.reset_index(drop=True)


def check_degeneracy(
    labels: Optional[Sequence] = None,
    feature_table: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    min_samples: int = 10,
) -> Dict[str, object]:
    warnings = []
    failures = []
    checks = {}

    if feature_table is not None:
        n = int(feature_table.shape[0])
        checks["n_samples"] = n
        if n < min_samples:
            failures.append(f"Too few samples ({n}) < min_samples ({min_samples})")
        work = (
            feature_table.drop(columns=[id_col])
            if (id_col and id_col in feature_table.columns)
            else feature_table.copy()
        )
        if work.shape[1] == 0:
            failures.append("No feature columns available after ID removal")
            checks["n_features"] = 0
        else:
            checks["n_features"] = int(work.shape[1])
            const_count = 0
            all_missing = 0
            binary_like_count = 0
            for c in work.columns:
                s = work[c]
                nonnull = s.dropna()
                if len(nonnull) == 0:
                    all_missing += 1
                    continue
                # determine if binary-like
                if s.dtype == bool:
                    vals = s.dropna().astype(int)
                    is_binary_like = True
                else:
                    if s.dtype == object:
                        low = s.astype(str).str.strip().str.lower()
                        mapping = {
                            "present": 1,
                            "absent": 0,
                            "yes": 1,
                            "no": 0,
                            "true": 1,
                            "false": 0,
                            "t": 1,
                            "f": 0,
                            "+": 1,
                            "-": 0,
                        }
                        coerced = pd.to_numeric(
                            low.map(mapping).fillna(low), errors="coerce"
                        )
                    else:
                        coerced = pd.to_numeric(s, errors="coerce")
                    cv = coerced.dropna()
                    is_binary_like = len(cv) > 0 and set(
                        pd.unique(cv.astype(float).round(6))
                    ).issubset({0.0, 1.0})
                    vals = cv.astype(int) if is_binary_like else nonnull.astype(str)
                if is_binary_like:
                    binary_like_count += 1
                if pd.Series(vals).nunique(dropna=True) <= 1:
                    const_count += 1
            checks["binary_like_feature_count"] = int(binary_like_count)
            checks["constant_feature_count"] = int(const_count)
            checks["all_missing_feature_count"] = int(all_missing)
            if const_count == work.shape[1]:
                failures.append(
                    "All features are constant (zero entropy / no variation)"
                )
            elif const_count / max(1, work.shape[1]) > 0.8:
                warnings.append(
                    f"High fraction of constant features ({const_count}/{work.shape[1]})"
                )
            if all_missing > 0:
                warnings.append(f"{all_missing} feature(s) are entirely missing")
    if labels is not None:
        ser = pd.Series(labels)
        n_labels = ser.notna().sum()
        uniq = ser.dropna().nunique()
        checks["n_labels_non_missing"] = int(n_labels)
        checks["n_unique_labels"] = int(uniq)
        if n_labels == 0:
            failures.append(
                "No labels available for clustering/network/MDR consistency"
            )
        elif uniq <= 1:
            failures.append("Degenerate labels: <=1 unique class/community/cluster")
        elif uniq / max(1, n_labels) > 0.5:
            warnings.append(
                f"Very fragmented labels: {uniq} unique labels among {n_labels} items"
            )
    status = "PASS"
    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
    }


def quality_gate(
    preflights: Sequence[DatasetPreflight],
    overlap_df: pd.DataFrame,
    config: Optional[Mapping[str, object]] = None,
    degeneracy_reports: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, object]:
    cfg = dict(config or {})
    thresholds = {
        "min_jaccard_warn": float(cfg.get("min_jaccard_warn", 0.20)),
        "min_jaccard_fail": float(cfg.get("min_jaccard_fail", 0.05)),
        "max_missing_id_fraction_warn": float(
            cfg.get("max_missing_id_fraction_warn", 0.05)
        ),
        "max_duplicate_id_fraction_warn": float(
            cfg.get("max_duplicate_id_fraction_warn", 0.05)
        ),
    }
    warnings = []
    failures = []

    pf_dicts = [asdict(p) for p in preflights]
    for p in preflights:
        miss_frac = (p.missing_id_rows / p.rows) if p.rows else 0.0
        dup_frac = (p.duplicate_id_rows / p.rows) if p.rows else 0.0
        if miss_frac > thresholds["max_missing_id_fraction_warn"]:
            warnings.append(f"{p.name}: high missing ID fraction ({miss_frac:.3f})")
        if dup_frac > thresholds["max_duplicate_id_fraction_warn"]:
            warnings.append(f"{p.name}: high duplicate ID fraction ({dup_frac:.3f})")
        if p.rows == 0:
            failures.append(f"{p.name}: dataset has zero rows")
        if p.cols <= 1:
            warnings.append(f"{p.name}: <=1 column")
    if overlap_df is not None and not overlap_df.empty:
        min_j = (
            overlap_df["jaccard"].dropna().min()
            if "jaccard" in overlap_df.columns
            else math.nan
        )
        if not math.isnan(min_j):
            if min_j < thresholds["min_jaccard_fail"]:
                failures.append(
                    f"Very low cross-layer ID overlap detected (min Jaccard={min_j:.3f})"
                )
            elif min_j < thresholds["min_jaccard_warn"]:
                warnings.append(
                    f"Low cross-layer ID overlap detected (min Jaccard={min_j:.3f})"
                )
    deg_reports = degeneracy_reports or {}
    for key, rep in deg_reports.items():
        if rep.get("status") == "FAIL":
            failures.append(
                f"{key}: degeneracy fail - " + "; ".join(rep.get("failures", []))
            )
        elif rep.get("status") == "WARN":
            warnings.append(
                f"{key}: degeneracy warnings - " + "; ".join(rep.get("warnings", []))
            )
    status = (
        "PASS" if not warnings and not failures else ("FAIL" if failures else "WARN")
    )
    return {
        "status": status,
        "thresholds": thresholds,
        "warnings": warnings,
        "failures": failures,
        "datasets": pf_dicts,
        "n_dataset_layers": len(preflights),
        "n_pairwise_overlap_checks": int(0 if overlap_df is None else len(overlap_df)),
    }


def build_run_manifest_extension(
    tool_name: str,
    version: str,
    input_paths: Mapping[str, str | Path],
    config_snapshot: Mapping[str, object],
    output_dir: str | Path,
) -> Dict[str, object]:
    out = {
        "tool_name": tool_name,
        "tool_version": version,
        "output_dir": str(_ensure_path(output_dir)),
        "config_snapshot": dict(config_snapshot),
        "input_files": {},
    }
    for key, p in input_paths.items():
        pp = _ensure_path(p)
        if pp.exists() and pp.is_file():
            out["input_files"][key] = {
                "path": str(pp),
                "sha256": sha256_file(pp),
                "size_bytes": pp.stat().st_size,
            }
        else:
            out["input_files"][key] = {"path": str(pp), "exists": False}
    return out


def write_reliability_bundle(
    outdir: str | Path,
    quality_gate_obj: Mapping[str, object],
    warnings_obj: Mapping[str, object],
    feature_info_map: Mapping[str, pd.DataFrame],
    manifest_extension: Mapping[str, object],
) -> Dict[str, str]:
    outdir = _ensure_path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["quality_gate.json"] = str(
        save_json(quality_gate_obj, outdir / "quality_gate.json")
    )
    paths["analysis_validity_warnings.json"] = str(
        save_json(warnings_obj, outdir / "analysis_validity_warnings.json")
    )
    fi_serializable = {
        k: v.to_dict(orient="records") for k, v in feature_info_map.items()
    }
    paths["feature_informativeness_index.json"] = str(
        save_json(fi_serializable, outdir / "feature_informativeness_index.json")
    )
    # Also save CSV per layer for convenience
    fi_csv_dir = outdir / "feature_informativeness"
    fi_csv_dir.mkdir(exist_ok=True)
    for k, v in feature_info_map.items():
        p = fi_csv_dir / f"{k}_feature_informativeness.csv"
        v.to_csv(p, index=False)
    paths["run_manifest.json"] = str(
        save_json(manifest_extension, outdir / "run_manifest.json")
    )
    return paths


def sensitivity_minirun_binary_features(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    n_runs: int = 10,
    subsample_fraction: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Lightweight stability proxy without invoking tool-specific heavy algorithms.
    Computes prevalence and entropy stability under row subsampling.
    """
    rng = np.random.default_rng(random_state)
    work = df.copy()
    if id_col and id_col in work.columns:
        work = work.drop(columns=[id_col])
    work = normalize_binary_like(work)
    n = len(work)
    if n == 0 or work.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "run",
                "n_rows",
                "mean_entropy",
                "median_prevalence",
                "constant_feature_fraction",
            ]
        )
    rows = []
    idx = np.arange(n)
    k = max(2, int(round(n * subsample_fraction)))
    for r in range(n_runs):
        pick = rng.choice(idx, size=min(k, n), replace=False)
        sub = work.iloc[pick]
        fi = feature_informativeness_index(sub, id_col=None)
        ent = pd.to_numeric(fi["entropy_bits"], errors="coerce")
        prev = pd.to_numeric(fi["prevalence_present"], errors="coerce")
        constv = pd.to_numeric(fi["constant_feature"], errors="coerce").astype(float)
        rows.append(
            {
                "run": r + 1,
                "n_rows": int(len(sub)),
                "mean_entropy": (
                    float(ent.dropna().mean()) if ent.notna().any() else math.nan
                ),
                "median_prevalence": (
                    float(prev.dropna().median()) if prev.notna().any() else math.nan
                ),
                "constant_feature_fraction": (
                    float(constv.dropna().mean()) if constv.notna().any() else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)

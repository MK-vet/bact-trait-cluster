
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from .core import (
    build_run_manifest_extension,
    check_degeneracy,
    feature_informativeness_index,
    layer_coverage_against_reference,
    pairwise_id_overlap,
    preflight_dataset,
    quality_gate,
    save_json,
    sensitivity_minirun_binary_features,
    write_reliability_bundle,
)


def _parse_dataset_args(dataset_args):
    parsed = {}
    for item in dataset_args or []:
        if "=" not in item:
            raise SystemExit(f"--dataset must be name=path, got: {item}")
        name, path = item.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def _parse_idcol_args(idcol_args):
    parsed = {}
    for item in idcol_args or []:
        if "=" not in item:
            raise SystemExit(f"--id-col must be name=column, got: {item}")
        name, col = item.split("=", 1)
        parsed[name.strip()] = col.strip()
    return parsed


def _load_config(path):
    if not path:
        return {}
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(txt) or {}
    return json.loads(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generic reliability hardening preflight for ssuis-* tools")
    ap.add_argument("--tool-name", required=True)
    ap.add_argument("--tool-version", default="0.0.0")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset", action="append", help="Dataset layer in name=path form", default=[])
    ap.add_argument("--id-col", action="append", help="Optional explicit ID column per dataset, name=column", default=[])
    ap.add_argument("--config", help="YAML/JSON config file with thresholds and preset settings")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--run-sensitivity", action="store_true")
    ap.add_argument("--reference-layer", default=None)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = _parse_dataset_args(args.dataset)
    idcol_overrides = _parse_idcol_args(args.id_col)
    cfg = _load_config(args.config)
    rel_cfg = cfg.get("reliability", cfg) if isinstance(cfg, dict) else {}

    preflights = []
    dfs: Dict[str, pd.DataFrame] = {}
    id_cols = {}
    feature_info = {}
    degeneracy = {}
    sensitivity = {}

    for name, path in datasets.items():
        pf, df = preflight_dataset(path, name=name, id_col=idcol_overrides.get(name))
        preflights.append(pf)
        dfs[name] = df
        id_cols[name] = pf.id_col
        feature_info[name] = feature_informativeness_index(df, id_col=pf.id_col)
        degeneracy[name] = check_degeneracy(feature_table=df, id_col=pf.id_col, min_samples=args.min_samples)
        if args.run_sensitivity:
            sensitivity[name] = sensitivity_minirun_binary_features(
                df, id_col=pf.id_col,
                n_runs=int(rel_cfg.get("sensitivity_n_runs", 10)),
                subsample_fraction=float(rel_cfg.get("sensitivity_subsample_fraction", 0.8)),
                random_state=int(rel_cfg.get("random_state", 42)),
            )

    overlap = pairwise_id_overlap(dfs, id_cols) if len(dfs) >= 2 else pd.DataFrame(
        columns=["dataset_a","dataset_b","n_ids_a","n_ids_b","intersection","union","jaccard","recall_a_in_b","recall_b_in_a"]
    )
    coverage = layer_coverage_against_reference(dfs, id_cols, reference_name=args.reference_layer)

    qg = quality_gate(
        preflights,
        overlap_df=overlap,
        config=rel_cfg,
        degeneracy_reports=degeneracy,
    )
    warnings_obj = {
        "status": "PASS" if qg["status"] == "PASS" else "WARN",
        "tool_name": args.tool_name,
        "warnings": qg.get("warnings", []),
        "failures": qg.get("failures", []),
        "degeneracy_reports": degeneracy,
    }
    manifest = build_run_manifest_extension(
        tool_name=args.tool_name,
        version=args.tool_version,
        input_paths=datasets,
        config_snapshot=cfg,
        output_dir=outdir,
    )
    bundle_paths = write_reliability_bundle(
        outdir=outdir,
        quality_gate_obj=qg,
        warnings_obj=warnings_obj,
        feature_info_map=feature_info,
        manifest_extension=manifest,
    )

    # Save tabular artifacts
    overlap.to_csv(outdir / "pairwise_id_overlap.csv", index=False)
    coverage.to_csv(outdir / "layer_coverage.csv", index=False)
    save_json([pf.__dict__ for pf in preflights], outdir / "data_preflight.json")
    if args.run_sensitivity:
        sens_dir = outdir / "sensitivity_miniruns"
        sens_dir.mkdir(exist_ok=True)
        for name, df in sensitivity.items():
            df.to_csv(sens_dir / f"{name}_sensitivity_miniruns.csv", index=False)

    # Minimal stdout summary (machine-friendly and human-readable)
    summary = {
        "tool": args.tool_name,
        "status": qg["status"],
        "n_datasets": len(preflights),
        "outdir": str(outdir),
        "artifacts": bundle_paths,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

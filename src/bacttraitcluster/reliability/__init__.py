from .core import (
    preflight_dataset,
    pairwise_id_overlap,
    layer_coverage_against_reference,
    feature_informativeness_index,
    check_degeneracy,
    quality_gate,
    build_run_manifest_extension,
    write_reliability_bundle,
    sensitivity_minirun_binary_features,
)

__all__ = [
    "preflight_dataset",
    "pairwise_id_overlap",
    "layer_coverage_against_reference",
    "feature_informativeness_index",
    "check_degeneracy",
    "quality_gate",
    "build_run_manifest_extension",
    "write_reliability_bundle",
    "sensitivity_minirun_binary_features",
    "build_consistency_report",
    "run_reliability_preflight",
    "run_consistency_from_spec",
]

def build_consistency_report(*args, **kwargs):
    from .consistency import build_consistency_report as _impl
    return _impl(*args, **kwargs)

# Integration layer imports
from pathlib import Path
import json
from typing import Any, Dict, Mapping
import yaml


# --- Integration layer (merged from reliability.py) ---

def _load_rel_cfg(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reliability config not found: {p}")
    text = p.read_text(encoding='utf-8')
    if p.suffix.lower() in {'.yaml', '.yml'}:
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _default_rel_outdir(cfg, override: str | None) -> Path:
    if override:
        return Path(override)
    base = Path(getattr(cfg, 'output_dir', 'results'))
    return base / '_reliability'


def _collect_table_inputs(cfg) -> Dict[str, str]:
    datasets: Dict[str, str] = {}
    if hasattr(cfg, 'layers') and getattr(cfg, 'layers', None):
        for sp in getattr(cfg, 'layers'):
            name = getattr(sp, 'name', None) or Path(getattr(sp, 'path')).stem
            datasets[str(name)] = str(getattr(sp, 'path'))
    if hasattr(cfg, 'input_csv') and getattr(cfg, 'input_csv', None):
        datasets.setdefault('MIC', str(getattr(cfg, 'input_csv')))
    if hasattr(cfg, 'gene_csv') and getattr(cfg, 'gene_csv', None):
        datasets.setdefault('AMR_genes', str(getattr(cfg, 'gene_csv')))
    # Optional common layers on some configs
    for attr in ('mic_csv', 'amr_genes_csv', 'virulence_csv', 'mge_csv', 'plasmid_csv', 'mlst_csv', 'serotype_csv'):
        if hasattr(cfg, attr) and getattr(cfg, attr, None):
            datasets.setdefault(attr.replace('_csv','').replace('_','').upper() if attr.endswith('_csv') else attr, str(getattr(cfg, attr)))
    return datasets


def _collect_extra_inputs(cfg) -> Dict[str, str]:
    extra = {}
    tree = getattr(cfg, 'tree', None)
    if tree is not None and hasattr(tree, 'path') and getattr(tree, 'path', None):
        extra['Tree'] = str(getattr(tree, 'path'))
    return extra


def run_reliability_preflight(
    cfg,
    *,
    tool_name: str,
    tool_version: str,
    cfg_path: str | None = None,
    reliability_config_path: str | None = None,
    reliability_outdir: str | None = None,
    fail_fast: bool = False,
    run_sensitivity: bool = False,
) -> Dict[str, Any]:
    rel_cfg = _load_rel_cfg(reliability_config_path)
    outdir = _default_rel_outdir(cfg, reliability_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = _collect_table_inputs(cfg)
    if not datasets:
        raise RuntimeError('No tabular input datasets detected for reliability preflight')

    preflights, dfs, id_cols, fi_map, deg = [], {}, {}, {}, {}
    sensitivity_dir = outdir / 'sensitivity_miniruns'
    sens_cfg = dict(rel_cfg.get('sensitivity') or {})
    run_sensitivity = bool(run_sensitivity or sens_cfg.get('enabled', False))

    min_samples = int(rel_cfg.get('min_samples', 10))
    for name, path in datasets.items():
        pf, df = preflight_dataset(path, name=name)
        preflights.append(pf)
        dfs[name] = df
        id_cols[name] = pf.id_col
        fi = feature_informativeness_index(df, id_col=pf.id_col)
        fi_map[name] = fi
        deg[name] = check_degeneracy(feature_table=df, id_col=pf.id_col, min_samples=min_samples)
        if run_sensitivity:
            try:
                s_df = sensitivity_minirun_binary_features(
                    df,
                    id_col=pf.id_col,
                    n_runs=int(sens_cfg.get('n_runs', 8)),
                    subsample_fraction=float(sens_cfg.get('subsample_fraction', 0.8)),
                    random_state=int(sens_cfg.get('random_state', 42)),
                )
                sensitivity_dir.mkdir(parents=True, exist_ok=True)
                s_df.to_csv(sensitivity_dir / f"{name}_sensitivity_minirun.csv", index=False)
            except Exception as e:
                deg[name].setdefault('warnings', []).append(f'sensitivity_minirun_failed: {e}')
                if deg[name].get('status') == 'PASS':
                    deg[name]['status'] = 'WARN'

    overlap = pairwise_id_overlap(dfs, id_cols)
    qg = quality_gate(preflights, overlap_df=overlap, config=rel_cfg, degeneracy_reports=deg)

    layer_cov = layer_coverage_against_reference(dfs, id_cols)
    layer_cov.to_csv(outdir / 'layer_coverage_against_reference.csv', index=False)
    overlap.to_csv(outdir / 'pairwise_id_overlap.csv', index=False)

    manifest_inputs = {}
    manifest_inputs.update(datasets)
    manifest_inputs.update(_collect_extra_inputs(cfg))
    manifest = build_run_manifest_extension(
        tool_name=tool_name,
        version=tool_version,
        input_paths=manifest_inputs,
        config_snapshot={
            'analysis_config_path': cfg_path,
            'reliability_config_path': reliability_config_path,
            'reliability_thresholds': {k:v for k,v in rel_cfg.items() if k != 'notes'},
        },
        output_dir=outdir,
    )
    warnings_obj = {
        'status': 'PASS' if qg.get('status') == 'PASS' else 'WARN',
        'warnings': qg.get('warnings', []),
        'failures': qg.get('failures', []),
        'degeneracy_reports': deg,
    }
    write_reliability_bundle(outdir, qg, warnings_obj, fi_map, manifest)

    status = qg.get('status', 'UNKNOWN')
    do_fail_fast = bool(fail_fast or rel_cfg.get('fail_fast_on_quality_gate_fail', False))
    if do_fail_fast and status == 'FAIL':
        raise RuntimeError('Reliability quality gate FAIL: ' + '; '.join(qg.get('failures', [])))

    return {
        'status': status,
        'outdir': str(outdir),
        'quality_gate': qg,
        'datasets': list(datasets.keys()),
        'sensitivity_enabled': run_sensitivity,
    }


def run_consistency_from_spec(spec_path: str, out_path: str | None = None) -> Dict[str, Any]:
    from bacttraitcluster.reliability.consistency import build_consistency_report
    p = Path(spec_path)
    if not p.exists():
        raise FileNotFoundError(f'Consistency spec not found: {p}')
    if p.suffix.lower() in {'.yaml', '.yml'}:
        spec = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    else:
        spec = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(spec, Mapping):
        raise ValueError('Consistency spec must be a mapping tool->spec')
    rep = build_consistency_report(spec, out_json=out_path)
    return rep

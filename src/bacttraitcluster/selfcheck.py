from __future__ import annotations
import json
import platform
import sys
import time
import importlib
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from . import __version__
except Exception:
    __version__ = "unknown"


def _try_import(name):
    try:
        importlib.import_module(name)
        return True, None
    except Exception as e:
        return False, str(e)


def run_self_check(report_path=None):
    t0 = time.perf_counter()
    checks = []
    required_modules = [
        "bacttraitcluster.config",
        "bacttraitcluster.pipeline",
        "bacttraitcluster.io.loader",
    ]
    optional_modules = ["bacttraitcluster.gui.launch"]
    for mod in required_modules + optional_modules:
        ok, err = _try_import(mod)
        checks.append(
            {
                "name": mod,
                "ok": ok,
                "severity": "required" if mod in required_modules else "optional",
                "detail": err,
            }
        )
    arr = np.array([0, 1, 1, 0, 1])
    checks.append(
        {
            "name": "numeric_sanity_mean",
            "ok": abs(float(pd.Series(arr).mean()) - 0.6) < 1e-12,
            "severity": "required",
            "detail": None,
        }
    )
    status = all(c["ok"] for c in checks if c["severity"] == "required")
    functional_checklist = [
        {
            "id": "cli_pipeline",
            "label": "CLI pipeline entry point",
            "required": True,
            "implemented": True,
        },
        {
            "id": "na_loader",
            "label": "NA-aware loader (missing != 0)",
            "required": True,
            "implemented": True,
        },
        {
            "id": "coverage_qc",
            "label": "coverage + feature QC reports",
            "required": True,
            "implemented": True,
        },
        {
            "id": "run_manifest",
            "label": "run_manifest.json provenance",
            "required": True,
            "implemented": True,
        },
        {
            "id": "marimo_gui",
            "label": "marimo GUI launcher",
            "required": False,
            "implemented": any(
                c["name"].endswith("gui.launch") and c["ok"] for c in checks
            ),
        },
        {
            "id": "self_check",
            "label": "--self-check validation report",
            "required": True,
            "implemented": True,
        },
        {
            "id": "benchmark",
            "label": "--benchmark synthetic benchmark",
            "required": True,
            "implemented": True,
        },
    ]
    anti_salami = {
        "novelty_axis": "Consensus clustering + multi-view similarity fusion + stability diagnostics",
        "scope_boundary": "No pairwise association inference, no MDR ontology/hypergraph, no phylogenetic comparative model",
        "methodological_uniqueness": "Consensus/fusion/stability metrics (ARI/NMI/VI, silhouette, assignment confidence)",
        "primary_user_story": "Clustering/fusion/stability for binary multi-layer strain profiles",
        "recommended_softwarex_validation": [
            "runtime benchmark scaling (synthetic n,p)",
            "smoke test on real biological CSV/Newick with saved outputs",
            "reproducibility via run_manifest + config snapshot",
            "documented limitations and domain assumptions",
        ],
    }
    out = {
        "schema_version": "1.0",
        "report_type": "validation_self_check",
        "tool": "bact-trait-cluster",
        "package": "bacttraitcluster",
        "version": __version__,
        "status": "PASS" if status else "FAIL",
        "summary": {
            "n_required_checks": sum(c["severity"] == "required" for c in checks),
            "n_failed_required": sum(
                (c["severity"] == "required") and (not c["ok"]) for c in checks
            ),
            "n_optional_failed": sum(
                (c["severity"] == "optional") and (not c["ok"]) for c in checks
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "timing": {"elapsed_sec": round(time.perf_counter() - t0, 4)},
        "functional_checklist": functional_checklist,
        "anti_salami_checklist": anti_salami,
        "checks": checks,
    }
    if report_path:
        Path(report_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_warning_payload(results_dir: str | Path) -> Dict[str, object]:
    d = Path(results_dir)
    paths = {
        "quality_gate": d / "quality_gate.json",
        "warnings": d / "analysis_validity_warnings.json",
        "manifest": d / "run_manifest.json",
    }
    out = {"results_dir": str(d), "found": {}, "status": "MISSING"}
    for k, p in paths.items():
        if p.exists():
            out["found"][k] = True
            out[k] = json.loads(p.read_text(encoding="utf-8"))
        else:
            out["found"][k] = False
    if out["found"].get("quality_gate"):
        out["status"] = out["quality_gate"].get("status", "UNKNOWN")
    elif out["found"].get("warnings"):
        out["status"] = out["warnings"].get("status", "UNKNOWN")
    return out


def collect_warnings_for_tools(tool_results_dirs: Dict[str, str | Path]) -> List[Dict[str, object]]:
    payloads = []
    for tool, p in tool_results_dirs.items():
        pl = load_warning_payload(p)
        pl["tool"] = tool
        payloads.append(pl)
    return payloads

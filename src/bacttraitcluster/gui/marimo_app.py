import marimo as mo
import io
import json
import tempfile
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from bacttraitcluster.reliability.marimo_warnings import load_warning_payload
except Exception:
    load_warning_payload = None

app = mo.App(title="# bact-trait-cluster — Interactive dashboard")


def _read_csv_safe(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep="	")


def _csvs_recursive(d: Path):
    return sorted([p for p in d.rglob("*.csv") if p.is_file()])


def _render_reliability_banner(d: Path):
    if load_warning_payload is None:
        return None
    try:
        payload = load_warning_payload(d)
    except Exception as e:
        return mo.md(f"Reliability warnings: parse error ({e})")
    if payload.get("status") == "MISSING":
        return None
    qg = payload.get("quality_gate", {}) or {}
    n_w = len(qg.get("warnings", []) or [])
    n_f = len(qg.get("failures", []) or [])
    status = payload.get("status", "UNKNOWN")
    return mo.md(
        f"**Reliability status:** `{status}` | warnings: **{n_w}** | failures: **{n_f}** (from `quality_gate.json`) "
    )


def _paged(df: pd.DataFrame):
    ps = mo.ui.number(value=25, start=10, stop=500, step=5, label="Rows/page")
    npages = max(1, int(np.ceil(len(df) / max(1, int(ps.value)))))
    pg = mo.ui.number(value=1, start=1, stop=npages, step=1, label="Page")
    mo.hstack([ps, pg])
    st = (int(pg.value) - 1) * int(ps.value)
    mo.ui.table(df.iloc[st : st + int(ps.value)], page_size=int(ps.value))


def _plot(df: pd.DataFrame):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return
    dff = df.sample(min(len(df), 5000), random_state=0) if len(df) > 5000 else df
    kind = mo.ui.dropdown(
        {"hist": "Histogram", "scatter": "Scatter"},
        value=("scatter" if len(num) > 1 else "hist"),
        label="Plot",
    )
    x = mo.ui.dropdown({c: c for c in num}, value=num[0], label="X")
    y = mo.ui.dropdown(
        {c: c for c in num}, value=(num[1] if len(num) > 1 else num[0]), label="Y"
    )
    mo.hstack([kind, x, y])
    fig = (
        px.histogram(dff, x=x.value)
        if kind.value == "hist"
        else px.scatter(dff, x=x.value, y=y.value, opacity=0.5)
    )
    mo.ui.plotly(fig)


def _guess_id_from_upload(upload):
    try:
        for f in upload.value or []:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(f.contents), nrows=3)
                for c in df.columns:
                    if c.lower() in (
                        "strain_id",
                        "strainid",
                        "isolate_id",
                        "sample_id",
                        "id",
                    ):
                        return c
                return df.columns[0]
    except Exception:
        pass
    return "Strain_ID"


def _save_uploads(upload):
    tmp = Path(tempfile.mkdtemp(prefix="ssuis_gui_"))
    out = []
    for f in upload.value or []:
        p = tmp / f.name
        p.write_bytes(f.contents)
        out.append(p)
    return tmp, out


@app.cell
def _():
    mo.md("# bact-trait-cluster — Interactive dashboard")
    return


@app.cell
def _():
    tabs = mo.ui.tabs(
        {
            "Explore results": mo.ui.markdown(
                "Browse output CSVs recursively (paged tables + downsampled plots)."
            ),
            "Run analysis": mo.ui.markdown(
                "Upload inputs, edit tool-specific settings, run CLI."
            ),
            "Data editor": mo.ui.markdown(
                "Rename/recode/drop columns and export edited CSV."
            ),
            "Domain editor": mo.ui.markdown(
                "Tool-specific domain settings (YAML) for universality across bacteria/species/ontologies."
            ),
            "Methods": mo.ui.markdown("Statistics used and interpretation notes."),
        },
        value="Explore results",
    )
    tabs
    return tabs


@app.cell
def _(tabs):
    if tabs.value != "Explore results":
        return
    outdir = mo.ui.text(value=str(Path.cwd()), label="Output directory")
    outdir
    return outdir


@app.cell
def _(tabs, outdir):
    if tabs.value != "Explore results" or outdir is None:
        return
    d = Path(outdir.value).expanduser().resolve()
    if not d.exists():
        mo.md("Directory does not exist")
        return
    banner = _render_reliability_banner(d)
    csvs = _csvs_recursive(d)
    if not csvs:
        mo.md("No CSV files found.")
        return
    sel = mo.ui.dropdown(
        {p.relative_to(d).as_posix(): str(p) for p in csvs},
        value=str(csvs[0]),
        label="Table",
    )
    if banner is not None:
        banner
    mo.md(f"Found **{len(csvs)}** CSV files under `{d}`.")
    sel
    return sel


@app.cell
def _(tabs, sel):
    if tabs.value != "Explore results" or sel is None:
        return
    p = Path(sel.value)
    df = _read_csv_safe(p)
    mo.md(f"### `{p.name}` — {df.shape[0]}×{df.shape[1]}")
    _paged(df)
    _plot(df)
    return


@app.cell
def _(tabs):
    if tabs.value != "Methods":
        return
    mo.md(
        "## Methods and interpretation — bact-trait-cluster\n\nConsensus clustering, stability diagnostics, and multi-view fusion for sparse microbial trait layers (MIC/AMR/virulence/MGE/plasmids). Missingness is preserved (NA != 0). Interpret clusters using stability + biological coherence, not only a single metric.\n\nUse the Domain editor to adapt layers and species-specific conventions."
    )
    return


@app.cell
def _(tabs):
    if tabs.value != "Domain editor":
        return
    mo.md(
        "## Cluster domain editor\nPaste/edit YAML that describes layers, fusion, and clustering ranges. This enables reuse across bacterial species and naming conventions."
    )
    domain_cfg_txt = mo.ui.text_area(
        value="layers:\n  - name: MIC\n    path: MIC.csv\n    id_column: Strain_ID\n  - name: AMR_genes\n    path: AMR_genes.csv\n    id_column: Strain_ID\n  - name: Virulence\n    path: Virulence.csv\n    id_column: Strain_ID\n",
        rows=20,
        label="Domain YAML (editable)",
    )
    # tool-specific dynamic widgets (optional)
    fusion = mo.ui.dropdown(
        {"true": "true", "false": "false"}, value="true", label="Fusion enabled"
    )
    kmin = mo.ui.number(value=2, start=2, stop=20, step=1, label="k min")
    kmax = mo.ui.number(value=8, start=2, stop=50, step=1, label="k max")
    mo.hstack([fusion, kmin, kmax])
    domain_cfg_txt.value = (
        domain_cfg_txt.value
        + f"\nconsensus:\n  k_range: [{int(kmin.value)}, {int(kmax.value)}]\nfusion:\n  enabled: {fusion.value}\n"
        if "consensus:" not in domain_cfg_txt.value
        else domain_cfg_txt.value
    )
    domain_cfg_txt
    return domain_cfg_txt


@app.cell
def _(tabs):
    if tabs.value != "Run analysis":
        return
    upload = mo.ui.file(
        multiple=True,
        filetypes=[".csv", ".newick", ".nwk", ".yaml", ".yml"],
        label="Upload inputs",
    )
    upload
    return upload


@app.cell
def _(tabs, upload):
    if tabs.value != "Run analysis" or upload is None:
        return
    outdir = mo.ui.text(
        value=str((Path.cwd() / "ssuis_output").resolve()), label="Output directory"
    )
    id_col = mo.ui.text(
        value=_guess_id_from_upload(upload), label="Strain/sample ID column"
    )
    species = mo.ui.text(value="Bacteria sp.", label="Species label")
    overrides = mo.ui.text_area(value="", rows=12, label="YAML overrides (advanced)")
    mo.vstack([outdir, mo.hstack([id_col, species]), overrides])
    return outdir, id_col, species, overrides


@app.cell
def _(tabs):
    if tabs.value != "Run analysis":
        return
    run_btn = mo.ui.button(label="Run analysis", kind="success")
    run_btn
    return run_btn


@app.cell
def _(tabs, run_btn, upload, outdir, id_col, species, overrides):
    if tabs.value != "Run analysis" or run_btn is None:
        return
    logs = mo.ui.text_area(value="", rows=18, label="Logs")
    if run_btn.value:
        import yaml as _yaml

        _, saved = _save_uploads(upload)
        cfg = {
            "output_dir": str(Path(outdir.value).expanduser().resolve()),
            "metadata": {"species": species.value},
        }
        # generic file mapping
        layers = []
        for p in saved:
            low = p.name.lower()
            if low.endswith((".newick", ".nwk")):
                cfg.setdefault("tree", {})["path"] = str(p)
                continue
            if p.suffix.lower() == ".csv":
                layers.append(
                    {"name": p.stem, "path": str(p), "id_column": id_col.value}
                )
        if True:
            cfg["layers"] = layers
        else:
            csvs = [Path(x["path"]) for x in layers]
            mic = next(
                (p for p in csvs if "mic" in p.stem.lower()),
                (csvs[0] if csvs else None),
            )
            genes = next(
                (
                    p
                    for p in csvs
                    if ("amr" in p.stem.lower() or "gene" in p.stem.lower())
                ),
                None,
            )
            if mic:
                cfg["input_csv"] = str(mic)
                cfg["input"] = {"id_column": id_col.value}
            if genes and genes != mic:
                cfg["gene_csv"] = str(genes)
        # merge domain editor YAML if available
        try:
            from marimo._runtime.state import (
                get_context,
            )  # may fail depending on version

            _ = get_context
        except Exception:
            pass
        # user can paste domain YAML into overrides if needed; also try to auto-read Domain editor via widget name if same kernel state keeps object in globals
        try:
            dcfg = globals().get("domain_cfg_txt")
            if dcfg is not None and getattr(dcfg, "value", "").strip():
                dv = _yaml.safe_load(dcfg.value)
                if isinstance(dv, dict):
                    cfg.update(dv)
        except Exception as e:
            logs.value += f"\n[domain YAML parse error] {e}\n"
        if overrides.value.strip():
            try:
                ov = _yaml.safe_load(overrides.value)
                if isinstance(ov, dict):
                    cfg.update(ov)
            except Exception as e:
                logs.value += f"\n[override parse error] {e}\n"
        tmp = Path(tempfile.mkdtemp(prefix="ssuis_cfg_"))
        cfg_path = tmp / "config.yaml"
        cfg_path.write_text(_yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "bacttraitcluster" + ".cli",
            "--config",
            str(cfg_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        logs.value = (proc.stdout or "") + "\n" + (proc.stderr or "")
    logs
    return


@app.cell
def _(tabs):
    if tabs.value != "Data editor":
        return
    up = mo.ui.file(multiple=False, filetypes=[".csv"], label="Upload CSV to edit")
    up
    return up


@app.cell
def _(tabs, up):
    if tabs.value != "Data editor" or up is None or not up.value:
        return
    df = pd.read_csv(io.BytesIO(up.value.contents))
    mo.md(f"Loaded `{up.value.name}`: **{df.shape[0]}×{df.shape[1]}**")
    _paged(df.head(1000))
    rename_map = mo.ui.text_area(value="{}", rows=5, label="Rename mapping JSON")
    recode_col = mo.ui.text(value="", label="Column to recode")
    recode_map = mo.ui.text_area(value="{}", rows=4, label="Recode mapping JSON")
    drop_cols = mo.ui.text(value="", label="Drop columns (comma-separated)")
    outname = mo.ui.text(value=f"edited_{up.value.name}", label="Output filename")
    save_btn = mo.ui.button(label="Export edited CSV", kind="success")
    mo.vstack([rename_map, recode_col, recode_map, drop_cols, outname, save_btn])
    if save_btn.value:
        df2 = df.copy()
        try:
            mp = json.loads(rename_map.value)
            if isinstance(mp, dict):
                df2 = df2.rename(columns=mp)
        except Exception:
            pass
        if recode_col.value and recode_col.value in df2.columns:
            try:
                rmp = json.loads(recode_map.value)
                if isinstance(rmp, dict):
                    df2[recode_col.value] = df2[recode_col.value].replace(rmp)
            except Exception:
                pass
        dc = [c.strip() for c in drop_cols.value.split(",") if c.strip()]
        if dc:
            df2 = df2.drop(columns=[c for c in dc if c in df2.columns], errors="ignore")
        outp = Path.cwd() / outname.value
        df2.to_csv(outp, index=False)
        mo.md(f"Saved `{outp}` ({df2.shape[0]}×{df2.shape[1]}).")
    return


if __name__ == "__main__":
    app.run()

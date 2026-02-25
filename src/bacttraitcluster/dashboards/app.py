import marimo


app = marimo.App(width="full")


@app.cell
def _():
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import yaml

    return Path, json, mo, pd, subprocess, sys, time, yaml


@app.cell
def _(mo):
    mo.md(
        """
        # bact-trait-cluster — interactive dashboard (marimo)

        This app supports:
        - config builder + editable YAML editor,
        - non-blocking run launcher (spawns the CLI and tails logs),
        - scalable result browsing (paged tables + preview row limits),
        - quick plots for the most common artifacts.
        """
    )
    return


@app.cell
def _(mo, Path):
    data_root = mo.ui.text(value=str(Path.cwd()), label="Data root")
    out_root = mo.ui.text(value=str(Path.cwd() / "bacttraitcluster_out"), label="Output directory")

    mic_path = mo.ui.text(value="MIC.csv", label="MIC CSV")
    amr_path = mo.ui.text(value="AMR_genes.csv", label="AMR genes CSV (optional)")
    vir_path = mo.ui.text(value="Virulence.csv", label="Virulence CSV (optional)")
    mge_path = mo.ui.text(value="MGE.csv", label="MGE CSV (optional, long)")
    plas_path = mo.ui.text(value="Plasmid.csv", label="Plasmid CSV (optional, long)")
    mlst_path = mo.ui.text(value="MLST.csv", label="MLST CSV (optional)")
    sero_path = mo.ui.text(value="Serotype.csv", label="Serotype CSV (optional)")

    align_mode = mo.ui.dropdown(options=["union", "intersection"], value="union", label="align_mode")
    schema_version = mo.ui.text(value="1.1", label="schema_version")
    config_strict = mo.ui.switch(value=False, label="config_strict")

    k_min = mo.ui.number(value=2, label="k_min", step=1)
    k_max = mo.ui.number(value=6, label="k_max", step=1)
    n_consensus = mo.ui.number(value=50, label="n_consensus_runs", step=10)
    enable_profiling = mo.ui.switch(value=True, label="profiling.enabled")

    mo.md("## Config builder")
    mo.vstack(
        [
            data_root,
            out_root,
            mo.hstack([mic_path, mlst_path, sero_path]),
            mo.hstack([amr_path, vir_path]),
            mo.hstack([mge_path, plas_path]),
            mo.hstack([align_mode, schema_version, config_strict]),
            mo.hstack([k_min, k_max, n_consensus, enable_profiling]),
        ]
    )

    return (
        Path,
        align_mode,
        amr_path,
        config_strict,
        data_root,
        enable_profiling,
        k_max,
        k_min,
        mic_path,
        mge_path,
        mlst_path,
        n_consensus,
        out_root,
        plas_path,
        schema_version,
        sero_path,
        vir_path,
    )


@app.cell
def _(Path, mo, yaml, align_mode, amr_path, config_strict, data_root, enable_profiling, k_max, k_min, mic_path, mge_path, mlst_path, n_consensus, out_root, plas_path, schema_version, sero_path, vir_path):
    def _abs(p: str) -> str:
        p = p.strip()
        if not p:
            return ""
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str(Path(data_root.value) / pp)

    layers = [{"name": "MIC", "path": _abs(mic_path.value), "format": "wide"}]
    for name, widget, fmt in [
        ("AMR_genes", amr_path, "wide"),
        ("Virulence", vir_path, "wide"),
        ("MGE", mge_path, "long"),
        ("Plasmid", plas_path, "long"),
        ("MLST", mlst_path, "wide"),
        ("Serotype", sero_path, "wide"),
    ]:
        p = _abs(widget.value)
        if p:
            layers.append({"name": name, "path": p, "format": fmt})

    cfg = {
        "schema_version": schema_version.value,
        "config_strict": bool(config_strict.value),
        "output_dir": str(Path(out_root.value)),
        "align_mode": align_mode.value,
        "layers": layers,
        "clustering": {
            "k_min": int(k_min.value),
            "k_max": int(k_max.value),
            "n_consensus_runs": int(n_consensus.value),
            "algorithms": ["agglomerative_hamming", "spectral_jaccard", "kmodes"],
        },
        "profiling": {
            "enabled": bool(enable_profiling.value),
            "fdr_alpha": 0.05,
            "effect_size": "cliffs_delta",
        },
        "reliability": {"enabled": True, "fail_fast": False},
    }

    editor = mo.ui.text_area(value=yaml.safe_dump(cfg, sort_keys=False), label="Config YAML (editable)", rows=22)
    mo.md("## Config editor")
    editor
    return editor


@app.cell
def _(Path, json, mo, subprocess, sys, time, editor, out_root):
    mo.md("## Run controls")
    start = mo.ui.button(label="Start analysis", kind="success")
    refresh = mo.ui.button(label="Refresh status")
    mo.hstack([start, refresh])

    out_dir = Path(out_root.value)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config_used.yaml"
    log_path = out_dir / "run.log"
    state_path = out_dir / "run_state.json"

    if start.value:
        cfg_path.write_text(editor.value, encoding="utf-8")
        cmd = [sys.executable, "-m", "bacttraitcluster.cli", "--config", str(cfg_path)]
        with open(log_path, "ab") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(out_dir))
        state_path.write_text(
            json.dumps({"pid": proc.pid, "cmd": cmd, "started_at": time.time(), "cwd": str(out_dir)}, indent=2),
            encoding="utf-8",
        )
    return cfg_path, log_path, refresh, state_path


@app.cell
def _(mo, log_path, refresh):
    if not log_path.exists():
        mo.md("No log yet.")
        return
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20000), 0)
        tail = f.read().decode("utf-8", errors="replace")
    mo.md("### Log (tail)")
    mo.code(tail)
    return


@app.cell
def _(Path, mo, out_root, pd):
    mo.md("## Results explorer")
    out_dir = Path(out_root.value)
    pattern = mo.ui.text(value="*.csv", label="File glob")
    limit = mo.ui.number(value=200, label="Preview rows", step=50)
    label_map = mo.ui.text_area(value="{}", label="Label map (JSON) — applied to displayed tables", rows=4)
    mo.vstack([mo.hstack([pattern, limit]), label_map])

    files = sorted([p for p in out_dir.rglob(pattern.value) if p.is_file()]) if out_dir.exists() else []
    if not files:
        mo.md("No matching files yet.")
        return
    display = files[:200]
    selector = mo.ui.dropdown(
        options=[str(p.relative_to(out_dir)) for p in display],
        value=str(display[0].relative_to(out_dir)),
        label=f"Select a file ({len(display)}/{len(files)})",
    )
    file_path = out_dir / selector.value
    selector
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, nrows=int(limit.value))
        # Apply optional display label map
        try:
            mapping = __import__("json").loads(label_map.value or "{}")
            if isinstance(mapping, dict) and mapping:
                df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
                # Replace exact string matches in object columns
                for c in df.columns:
                    if df[c].dtype == object:
                        df[c] = df[c].replace(mapping)
        except Exception:
            pass
        mo.ui.table(df, pagination=True, label=f"Preview: {selector.value}")
    else:
        mo.md(f"Selected: `{selector.value}`")
    return


@app.cell
def _(Path, mo, out_root, pd):
    mo.md("## Quick plots")
    out_dir = Path(out_root.value)
    stab = out_dir / "stability_path.csv"
    if not stab.exists():
        mo.md("`stability_path.csv` not found yet.")
        return
    try:
        df = pd.read_csv(stab)
    except Exception as e:
        mo.md(f"Failed to read stability_path.csv: `{e}`")
        return

    k_col = "k" if "k" in df.columns else df.columns[0]
    mean_col = "nvi_mean" if "nvi_mean" in df.columns else ("mean" if "mean" in df.columns else None)
    lo_col = "nvi_ci_low" if "nvi_ci_low" in df.columns else None
    hi_col = "nvi_ci_high" if "nvi_ci_high" in df.columns else None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if mean_col is None:
        ax.plot(df[k_col], df[df.columns[1]], marker="o")
        ax.set_ylabel(df.columns[1])
    else:
        ax.plot(df[k_col], df[mean_col], marker="o")
        if lo_col and hi_col:
            ax.fill_between(df[k_col], df[lo_col], df[hi_col], alpha=0.2)
        ax.set_ylabel("NVI")
    ax.set_xlabel("k")
    ax.set_title("Stability path")
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("## Output reference")
    artifacts = [
        {"file": "stability_path.csv", "meaning": "Stability vs k (NVI). Lower NVI => more stable partitions across bootstrap runs.", "interpretation": "Pick k at a stability elbow; avoid k where CI is wide."},
        {"file": "clusters.csv", "meaning": "Per-layer cluster labels.", "interpretation": "Check if clusters match known biology; validate with enrichment/effect sizes."},
        {"file": "consensus_matrix.npy", "meaning": "Co-association probabilities from consensus clustering.", "interpretation": "Block-diagonal structure indicates stable clusters."},
        {"file": "feature_qc.csv", "meaning": "Feature prevalence/missingness filters applied.", "interpretation": "If key features are filtered, adjust thresholds."},
        {"file": "layer_coverage.csv", "meaning": "Sample overlap per layer.", "interpretation": "Low overlap => fused clusters may be driven by a subset."},
        {"file": "integrated_clusters.csv", "meaning": "All layers' cluster labels aligned by Strain_ID.", "interpretation": "Use to compare phenotypic vs genotypic groupings."},
        {"file": "layer_concordance.csv", "meaning": "Agreement (ARI/NMI/VI) between layers.", "interpretation": "High concordance suggests shared structure; discordance suggests recombination/HGT or measurement differences."},
        {"file": "layer_weights.csv", "meaning": "Weights used in multi-view fusion.", "interpretation": "Higher weight => layer contributes more to fused consensus."},
        {"file": "fused_clusters.csv", "meaning": "Final fused clustering.", "interpretation": "Treat as a hypothesis; confirm with independent signals (phylogeny, metadata)."},
        {"file": "prediction_strength.csv", "meaning": "Out-of-bootstrap replicability metric.", "interpretation": "Low values indicate unstable clustering; reduce k or adjust filtering."},
        {"file": "enrichment_z.csv", "meaning": "Cluster-wise enrichment z-scores with BH-FDR.", "interpretation": "Focus on effect size + prevalence, not only q-values."},
        {"file": "cliff_delta.csv", "meaning": "Nonparametric effect size per feature (Cliff's delta) with CI.", "interpretation": "Prioritise large |delta| with tight CI."},
        {"file": "config_validation.json", "meaning": "Config schema + unknown-key report.", "interpretation": "If strict mode fails, fix config keys."},
        {"file": "run_manifest.json", "meaning": "Input hashes + environment + algorithms used/missing.", "interpretation": "Use for reproducibility and audit trails."},
    ]
    mo.ui.table(artifacts, pagination=True, label="Artifacts")

    mo.md(
        """
        ## Statistical assumptions (practical)
        - Binary features are treated as presence/absence; prevalence filtering is essential to avoid sparse-feature artifacts.
        - Stability (NVI) is estimated via bootstrap/resampling; it is not a p-value.
        - Enrichment is multiple-tested; interpret with biological plausibility and effect sizes.
        """
    )
    return


if __name__ == "__main__":
    app.run()

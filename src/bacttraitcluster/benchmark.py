
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
def _assoc(df, max_pairs=2000):
    a=df.to_numpy(dtype=np.uint8); n=a.shape[1]; cnt=0; checksum=0.0
    for i in range(min(n,2000)):
        xi=a[:,i]
        for j in range(i+1,min(n,2000)):
            xj=a[:,j]
            q=((xi==1)&(xj==1)).sum(); w=((xi==1)&(xj==0)).sum(); e=((xi==0)&(xj==1)).sum(); r=((xi==0)&(xj==0)).sum()
            den=((q+w)*(e+r)*(q+e)*(w+r))**0.5
            if den: checksum += (q*r-w*e)/den
            cnt += 1
            if cnt>=max_pairs: return cnt, float(checksum)
    return cnt, float(checksum)
def run_benchmark(out_path=None, n_rows=300, p_values=(1000,5000,20000), seed=123):
    rng=np.random.default_rng(seed); rows=[]
    for p in p_values:
        X=(rng.random((n_rows,p))<0.08).astype(np.uint8)
        df=pd.DataFrame(X)
        t=time.perf_counter(); prev=df.mean(0); informative=int(((prev>0.01)&(prev<0.99)).sum()); qc=time.perf_counter()-t
        t=time.perf_counter(); pairs, chk=_assoc(df); assoc=time.perf_counter()-t
        rows.append({'n_rows':n_rows,'p':int(p),'informative_features':informative,'qc_sec':round(qc,4),
                      'assoc2000pairs_sec':round(assoc,4),'pairs_eval':pairs,'checksum':round(chk,6)})
    out={'tool':'bact-trait-cluster','results':rows}
    if out_path: Path(out_path).write_text(json.dumps(out,indent=2))
    return out


def run_benchmark_paper_ready(out_dir, n_rows=300, p_values=(1000,5000,20000), seed=123):
    from pathlib import Path
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = run_benchmark(None, n_rows=n_rows, p_values=p_values, seed=seed)
    (out_dir / "benchmark_synthetic.json").write_text(json.dumps(js, indent=2))
    df = pd.DataFrame(js.get("results", []))
    if not df.empty:
        df.insert(0, "tool", js.get("tool", "unknown"))
        df.to_csv(out_dir / "benchmark_synthetic.csv", index=False)
        fig = plt.figure(figsize=(6,4))
        plt.plot(df["p"], df["qc_sec"], marker="o", label="QC")
        plt.plot(df["p"], df["assoc2000pairs_sec"], marker="o", label="Assoc (2000 pairs)")
        plt.xlabel("Number of features (p)")
        plt.ylabel("Time (s)")
        plt.xscale("log")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "benchmark_runtime_vs_p.png", dpi=160)
        plt.close(fig)
        fig = plt.figure(figsize=(6,4))
        plt.plot(df["p"], df["informative_features"], marker="o")
        plt.xlabel("Number of features (p)")
        plt.ylabel("Informative features")
        plt.xscale("log")
        plt.tight_layout()
        fig.savefig(out_dir / "benchmark_informative_vs_p.png", dpi=160)
        plt.close(fig)
    return js

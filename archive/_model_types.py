"""Architecture (model-type) analysis on stage-1 pixel-val ROC.
Normalises each config as Δ vs THAT pool's incumbent (removes per-pool val-distribution
differences), then averages Δ across pools. Ranks configs + knob main-effects.
NOTE: pixel-val is the shortlisting metric; only ne30_mf32 was carried to country validation
(where it gave ~+0.010 n-wtd over incumbent).
"""
import glob, os, numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/Solar_Workspace"))
D = "Solar-Siting/artifacts/paper_v3/budget_sweep"
PL = {"px_v3":"cap120","T1000000":"1M","T2000000":"2M","T4000000":"4M",
      "T6000000":"6M","T10000000":"10M","T200000000":"uncap"}

frames = []
for f in sorted(glob.glob(f"{D}/*_stage1.csv")):
    d = pd.read_csv(f)
    pool = d["pool"].iloc[0]
    inc = d.loc[d.tag == "incumbent", "val_roc"]
    if not len(inc): continue
    d["delta"] = d["val_roc"] - float(inc.iloc[0])
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
sweep = df[df.tag != "incumbent"].copy()
print(f"pools: {df.pool.nunique()}  |  configs/pool completed: "
      + ", ".join(f"{PL[p]}={ (df.pool==p).sum() }" for p in PL))

# ---- config ranking: mean Δ vs incumbent, across pools (only configs in >=5 pools) ----
g = sweep.groupby("tag").agg(mean_delta=("delta","mean"), sd=("delta","std"),
                             npools=("delta","size"), mean_roc=("val_roc","mean"),
                             ne=("n_estimators","first"), mf=("max_features","first"),
                             ms=("max_samples","first"), crit=("criterion","first"),
                             leaves=("mean_leaves","mean"), mb=("size_mb","mean"))
g = g[g.npools >= 5].sort_values("mean_delta", ascending=False)
pd.set_option("display.width", 200)
print("\n=== TOP 12 architectures by mean Δpixel-ROC vs incumbent (avg across pools) ===")
print(g.head(12).round(4).to_string())
print("\n=== BOTTOM 5 ===")
print(g.tail(5).round(4).to_string())

# ---- knob main-effects (mean Δ, averaging over everything else) ----
print("\n=== knob main-effects on Δpixel-ROC vs incumbent (mean across all configs/pools) ===")
for knob in ["n_estimators","max_features","max_samples","criterion"]:
    m = sweep.groupby(knob)["delta"].mean().round(4)
    print(f"\n {knob}:")
    print(m.to_string())

# ---- incumbent reference: its own pixel ROC per pool ----
print("\n=== incumbent pixel-val ROC per pool (the baseline Δ=0 is measured against) ===")
print(df[df.tag=="incumbent"].set_index("pool")["val_roc"].round(4).to_string())

"""Per-country BEST-model ROC across all 7 data pools (t-2 country scores, cached npz, no GEE).
Answers: is cap120 best in EVERY country, or only n-weighted? Marks each country's winning pool."""
import glob, os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace"))
SC = "Solar-Siting/artifacts/paper_v3/budget_sweep/scores/country"
PL = {"px_v3":"cap120","T1000000":"1M","T2000000":"2M","T4000000":"4M",
      "T6000000":"6M","T10000000":"10M","T200000000":"uncap"}
order = ["cap120","1M","2M","4M","6M","10M","uncap"]
rows = {}; nsite = {}
for f in glob.glob(f"{SC}/*_BEST_*_2019_scores.npz"):
    b = os.path.basename(f)
    m = re.match(r"(?P<c>[a-z_]+)__(?P<pool>px_v3|T\d+)_BEST_.*_2019_scores\.npz", b)
    if not m: continue
    c, pool = m["c"], PL[m["pool"]]
    z = np.load(f); s, r = z["site_s"], z["rand_s"]
    roc = roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(r))], np.r_[s, r])
    rows.setdefault(c, {})[pool] = round(roc, 4); nsite[c] = len(s)
df = pd.DataFrame(rows).T
df = df[[c for c in order if c in df.columns]]
df["n"] = pd.Series(nsite)
df = df.sort_values("n", ascending=False)
# winning pool per country (among pools present)
poolcols = [c for c in order if c in df.columns]
df["BEST_POOL"] = df[poolcols].idxmax(axis=1)
df["cap120_gap"] = (df["cap120"] - df[poolcols].max(axis=1)).round(4)  # 0 => cap120 wins; neg => another pool better
pd.set_option("display.width", 220)
print("=== per-country BEST-model ROC by data pool (t-2) ===")
print(df.to_string())
print("\n=== where does cap120 NOT win? (cap120_gap < 0) ===")
lose = df[df.cap120_gap < 0]
print(lose[["n","cap120","BEST_POOL","cap120_gap"]].to_string() if len(lose) else "  cap120 wins or ties in ALL countries")
print("\n=== count of countries won by each pool ===")
print(df.BEST_POOL.value_counts().to_string())
# n-weighted per pool
print("\n=== n-weighted ROC per pool ===")
for c in poolcols:
    sub = df[[c,"n"]].dropna(); print(f"  {c:>7}: {np.average(sub[c],weights=sub['n']):.4f}")

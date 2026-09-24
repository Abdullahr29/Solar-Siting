"""Reconstruct the full country ROC table directly from saved npz score arrays
(bypasses the CSV, which is missing rows dropped by the png-thumbnail bug in run_country_rf).
ROC = roc_auc_score([1]*site + [0]*rand, concat(site_s, rand_s)) -- identical to run_country line 157.
"""
import glob, os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace"))
D = "Solar-Siting/artifacts/paper_v3/budget_sweep/scores/country"
POOLS = ["px_v3", "T1000000", "T2000000", "T4000000", "T6000000", "T10000000", "T200000000"]
PLABEL = {"px_v3":"cap120", "T1000000":"1M", "T2000000":"2M", "T4000000":"4M",
          "T6000000":"6M", "T10000000":"10M", "T200000000":"uncap"}

rows = {}
nsites = {}
for f in sorted(glob.glob(f"{D}/*.npz")):
    b = os.path.basename(f)
    m = re.match(r"(?P<c>[a-z_]+)__(?P<pool>px_v3|T\d+)_(?P<kind>BEST_.*|incumbent)_2019_scores\.npz", b)
    if not m:
        print("SKIP", b); continue
    c, pool, kind = m["c"], m["pool"], m["kind"]
    z = np.load(f)
    s, r = z["site_s"], z["rand_s"]
    y = np.r_[np.ones(len(s)), np.zeros(len(r))]
    x = np.r_[s, r]
    roc = roc_auc_score(y, x)
    col = f"{PLABEL[pool]}_{'BEST' if kind.startswith('BEST') else 'inc'}"
    rows.setdefault(c, {})[col] = round(roc, 4)
    nsites[c] = len(s)

df = pd.DataFrame(rows).T
df["n"] = pd.Series(nsites)
# column order
cols = []
for p in POOLS:
    cols += [f"{PLABEL[p]}_inc", f"{PLABEL[p]}_BEST"]
cols = [c for c in cols if c in df.columns]
df = df[cols + ["n"]]
df = df.sort_values("n", ascending=False)
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
print("=== per-country ROC (inc = that pool's incumbent arch; BEST = ne30_mf32 winner) ===")
print(df.to_string())

# n-weighted means (only countries present for that column)
print("\n=== n-weighted mean ROC per column ===")
wm = {}
for c in cols:
    sub = df[[c, "n"]].dropna()
    wm[c] = round(np.average(sub[c], weights=sub["n"]), 4)
print(pd.Series(wm).to_string())

# headline: deployed model (cap120_inc) vs each pool's BEST, n-weighted, common countries
print("\n=== HEADLINE: best-per-pool BEST vs the DEPLOYED model (cap120_inc), n-weighted ===")
base = "cap120_inc"
for p in POOLS:
    bc = f"{PLABEL[p]}_BEST"
    if bc not in df.columns: continue
    sub = df[[base, bc, "n"]].dropna()
    d = np.average(sub[bc]-sub[base], weights=sub["n"])
    print(f"  {bc:>12} - {base}: {d:+.4f}  (n-wtd over {len(sub)} countries)")

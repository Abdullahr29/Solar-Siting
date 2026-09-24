"""Probe n_estimators BELOW 30 at the fixed ~9MB leaf budget (the one untested lever).
For ne in {15,20,25,30}: max_leaf_nodes = BUDGET//ne (fewer trees -> richer trees), mf=32.
Train on POOL (default px_v3=cap120), evaluate COUNTRY ROC locally on cached AEF-2019 val
embeddings (n-weighted AND macro-avg), report GEE string MB. Saves models. No GEE for scoring.
"""
import os, sys, time, glob, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, "Solar-Siting")
os.environ.setdefault("POOL", "px_v3")
os.environ.setdefault("BUDGET_LEAVES", "63000")
import gee_tree_fix; gee_tree_fix.apply()
import geemap.ml as ml
BANDS = [f"A{i:02d}" for i in range(64)]
BUDGET = int(os.environ["BUDGET_LEAVES"]); POOL = os.environ["POOL"]; NJOBS = 16
POOLD = "Solar-Siting/artifacts/paper_v3/pools"
OUT = "Solar-Siting/artifacts/paper_v3/pools_t3/treecount"; os.makedirs(OUT, exist_ok=True)
VAL = "Solar-Siting/artifacts/paper_v3/pools_t3/val_emb"

def load_pool():
    """Inline (rf_budget_sweep.py is not import-safe). cap120 = px_v3_train.npz."""
    if POOL != "px_v3":
        raise SystemExit(f"treecount currently supports POOL=px_v3 only (got {POOL})")
    z = np.load(f"{POOLD}/px_v3_train.npz")
    Xs, Xn = z["X_sol"], z["X_neg"]
    X = np.concatenate([Xs, Xn]).astype(np.float32)
    y = np.r_[np.ones(len(Xs), np.int8), np.zeros(len(Xn), np.int8)]
    keep = np.isfinite(X).all(1)
    return X[keep], y[keep]

print(f"POOL={POOL} BUDGET_LEAVES={BUDGET}", flush=True)
X, y = load_pool()
print(f"pool: {len(X):,} rows ({int(y.sum()):,} pos)", flush=True)

# cached val embeddings per country
val = {}
for f in glob.glob(f"{VAL}/*_val2019.npz"):
    c = os.path.basename(f).split("_val2019")[0]; z = np.load(f)
    Xs = z["site_emb"]; Xr = z["rand_emb"]
    val[c] = (Xs[np.isfinite(Xs).all(1)], Xr[np.isfinite(Xr).all(1)])
def country_roc(rf):
    per = {}
    for c,(Xs,Xr) in val.items():
        yy = np.r_[np.ones(len(Xs)), np.zeros(len(Xr))]
        xx = np.r_[rf.predict_proba(Xs)[:,1], rf.predict_proba(Xr)[:,1]]
        per[c] = (roc_auc_score(yy,xx), len(Xs))
    rocs = np.array([v[0] for v in per.values()]); ns = np.array([v[1] for v in per.values()])
    return np.average(rocs, weights=ns), rocs.mean(), per

rows = []
for ne in [15, 20, 25, 30]:
    mln = BUDGET // ne
    t = time.time()
    rf = RandomForestClassifier(n_estimators=ne, max_leaf_nodes=mln, max_features=32,
                                min_samples_leaf=20, class_weight="balanced",
                                n_jobs=NJOBS, random_state=0).fit(X, y)
    joblib.dump(rf, f"{OUT}/{POOL}_ne{ne}_mf32.joblib", compress=3)
    tr = ml.rf_to_strings(rf, BANDS, processes=8, output_mode="PROBABILITY")
    mb = sum(len(s) for s in tr)/1e6
    nw, macro, _ = country_roc(rf)
    leaves = np.mean([e.get_n_leaves() for e in rf.estimators_])
    print(f"ne{ne:>3} mln={mln:>4} leaves={leaves:.0f} GEE={mb:.2f}MB  country n-wtd={nw:.4f} macro={macro:.4f}  ({time.time()-t:.0f}s)", flush=True)
    rows.append(dict(ne=ne, max_leaf_nodes=mln, mean_leaves=round(leaves), gee_mb=round(mb,2),
                     country_nwtd=round(nw,4), country_macro=round(macro,4)))
import pandas as pd
pd.DataFrame(rows).to_csv(f"{OUT}/{POOL}_treecount.csv", index=False)
print(f"wrote {OUT}/{POOL}_treecount.csv", flush=True)

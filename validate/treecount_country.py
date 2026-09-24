"""Full country validation for ne15/20/25/30 (cap120, mf32) on cached AEF-2019 val embeddings.
Per-country ROC + n-weighted + macro-avg + bootstrap over countries (which ne wins, robustly)."""
import os, glob, numpy as np, joblib, pandas as pd
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
TC = "artifacts/paper_v3/pools_t3/treecount"; VAL = "artifacts/paper_v3/pools_t3/val_emb"
MODELS = {f"ne{ne}": f"{TC}/px_v3_ne{ne}_mf32.joblib" for ne in [15,20,25,30]}
rfs = {k: joblib.load(v) for k,v in MODELS.items()}
val = {}
for f in glob.glob(f"{VAL}/*_val2019.npz"):
    c = os.path.basename(f).split("_val2019")[0]; z = np.load(f)
    Xs = z["site_emb"]; Xr = z["rand_emb"]
    val[c] = (Xs[np.isfinite(Xs).all(1)], Xr[np.isfinite(Xr).all(1)])
rows = {}; nsite = {}
for c,(Xs,Xr) in val.items():
    y = np.r_[np.ones(len(Xs)), np.zeros(len(Xr))]; nsite[c] = len(Xs)
    for k,rf in rfs.items():
        x = np.r_[rf.predict_proba(Xs)[:,1], rf.predict_proba(Xr)[:,1]]
        rows.setdefault(c, {})[k] = round(roc_auc_score(y,x),4)
df = pd.DataFrame(rows).T[list(MODELS)]; df["n"] = pd.Series(nsite)
df = df.sort_values("n", ascending=False)
df["best_ne"] = df[list(MODELS)].idxmax(1)
pd.set_option("display.width",160)
print("=== per-country ROC by tree count (cap120, mf32) ===")
print(df.to_string())
M = df[list(MODELS)].to_numpy(); n = df["n"].to_numpy(); cols=list(MODELS)
print("\n=== n-weighted / macro-avg ===")
for i,k in enumerate(cols):
    print(f"  {k}: n-wtd={np.average(M[:,i],weights=n):.4f}  macro={M[:,i].mean():.4f}")
print(f"\nper-country winner counts:\n{df.best_ne.value_counts().to_string()}")
rng=np.random.default_rng(0); B=10000; win={k:0 for k in cols}; winm={k:0 for k in cols}
for _ in range(B):
    bi=rng.integers(0,len(df),len(df))
    win[cols[int(np.argmax(np.average(M[bi],axis=0,weights=n[bi])))]]+=1
    winm[cols[int(np.argmax(M[bi].mean(0)))]]+=1
print("\n=== bootstrap over countries: P(best) ===")
for k in cols: print(f"  {k}: n-wtd {100*win[k]/B:5.1f}%   macro {100*winm[k]/B:5.1f}%")

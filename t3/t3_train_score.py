"""Train t-3 models on cap120_t3, score them + the existing t-2 cap120 models LOCALLY on the
AEF-2019 validation embeddings, and print the t-2 vs t-3 country ROC comparison (n-weighted).
Local scoring == GEE (confirmed exact). Both lags scored on the SAME validation points."""
import os, glob, time, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
POOLD_T3 = "Solar-Siting/artifacts/paper_v3/pools_t3"
MODELS_T2 = "Solar-Siting/artifacts/paper_v3/budget_sweep/models"
VAL = f"{POOLD_T3}/val_emb"
NJOBS = 12

CFG = {
 "incumbent":  dict(n_estimators=30, max_depth=12, min_samples_leaf=1500, max_features="sqrt",
                    max_samples=0.34, criterion="gini"),
 "ne30_mf32":  dict(n_estimators=30, max_leaf_nodes=2100, max_features=32, min_samples_leaf=20,
                    max_depth=None, criterion="gini"),
 "ne30_mf24":  dict(n_estimators=30, max_leaf_nodes=2100, max_features=24, min_samples_leaf=20,
                    max_depth=None, criterion="gini"),
 "ne30_mf48":  dict(n_estimators=30, max_leaf_nodes=2100, max_features=48, min_samples_leaf=20,
                    max_depth=None, criterion="gini"),
}
T2_MODEL = {"incumbent": f"{MODELS_T2}/px_v3_incumbent.joblib",
            "ne30_mf32": f"{MODELS_T2}/px_v3_ne30_mf32_ms0.34_gini.joblib"}

def train_t3():
    z = np.load(f"{POOLD_T3}/cap120_t3_train.npz")
    Xs, Xn = z["X_sol"], z["X_neg"]
    X = np.concatenate([Xs, Xn]).astype(np.float32)
    y = np.r_[np.ones(len(Xs), np.int8), np.zeros(len(Xn), np.int8)]
    keep = np.isfinite(X).all(1); X, y = X[keep], y[keep]
    print(f"t3 pool: {len(Xs):,} pos / {len(Xn):,} neg -> {len(X):,} finite rows", flush=True)
    os.makedirs(f"{POOLD_T3}/models", exist_ok=True)
    out = {}
    for tag, kw in CFG.items():
        p = f"{POOLD_T3}/models/cap120_t3_{tag}.joblib"
        if os.path.exists(p):
            out[tag] = joblib.load(p); print(f"  [load] {tag}"); continue
        t = time.time()
        rf = RandomForestClassifier(class_weight="balanced", n_jobs=NJOBS, random_state=0, **kw).fit(X, y)
        joblib.dump(rf, p, compress=3); out[tag] = rf
        print(f"  trained t3 {tag} ({time.time()-t:.0f}s)", flush=True)
    return out

def score(models_by_lag):
    files = sorted(glob.glob(f"{VAL}/*_val2019.npz"))
    rows = {}
    for f in files:
        c = os.path.basename(f).split("_val2019")[0]
        z = np.load(f); Xs, Xr = z["site_emb"], z["rand_emb"]
        fs = np.isfinite(Xs).all(1); fr = np.isfinite(Xr).all(1)
        Xs, Xr = Xs[fs], Xr[fr]
        y = np.r_[np.ones(len(Xs)), np.zeros(len(Xr))]
        rows[c] = {"n": len(Xs)}
        for lag, models in models_by_lag.items():
            for tag, rf in models.items():
                x = np.r_[rf.predict_proba(Xs)[:,1], rf.predict_proba(Xr)[:,1]]
                rows[c][f"{lag}_{tag}"] = roc_auc_score(y, x)
    return rows

if __name__ == "__main__":
    t3 = train_t3()
    t2 = {tag: joblib.load(p) for tag, p in T2_MODEL.items()}
    rows = score({"t2": t2, "t3": t3})
    import pandas as pd
    df = pd.DataFrame(rows).T.sort_values("n", ascending=False)
    cols = ["n","t2_incumbent","t3_incumbent","t2_ne30_mf32","t3_ne30_mf32","t3_ne30_mf24","t3_ne30_mf48"]
    df = df[[c for c in cols if c in df.columns]]
    pd.set_option("display.width",220)
    print("\n=== per-country ROC: t-2 vs t-3 (cap120) ===")
    print(df.round(4).to_string())
    print("\n=== n-weighted means ===")
    for c in df.columns:
        if c=="n": continue
        sub=df[[c,"n"]].dropna(); print(f"  {c:>14}: {np.average(sub[c],weights=sub['n']):.4f}")
    df.to_csv(f"{POOLD_T3}/t3_vs_t2_country.csv")
    print(f"\nwrote {POOLD_T3}/t3_vs_t2_country.csv", flush=True)

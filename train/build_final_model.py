"""Build + validate the candidate FINAL model: cap120 . ne15 . mf32 . t-3.
Train ne15_mf32 on the cap120 t-3 pool, then validate LOCALLY (no GEE): country ROC (cached
AEF-2019 val embeddings), deep-lookback (cached AEF-2017), + GEE string size. Confirms the
ne15 x t-3 combination behaves as the separately-validated pieces predict."""
import os, time, glob, re, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
import sys; sys.path.insert(0, ".")
import gee_tree_fix; gee_tree_fix.apply()
import geemap.ml as ml
BANDS = [f"A{i:02d}" for i in range(64)]
P3 = "artifacts/paper_v3/pools_t3"; VAL = f"{P3}/val_emb"
OUTM = f"{P3}/models/cap120_t3_ne15_mf32.joblib"

# ---- 1. TRAIN ----
z = np.load(f"{P3}/cap120_t3_train.npz"); Xs, Xn = z["X_sol"], z["X_neg"]
X = np.concatenate([Xs, Xn]).astype(np.float32)
y = np.r_[np.ones(len(Xs), np.int8), np.zeros(len(Xn), np.int8)]
keep = np.isfinite(X).all(1); X, y = X[keep], y[keep]
print(f"cap120 t-3 pool: {len(X):,} rows ({int(y.sum()):,} pos)", flush=True)
t = time.time()
rf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=4200, max_features=32,
                            min_samples_leaf=20, class_weight="balanced", n_jobs=16, random_state=0).fit(X, y)
joblib.dump(rf, OUTM, compress=3)
print(f"trained + saved cap120_t3_ne15_mf32 ({time.time()-t:.0f}s)", flush=True)
tr = ml.rf_to_strings(rf, BANDS, processes=8, output_mode="PROBABILITY"); mb = sum(len(s) for s in tr)/1e6
print(f"GEE string: {mb:.2f} MB  (ne15-t2 was 9.09MB and deployed OK)", flush=True)

# ---- 2. COUNTRY (cached AEF-2019 val embeddings) ----
rows = {}; nsite = {}
for f in glob.glob(f"{VAL}/*_val2019.npz"):
    c = os.path.basename(f).split("_val2019")[0]; d = np.load(f)
    Xso, Xr = d["site_emb"], d["rand_emb"]; Xso, Xr = Xso[np.isfinite(Xso).all(1)], Xr[np.isfinite(Xr).all(1)]
    yy = np.r_[np.ones(len(Xso)), np.zeros(len(Xr))]
    rows[c] = roc_auc_score(yy, np.r_[rf.predict_proba(Xso)[:,1], rf.predict_proba(Xr)[:,1]]); nsite[c] = len(Xso)
s = pd.Series(rows); n = pd.Series(nsite)
print("\n=== COUNTRY (cap120.ne15.t-3) ===")
print(s.round(4).sort_index().to_string())
print(f"n-weighted={np.average(s, weights=n.reindex(s.index)):.4f}  macro={s.mean():.4f}")
print("  PROGRESSION (n-wtd country ROC): deployed-incumbent 0.8736 -> ne30_mf32.t-2 0.8844")
print("    -> ne15_mf32.t-2 0.8884 -> ne30.t-3 0.8850 -> [THIS: cap120.ne15.t-3]")
print("  best-so-far bar to beat/match = 0.8884 (ne15.t-2); want t-3's flatter deep-lookback too")

# ---- 3. DEEP-LOOKBACK (cached AEF-2017) ----
import geopandas as gpd
DB = "artifacts/paper_v3/temporal_probe/big"
base = gpd.read_file("global_solar_ml_pipeline.gpkg", layer="base_train"); bid = set(base["PV_ID"].tolist())
inv = gpd.read_file("global_pv_facility_inventory.gpkg")
held = inv[(~inv["PV_ID"].isin(bid)) & inv.year.between(2019, 2024)]
ev = pd.concat([sub.sample(min(1000, len(sub)), random_state=0) for _, sub in held.groupby("year")]).reset_index(drop=True)
Es = np.load(f"{DB}/emb_deepsite_2017.npy"); Er = np.load(f"{DB}/emb_deeprand_2017.npy")
oks = np.isfinite(Es).all(1); okr = np.isfinite(Er).all(1); lead = (ev.year.to_numpy() - 2017)[oks]
ps = rf.predict_proba(Es[oks])[:,1]; pr = rf.predict_proba(Er[okr])[:,1]
def roc(a, b): return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
print("\n=== DEEP-LOOKBACK (cap120.ne15.t-3) ===")
print(f"roc_all={roc(ps,pr):.4f}  " + " ".join(f"L{L}={roc(ps[lead==L],pr):.4f}" for L in range(2,8) if (lead==L).sum()>=30))
print("  (ref: ne15-t2 roc_all 0.9478 L7 0.9126 | ne30-t3 L7 0.9165)")
print("\nDONE build_final", flush=True)

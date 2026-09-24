"""Temporal-holdout training: train ne15/20/30 (mf32, 9MB budget) on the canonical temporal pool
px_temporal_le2021.npz (cap120 restricted to installs<=2023 / AEF<=2021 — never sees 2024 sites).
Scoring on 2024 installs is done separately (treecount_temporal_score.py)."""
import os, time, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
POOL = "artifacts/stats/px_temporal_le2021.npz"
OUT = "artifacts/paper_v3/pools_t3/treecount/temporal"; os.makedirs(OUT, exist_ok=True)
BUDGET = 63000; NJOBS = 16
z = np.load(POOL); Xs, Xn = z["X_sol"], z["X_neg"]
X = np.concatenate([Xs, Xn]).astype(np.float32)
y = np.r_[np.ones(len(Xs), np.int8), np.zeros(len(Xn), np.int8)]
keep = np.isfinite(X).all(1); X, y = X[keep], y[keep]
print(f"temporal pool (installs<=2023): {len(X):,} rows ({int(y.sum()):,} pos)", flush=True)
for ne in [15, 20, 30]:
    p = f"{OUT}/temporal_ne{ne}_mf32.joblib"
    if os.path.exists(p): print(f"  [exists] ne{ne}"); continue
    t = time.time()
    rf = RandomForestClassifier(n_estimators=ne, max_leaf_nodes=BUDGET//ne, max_features=32,
                                min_samples_leaf=20, class_weight="balanced",
                                n_jobs=NJOBS, random_state=0).fit(X, y)
    joblib.dump(rf, p, compress=3)
    print(f"  trained temporal ne{ne} ({time.time()-t:.0f}s)", flush=True)
print("DONE temporal train", flush=True)

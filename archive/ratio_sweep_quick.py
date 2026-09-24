"""Quick train-ratio sensitivity check (IMPROVEMENTS.md Track 1).

Isolates the ONE question: with class_weight="balanced", does the raw solar:non-solar TRAIN
ratio move the metric? Fix all positives; subsample negatives to several ratios; train the
deployed RF recipe (min_leaf=1500 fixed — no GEE squeeze, just comparability); score the FIXED
held-out test pool (px_v3_test, unchanged) so ROC is comparable across ratios (ROC is
prevalence-invariant anyway). Pixel-ROC is the fast proxy here; the definitive check reruns on
the master pool at T=2M with country forward-ROC.
"""
import os, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

WS = os.path.expanduser("~/Solar_Workspace")
POOL = f"{WS}/Solar-Siting/artifacts/paper_v3/pools"
RATIOS = [0.5, 1, 2, 3, 5]      # neg = ratio * n_pos (capped by what's available ~1:5)

tr = np.load(f"{POOL}/px_v3_train.npz")
Xs = tr["X_sol"].astype(np.float32)
Xn = tr["X_neg"].astype(np.float32)
te = np.load(f"{POOL}/px_v3_test.npz")
Xt = np.concatenate([te["X_sol"], te["X_neg"]]).astype(np.float32)
yt = np.r_[np.ones(te["X_sol"].shape[0]), np.zeros(te["X_neg"].shape[0])]
keep = np.isfinite(Xt).all(1); Xt, yt = Xt[keep], yt[keep]
print(f"pos {len(Xs):,} | neg avail {len(Xn):,} | test {len(yt):,} (chance {yt.mean():.3f})", flush=True)

rng = np.random.default_rng(0)
print(f"\n{'target':>7} {'actual':>7} {'n_neg':>12} {'ROC':>7} {'PR':>7} {'fit_s':>6}")
for r in RATIOS:
    nneg = min(int(r * len(Xs)), len(Xn))
    idx = rng.choice(len(Xn), nneg, replace=False)
    X = np.concatenate([Xs, Xn[idx]])
    y = np.r_[np.ones(len(Xs)), np.zeros(nneg)]
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=30, max_depth=12, min_samples_leaf=1500,
                                max_samples=0.34, class_weight="balanced", n_jobs=8,
                                random_state=0).fit(X, y)
    p = rf.predict_proba(Xt)[:, 1]
    roc, pr = roc_auc_score(yt, p), average_precision_score(yt, p)
    print(f"  1:{r:<4g} 1:{nneg/len(Xs):<4.2f} {nneg:>12,} {roc:>7.4f} {pr:>7.4f} {time.time()-t0:>6.0f}", flush=True)
    del X, y
print("\nverdict: if ROC spread across ratios is within ~0.005, the balanced-weights assumption holds.", flush=True)

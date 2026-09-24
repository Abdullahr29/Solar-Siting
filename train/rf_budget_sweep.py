"""Joint capacity x data-size sweep for the BEST GEE-deployable (<=9 MB) RF, per data size.

Research-backed design (Probst 2019 + sklearn docs):
  * `max_features` is the top accuracy lever and is FREE on model size -> swept full range.
  * Model size is controlled by `max_leaf_nodes` (best-first growth: keeps the most informative
    splits at a fixed leaf budget) -- strictly better than inflating `min_samples_leaf`.
  * GEE string size ~= n_estimators * leaves/tree * ~135 B, so the 9 MB budget is a hyperbola:
    n_estimators * max_leaf_nodes ~= BUDGET_LEAVES. Every config ships ~9 MB BY CONSTRUCTION,
    so exactly ONE fit per config (no squeeze-refit loop).

One job == one data size (POOL): "px_v3" (incumbent cap_sol=120 6M pool) or "T<pixels>" (sampled
from the master pool with the per-country spatial sampler). Stage-1 metric = held-out 15% pixel
ROC (shortlisting only); the honest country metric is Stage 2 (rf_budget_country.py).
"""
import os, sys, time, json, resource
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

WS   = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
SS   = f"{WS}/Solar-Siting"
POOLD = f"{SS}/artifacts/paper_v3/pools"
STAT = f"{SS}/artifacts/stats"
OUT  = f"{SS}/artifacts/paper_v3/budget_sweep"
os.makedirs(f"{OUT}/models", exist_ok=True)

from geemap import ml
BANDS = [f"A{i:02d}" for i in range(64)]
MB_CAP = 9.0
BUDGET_LEAVES = int(os.environ.get("BUDGET_LEAVES", 60000))   # ~8.1 MB at ~135 B/leaf (safe < 9)
POOL_SPEC = os.environ["POOL"]                                 # "px_v3" | "T6000000" | ...
NJOBS = int(os.environ.get("NJOBS", 12))
VAL_FRAC = 0.15
SEED = 0
rng = np.random.default_rng(SEED)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


# ---- per-country spatial sampler (identical to tsweep_v3 / train_candidates) ----
def even_alloc(counts, budget):
    counts = np.asarray(counts, np.int64); alloc = np.zeros_like(counts)
    active = counts > 0; remaining = int(min(budget, counts.sum()))
    while remaining > 0 and active.any():
        n = int(active.sum()); share = remaining // n
        if share == 0:
            cap = (counts - alloc) * active
            take = np.argsort(-cap)[:remaining]; alloc[take] += 1; remaining = 0; break
        cap = counts - alloc; take = np.minimum(cap, share) * active
        alloc += take; remaining -= int(take.sum()); active = (counts - alloc) > 0
    return alloc


def sample_country(idx, cell, clus, T):
    if len(idx) <= T:
        return idx
    ucell, cinv = np.unique(cell, return_inverse=True)
    cell_bud = even_alloc(np.bincount(cinv, minlength=len(ucell)), T)
    out = []
    for ci in range(len(ucell)):
        b = int(cell_bud[ci])
        if b == 0: continue
        m = cinv == ci; ii, cl = idx[m], clus[m]
        if b >= len(ii): out.append(ii); continue
        usite, sinv = np.unique(cl, return_inverse=True)
        site_bud = even_alloc(np.bincount(sinv, minlength=len(usite)), b)
        for si in range(len(usite)):
            sb = int(site_bud[si])
            if sb == 0: continue
            pix = ii[sinv == si]
            out.append(pix if sb >= len(pix) else rng.choice(pix, sb, replace=False))
    return np.concatenate(out)


def build_pool():
    if POOL_SPEC == "px_v3":
        z = np.load(f"{POOLD}/px_v3_train.npz")
        Xs, Xn = z["X_sol"], z["X_neg"]
        X = np.concatenate([Xs, Xn]).astype(np.float32)
        y = np.r_[np.ones(len(Xs), np.int8), np.zeros(len(Xn), np.int8)]
    else:
        T = int(POOL_SPEC[1:])
        z = np.load(f"{POOLD}/px_master_train.npz", allow_pickle=False)
        Xs = z["X_sol"]; iso = z["iso_sol"]; src = z["src_sol"]; pos_chips = z["pos_chips"]
        Xn = z["X_neg"]
        geo = pd.read_csv(f"{STAT}/chip_geo_pos.csv").set_index("chip")
        gname = geo.index.to_numpy()
        lat = np.floor(geo["lat"].to_numpy()).astype(np.int64)
        lon = np.floor(geo["lon"].to_numpy()).astype(np.int64)
        cell_by = dict(zip(gname, (lat + 90) * 360 + (lon + 180)))
        clus_by = dict(zip(gname, geo["cluster"].to_numpy().astype(np.int64)))
        cell_src = np.array([cell_by.get(n, -1) for n in pos_chips], np.int64)[src]
        clus_src = iso.astype(np.int64) * 10_000_000 + \
            np.array([clus_by.get(n, -1) for n in pos_chips], np.int64)[src]
        fin_s = np.isfinite(Xs).all(1)
        sel_parts = []
        for u in np.unique(iso):
            idx = np.where((iso == u) & fin_s)[0]
            sel_parts.append(idx if len(idx) <= T
                             else sample_country(idx, cell_src[idx], clus_src[idx], T))
        sel = np.sort(np.concatenate(sel_parts))
        fin_n = np.isfinite(Xn).all(1); nk = np.where(fin_n)[0]
        Xp = Xs[sel]
        X = np.concatenate([Xp, Xn[nk]]).astype(np.float32); del Xp
        y = np.r_[np.ones(len(sel), np.int8), np.zeros(len(nk), np.int8)]
    keep = np.isfinite(X).all(1); X, y = X[keep], y[keep]
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]   # shuffle once so val slice is mixed
    return X, y


def make_configs():
    """Coverage-maximal grid on the 9 MB budget line (max_leaf_nodes = BUDGET_LEAVES/n_est).
    max_features x n_estimators x max_samples fully crossed (gini); entropy probed on the
    ms=0.6 slice. Plus the exact incumbent (min_samples_leaf size control) as a control."""
    MF   = [8, 16, 24, 32, 48, 64]          # feature counts (sqrt(64)=8 is the incumbent's)
    NEST = [30, 45, 60, 80, 100, 150]
    MS   = [0.34, 0.6, 1.0]
    cfgs = [dict(tag="incumbent", n_estimators=30, max_depth=12, min_samples_leaf=1500,
                 max_features="sqrt", max_samples=0.34, criterion="gini", max_leaf_nodes=None)]
    for ne in NEST:
        mln = BUDGET_LEAVES // ne
        for mf in MF:
            for ms in MS:
                cfgs.append(dict(tag=f"ne{ne}_mf{mf}_ms{ms}_gini", n_estimators=ne,
                                 max_features=mf, max_samples=ms, criterion="gini",
                                 min_samples_leaf=20, max_depth=None, max_leaf_nodes=mln))
            # entropy probe on the ms=0.6 slice
            cfgs.append(dict(tag=f"ne{ne}_mf{mf}_ms0.6_entropy", n_estimators=ne,
                             max_features=mf, max_samples=0.6, criterion="entropy",
                             min_samples_leaf=20, max_depth=None, max_leaf_nodes=mln))
    return cfgs


SMOKE = os.environ.get("SMOKE", "") == "1"
print(f"POOL={POOL_SPEC} BUDGET_LEAVES={BUDGET_LEAVES} njobs={NJOBS} smoke={SMOKE}", flush=True)

# early exit (for dependency-chain successors): if the CSV already has every config, do nothing.
_csv = f"{OUT}/{POOL_SPEC}_stage1.csv"
if os.path.exists(_csv) and not SMOKE:
    _ncfg = len(make_configs())
    if len(pd.read_csv(_csv)) >= _ncfg:
        print(f"all {_ncfg} configs already done — exiting without rebuilding pool", flush=True)
        sys.exit(0)

t0 = time.time()
X, y = build_pool()
if SMOKE:
    si = rng.choice(len(X), min(300_000, len(X)), replace=False)
    X, y = X[si], y[si]
nval = int(len(X) * VAL_FRAC)
Xtr, ytr, Xva, yva = X[:-nval], y[:-nval], X[-nval:], y[-nval:]
print(f"pool built: {len(X):,} rows ({int(y.sum()):,} pos), val {len(yva):,}  ({time.time()-t0:.0f}s)",
      flush=True)

# resume: skip configs already recorded in the stage-1 CSV
csv_path = f"{OUT}/{POOL_SPEC}_stage1.csv"
done = set()
rows = []
if os.path.exists(csv_path) and not SMOKE:
    prev = pd.read_csv(csv_path)
    rows = prev.to_dict("records")
    done = set(prev.tag)
    print(f"resume: {len(done)} configs already done, skipping them", flush=True)
print(f"\n{'tag':>26} {'ne':>4} {'mln':>5} {'mf':>5} {'ms':>4} {'crit':>7} "
      f"{'leaves':>7} {'MB':>5} {'valROC':>7} {'fit_s':>6}", flush=True)
_configs = make_configs()[:3] if SMOKE else make_configs()
for i, c in enumerate(_configs):
    if c["tag"] in done:
        continue
    kw = {k: c[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features",
                            "max_samples", "criterion", "max_leaf_nodes") if k in c}
    t1 = time.time()
    rf = RandomForestClassifier(class_weight="balanced", n_jobs=NJOBS, random_state=0, **kw).fit(Xtr, ytr)
    p = rf.predict_proba(Xva)[:, 1]
    roc, pr = float(roc_auc_score(yva, p)), float(average_precision_score(yva, p))
    trees = ml.rf_to_strings(rf, BANDS, processes=32, output_mode="PROBABILITY")
    mb = sum(len(s) for s in trees) / 1e6
    leaves = int(np.mean([t.get_n_leaves() for t in rf.estimators_]))
    feasible = mb <= MB_CAP
    mpath = f"Solar-Siting/artifacts/paper_v3/budget_sweep/models/{POOL_SPEC}_{c['tag']}.joblib"
    joblib.dump(rf, os.path.join(WS, mpath), compress=3)
    print(f"{c['tag']:>26} {c['n_estimators']:>4} {str(c.get('max_leaf_nodes')):>5} "
          f"{str(c['max_features']):>5} {c['max_samples']:>4} {c['criterion']:>7} "
          f"{leaves:>7} {mb:>5.1f} {roc:>7.4f} {time.time()-t1:>6.0f}", flush=True)
    rows.append(dict(pool=POOL_SPEC, tag=c["tag"], **kw, mean_leaves=leaves, size_mb=round(mb, 2),
                     feasible=bool(feasible), val_roc=round(roc, 4), val_pr=round(pr, 4),
                     fit_s=round(time.time()-t1), model_path=mpath, peak_gb=round(peak_gb(), 0)))
    pd.DataFrame(rows).to_csv(f"{OUT}/{POOL_SPEC}_stage1.csv", index=False)   # checkpoint

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/{POOL_SPEC}_stage1.csv", index=False)
best = df[df.feasible].sort_values("val_roc", ascending=False).head(1)
print(f"\nBEST feasible for {POOL_SPEC}: {best.tag.iloc[0]} valROC={best.val_roc.iloc[0]} "
      f"({best.size_mb.iloc[0]}MB)  | total {time.time()-t0:.0f}s", flush=True)

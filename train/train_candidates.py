"""S2 step 1: train + SQUEEZE candidate RF models at chosen per-country budgets T, from the
master pool, using the tsweep_v3 spatial sampler and the deployed RF recipe. Each model is
grown until its geemap tree-strings fit the GEE cap (mb_cap=9.0), i.e. GEE-DEPLOYABLE, so the
country forward-validation (run_country_rf.py, server-side GEE scoring) is a fair shipped-vs-
shipped comparison against the incumbent rf30_final. Loads the 40 GB master pool ONCE; all
64.65M negatives fixed for every T. Saves one squeezed joblib per T + a manifest.
"""
import os, time, json, resource
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

WS   = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
SS   = f"{WS}/Solar-Siting"
POOL = f"{SS}/artifacts/paper_v3/pools"
STAT = f"{SS}/artifacts/stats"
MODELS = f"{SS}/artifacts/paper_v3/models"
os.makedirs(MODELS, exist_ok=True)

from geemap import ml
BANDS  = [f"A{i:02d}" for i in range(64)]
MB_CAP = float(os.environ.get("MB_CAP", 9.0))
TS     = [int(x) for x in os.environ.get("TS", "10000000,6000000,4000000,1000000").split(",")]
NJOBS  = int(os.environ.get("NJOBS", 12))
SEED   = 0
rng = np.random.default_rng(SEED)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


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


print("loading master pool ...", flush=True); t0 = time.time()
z = np.load(f"{POOL}/px_master_train.npz", allow_pickle=False)
Xs = z["X_sol"]; iso = z["iso_sol"]; src = z["src_sol"]
names = z["iso_names"]; pos_chips = z["pos_chips"]
Xn = z["X_neg"]
print(f"  pos {len(Xs):,}  neg {len(Xn):,}  ({time.time()-t0:.0f}s)", flush=True)

geo = pd.read_csv(f"{STAT}/chip_geo_pos.csv").set_index("chip")
gname = geo.index.to_numpy()
lat = np.floor(geo["lat"].to_numpy()).astype(np.int64)
lon = np.floor(geo["lon"].to_numpy()).astype(np.int64)
cell_by_name = dict(zip(gname, (lat + 90) * 360 + (lon + 180)))
clus_by_name = dict(zip(gname, geo["cluster"].to_numpy().astype(np.int64)))
cell_of_src = np.array([cell_by_name.get(n, -1) for n in pos_chips], np.int64)
clus_of_src = np.array([clus_by_name.get(n, -1) for n in pos_chips], np.int64)
cell_sol = cell_of_src[src]
clus_sol = iso.astype(np.int64) * 10_000_000 + clus_of_src[src]

fin_s = np.isfinite(Xs).all(1)
fin_n = np.isfinite(Xn).all(1); nkeep = np.where(fin_n)[0]
Xn_fix = np.ascontiguousarray(Xn[nkeep]); del Xn
print(f"  finite pos {int(fin_s.sum()):,} | fixed neg {len(Xn_fix):,}", flush=True)

# test pool for a reference pixel-ROC on each squeezed model
te = np.load(f"{POOL}/px_v3_test.npz")
Xt = np.concatenate([te["X_sol"], te["X_neg"]]).astype(np.float32)
yt = np.r_[np.ones(te["X_sol"].shape[0]), np.zeros(te["X_neg"].shape[0])]
keepm = np.isfinite(Xt).all(1); Xt, yt = Xt[keepm], yt[keepm]

uiso = np.unique(iso)
manifest = []
for T in TS:
    sel_parts, capped = [], 0
    for u in uiso:
        idx = np.where((iso == u) & fin_s)[0]
        if len(idx) <= T:
            sel_parts.append(idx)
        else:
            capped += 1
            sel_parts.append(sample_country(idx, cell_sol[idx], clus_sol[idx], T))
    sel = np.sort(np.concatenate(sel_parts))
    Xp = Xs[sel]
    X = np.concatenate([Xp, Xn_fix]); del Xp
    y = np.r_[np.ones(len(sel), np.int8), np.zeros(len(Xn_fix), np.int8)]
    n_train = len(y)
    # start leaf scaled to n_train (deployed: ~1500 fit <=9MB at ~36M rows) -> ~1 refit
    min_leaf = max(1500, int(1500 * n_train / 36e6))
    print(f"\n=== T={T/1e6:.1f}M  pos={len(sel):,} capped={capped} n_train={n_train:,} "
          f"start_leaf={min_leaf} ===", flush=True)
    while True:
        t1 = time.time()
        rf = RandomForestClassifier(n_estimators=30, max_depth=12, min_samples_leaf=min_leaf,
                                    max_samples=0.34, class_weight="balanced", n_jobs=NJOBS,
                                    random_state=0).fit(X, y)
        trees = ml.rf_to_strings(rf, BANDS, processes=32, output_mode="PROBABILITY")
        mb = sum(len(s) for s in trees) / 1e6
        print(f"  min_leaf={min_leaf}: {mb:.1f} MB ({time.time()-t1:.0f}s) peakRSS={peak_gb():.0f}GB",
              flush=True)
        if mb <= MB_CAP:
            break
        min_leaf = int(min_leaf * mb / (MB_CAP - 0.5))
    p = rf.predict_proba(Xt)[:, 1]
    roc, pr = float(roc_auc_score(yt, p)), float(average_precision_score(yt, p))
    out = f"{MODELS}/rf30_T{T//1000000}M.joblib"
    joblib.dump(rf, out, compress=3)
    sz = os.path.getsize(out) / 1e6
    print(f"  saved {out} ({sz:.1f}MB joblib, {mb:.1f}MB strings, leaf={min_leaf}) "
          f"pxROC={roc:.4f} PR={pr:.4f}", flush=True)
    manifest.append(dict(T=T, model=out, pos=int(len(sel)), capped=capped, neg=int(len(Xn_fix)),
                         n_train=int(n_train), min_leaf=int(min_leaf), mb_strings=round(mb, 1),
                         joblib_mb=round(sz, 1), px_roc=round(roc, 4), px_pr=round(pr, 4)))
    with open(f"{MODELS}/candidates_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    del X, y, rf, p

print(f"\nwrote {MODELS}/candidates_manifest.json  (total {time.time()-t0:.0f}s)", flush=True)

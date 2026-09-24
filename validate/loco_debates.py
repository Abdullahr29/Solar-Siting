"""LOCO (leave-one-country-out) for the three open debates, scored LOCALLY on cached val embeddings.
Excludes a country's POS+NEG from training (matches loco_country.py --exclude-iso3), retrains, scores
that country's post-2021 forward installs. Compares LOCO ROC across:
  ne15 vs ne30 (cap120), t-2 vs t-3 (cap120 ne30), 6M vs cap120 (ne30).
Held-out countries: Greece, Spain, Poland (the paper's LOCO set; ESP/POL also stress the pool debate).
Resumable per (config,country). Run on LOTUS (loads 40GB master for the 6M rebuilds)."""
import os, sys, json, time, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
POOLD = "artifacts/paper_v3/pools"; STAT = "artifacts/stats"
VAL = "artifacts/paper_v3/pools_t3/val_emb"
OUT = "artifacts/paper_v3/pools_t3/loco"; os.makedirs(OUT, exist_ok=True)
BUDGET = 63000; NJOBS = 16
COUNTRIES = {"greece": ("GRC", 55), "spain": ("ESP", 47), "poland": ("POL", 115)}
CONFIGS = {"cap120_ne30_t2": ("px_v3", 30), "cap120_ne15_t2": ("px_v3", 15),
           "cap120_ne30_t3": ("cap120_t3", 30), "6M_ne30_t2": ("T6000000", 30)}

# --- master sampler (copied from rf_budget_sweep.build_pool) ---
def even_alloc(counts, budget):
    counts = np.asarray(counts, np.int64); alloc = np.zeros_like(counts)
    active = counts > 0; remaining = int(min(budget, counts.sum()))
    while remaining > 0 and active.any():
        share = remaining // int(active.sum())
        if share == 0:
            take = np.argsort(-((counts-alloc)*active))[:remaining]; alloc[take] += 1; break
        take = np.minimum(counts-alloc, share)*active; alloc += take
        remaining -= int(take.sum()); active = (counts-alloc) > 0
    return alloc
_rng = np.random.default_rng(0)
def sample_country(idx, cell, clus, T):
    if len(idx) <= T: return idx
    uc, ci = np.unique(cell, return_inverse=True); cb = even_alloc(np.bincount(ci, minlength=len(uc)), T)
    out = []
    for k in range(len(uc)):
        b = int(cb[k])
        if b == 0: continue
        m = ci == k; ii, cl = idx[m], clus[m]
        if b >= len(ii): out.append(ii); continue
        us, si = np.unique(cl, return_inverse=True); sb = even_alloc(np.bincount(si, minlength=len(us)), b)
        for s in range(len(us)):
            if sb[s] == 0: continue
            pix = ii[si == s]; out.append(pix if sb[s] >= len(pix) else _rng.choice(pix, int(sb[s]), replace=False))
    return np.concatenate(out)

_master = {}
def build_pool(pool, exclude_code):
    if pool in ("px_v3", "cap120_t3"):
        if pool == "px_v3":
            z = np.load(f"{POOLD}/px_v3_train.npz"); Xs, iso_s = z["X_sol"], z["iso_sol"]; Xn, iso_n = z["X_neg"], z["iso_neg"]
        else:
            z = np.load("artifacts/paper_v3/pools_t3/cap120_t3_train.npz")
            Xs, iso_s = z["X_sol"], z["iso_sol"]; Xn = z["X_neg"]; iso_n = np.load(f"{POOLD}/px_v3_train.npz")["iso_neg"]
        ps = iso_s != exclude_code; ns = iso_n != exclude_code
        X = np.concatenate([Xs[ps], Xn[ns]]).astype(np.float32)
        y = np.r_[np.ones(int(ps.sum()), np.int8), np.zeros(int(ns.sum()), np.int8)]
    else:                                             # T6000000 from master, excluding country
        T = int(pool[1:])
        if not _master:
            z = np.load(f"{POOLD}/px_master_train.npz", allow_pickle=False)
            _master.update(Xs=z["X_sol"], iso=z["iso_sol"], src=z["src_sol"], pos_chips=z["pos_chips"],
                           Xn=z["X_neg"], iso_n=z["iso_neg"])
            geo = pd.read_csv(f"{STAT}/chip_geo_pos.csv").set_index("chip"); g = geo.index.to_numpy()
            lat = np.floor(geo.lat.to_numpy()).astype(np.int64); lon = np.floor(geo.lon.to_numpy()).astype(np.int64)
            cby = dict(zip(g, (lat+90)*360+(lon+180))); clby = dict(zip(g, geo.cluster.to_numpy().astype(np.int64)))
            pc = _master["pos_chips"]
            _master["cell"] = np.array([cby.get(n,-1) for n in pc], np.int64)[_master["src"]]
            _master["clus"] = _master["iso"].astype(np.int64)*10_000_000 + np.array([clby.get(n,-1) for n in pc], np.int64)[_master["src"]]
            _master["fin_s"] = np.isfinite(_master["Xs"]).all(1)
        M = _master; parts = []
        for u in np.unique(M["iso"]):
            if u == exclude_code: continue
            idx = np.where((M["iso"] == u) & M["fin_s"])[0]
            parts.append(idx if len(idx) <= T else sample_country(idx, M["cell"][idx], M["clus"][idx], T))
        sel = np.sort(np.concatenate(parts))
        nk = np.where(np.isfinite(M["Xn"]).all(1) & (M["iso_n"] != exclude_code))[0]
        X = np.concatenate([M["Xs"][sel], M["Xn"][nk]]).astype(np.float32)
        y = np.r_[np.ones(len(sel), np.int8), np.zeros(len(nk), np.int8)]
    keep = np.isfinite(X).all(1); return X[keep], y[keep]

def val_roc(rf, cname):
    z = np.load(f"{VAL}/{cname}_val2019.npz"); Xs, Xr = z["site_emb"], z["rand_emb"]
    Xs, Xr = Xs[np.isfinite(Xs).all(1)], Xr[np.isfinite(Xr).all(1)]
    y = np.r_[np.ones(len(Xs)), np.zeros(len(Xr))]
    return roc_auc_score(y, np.r_[rf.predict_proba(Xs)[:,1], rf.predict_proba(Xr)[:,1]])

# cap120-based configs first (cheap), then 6M (loads master once)
order = sorted(CONFIGS, key=lambda c: CONFIGS[c][0] == "T6000000")
for cfg in order:
    pool, ne = CONFIGS[cfg]
    for cname, (iso3, code) in COUNTRIES.items():
        ck = f"{OUT}/{cfg}__{cname}.json"
        if os.path.exists(ck): print(f"[skip] {cfg} {cname}", flush=True); continue
        t = time.time()
        X, y = build_pool(pool, code)
        rf = RandomForestClassifier(n_estimators=ne, max_leaf_nodes=BUDGET//ne, max_features=32,
                                    min_samples_leaf=20, class_weight="balanced", n_jobs=NJOBS, random_state=0).fit(X, y)
        lroc = float(val_roc(rf, cname))
        json.dump(dict(config=cfg, country=cname, loco_roc=round(lroc,4), n_train=int(len(y)), pos=int(y.sum())), open(ck,"w"))
        print(f"{cfg} LOCO-{cname}: ROC={lroc:.4f}  (train {len(y):,}, {time.time()-t:.0f}s)", flush=True)
print("DONE loco", flush=True)

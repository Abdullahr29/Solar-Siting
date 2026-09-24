"""Per-country pixel-budget T-sweep with spatial diversity (IMPROVEMENTS.md Track 1, S1).

Retires the per-chip cap. For each budget T (px/country):
  - countries with <= T positive px are used ENTIRELY;
  - countries with  > T are subsampled to T px spread as evenly as possible across
    1x1 deg lat/lon CELLS -> SITES (clusters) -> pixels (water-filling at each level,
    random within a site). The redundancy unit is the SITE, so mega-farms can't dominate.
Negatives are held FIXED (one large diverse pool, ratio floats) -- justified by the ratio
sweep (0.0054 ROC spread over 2:1..1:5). Each T trains the deployed RF recipe (min_leaf=1500,
no GEE squeeze here -- comparability) and is scored on the FIXED held-out test pool
(px_v3_test) as a fast pixel-ROC pre-screen. Survivors go to S2 country forward-ROC.
"""
import os, time, json, resource
import numpy as np
import pandas as pd

def peak_gb():
    # ru_maxrss is KB on Linux -> GB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

WS   = os.path.expanduser("~/Solar_Workspace")
SS   = f"{WS}/Solar-Siting"
POOL = f"{SS}/artifacts/paper_v3/pools"
STAT = f"{SS}/artifacts/stats"
OUT  = f"{SS}/artifacts/paper_v3"
os.makedirs(f"{OUT}/pools", exist_ok=True)

TS     = [10_000_000, 6_000_000, 4_000_000, 2_000_000, 1_000_000, 500_000]
N_NEG  = int(os.environ.get("N_NEG", 0)) or None   # None -> use ALL negatives (fixed diverse pool)
SEED   = 0
RECIPE = dict(n_estimators=30, max_depth=12, min_samples_leaf=1500, max_samples=0.34,
              class_weight="balanced", n_jobs=int(os.environ.get("NJOBS", 32)), random_state=0)
rng = np.random.default_rng(SEED)


def even_alloc(counts, budget):
    """Water-fill `budget` across groups as evenly as possible, capped by `counts` available."""
    counts = np.asarray(counts, np.int64)
    alloc  = np.zeros_like(counts)
    active = counts > 0
    remaining = int(min(budget, counts.sum()))
    while remaining > 0 and active.any():
        n = int(active.sum())
        share = remaining // n
        if share == 0:                       # fewer left than active groups: 1 each to biggest cap
            cap = (counts - alloc) * active
            take = np.argsort(-cap)[:remaining]
            alloc[take] += 1
            remaining = 0
            break
        cap  = (counts - alloc)
        take = np.minimum(cap, share) * active
        alloc += take
        remaining -= int(take.sum())
        active = (counts - alloc) > 0
    return alloc


def sample_country(idx, cell, clus, T):
    """Return <=T selected pixel row-indices from a country's pixels, spread cell->site->pixel."""
    if len(idx) <= T:
        return idx
    # 1) allocate T across cells
    ucell, cinv = np.unique(cell, return_inverse=True)
    cell_cnt = np.bincount(cinv, minlength=len(ucell))
    cell_bud = even_alloc(cell_cnt, T)
    out = []
    for ci in range(len(ucell)):
        b = int(cell_bud[ci])
        if b == 0:
            continue
        m = cinv == ci
        ii, cl = idx[m], clus[m]
        if b >= len(ii):
            out.append(ii); continue
        # 2) allocate cell budget across sites
        usite, sinv = np.unique(cl, return_inverse=True)
        site_cnt = np.bincount(sinv, minlength=len(usite))
        site_bud = even_alloc(site_cnt, b)
        for si in range(len(usite)):
            sb = int(site_bud[si])
            if sb == 0:
                continue
            pix = ii[sinv == si]
            # 3) random within site
            out.append(pix if sb >= len(pix) else rng.choice(pix, sb, replace=False))
    return np.concatenate(out)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    return float((2*np.arange(1, n+1) - n - 1).dot(x) / (n * x.sum()))


print("loading master pool ...", flush=True); t0 = time.time()
z    = np.load(f"{POOL}/px_master_train.npz", allow_pickle=False)
Xs   = z["X_sol"]; iso = z["iso_sol"]; src = z["src_sol"]
Xn   = z["X_neg"]
names = z["iso_names"]; pos_chips = z["pos_chips"]
print(f"  pos {len(Xs):,}  neg {len(Xn):,}  ({time.time()-t0:.0f}s)", flush=True)

# per-source-chip -> (cell, cluster) via chip_geo
geo = pd.read_csv(f"{STAT}/chip_geo_pos.csv").set_index("chip")
gname = geo.index.to_numpy()
lat = np.floor(geo["lat"].to_numpy()).astype(np.int64)
lon = np.floor(geo["lon"].to_numpy()).astype(np.int64)
cell_by_name = dict(zip(gname, (lat + 90) * 360 + (lon + 180)))
clus_by_name = dict(zip(gname, geo["cluster"].to_numpy().astype(np.int64)))
# map src index -> chip name -> cell/cluster
cell_of_src = np.array([cell_by_name.get(n, -1) for n in pos_chips], np.int64)
clus_of_src = np.array([clus_by_name.get(n, -1) for n in pos_chips], np.int64)
cell_sol = cell_of_src[src]
# cluster must be globally unique across countries -> combine iso + cluster
clus_sol = iso.astype(np.int64) * 10_000_000 + clus_of_src[src]
miss = int((cell_sol < 0).sum())
print(f"  geo-mapped positives; unresolved cells: {miss:,}", flush=True)

# finiteness precomputed ONCE (avoids a full ~35GB isfinite copy per T)
fin_s = np.isfinite(Xs).all(1)
print(f"  finite positives: {int(fin_s.sum()):,} / {len(Xs):,}", flush=True)

# fixed negative pool (finite only); free the raw negatives afterwards to cap peak memory
fin_n = np.isfinite(Xn).all(1)
nkeep = np.where(fin_n)[0]
if N_NEG and N_NEG < len(nkeep):
    nkeep = np.sort(rng.choice(nkeep, N_NEG, replace=False))
Xn_fix = np.ascontiguousarray(Xn[nkeep])
del Xn
print(f"  fixed negatives: {len(Xn_fix):,}", flush=True)

# test pool (fixed)
te = np.load(f"{POOL}/px_v3_test.npz")
Xt = np.concatenate([te["X_sol"], te["X_neg"]]).astype(np.float32)
yt = np.r_[np.ones(te["X_sol"].shape[0]), np.zeros(te["X_neg"].shape[0])]
keep = np.isfinite(Xt).all(1); Xt, yt = Xt[keep], yt[keep]
print(f"  test {len(yt):,} (chance {yt.mean():.3f})\n", flush=True)

# country ids present
uiso = np.unique(iso)

rows = []
hdr = f"{'T':>6} {'pos_used':>12} {'capped':>6} {'gini':>6} {'neg':>12} {'n_train':>12} {'ROC':>7} {'PR':>7} {'fit_s':>6}"
print(hdr, flush=True)
for T in TS:
    tt = time.time()
    sel_parts, capped, per_country = [], 0, {}
    for u in uiso:
        idx = np.where((iso == u) & fin_s)[0]
        if len(idx) <= T:
            sel_parts.append(idx); per_country[int(u)] = len(idx); continue
        capped += 1
        s = sample_country(idx, cell_sol[idx], clus_sol[idx], T)
        sel_parts.append(s); per_country[int(u)] = len(s)
    sel = np.concatenate(sel_parts)
    g = gini(list(per_country.values()))
    n_pos = len(sel)
    Xp = Xs[np.sort(sel)]                       # positives already finite (fin_s)
    X  = np.concatenate([Xp, Xn_fix]); del Xp   # single train copy; no shuffle (RF bootstraps)
    y  = np.r_[np.ones(n_pos, np.int8), np.zeros(len(Xn_fix), np.int8)]
    t1 = time.time()
    rf = RandomForestClassifier(**RECIPE).fit(X, y)
    p = rf.predict_proba(Xt)[:, 1]
    roc, pr = roc_auc_score(yt, p), average_precision_score(yt, p)
    print(f"{T/1e6:>5.1f}M {n_pos:>12,} {capped:>6} {g:>6.3f} {len(Xn_fix):>12,} "
          f"{len(y):>12,} {roc:>7.4f} {pr:>7.4f} {time.time()-t1:>6.0f}  peakRSS={peak_gb():.0f}GB", flush=True)
    del X, rf, p
    rows.append(dict(T=T, pos_used=int(n_pos), capped=capped, gini=round(g, 4),
                     neg=int(len(Xn_fix)), n_train=int(len(y)), roc=round(float(roc), 4),
                     pr=round(float(pr), 4), fit_s=round(time.time()-t1, 0),
                     per_country=per_country))
    del y

df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_country"} for r in rows])
df.to_csv(f"{OUT}/tsweep_pixel_screen.csv", index=False)
with open(f"{OUT}/tsweep_pixel_screen.json", "w") as f:
    json.dump({"iso_names": names.tolist(), "rows": rows}, f)
print(f"\nwrote {OUT}/tsweep_pixel_screen.csv  (total {time.time()-t0:.0f}s)", flush=True)

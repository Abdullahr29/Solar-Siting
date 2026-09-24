"""Decide whether a 6M . t-3 pool can be built WITHOUT new GEE.

t-3 (install-3 AEF) exists locally ONLY for the cap120 selection (<=120 finite masked px per chip,
seed=train order) that t3_sample_pool already sampled. A pixel from the 6M per-country sample is
already covered iff its chip has <=120 stored positives (cap120 then took ALL of them). In a chip
with >120 positives, cap120 kept a random 120 and the 6M sampler can pick others we never sampled
-> those need fresh GEE at install-3 (unless AEF-year==2017, which reuses t-2 for free).

Reads only the small index arrays from master (no X_sol read). Reconstructs the exact 6M sample
from loco_debates.build_pool (T6000000 branch, no exclusion). Reports the GEE gap in pixels.
"""
import os, re, time, numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
POOLD = "artifacts/paper_v3/pools"; STAT = "artifacts/stats"
T = 6_000_000; CAP = 120
t0 = time.time()

z = np.load(f"{POOLD}/px_master_train.npz", allow_pickle=False)
print("keys:", list(z.keys()), flush=True)
iso = z["iso_sol"]; src = z["src_sol"]; pos_chips = z["pos_chips"]
iso_n = z["iso_n"] if "iso_n" in z.files else z["iso_neg"]
Npos = len(iso); print(f"master positives={Npos:,}  master negatives={len(iso_n):,}  chips={len(pos_chips):,}  ({time.time()-t0:.0f}s)", flush=True)

# per-chip stored positive counts + AEF year parsed from chip name POS_CCC_id_YEAR_n.tif
chipcount = np.bincount(src, minlength=len(pos_chips))
def yr_of(name):
    m = re.search(r"_(\d{4})_\d+\.tif$", str(name))
    return int(m.group(1)) if m else -1
chip_yr = np.array([yr_of(n) for n in pos_chips], np.int16)

# cell/clus per pixel (matches loco_debates.build_pool)
geo = pd.read_csv(f"{STAT}/chip_geo_pos.csv").set_index("chip")
g = geo.index.to_numpy()
lat = np.floor(geo.lat.to_numpy()).astype(np.int64); lon = np.floor(geo.lon.to_numpy()).astype(np.int64)
cby = dict(zip(g, (lat+90)*360+(lon+180))); clby = dict(zip(g, geo.cluster.to_numpy().astype(np.int64)))
cell = np.array([cby.get(n,-1) for n in pos_chips], np.int64)[src]
clus = iso.astype(np.int64)*10_000_000 + np.array([clby.get(n,-1) for n in pos_chips], np.int64)[src]

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
def sample_country(idx, cel, cl, T):
    if len(idx) <= T: return idx
    uc, ci = np.unique(cel, return_inverse=True); cb = even_alloc(np.bincount(ci, minlength=len(uc)), T)
    out = []
    for k in range(len(uc)):
        b = int(cb[k])
        if b == 0: continue
        m = ci == k; ii, clq = idx[m], cl[m]
        if b >= len(ii): out.append(ii); continue
        us, si = np.unique(clq, return_inverse=True); sb = even_alloc(np.bincount(si, minlength=len(us)), b)
        for s in range(len(us)):
            if sb[s] == 0: continue
            pix = ii[si == s]; out.append(pix if sb[s] >= len(pix) else _rng.choice(pix, int(sb[s]), replace=False))
    return np.concatenate(out)

# NOTE: finiteness of X_sol is skipped (would require reading the 64-d block); it removes only a
# small fraction and barely moves the coverage ratio. Counts here are therefore an upper bound on pool size.
parts = []
for u in np.unique(iso):
    idxc = np.where(iso == u)[0]
    parts.append(idxc if len(idxc) <= T else sample_country(idxc, cell[idxc], clus[idxc], T))
sel = np.concatenate(parts)
print(f"\n6M-pool positives (T={T:,}/country, no-exclusion)= {len(sel):,}  ({time.time()-t0:.0f}s)", flush=True)

sc = chipcount[src[sel]]; sy = chip_yr[src[sel]]
covered = sc <= CAP
need = ~covered
need_gee = need & (sy >= 2018)          # real GEE work (>120 chip AND AEF-year>=2018)
need_free = need & (sy == 2017)         # in >120 chip but 2019-install cohort -> reuse t-2 free
print(f"  in <=120 chips (already have t-3)  : {covered.sum():,}  ({100*covered.mean():.1f}%)")
print(f"  in >120 chips  -> not cached       : {need.sum():,}  ({100*need.mean():.1f}%)")
print(f"     of which AEF-year>=2018 (GEE)   : {need_gee.sum():,}  <-- fresh GEE point-samples required")
print(f"     of which AEF-year==2017 (free)  : {need_free.sum():,}")
# for context: how the master positives split by chip density
print(f"\nmaster positives in >120 chips: {(chipcount[src]>CAP).sum():,} / {Npos:,} ({100*(chipcount[src]>CAP).mean():.1f}%)")
print(f"cap120 t-3 pool size (ref)     : 6,193,548 positives")
gee = int(need_gee.sum())
rate = 40  # GEE reduceRegions pts/sec in restricted mode (observed)
print(f"\nEST GEE time for 6M.t-3 gap @ ~{rate} pts/s: {gee/rate/3600:.0f} h  ({gee/rate/3600/24:.1f} days)")
print(f"DONE coverage ({time.time()-t0:.0f}s)", flush=True)

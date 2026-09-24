"""Join per-facility generation (Song 2026) to our saved per-site RF validation scores,
and to the land-score distribution (rand_s), to build a REAL generation-weighted validation.

For each validation country we already have (paper_v3/scores/country/<c>_R_2019_scores.npz):
  site_s  : RF suitability at each real solar site (2019 AEF)
  site_xy : lon/lat of each site
  rand_s  : RF suitability at 6000 random land points  = the land-score distribution
We match site_xy to the nearest generation facility (KDTree, equal-area, <=1 km) to attach
yield (kWh/m2/yr), install year and 2023 generation. Then:
  (a) corr(site_s, yield)  overall / within-country / pre-install (install>=2020)
  (b) generation-coverage curve: %generation captured vs top-X% land (rand_s percentiles)
  (c) generation-weighted vs unweighted site-vs-random AUC
Saves artifacts/paper_v3/generation/gen_join.npz + prints a report."""
import os, glob
import numpy as np, pandas as pd
os.environ["PROJ_DATA"] = os.path.expanduser("~/Solar_Workspace/envs/env_solar/share/proj")
WS = os.path.expanduser("~/Solar_Workspace/Solar-Siting")
os.chdir(WS)
GD = "Data/external/pv_generation"
OUT = "artifacts/paper_v3/generation"
os.makedirs(OUT, exist_ok=True)

# ---- build facility table across years: latest-year yield + generation, install year ----
frames = []
for gy in range(2019, 2024):
    d = pd.read_csv(f"{GD}/PV_facility_generation_year_{gy}.csv")
    d["gen_year"] = gy
    frames.append(d)
allg = pd.concat(frames, ignore_index=True)
for c in ["power_POA (kWh)", "power_POA_cln (kWh)", "aerosol_loss (kWh)", "area_m2", "latitude", "longitude"]:
    allg[c] = pd.to_numeric(allg[c], errors="coerce")
allg = allg[(allg.area_m2 > 0) & (allg["power_POA (kWh)"] > 0)].copy()
# per facility: take the LATEST generation-year record (full operating year, most recent)
allg = allg.sort_values("gen_year").groupby("PV_ID", as_index=False).last()
allg["yield"] = allg["power_POA (kWh)"] / allg.area_m2
allg["gen_kwh"] = allg["power_POA (kWh)"]
allg["install_year"] = pd.to_numeric(allg["year"], errors="coerce")
print("facility table:", allg.shape, "| yield med", round(allg["yield"].median(), 1))

# project facility coords to equal-area once
import pyproj
tf = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
fx, fy = tf.transform(allg.longitude.values, allg.latitude.values)
from scipy.spatial import cKDTree
tree = cKDTree(np.c_[fx, fy])
yld = allg["yield"].values; gen = allg["gen_kwh"].values; iyr = allg["install_year"].values

CN = {"united_states_of_america": "USA", "south_africa": "ZAF"}
rows = []
cov_country = {}
pooled_site_s = []; pooled_yld = []; pooled_iyr = []; pooled_ctry = []; pooled_rand = []
for f in sorted(glob.glob("artifacts/paper_v3/scores/country/*_R_2019_scores.npz")):
    c = os.path.basename(f).replace("_R_2019_scores.npz", "")
    z = np.load(f)
    site_s, site_xy, rand_s = z["site_s"], z["site_xy"], z["rand_s"]
    m = min(len(site_s), len(site_xy))
    if len(site_s) != len(site_xy):
        print(f"  {c}: len mismatch site_s={len(site_s)} site_xy={len(site_xy)} -> using {m}")
    site_s, site_xy = site_s[:m], site_xy[:m]
    sx, sy = tf.transform(site_xy[:, 0], site_xy[:, 1])
    dist, idx = tree.query(np.c_[sx, sy], k=1)
    ok = dist < 1000.0  # <=1 km match
    s_s = site_s[ok]; s_y = yld[idx[ok]]; s_i = iyr[idx[ok]]
    if ok.sum() < 10:
        print(f"  {c}: only {ok.sum()} matched, skip"); continue
    # (a) correlation
    from scipy.stats import spearmanr
    rho_all = spearmanr(s_s, s_y).correlation
    pre = s_i >= 2020
    rho_pre = spearmanr(s_s[pre], s_y[pre]).correlation if pre.sum() > 10 else np.nan
    # (b) coverage: for each site, land-percentile of its score vs rand_s
    order = np.argsort(rand_s)
    # fraction of land with score <= site score  -> site sits in top (1-frac)
    land_pct = np.searchsorted(np.sort(rand_s), s_s) / len(rand_s)
    rows.append(dict(country=c, n=int(ok.sum()), matched_frac=round(ok.mean(), 2),
                     rho_yield=round(rho_all, 3), rho_yield_pre=round(rho_pre, 3),
                     med_yield=round(np.nanmedian(s_y), 1),
                     med_site_s=round(np.median(s_s), 3), med_rand_s=round(np.median(rand_s), 3)))
    cov_country[c] = dict(site_s=s_s, yld=s_y, gen=gen[idx[ok]], rand_s=rand_s, land_pct=land_pct)
    pooled_site_s.append(s_s); pooled_yld.append(s_y); pooled_iyr.append(s_i)
    pooled_ctry += [c] * ok.sum(); pooled_rand.append(rand_s)

rep = pd.DataFrame(rows)
print("\n=== per-country RF-score vs generation-yield ===")
print(rep.to_string(index=False))

# pooled correlation (within-country z-scored to remove climate confound)
ps = np.concatenate(pooled_site_s); py = np.concatenate(pooled_yld); pi = np.concatenate(pooled_iyr)
pc = np.array(pooled_ctry)
from scipy.stats import spearmanr
print("\npooled rho(site_s, yield)         :", round(spearmanr(ps, py).correlation, 3), "n=", len(ps))
pre = pi >= 2020
print("pooled rho pre-install(>=2020)     :", round(spearmanr(ps[pre], py[pre]).correlation, 3), "n=", pre.sum())
# within-country z-scored (partial out climate/country)
def zc(v, c):
    out = np.full(len(v), np.nan)
    for cc in np.unique(c):
        m = c == cc
        if m.sum() > 2 and np.nanstd(v[m]) > 0:
            out[m] = (v[m] - np.nanmean(v[m])) / np.nanstd(v[m])
    return out
zs, zy = zc(ps, pc), zc(py, pc)
good = np.isfinite(zs) & np.isfinite(zy)
print("within-country rho (climate-adj)   :", round(spearmanr(zs[good], zy[good]).correlation, 3), "n=", good.sum())

# (c) generation-weighted vs unweighted AUC pooled (site vs random)
def auc_w(pos, neg, wpos=None):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    w = np.r_[(np.ones(len(pos)) if wpos is None else wpos), np.ones(len(neg))]
    return roc_auc_score(y, s, sample_weight=w)
allrand = np.concatenate(pooled_rand)
pg = []
for cc in cov_country.values():
    pg.append(cc["gen"])
pg = np.concatenate(pg)
print("\npooled site-vs-random AUC unweighted        :", round(auc_w(ps, allrand), 3))
print("pooled site-vs-random AUC generation-weighted:", round(auc_w(ps, allrand, wpos=pg), 3))

# (b) pooled generation-coverage curve: rank land by score; cumulative generation vs land fraction
all_land_pct = np.concatenate([cov_country[c]["land_pct"] for c in cov_country])
# top land fraction each site sits in = 1 - land_pct ; capture curve
Xf = np.linspace(0, 1, 101)
capt = []
for xf in Xf:
    thr = np.quantile(allrand, 1 - xf) if xf > 0 else np.inf
    inc = ps >= thr
    capt.append(pg[inc].sum() / pg.sum())
capt = np.array(capt)
for xf in [0.05, 0.1, 0.2, 0.3, 0.5]:
    thr = np.quantile(allrand, 1 - xf); print(f"top {int(xf*100):2d}% land -> {ps[ps>=thr].size} sites, {pg[ps>=thr].sum()/pg.sum()*100:.1f}% of generation")

np.savez(f"{OUT}/gen_join.npz", Xf=Xf, capt=capt, ps=ps, py=py, pi=pi, pc=pc,
         allrand=allrand, pg=pg, report=rep.to_dict("list"))
rep.to_csv(f"{OUT}/gen_join_by_country.csv", index=False)
print("\nsaved", OUT)

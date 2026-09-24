"""Global suitable-land fraction via POINT-SAMPLING (no raster, restricted-mode-safe).

Justified by the pooled-ROC comparability check (global-pooled ROC 0.868 ~ per-country mean): RF-R
scores are comparable enough across countries to apply a single global threshold.

Method:
  * Global LAND scores: score.sample() random ~1 km land pixels worldwide (server-side; nearest-
    neighbour subsample of the native-10 m RF-R classify, NOT pooling) -> ~40k land scores.
  * Global SITE threshold: score ~10k real solar sites worldwide (TZ-SAM new sites, constructed_after
    >= 2020 so bare in 2019 -> clean pre-solar suitability) -> threshold = 10th pct of site scores
    ("as suitable as 90% of real installations").
  * global_suitable_fraction = fraction of global land >= threshold; also global-pooled ROC.

Out -> Solar-Siting/artifacts/paper_v3/figA_combined/global_suitable.npz + printed summary.
"""
import csv, os, time
from datetime import datetime
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
TZ_NEW = "Data/external/tz_sam_q1_2026/overlap_analysis/tz_new_sites.csv"
OUT = "Solar-Siting/artifacts/paper_v3/figA_combined/global_suitable.npz"
YEAR = 2019
N_LAND_TARGET = 40000
N_SITES = 10000
MIN_CA = 2020
SEED = 42


def gi(obj, tries=6, base=8):
    import time as _t
    for k in range(tries):
        try:
            return obj.getInfo()
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  getInfo retry {k+1} in {base*2**k}s: {str(e)[:70]}", flush=True)
            _t.sleep(base * 2 ** k)


def load_sites():
    xy = []
    with open(os.path.join(WS, TZ_NEW), newline="") as f:
        for row in csv.DictReader(f):
            ca = (row.get("constructed_after") or "").strip()
            if not ca:
                continue
            try:
                if datetime.fromisoformat(ca[:10]).year < MIN_CA:
                    continue
                xy.append((float(row["longitude"]), float(row["latitude"])))
            except (ValueError, KeyError):
                continue
    xy = np.array(xy, float)
    rng = np.random.default_rng(SEED)
    if len(xy) > N_SITES:
        xy = xy[rng.choice(len(xy), N_SITES, replace=False)]
    return xy


def main():
    os.chdir(WS)
    import ee, joblib
    from geemap import ml
    ee.Initialize(project="ee-abdullahr-solar")
    bands = [f"A{i:02d}" for i in range(64)]
    rf = joblib.load(MODEL)
    clf = ml.strings_to_classifier(ml.rf_to_strings(rf, bands, processes=32, output_mode="PROBABILITY"))
    print("classifier injected", flush=True)
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01").mosaic().select(bands))
    score = img.classify(clf).rename("score")

    def score_points(xy):
        """RF-R score at points via reduceRegions (point sampling — reliable in restricted mode;
        ocean/no-AEF points return null and cost nothing). Returns only the finite land scores."""
        out = []
        for i in range(0, len(xy), 1000):
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]))
                                       for x, y in xy[i:i+1000]])
            vals = gi(score.reduceRegions(fc, ee.Reducer.first(), scale=10).aggregate_array("first"))
            out += [v for v in vals if v is not None]
        return out

    # ---- global land scores: uniform-on-sphere candidates (equal-area), score, keep land ----
    rng = np.random.default_rng(SEED)
    land_scores = []
    t0 = time.time()
    smin, smax = np.sin(np.deg2rad(-56)), np.sin(np.deg2rad(72))
    batch = 0
    while len(land_scores) < N_LAND_TARGET and batch < 200:
        u = rng.uniform(smin, smax, 3000)
        lat = np.degrees(np.arcsin(u)); lon = rng.uniform(-180, 180, 3000)
        got = score_points(np.c_[lon, lat])
        land_scores += got
        batch += 1
        print(f"  land batch {batch}: +{len(got)} land of 3000 -> {len(land_scores)} total ({time.time()-t0:.0f}s)", flush=True)
    land_scores = np.array(land_scores, float)

    # ---- global site scores ----
    site_xy = load_sites()
    print(f"scoring {len(site_xy)} global solar sites...", flush=True)
    site_scores = []
    for i in range(0, len(site_xy), 1000):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]))
                                   for x, y in site_xy[i:i+1000]])
        vals = gi(score.reduceRegions(fc, ee.Reducer.first(), scale=10).aggregate_array("first"))
        site_scores += [v for v in vals if v is not None]
        print(f"  sites {min(i+1000,len(site_xy))}/{len(site_xy)}", flush=True)
    site_scores = np.array(site_scores, float)

    # ---- metrics ----
    def auc(pos, neg):
        ns = np.sort(neg); lo = np.searchsorted(ns, pos, "left"); hi = np.searchsorted(ns, pos, "right")
        return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))
    thr = np.percentile(site_scores, 10)
    frac = float(np.mean(land_scores >= thr))
    pooled = auc(site_scores, land_scores)
    med_land, med_site = np.median(land_scores), np.median(site_scores)

    np.savez(OUT, land_scores=land_scores, site_scores=site_scores, threshold=thr,
             suitable_fraction=frac, pooled_auc=pooled)
    print("\n===== GLOBAL SUITABLE-LAND SUMMARY =====")
    print(f"n_land={len(land_scores)}  n_sites={len(site_scores)}")
    print(f"median land score={med_land:.3f}  median site score={med_site:.3f}")
    print(f"threshold (10th pct of real sites)={thr:.3f}")
    print(f"GLOBAL SUITABLE-LAND FRACTION = {frac*100:.1f}%")
    print(f"global-pooled ROC (sites vs land) = {pooled:.3f}")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

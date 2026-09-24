"""Grid/terrain/land-cover-constrained REALISTIC solar potential (global point-sampling).

Extends the global suitable-fraction with standard infrastructure/terrain/exclusion constraints,
with literature-backed cutoffs (documented below). Road is omitted (no global roads dataset; grid
proximity is the dominant infrastructure constraint anyway).

CUTOFFS (public-knowledge / literature-backed):
  * RF-suitable      : RF-R score >= 0.436 (10th pct of real global sites; matches global_suitable.py)
  * Slope            : <= 5 deg        (utility-scale PV standard; NREL/IRENA GIS-suitability studies)
  * Grid proximity   : <= 10 km        (grid-connection economic-viability threshold, common in
                                        techno-economic potential studies; gridfinder MV grid)
  * Protected areas  : exclude WDPA    (standard conservation exclusion)
  * Land cover (ESA WorldCover): eligible = shrub(20)/grass(30)/bare(60); exclude tree/built/water/
                                  wetland/snow. Cropland(40) reported separately (food-security).
Method: score global-land points + real sites for RF/slope/worldcover/WDPA (GEE, one multiband
reduceRegions) + grid distance (local gridfinder, EPSG:6933). Report constrained fraction of global
land, sanity-check the fraction of real sites passing each gate, and the capacity potential.

Out -> Solar-Siting/artifacts/paper_v3/figA_combined/grid_constrained.npz + summary.
"""
import csv, os, time
from datetime import datetime
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
TZ_NEW = "Data/external/tz_sam_q1_2026/overlap_analysis/tz_new_sites.csv"
GRID = "Solar-Siting/covariate_data/gridfinder/grid.gpkg"
OUT = "Solar-Siting/artifacts/paper_v3/figA_combined/grid_constrained.npz"
YEAR = 2019
SUIT_THR = 0.436
SLOPE_MAX = 5.0
GRID_MAX_KM = 10.0
N_LAND_TARGET = 40000
N_SITES = 10000
ELIGIBLE_WC = {20, 30, 60}          # shrub, grass, bare/sparse
ELIGIBLE_WC_CROP = {20, 30, 40, 60}  # + cropland (sensitivity)
GLOBAL_LAND_KM2 = 130e6             # land in the -56..72 band (approx, Antarctica excluded)
DENSITY_MW_KM2 = 39.4               # TZ-SAM area-weighted aggregate (see tz-sam-methodology)
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
                if datetime.fromisoformat(ca[:10]).year < 2020:
                    continue
                xy.append((float(row["longitude"]), float(row["latitude"])))
            except (ValueError, KeyError):
                continue
    xy = np.array(xy, float)
    rng = np.random.default_rng(SEED)
    return xy[rng.choice(len(xy), N_SITES, replace=False)] if len(xy) > N_SITES else xy


def build_multiband(clf, bands):
    import ee
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01").mosaic().select(bands))
    score = img.classify(clf).rename("score")
    # slope: setDefaultProjection so ee.Terrain is at native ~30 m (not the ~1 deg mosaic artifact)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().setDefaultProjection("EPSG:4326", None, 30)
    slope = ee.Terrain.slope(dem).rename("slope")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("wc")
    try:
        wdpa = ee.Image().byte().paint(ee.FeatureCollection("WCMC/WDPA/current/polygons"), 1).unmask(0).rename("wdpa")
        return score.addBands([slope, wc, wdpa]), True
    except Exception as e:
        print(f"WDPA unavailable ({str(e)[:60]}); proceeding without protected-area gate", flush=True)
        return score.addBands([slope, wc]), False


def sample(stack, xy, has_wdpa):
    import ee
    keys = ["score", "slope", "wc"] + (["wdpa"] if has_wdpa else [])
    rows = {k: [] for k in keys}
    rows["lon"] = []; rows["lat"] = []
    for i in range(0, len(xy), 1000):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]))
                                   for x, y in xy[i:i+1000]])
        res = gi(stack.reduceRegions(fc, ee.Reducer.first(), scale=10))
        for feat, (x, y) in zip(res["features"], xy[i:i+1000]):
            p = feat["properties"]
            if p.get("score") is None:
                continue
            rows["lon"].append(x); rows["lat"].append(y)
            for k in keys:
                rows[k].append(p.get(k, np.nan))
    return {k: np.array(v, float) for k, v in rows.items()}


def grid_dist_km(lon, lat):
    import geopandas as gpd
    from shapely import STRtree
    g = gpd.read_file(os.path.join(WS, GRID)).to_crs("EPSG:6933")
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs("EPSG:6933")
    tree = STRtree(g.geometry.values)
    idx = tree.nearest(pts.values)
    return np.array([pts.values[i].distance(g.geometry.values[j]) for i, j in enumerate(idx)]) / 1000.0


def main():
    os.chdir(WS)
    import ee, joblib
    from geemap import ml
    ee.Initialize(project="ee-abdullahr-solar")
    bands = [f"A{i:02d}" for i in range(64)]
    clf = ml.strings_to_classifier(ml.rf_to_strings(joblib.load(MODEL), bands, processes=32, output_mode="PROBABILITY"))
    stack, has_wdpa = build_multiband(clf, bands)
    print(f"classifier injected; WDPA gate = {has_wdpa}", flush=True)

    # global land points
    rng = np.random.default_rng(SEED)
    smin, smax = np.sin(np.deg2rad(-56)), np.sin(np.deg2rad(72))
    land = {k: [] for k in (["score", "slope", "wc", "lon", "lat"] + (["wdpa"] if has_wdpa else []))}
    t0 = time.time(); b = 0
    while len(land["score"]) < N_LAND_TARGET and b < 200:
        u = rng.uniform(smin, smax, 3000)
        xy = np.c_[rng.uniform(-180, 180, 3000), np.degrees(np.arcsin(u))]
        got = sample(stack, xy, has_wdpa)
        for k in land:
            land[k] += list(got[k])
        b += 1
        print(f"  land batch {b}: +{len(got['score'])} -> {len(land['score'])} ({time.time()-t0:.0f}s)", flush=True)
    land = {k: np.array(v, float) for k, v in land.items()}

    print("scoring sites...", flush=True)
    sites = sample(stack, load_sites(), has_wdpa)

    print("grid distance (local gridfinder)...", flush=True)
    land["grid_km"] = grid_dist_km(land["lon"], land["lat"])
    sites["grid_km"] = grid_dist_km(sites["lon"], sites["lat"])

    def gates(d, wc_set):
        g_suit = d["score"] >= SUIT_THR
        g_slope = d["slope"] <= SLOPE_MAX
        g_grid = d["grid_km"] <= GRID_MAX_KM
        g_wc = np.isin(np.nan_to_num(d["wc"], nan=-1).astype(int), list(wc_set))
        g_wdpa = (d["wdpa"] < 0.5) if has_wdpa else np.ones(len(d["score"]), bool)
        allg = g_suit & g_slope & g_grid & g_wc & g_wdpa
        return dict(suit=g_suit, slope=g_slope, grid=g_grid, wc=g_wc, wdpa=g_wdpa, all=allg)

    lg = gates(land, ELIGIBLE_WC); sg = gates(sites, ELIGIBLE_WC)
    lg_c = gates(land, ELIGIBLE_WC_CROP)
    frac = lg["all"].mean(); frac_crop = lg_c["all"].mean()
    area = frac * GLOBAL_LAND_KM2
    cap_tw = area * DENSITY_MW_KM2 / 1e6  # MW -> TW

    np.savez(OUT, **{f"land_{k}": v for k, v in land.items()},
             constrained_fraction=frac, constrained_fraction_cropland=frac_crop)
    print("\n===== GRID-CONSTRAINED REALISTIC POTENTIAL =====")
    print(f"n_land={len(land['score'])}  n_sites={len(sites['score'])}  WDPA_gate={has_wdpa}")
    print("--- per-gate PASS rate: global land vs real sites (sanity: sites should mostly pass) ---")
    for g in ["suit", "slope", "grid", "wc", "wdpa", "all"]:
        print(f"  {g:6}: land {lg[g].mean()*100:5.1f}%   sites {sg[g].mean()*100:5.1f}%")
    print(f"\nUNCONSTRAINED suitable fraction (score only) = {lg['suit'].mean()*100:.1f}%")
    print(f"CONSTRAINED realistic fraction               = {frac*100:.1f}%  (+cropland {frac_crop*100:.1f}%)")
    print(f"-> realistic developable area ~ {area/1e6:.1f}M km2")
    print(f"-> capacity potential @ {DENSITY_MW_KM2} MW/km2 = {cap_tw:.0f} TW  "
          f"(vs 1.33 TW installed = {cap_tw/1.33:.0f}x); apply a land-use availability factor for a realistic figure")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

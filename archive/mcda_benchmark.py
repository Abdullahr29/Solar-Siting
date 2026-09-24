"""GIS-MCDA benchmark: does our AlphaEarth RF rank real installs better than a standard
literature-weighted multi-criteria suitability model?

Fair fight, identical points: the 20k random-land covariate sample is the shared NEGATIVE
set (it already carries every criterion + the RF score); we sample the same criteria + RF
score at real FORWARD-install sites (installed >= year+2, unseen by the imagery) as the
POSITIVES. Then, on those same points, we compute:
  * RF ROC   = roc(label, RF score)
  * MCDA ROC = roc(label, weighted-linear-combination suitability)   for each weight vector

MCDA design (fixed published weights, never tuned to our data):
  criteria (linear min-max, normalised against the country's random-land distribution):
    GHI (benefit), slope (cost), grid distance (cost), road distance (cost),
    land-cover suitability (class lookup)
  weight band (each sums to 1): balanced / irradiance-heavy / infrastructure-heavy
  hard exclusions -> suitability 0: WDPA protected areas OR water/wetland land cover

Usage (JASMIN sci node, env_solar):  python Solar-Siting/mcda_benchmark.py "Germany" --year 2021
Runs after the covariate pipeline + roads pass (needs each country's _covsample_full.npz
with gsa_ghi / grid_dist_m / road_dist_m).
"""
import argparse, csv, os
import numpy as np

REFERENCE_MODEL = "Solar-Siting/artifacts/models/rf30_final.joblib"
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
INV_NAME = {"United States": "United States of America"}   # LSIB -> inventory name

WEIGHT_SETS = {   # criteria order: ghi, slope, grid, road, land  (each sums to 1.0)
    "balanced":       dict(ghi=.35, slope=.20, grid=.15, road=.10, land=.20),
    "irradiance":     dict(ghi=.45, slope=.20, grid=.15, road=.10, land=.10),
    "infrastructure": dict(ghi=.30, slope=.15, grid=.25, road=.15, land=.15),
}
# ESA WorldCover class -> suitability (0..1); water(80)/wetland(90) are hard-excluded anyway
LAND_SUIT = {10: 0.3, 20: 0.8, 30: 0.9, 40: 0.6, 50: 0.1, 60: 1.0,
             70: 0.2, 80: 0.0, 90: 0.0, 95: 0.0, 100: 0.5}
EXCLUDE_WC = {80, 90}   # water, wetland


def nb(x, lo, hi):  return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)      # benefit (up)
def nc(x, lo, hi):  return np.clip((hi - x) / (hi - lo + 1e-12), 0, 1)      # cost (down)


def sample_sites(country, year, site_xy):
    """Sample RF score + MCDA criteria at given site points (GEE stack + GSA + grid + road)."""
    import ee
    from covariate_sample import build_stack, sample_stack
    from covariate_append_manual import ensure_gsa_tif, sample_raster, nearest_distance, \
        OSM_SHP, OSM_PBF, roads_from_pbf, CD
    import geopandas as gpd

    stack, geom = build_stack(country, year, REFERENCE_MODEL)
    d = sample_stack(stack, site_xy)          # score, slope, elevation, worldcover, dw_label, in_wdpa
    lon, lat = site_xy[:, 0], site_xy[:, 1]
    out = {k: d[k] for k in ("score", "slope", "worldcover", "in_wdpa")}
    out["gsa_ghi"] = sample_raster(ensure_gsa_tif("World_GHI_"), lon, lat)

    slug = country.lower().replace(" ", "_")
    pad = 0.6
    bbox = (lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)
    g = gpd.read_file(f"{CD}/gridfinder/grid.gpkg", bbox=bbox)
    out["grid_dist_m"] = nearest_distance(lon, lat, g, "grid@sites") if len(g) else np.full(len(lon), np.nan)

    key = slug.replace("_", " ")
    gpkg = f"{CD}/osm/{slug}_roads.gpkg"
    if os.path.exists(gpkg):
        r = gpd.read_file(gpkg, bbox=bbox)
    elif key in OSM_SHP:
        r = gpd.read_file(f"/vsizip/{os.path.abspath(CD)}/osm/{OSM_SHP[key]}/gis_osm_roads_free_1.shp", bbox=bbox)
    else:
        r = None
    out["road_dist_m"] = nearest_distance(lon, lat, r, "roads@sites") if (r is not None and len(r)) \
        else np.full(len(lon), np.nan)
    return out


def mcda_scores(cols, ranges):
    """Return {weight_name: mcda suitability array} + exclusion mask, from criterion columns."""
    c_ghi = nb(cols["gsa_ghi"], *ranges["gsa_ghi"])
    c_slp = nc(cols["slope"], *ranges["slope"])
    c_grd = nc(cols["grid_dist_m"], *ranges["grid_dist_m"])
    c_rd = nc(cols["road_dist_m"], *ranges["road_dist_m"])
    wc = np.where(np.isfinite(cols["worldcover"]), cols["worldcover"], -1).astype(int)
    c_land = np.array([LAND_SUIT.get(int(v), 0.3) for v in wc])
    excl = (cols["in_wdpa"] >= 0.5) | np.isin(wc, list(EXCLUDE_WC))
    out = {}
    for name, w in WEIGHT_SETS.items():
        s = w["ghi"]*c_ghi + w["slope"]*c_slp + w["grid"]*c_grd + w["road"]*c_rd + w["land"]*c_land
        s = s.copy(); s[excl] = 0.0
        out[name] = s
    return out, excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--max-sites", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.chdir(os.path.expanduser("~/Solar_Workspace"))
    from sklearn.metrics import roc_auc_score
    import ee, geopandas as gpd
    ee.Initialize(project="ee-abdullahr-solar")

    slug = args.country.lower().replace(" ", "_")
    # Prefer the CORRECTED negatives table (native-30 m slope). The old _covsample_full.npz
    # carries the ~0-deg slope artifact, which — combined with correctly-sloped positives
    # from the now-fixed covariate_sample.build_stack — silently broke MCDA's slope cost.
    base = f"Solar-Siting/artifacts/covariate/{slug}/{slug}_{args.year}"
    neg_path = f"{base}_covsample_corrected.npz" if os.path.exists(f"{base}_covsample_corrected.npz") \
        else f"{base}_covsample_full.npz"
    print(f"negatives: {os.path.basename(neg_path)}", flush=True)
    neg = dict(np.load(neg_path))
    crit = ["gsa_ghi", "slope", "grid_dist_m", "road_dist_m", "worldcover", "in_wdpa", "score"]
    missing = [c for c in crit if c not in neg]
    if missing:
        raise SystemExit(f"{args.country}: covsample_full missing {missing} — run append first")

    # forward-install positives (installed >= year+2, unseen by imagery)
    inv_name = INV_NAME.get(args.country, args.country)
    min_install = args.year + 2
    sites = gpd.read_file(INV, where=f"country = '{inv_name}' AND year >= {min_install}")
    if len(sites) == 0:
        raise SystemExit(f"{args.country}: no inventory sites (country='{inv_name}', year>={min_install})")
    pts = sites.geometry.representative_point().to_crs("EPSG:4326")
    site_xy = np.c_[pts.x, pts.y]
    rng = np.random.default_rng(args.seed)
    if len(site_xy) > args.max_sites:
        site_xy = site_xy[rng.choice(len(site_xy), args.max_sites, replace=False)]
    print(f"{args.country}: {len(site_xy)} forward-install sites, {len(neg['score'])} random-land negatives", flush=True)

    pos = sample_sites(args.country, args.year, site_xy)

    # normalise against the country's random-land (negative) distribution, robust 2-98 pct
    ranges = {c: tuple(np.nanpercentile(neg[c], [2, 98])) for c in ["gsa_ghi", "slope", "grid_dist_m", "road_dist_m"]}

    # assemble pos+neg, drop rows with any NaN criterion (consistently for both models)
    P = {c: np.asarray(pos[c], float) for c in crit}
    N = {c: np.asarray(neg[c], float) for c in crit}
    cols = {c: np.concatenate([P[c], N[c]]) for c in crit}
    label = np.r_[np.ones(len(P["score"])), np.zeros(len(N["score"]))]
    need = ["gsa_ghi", "slope", "grid_dist_m", "road_dist_m", "score"]
    good = np.all(np.isfinite(np.column_stack([cols[c] for c in need])), axis=1)
    for c in crit:
        cols[c] = cols[c][good]
    label = label[good]

    mcda, excl = mcda_scores(cols, ranges)
    rf_roc = roc_auc_score(label, cols["score"])
    mroc = {name: roc_auc_score(label, s) for name, s in mcda.items()}
    mvals = list(mroc.values())

    row = dict(country=args.country, year=args.year,
               n_sites=int(label.sum()), n_rand=int((label == 0).sum()),
               n_excluded=int(excl.sum()),   # excl computed on already-filtered cols
               rf_roc=round(rf_roc, 4),
               mcda_roc_min=round(min(mvals), 4), mcda_roc_mean=round(float(np.mean(mvals)), 4),
               mcda_roc_max=round(max(mvals), 4),
               rf_minus_best_mcda=round(rf_roc - max(mvals), 4),
               **{f"mcda_{k}": round(v, 4) for k, v in mroc.items()})

    out = "Solar-Siting/artifacts/results/mcda_benchmark_corrected.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\n=== {args.country} MCDA benchmark ===")
    print(f"RF ROC {rf_roc:.3f}  vs  MCDA ROC {min(mvals):.3f}-{max(mvals):.3f} "
          f"(band: {', '.join(f'{k} {v:.3f}' for k, v in mroc.items())})")
    print(f"RF minus best MCDA = {rf_roc - max(mvals):+.3f}  "
          f"({'RF wins' if rf_roc > max(mvals) else 'MCDA wins'})  | appended {out}", flush=True)


if __name__ == "__main__":
    main()

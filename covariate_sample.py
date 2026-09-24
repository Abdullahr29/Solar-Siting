"""Covariate-analysis SAMPLING pass (GEE-native half).

Scores a country's AlphaEarth mosaic with the frozen reference RF (server-side, no
download), stacks GEE-native covariates alongside the score, and samples score +
every covariate at a set of random-land points in one Earth Engine pass. Writes a
tidy per-point table for the offline univariate/multivariate analysis step.

GEE-native covariates (available now, no transfer needed):
  score        RF suitability probability (rf30_final on AEF <year>)
  slope        Copernicus GLO-30 DEM slope (deg)
  elevation    Copernicus GLO-30 DEM elevation (m)
  worldcover   ESA WorldCover v200 (2021) land-cover class (categorical)
  dw_label     Dynamic World <year> annual modal class (categorical)
  in_wdpa      inside a WDPA protected-area polygon (0/1)

Manually-transferred covariates (GSA irradiance, Gridfinder grid-distance, OSM
road-distance) are appended to this same point table by a later step, keyed on
(lon, lat), so this pass does not wait on the ~40 GB transfer.

Usage (JASMIN sci node, env_solar):
    python Solar-Siting/covariate_sample.py Greece --year 2021 --n 20000

Epoch note: covariate epoch locked to 2021 (WorldCover v200 = 2021, DW annual 2021,
AEF-2021 map); DEM / WDPA are static/current.
"""
import argparse, json, os, sys, time
import numpy as np

REFERENCE_MODEL = "Solar-Siting/artifacts/models/rf30_final.joblib"


def build_stack(country, year, model_path):
    import ee, joblib
    from geemap import ml
    bands = [f"A{i:02d}" for i in range(64)]

    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    fc = lsib.filter(ee.Filter.eq("country_na", country))
    if fc.size().getInfo() == 0:
        names = lsib.aggregate_array("country_na").distinct().getInfo()
        close = [n for n in names if country.lower() in n.lower()]
        sys.exit(f"'{country}' not in LSIB. Close matches: {close}")
    geom = fc.geometry()

    rf = joblib.load(model_path)
    t0 = time.time()
    trees = ml.rf_to_strings(rf, bands, processes=32, output_mode="PROBABILITY")
    clf = ml.strings_to_classifier(trees)
    print(f"classifier injected: {len(trees)} trees ({time.time()-t0:.0f}s)", flush=True)

    aef = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{year}-01-01", f"{year+1}-01-01").mosaic().select(bands))
    score = aef.classify(clf).rename("score")

    # --- GEE-native covariates ---
    # NOTE: .mosaic() drops the projection, so ee.Terrain on the bare mosaic computes at
    # EE's default ~1deg grid and slope collapses to ~0 everywhere (this silently made the
    # original "slope is null" result an artifact). Restore the tiles' native ~30 m
    # projection with setDefaultProjection before Terrain so slope is real.
    dem_coll = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")
    dem = dem_coll.mosaic().setDefaultProjection(dem_coll.first().projection())
    slope = ee.Terrain.slope(dem).rename("slope")
    elevation = dem.rename("elevation")
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover")
    dw_label = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(geom)
                .select("label").reduce(ee.Reducer.mode()).rename("dw_label"))
    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(geom)
    in_wdpa = ee.Image(0).paint(wdpa, 1).rename("in_wdpa").toByte()

    stack = score.addBands([slope, elevation, worldcover, dw_label, in_wdpa])
    return stack, geom


def random_land_points(geom, n, seed):
    """Uniform random points inside the country polygon, sampled server-side in EE.

    EE randomPoints samples within the actual geometry (not a bbox), so it is robust for
    large / antimeridian-spanning countries (e.g. the US with Alaska) where client-side
    bbox rejection sampling would almost never land inside.
    """
    import ee
    rp = ee.FeatureCollection.randomPoints(region=geom, points=n, seed=seed)
    coords = rp.geometry().coordinates().getInfo()   # [[lon,lat], ...]
    return np.array(coords, dtype=float)


def sample_stack(stack, xy, scale=10, chunk=400, max_retries=4):
    """reduceRegions the stack at each point, chunked, resilient to EE "Computation timed
    out": a timing-out batch is recursively halved and retried (big-country tiles make a
    1000-pt batch too heavy), with backoff on the smallest batches."""
    import ee, time
    cols = ["score", "slope", "elevation", "worldcover", "dw_label", "in_wdpa"]

    def fetch(sub):
        for attempt in range(max_retries):
            try:
                feats = [ee.Feature(ee.Geometry.Point([float(p[0]), float(p[1])]),
                                    {"lon": float(p[0]), "lat": float(p[1])}) for p in sub]
                fc = ee.FeatureCollection(feats)
                return stack.reduceRegions(fc, ee.Reducer.first(), scale=scale).getInfo()["features"]
            except ee.ee_exception.EEException as e:
                msg = str(e).lower()
                if ("timed out" in msg or "too many" in msg or "computation" in msg):
                    if len(sub) > 50:                          # split heavy batch and recurse
                        m = len(sub) // 2
                        return fetch(sub[:m]) + fetch(sub[m:])
                    time.sleep(5 * (attempt + 1))              # small batch: back off, retry
                    continue
                raise
        print(f"  WARN: dropped {len(sub)} pts after {max_retries} retries", flush=True)
        return []

    rows, t0 = [], time.time()
    for i in range(0, len(xy), chunk):
        for ft in fetch(xy[i:i+chunk]):
            pr = ft["properties"]
            rows.append([pr.get("lon"), pr.get("lat")] + [pr.get(c) for c in cols])
        print(f"  sampled {min(i+chunk, len(xy))}/{len(xy)} -> {len(rows)} ok ({time.time()-t0:.0f}s)", flush=True)
    arr = np.array(rows, dtype=object)
    out = {"lon": arr[:, 0].astype(float), "lat": arr[:, 1].astype(float)}
    for j, c in enumerate(cols):
        out[c] = np.array([np.nan if v is None else v for v in arr[:, 2 + j]], dtype=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--n", type=int, default=20000, help="random land points")
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=REFERENCE_MODEL)
    args = ap.parse_args()

    import ee
    os.chdir(os.path.expanduser("~/Solar_Workspace"))
    ee.Initialize(project="ee-abdullahr-solar")

    stack, geom = build_stack(args.country, args.year, args.model)
    xy = random_land_points(geom, args.n, args.seed)
    print(f"{args.country}: {len(xy)} random land points; sampling stack...", flush=True)
    data = sample_stack(stack, xy, scale=args.scale)

    slug = args.country.lower().replace(" ", "_")
    outdir = f"Solar-Siting/artifacts/covariate/{slug}"
    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/{slug}_{args.year}_covsample.npz"
    np.savez(out, **data)
    n_valid = np.isfinite(data["score"]).sum()
    manifest = dict(country=args.country, year=args.year, n_requested=args.n,
                    n_rows=len(data["score"]), n_valid_score=int(n_valid),
                    columns=list(data.keys()), gee_native=True,
                    covariates_pending=["gsa_*", "grid_dist", "road_dist"])
    with open(f"{outdir}/{slug}_{args.year}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nsaved {out}  ({len(data['score'])} rows, {n_valid} valid scores)", flush=True)
    for c in ["score", "slope", "elevation", "worldcover", "dw_label", "in_wdpa"]:
        v = data[c][np.isfinite(data[c])]
        if len(v):
            print(f"  {c:11s} n={len(v):6d} min={v.min():.3g} med={np.median(v):.3g} max={v.max():.3g}")


if __name__ == "__main__":
    main()

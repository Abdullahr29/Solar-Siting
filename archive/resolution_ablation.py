"""Sampling-resolution ablation for the embedding-model deployment.

QUESTION
--------
We have ALWAYS scored solar suitability at AEF-native 10 m, single pixel:
`img.classify(clf)` runs the RF per 10 m pixel and `reduceRegions(first, scale=10)`
grabs the one pixel under each point (see run_country_rf.py / covariate_sample.py).
That is the noisiest, most context-free point on the resolution axis and we have
never tested any other. Does aggregating the embedding over a coarser cell before
scoring help (denoising / context) or hurt (diluting the site into its surroundings),
and where is the accuracy-vs-resolution sweet spot?

THREE CURVES (per resolution R, output grid = EPSG:6933 equal-area at R metres)
  A_raw       mean of the 64-D embedding over the R x R cell  ->  classify
              (deployment-realistic: exactly "average the AEF embedding in the cell";
               note the mean of unit vectors has norm < 1, so this is OFF the unit
               sphere the RF was trained on -> a possible out-of-distribution penalty)
  A_renorm    same mean embedding, RENORMALISED to unit length  ->  classify
              (puts the aggregate back on the training sphere; isolates "which way
               does this neighbourhood point" from the norm/heterogeneity shift)
  B_scoremean classify at native 10 m, then mean the PROBABILITY over the R x R cell
              (pure spatial smoothing of the score; no embedding aggregation)
R = 10 m is the current-pipeline baseline (native classify, no aggregation); all three
curves share that anchor.

METRIC
  Positives = forward installs (year >= YEAR+2, unseen by the imagery), same as the
  MCDA benchmark and country runs. Negatives = 20k random-land points (same seed as the
  covariate sample, so it ties to the frozen Greece baseline). ROC / PR-AUC / median
  site percentile per (resolution, method).

Model: the frozen reference RF (rf30_final), injected into EE unchanged. This is a
DEPLOYMENT-resolution ablation (fixed model, vary inference aggregation) -- it does not
retrain per resolution.

Usage (JASMIN sci node, env_solar):
  python Solar-Siting/resolution_ablation.py Greece --year 2021 --n-rand 20000
  python Solar-Siting/resolution_ablation.py Greece --smoke        # fast mechanics check
"""
import argparse, csv, json, os, sys, time
import numpy as np

REFERENCE_MODEL = "Solar-Siting/artifacts/models/rf30_final.joblib"
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
INV_NAME = {"United States": "United States of America"}   # LSIB -> inventory name
BANDS = [f"A{i:02d}" for i in range(64)]
OUT_CRS = None                                             # set at runtime = country UTM zone
MAXPX = 65536                                              # reduceResolution input-pixel cap
RESOLUTIONS = [10, 30, 50, 100, 200, 500, 1000, 2000]


# ----------------------------------------------------------------------------- EE builders
def build_classifier(model_path):
    import joblib
    from geemap import ml
    rf = joblib.load(model_path)
    t0 = time.time()
    trees = ml.rf_to_strings(rf, BANDS, processes=32, output_mode="PROBABILITY")
    clf = ml.strings_to_classifier(trees)
    print(f"classifier injected: {len(trees)} trees, "
          f"{sum(len(s) for s in trees)/1e6:.1f} MB ({time.time()-t0:.0f}s)", flush=True)
    return clf


def utm_epsg(geom):
    """EE-supported metric grid for aggregation: the country-centroid UTM zone (true metres).

    EPSG:6933 (our offline equal-area CRS) is NOT in EE's projection database ("CRS could not
    be parsed"), and lifting a global AEF tile's UTM zone is invalid off-zone. The centroid
    UTM zone is valid across the country and lets reduceResolution average in real metres.
    """
    c = geom.centroid(maxError=1).coordinates().getInfo()
    lon, lat = c[0], c[1]
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def proj10():
    """The fine INPUT grid for aggregation: the country UTM CRS at 10 m.

    We deliberately do NOT lift the projection off a constituent AEF tile -- `coll.first()`
    is an arbitrary global tile whose UTM zone is invalid over the target country ("invalid
    area of projection"). A single global metric CRS (6933) at 10 m is valid everywhere, and
    reduceResolution then averages ~(R/10)^2 pixels into each R-metre output cell.
    """
    import ee
    return ee.Projection(OUT_CRS).atScale(10)


def native_aef(year):
    """Plain 64-band AEF mosaic. When sampled/classified at scale=10 the pixels keep their
    true native projection (this is exactly what run_country_rf / covariate_sample do); we
    only impose a projection on the reduceResolution path, via `coarsen`."""
    import ee
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f"{year}-01-01", f"{year+1}-01-01").mosaic().select(BANDS))


def coarsen(img, scale_m):
    """Mean-aggregate a (10 m) image onto the EPSG:6933 grid at scale_m metres."""
    import ee
    return (img.setDefaultProjection(proj10())
               .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=MAXPX)
               .reproject(crs=OUT_CRS, scale=scale_m))


def renorm(img):
    """Rescale each pixel's 64-D vector back to unit L2 norm."""
    import ee
    n = img.pow(2).reduce(ee.Reducer.sum()).sqrt()          # single band
    return img.divide(n).rename(BANDS)                      # single-band divisor broadcasts


def images_for(aef, clf, native_score, R):
    """(ee.Image with one band per method, sample_scale, [method band-names]) for one R.

    All methods for a resolution are packed into ONE multi-band image so a single
    reduceRegions pass scores every curve at once (3x fewer server round-trips).
    """
    if R == 10:                                            # current pipeline, no aggregation
        return native_score.rename("native"), 10, ["native"]
    emb = coarsen(aef, R)
    a_raw = emb.classify(clf).rename("A_raw")
    a_ren = renorm(emb).classify(clf).rename("A_renorm")
    b = coarsen(native_score, R).rename("B_scoremean")
    return a_raw.addBands([a_ren, b]), R, ["A_raw", "A_renorm", "B_scoremean"]


# ----------------------------------------------------------------------------- sampling
def sample_multi(image, xy, bands, scale, chunk=1000, max_retries=4):
    """reduceRegions(first) of a multi-band image at each point, index-aligned; NaN where
    dropped/nodata. Coarse-R points are heavy (each triggers a reduceResolution over up to
    ~(R/10)^2 pixels x 64 bands + classify), so we run with tileScale to cut EE peak memory
    and, as a backstop, recursively halve a batch that still times out OR blows the memory
    limit -- same resilience pattern as covariate_sample.sample_stack.
    """
    import ee
    out = {b: np.full(len(xy), np.nan) for b in bands}

    def fetch(idxs):
        for attempt in range(max_retries):
            try:
                feats = [ee.Feature(ee.Geometry.Point([float(xy[i][0]), float(xy[i][1])]),
                                    {"idx": int(i)}) for i in idxs]
                fc = ee.FeatureCollection(feats)
                res = image.reduceRegions(fc, ee.Reducer.first(), scale=scale,
                                          tileScale=16).getInfo()
                for ft in res["features"]:
                    pr = ft["properties"]; k = int(pr["idx"])
                    for b in bands:
                        # multi-band -> band-named property; single band -> Reducer.first()
                        # emits "first" (as run_country_rf.py reads), so fall back to it.
                        v = pr.get(b) if pr.get(b) is not None else (
                            pr.get("first") if len(bands) == 1 else None)
                        if v is not None:
                            out[b][k] = v
                return
            except ee.ee_exception.EEException as e:
                msg = str(e).lower()
                if any(s in msg for s in ("timed out", "too many", "computation",
                                          "memory", "limit exceeded")):
                    if len(idxs) > 25:
                        m = len(idxs) // 2
                        fetch(idxs[:m]); fetch(idxs[m:]); return
                    time.sleep(5 * (attempt + 1)); continue
                raise
        print(f"  WARN dropped {len(idxs)} pts after {max_retries} retries", flush=True)

    t0 = time.time()
    for i in range(0, len(xy), chunk):
        fetch(list(range(i, min(i + chunk, len(xy)))))
        print(f"    sampled {min(i+chunk, len(xy))}/{len(xy)} ({time.time()-t0:.0f}s)", flush=True)
    return out


# ----------------------------------------------------------------------------- points
def random_land_points(geom, n, seed):
    import ee
    rp = ee.FeatureCollection.randomPoints(region=geom, points=n, seed=seed)
    return np.array(rp.geometry().coordinates().getInfo(), dtype=float)


def forward_install_points(country, year, max_sites, seed):
    import geopandas as gpd
    inv_name = INV_NAME.get(country, country)
    sites = gpd.read_file(INV, where=f"country = '{inv_name}' AND year >= {year + 2}")
    if len(sites) == 0:
        sys.exit(f"{country}: no inventory sites (country='{inv_name}', year>={year+2})")
    pts = sites.geometry.representative_point().to_crs("EPSG:4326")
    xy = np.c_[pts.x, pts.y]
    if len(xy) > max_sites:
        xy = xy[np.random.default_rng(seed).choice(len(xy), max_sites, replace=False)]
    return xy


# ----------------------------------------------------------------------------- metrics
def metrics(site_s, rand_s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    site_s = site_s[np.isfinite(site_s)]
    rand_s = rand_s[np.isfinite(rand_s)]
    y = np.r_[np.ones(len(site_s)), np.zeros(len(rand_s))]
    x = np.r_[site_s, rand_s]
    pct = np.array([(rand_s < s).mean() for s in site_s])
    return dict(n_sites=len(site_s), n_rand=len(rand_s),
                roc=round(roc_auc_score(y, x), 4),
                pr=round(average_precision_score(y, x), 4),
                pr_chance=round(len(site_s) / len(x), 4),
                median_site_pct=round(float(np.median(pct) * 100), 2),
                top10=round(float((pct >= 0.9).mean() * 100), 2))


def norm_check(aef, xy):
    """Native-pixel L2-norm distribution -- confirms AEF pixels are (quantised) unit-norm,
    so the renorm control is well posed and the norm signal only appears under aggregation."""
    import ee
    sub = xy[:min(len(xy), 600)]
    feats = [ee.Feature(ee.Geometry.Point([float(p[0]), float(p[1])])) for p in sub]
    res = aef.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.first(), scale=10).getInfo()
    norms = []
    for ft in res["features"]:
        v = [ft["properties"].get(b) for b in BANDS]
        if all(x is not None for x in v):
            norms.append(float(np.linalg.norm(v)))
    norms = np.array(norms)
    pcts = np.percentile(norms, [1, 5, 50, 95, 99]) if len(norms) else []
    print(f"\nnative-pixel ||v|| over {len(norms)} pts: "
          f"p1/5/50/95/99 = {', '.join(f'{p:.4f}' for p in pcts)}", flush=True)
    return norms


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--n-rand", type=int, default=20000)
    ap.add_argument("--max-sites", type=int, default=3000)
    ap.add_argument("--resolutions", default=None, help="comma list, metres (default: full set)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=REFERENCE_MODEL)
    ap.add_argument("--smoke", action="store_true", help="n-rand=400, resolutions=10,100,2000")
    args = ap.parse_args()

    os.chdir(os.path.expanduser("~/Solar_Workspace"))
    import ee
    ee.Initialize(project="ee-abdullahr-solar")

    resolutions = ([int(r) for r in args.resolutions.split(",")] if args.resolutions
                   else ([10, 100, 2000] if args.smoke else RESOLUTIONS))
    n_rand = 400 if args.smoke else args.n_rand
    slug = args.country.lower().replace(" ", "_")

    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.eq("country_na", args.country))
    if lsib.size().getInfo() == 0:
        sys.exit(f"'{args.country}' not in LSIB")
    geom = lsib.geometry()

    global OUT_CRS
    OUT_CRS = utm_epsg(geom)
    print(f"aggregation grid CRS: {OUT_CRS}", flush=True)

    clf = build_classifier(args.model)
    aef = native_aef(args.year)
    native_score = aef.classify(clf).rename("score")

    site_xy = forward_install_points(args.country, args.year, args.max_sites, args.seed)
    rand_xy = random_land_points(geom, n_rand, args.seed)
    print(f"{args.country} {args.year}: {len(site_xy)} forward-install sites, "
          f"{len(rand_xy)} random-land points; resolutions {resolutions}", flush=True)

    norm_check(aef, rand_xy)

    rows, scores = [], {}
    for R in resolutions:
        print(f"\n=== resolution {R} m ===", flush=True)
        image, scl, methods = images_for(aef, clf, native_score, R)
        print(f"  sites ({len(methods)} band(s): {methods})...", flush=True)
        site = sample_multi(image, site_xy, methods, scl)
        print("  random...", flush=True)
        rand = sample_multi(image, rand_xy, methods, scl)
        for name in methods:
            m = metrics(site[name], rand[name])
            rows.append(dict(country=args.country, year=args.year, resolution_m=R, method=name, **m))
            scores[f"{R}_{name}_site"] = site[name]
            scores[f"{R}_{name}_rand"] = rand[name]
            print(f"  -> {name} R={R}m: ROC {m['roc']} | PR {m['pr']} "
                  f"(chance {m['pr_chance']}) | median pct {m['median_site_pct']} "
                  f"| n {m['n_sites']}+{m['n_rand']}", flush=True)

    # ---- persist ----
    tag = "_smoke" if args.smoke else ""
    resdir = "Solar-Siting/artifacts/results"
    os.makedirs(resdir, exist_ok=True)
    csv_path = f"{resdir}/resolution_ablation_{slug}{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez(f"{resdir}/resolution_ablation_{slug}{tag}_scores.npz", **scores)
    with open(f"{resdir}/resolution_ablation_{slug}{tag}_manifest.json", "w") as fh:
        json.dump(dict(country=args.country, year=args.year, n_rand=n_rand,
                       n_sites=len(site_xy), resolutions=resolutions, model=args.model,
                       out_crs=OUT_CRS, methods=["A_raw", "A_renorm", "B_scoremean"]),
                  fh, indent=2)
    print(f"\nwrote {csv_path} (+ _scores.npz, _manifest.json)", flush=True)

    # ---- table ----
    print("\n  res(m)  A_renorm  A_raw   B_smean")
    base = next(r["roc"] for r in rows if r["resolution_m"] == 10)
    for R in resolutions:
        d = {r["method"]: r["roc"] for r in rows if r["resolution_m"] == R}
        if R == 10:
            print(f"  {R:>5}   {base:.4f}   (baseline: native single-pixel)")
        else:
            print(f"  {R:>5}   {d.get('A_renorm','   -  ')}   {d.get('A_raw','  -  ')}  "
                  f"{d.get('B_scoremean','  -  ')}")


if __name__ == "__main__":
    main()

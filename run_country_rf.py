"""Country-scale solar-suitability run with the GEE-injected random forest.

Scores a whole country's AlphaEarth (AEF) mosaic with the trained RF — entirely inside
Earth Engine, no imagery download — then validates the map against inventory sites that
did not yet exist in the imagery year, and renders a national heatmap with those sites
laid on top.

Usage (from a JASMIN terminal or notebook, in the Solar_Workspace root):

    python Solar-Siting/run_country_rf.py Greece
    python Solar-Siting/run_country_rf.py Germany --year 2019
    python Solar-Siting/run_country_rf.py China --n-rand 8000 --max-sites 3000

or from Python:

    from run_country_rf import run_country
    run_country("Spain", year=2019)

Method / caveats:
  * The map year is the AEF embedding year (default 2019). Validation uses inventory
    sites installed >= year+2, matching the 2-year pre-installation gap the model was
    trained with — so validation sites did NOT exist in the scored imagery.
  * "Random land" points estimate the country-wide score distribution (percentile
    thresholds converge fast; 6,000 points give sub-percentile accuracy). Random land
    is NOT confirmed non-solar: some high-scoring points are genuinely suitable but
    undeveloped land, so country ROC under-states the model.
  * Known limitation (under discussion): validation sites are not filtered against the
    RF's global training sample, so a few percent of them may overlap training scenes.
  * The RF must be trained WITHOUT max_leaf_nodes: that flag switches sklearn to
    best-first tree building, and geemap converts those trees into strings GEE
    rejects ("Error parsing line N: expected 8, got 3"). Size trees with
    min_samples_leaf / max_depth instead.
  * Country name must match both the LSIB boundary table ("country_na") and the
    inventory's "country" field; the script reports candidates if a name misses.
"""
import argparse, io, os, sys, time

import numpy as np


#: THE reference model. One model, everything scored against it — superseded forests live in
#: Solar-Siting/legacy/. This default was previously rf30_masked_5pct.joblib (px ROC 0.8907)
#: while the published Germany/China numbers actually came from rf30_coverage.joblib (0.9014),
#: so anyone running the CLI bare silently got the WEAKER forest. Every LOCO/temporal run goes
#: through here, so a stale default makes every ablation gap meaningless. Keep this pointed at
#: the frozen reference and change it only when a new model is blessed.
REFERENCE_MODEL = "Solar-Siting/artifacts/models/rf30_final.joblib"


def run_country(country, year=2019, min_install=None, n_rand=6000, max_sites=3000,
                model_path=REFERENCE_MODEL,
                out_prefix=None, workspace=None, seed=42, inv_country=None,
                exclude_pv_ids=None):
    """Score one country and write {prefix}_scores.npz, {prefix}_fig.png, {prefix}.png.

    Returns a dict of metrics. See module docstring for method and caveats.

    `inv_country` overrides the name used to query the inventory when it differs from the
    LSIB boundary name (e.g. LSIB "United States" vs inventory "United States of America").
    Defaults to `country`.
    """
    inv_country = inv_country or country
    import ee, geopandas as gpd, joblib, requests
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from geemap import ml
    from PIL import Image
    from shapely.geometry import shape, Point
    from sklearn.metrics import roc_auc_score, average_precision_score

    if workspace is None:
        workspace = os.path.expanduser("~/Solar_Workspace")
    os.chdir(workspace)
    if min_install is None:
        min_install = year + 2
    if out_prefix is None:
        out_prefix = f"Solar-Siting/artifacts/country/{country.lower().replace(' ', '_')}_rf_{year}"

    ee.Initialize(project="ee-abdullahr-solar")
    bands = [f"A{i:02d}" for i in range(64)]

    # ---- country boundary ----
    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    fc = lsib.filter(ee.Filter.eq("country_na", country))
    if fc.size().getInfo() == 0:
        names = lsib.aggregate_array("country_na").distinct().getInfo()
        close = [n for n in names if country.lower() in n.lower()]
        sys.exit(f"'{country}' not in LSIB. Close matches: {close}")
    geom = fc.geometry()

    # ---- RF -> GEE classifier (train locally once; inject as tree strings) ----
    rf = joblib.load(model_path)
    t0 = time.time()
    trees = ml.rf_to_strings(rf, bands, processes=32, output_mode="PROBABILITY")
    clf = ml.strings_to_classifier(trees)
    print(f"classifier injected: {len(trees)} trees, "
          f"{sum(len(s) for s in trees)/1e6:.1f} MB ({time.time()-t0:.0f}s)", flush=True)

    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{year}-01-01", f"{year+1}-01-01").mosaic().select(bands))
    score = img.classify(clf).clip(geom).rename("score")

    # ---- validation sites: installed after the imagery, so unseen by the map ----
    inv = gpd.read_file("Solar-Siting/global_pv_facility_inventory.gpkg",
                        where=f"country = '{inv_country}' AND year >= {min_install}")
    if len(inv) == 0:
        sys.exit(f"no inventory sites for country='{inv_country}' with year >= {min_install} "
                 f"(check the inventory's country spelling)")
    if exclude_pv_ids is not None and "PV_ID" in inv.columns:
        before = len(inv)
        inv = inv[~inv["PV_ID"].isin(set(exclude_pv_ids))]
        print(f"dedup: dropped {before-len(inv):,} of {before:,} validation sites seen in training", flush=True)
        if len(inv) == 0:
            sys.exit("all validation sites were in training after dedup")
    pts = inv.geometry.representative_point().to_crs("EPSG:4326")
    site_xy = np.c_[pts.x, pts.y]
    if len(site_xy) > max_sites:
        site_xy = site_xy[np.random.default_rng(seed).choice(len(site_xy), max_sites, replace=False)]
        print(f"validation sites: {len(pts):,} -> sampled {max_sites:,}", flush=True)
    else:
        print(f"validation sites (installed >= {min_install}): {len(site_xy):,}", flush=True)

    # ---- random land points (country-wide score distribution) ----
    shp = shape(geom.getInfo())
    minx, miny, maxx, maxy = shp.bounds
    rng = np.random.default_rng(seed)
    rand_xy = []
    while len(rand_xy) < n_rand:
        cand = np.c_[rng.uniform(minx, maxx, 4000), rng.uniform(miny, maxy, 4000)]
        rand_xy += [tuple(c) for c in cand if shp.contains(Point(c))]
    rand_xy = np.array(rand_xy[:n_rand])

    # ---- score all points (chunked reduceRegions: only touched tiles compute) ----
    def pts_scores(xy, tag):
        out = []
        for i in range(0, len(xy), 1000):
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(p[0]), float(p[1])]))
                                       for p in xy[i:i+1000]])
            vals = (score.reduceRegions(fc, ee.Reducer.first(), scale=10)
                    .aggregate_array("first").getInfo())
            out += [v for v in vals if v is not None]
            print(f"  {tag} {min(i+1000, len(xy))}/{len(xy)}", flush=True)
        return np.array(out, dtype=float)

    t0 = time.time()
    site_s = pts_scores(site_xy, "sites")
    rand_s = pts_scores(rand_xy, "random")
    print(f"scored {len(site_s)} sites + {len(rand_s)} random ({time.time()-t0:.0f}s)", flush=True)

    # ---- metrics ----
    y = np.r_[np.ones(len(site_s)), np.zeros(len(rand_s))]
    x = np.r_[site_s, rand_s]
    pct = np.array([(rand_s < s).mean() for s in site_s])
    m = dict(country=country, year=year, min_install=min_install,
             n_sites=len(site_s), n_rand=len(rand_s),
             ROC=roc_auc_score(y, x), PR=average_precision_score(y, x),
             PR_chance=len(site_s) / len(x), median_site_pct=np.median(pct) * 100,
             top20=(pct >= 0.8).mean() * 100, top10=(pct >= 0.9).mean() * 100,
             top5=(pct >= 0.95).mean() * 100)
    print(f"\n=== {country} {year} map vs sites installed >= {min_install} ===")
    print(f"ROC {m['ROC']:.3f} | PR-AUC {m['PR']:.3f} (chance {m['PR_chance']:.3f}) | "
          f"median site percentile {m['median_site_pct']:.1f}")
    print(f"sites found in top 20/10/5% of ranked land: "
          f"{m['top20']:.1f}% / {m['top10']:.1f}% / {m['top5']:.1f}%", flush=True)

    # ---- save scores FIRST (rendering can fail for huge countries; data must survive) ----
    b = geom.bounds().coordinates().getInfo()[0]
    xs, ys = [c[0] for c in b], [c[1] for c in b]
    ext = [min(xs), max(xs), min(ys), max(ys)]
    np.savez(f"{out_prefix}_scores.npz", site_s=site_s, rand_s=rand_s,
             site_xy=site_xy, rand_xy=rand_xy, ext=np.array(ext))
    print(f"saved {out_prefix}_scores.npz", flush=True)

    # ---- national heatmap + distributions figure ----
    lo, hi = np.percentile(rand_s, [2, 98])
    palette = ["000004", "2c115f", "721f81", "b73779", "f1605d", "feb078", "fcfdbf"]
    im = None
    for dim in (1400, 900, 600):
        try:
            url = score.getThumbURL({"region": geom.bounds(), "dimensions": dim,
                                     "min": float(lo), "max": float(hi), "palette": palette})
            png = requests.get(url, timeout=600).content
            im = np.array(Image.open(io.BytesIO(png)))
            with open(f"{out_prefix}.png", "wb") as f:
                f.write(png)
            break
        except Exception as e:
            snippet = png[:200] if isinstance(png, bytes) else b""
            print(f"thumbnail at {dim}px failed ({e}); payload head: {snippet!r}", flush=True)
    if im is None:
        print("no thumbnail could be rendered; skipping figure (scores npz already saved)", flush=True)
        return m

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].imshow(im, extent=ext, aspect="auto")
    axes[0].scatter(site_xy[:, 0], site_xy[:, 1], s=4, c="cyan", alpha=0.6,
                    label=f"sites installed >= {min_install} (n={len(site_xy):,})")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title(f"{country} — RF suitability score, AEF {year} (thumbnail render)")
    axes[1].hist(rand_s, bins=60, density=True, alpha=0.6, label=f"random {country} land")
    axes[1].hist(site_s, bins=60, density=True, alpha=0.6, label="future solar sites")
    axes[1].axvline(np.median(rand_s), ls="--", c="C0")
    axes[1].axvline(np.median(site_s), ls="--", c="C1")
    axes[1].set_xlabel("RF probability score")
    axes[1].set_ylabel("probability density (each histogram integrates to 1)")
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"ROC {m['ROC']:.3f} | PR-AUC {m['PR']:.3f} (chance {m['PR_chance']:.3f}) | "
                      f"median site pct {m['median_site_pct']:.0f}")
    fig.savefig(f"{out_prefix}_fig.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_prefix}_scores.npz / _fig.png / .png")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Country-scale RF suitability run (GEE-native)")
    ap.add_argument("country", help="country name as in LSIB / the inventory (e.g. Greece)")
    ap.add_argument("--year", type=int, default=2019, help="AEF embedding year of the map")
    ap.add_argument("--min-install", type=int, default=None,
                    help="validate sites installed >= this year (default: year+2)")
    ap.add_argument("--n-rand", type=int, default=6000, help="random land points")
    ap.add_argument("--max-sites", type=int, default=3000, help="cap on validation sites")
    ap.add_argument("--model", default=REFERENCE_MODEL,
                    help="path to the trained sklearn RF joblib (default: the frozen reference)")
    ap.add_argument("--inv-country", default=None,
                    help="inventory country name if it differs from the LSIB name "
                         "(e.g. 'United States of America' vs LSIB 'United States')")
    args = ap.parse_args()
    run_country(args.country, year=args.year, min_install=args.min_install,
                n_rand=args.n_rand, max_sites=args.max_sites, model_path=args.model,
                inv_country=args.inv_country)

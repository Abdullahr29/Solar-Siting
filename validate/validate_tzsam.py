"""External forward-validation of the RF-R model against TZ-SAM Q1-2026 solar sites.

Rationale
---------
Our inventory-based country validation is thin for several countries (South Africa 28,
Colombia 23, Philippines 22, Malaysia 32 positives). TZ-SAM (TransitionZero Solar Asset
Mapper, Q1-2026) is an INDEPENDENT, SAM-detected global solar dataset that reaches 2026 and
contains ~47.8k sites NOT in our supervisor's inventory. Using it as a second, independent
forward-validation source multiplies n_pos 1.4-6.5x per country and lifts the fragile
small-n countries into statistically meaningful territory.

Method (identical scoring to run_country_rf.py, so numbers are comparable)
-------------------------------------------------------------------------
  * Positives = TZ-SAM "new" sites (deduplicated against our inventory by nearest-neighbour
    distance) for the country, filtered to constructed_after >= 2020 so the ground was
    verifiably BARE in 2019 -> the fixed 2019 AEF embedding is guaranteed pre-solar
    (clean forward test, zero target leakage), AND >= 1 km from any training positive.
  * Each TZ point is scored by RF-R on the 2019 AlphaEarth mosaic, exactly as the inventory
    validation does: score = AEF2019.classify(rf_to_strings(RF-R)); reduceRegions(first, 10m).
  * Negatives are REUSED from the cached country run (scores/country/<c>_R_2019_scores.npz,
    key `rand_s`) — same model, same AEF year, same random-land sampler — so the TZ-SAM ROC
    sits on the identical negative distribution as the inventory ROC. Only the new positives
    hit GEE (cheap: point sampling, not raster classify — this is NOT the operation that blew
    the quota).
  * Reports THREE ROCs per country: inventory-only (from cache), TZ-SAM-only, and combined.

Usage
-----
    python Solar-Siting/validate_tzsam.py "South Africa"
    python Solar-Siting/validate_tzsam.py Greece Germany China ...     # several at once
    python Solar-Siting/validate_tzsam.py --all                        # the 13 validation countries

Outputs (per country) -> Solar-Siting/artifacts/paper_v3/tzsam/
    <c>_tzsam_scores.npz   (tz_s, tz_xy, and echoed inv site_s / rand_s for convenience)
    appends a row to       tzsam_validation.csv
"""
import argparse, csv, os, sys, time
from datetime import datetime

import numpy as np

WORKSPACE = os.path.expanduser("~/Solar_Workspace")
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"   # the R reference (paper_v3)
TZ_NEW = "Data/external/tz_sam_q1_2026/overlap_analysis/tz_new_sites.csv"
CACHE_DIR = "Solar-Siting/artifacts/paper_v3/scores/country"
OUT_DIR = "Solar-Siting/artifacts/paper_v3/tzsam"

# The 13 repeatedly-considered validation countries. Names must match the TZ-SAM `country`
# field AND the cached-npz basename (country.lower().replace(' ','_')).
VAL_COUNTRIES = ["Greece", "Germany", "China", "Spain", "Poland", "Japan", "India",
                 "United States", "South Africa", "Colombia", "Philippines",
                 "Malaysia", "Chile"]

MIN_CONSTRUCTED_YEAR = 2020   # constructed_after >= this -> bare in 2019 -> 2019 AEF pre-solar
MIN_NN_M = 1000.0             # >= this metres from any training positive (belt-and-braces)
AEF_YEAR = 2019


def _slug(country):
    return country.lower().replace(" ", "_")


def load_tz_positives(country):
    """Return Nx2 lon/lat array of clean, post-2019 TZ-SAM new sites for `country`."""
    xy = []
    kept = dropped_date = dropped_nn = 0
    with open(os.path.join(WORKSPACE, TZ_NEW), newline="") as f:
        for row in csv.DictReader(f):
            if row["country"] != country:
                continue
            # date gate: require a constructed_after date whose year >= MIN_CONSTRUCTED_YEAR
            ca = (row.get("constructed_after") or "").strip()
            if not ca:
                dropped_date += 1
                continue
            try:
                cay = datetime.fromisoformat(ca[:10]).year
            except ValueError:
                dropped_date += 1
                continue
            if cay < MIN_CONSTRUCTED_YEAR:
                dropped_date += 1
                continue
            # spatial gate: far from any training positive
            try:
                nn = float(row.get("nn_m") or 0.0)
            except ValueError:
                nn = 0.0
            if nn < MIN_NN_M:
                dropped_nn += 1
                continue
            try:
                lon = float(row["longitude"]); lat = float(row["latitude"])
            except (ValueError, KeyError):
                continue
            xy.append((lon, lat)); kept += 1
    print(f"  TZ-SAM {country}: kept {kept}  (dropped {dropped_date} pre-{MIN_CONSTRUCTED_YEAR}/no-date, "
          f"{dropped_nn} within {MIN_NN_M:.0f} m of training)", flush=True)
    return np.array(xy, dtype=float) if xy else np.empty((0, 2))


def load_cache(country):
    """Return (inv_site_s, rand_s) from the cached RF-R country run."""
    p = os.path.join(WORKSPACE, CACHE_DIR, f"{_slug(country)}_R_{AEF_YEAR}_scores.npz")
    if not os.path.exists(p):
        sys.exit(f"no cached RF-R country run at {p} — run run_country_rf.py first")
    d = np.load(p)
    return d["site_s"].astype(float), d["rand_s"].astype(float)


def run(countries):
    import ee, joblib
    from geemap import ml
    from sklearn.metrics import roc_auc_score

    os.chdir(WORKSPACE)
    os.makedirs(OUT_DIR, exist_ok=True)
    ee.Initialize(project="ee-abdullahr-solar")
    bands = [f"A{i:02d}" for i in range(64)]

    # RF-R -> GEE classifier, once
    rf = joblib.load(MODEL)
    t0 = time.time()
    trees = ml.rf_to_strings(rf, bands, processes=32, output_mode="PROBABILITY")
    clf = ml.strings_to_classifier(trees)
    print(f"classifier injected: {len(trees)} trees ({time.time()-t0:.0f}s)", flush=True)

    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{AEF_YEAR}-01-01", f"{AEF_YEAR+1}-01-01").mosaic().select(bands))
    score = img.classify(clf).rename("score")   # global; lazy — only sampled points compute

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

    rows = []
    for country in countries:
        print(f"\n=== {country} ===", flush=True)
        tz_xy = load_tz_positives(country)
        if len(tz_xy) == 0:
            print("  no TZ-SAM positives after filtering — skipping", flush=True)
            continue
        inv_s, rand_s = load_cache(country)

        t0 = time.time()
        tz_s = pts_scores(tz_xy, "tz")
        print(f"  scored {len(tz_s)}/{len(tz_xy)} TZ points ({time.time()-t0:.0f}s)", flush=True)

        def roc(pos):
            y = np.r_[np.ones(len(pos)), np.zeros(len(rand_s))]
            x = np.r_[pos, rand_s]
            return roc_auc_score(y, x)

        roc_inv = roc(inv_s) if len(inv_s) else float("nan")
        roc_tz = roc(tz_s)
        comb = np.r_[inv_s, tz_s]
        roc_comb = roc(comb)

        print(f"  ROC  inventory={roc_inv:.4f} (n={len(inv_s)})  "
              f"TZ-SAM={roc_tz:.4f} (n={len(tz_s)})  "
              f"combined={roc_comb:.4f} (n={len(comb)})", flush=True)

        out = os.path.join(OUT_DIR, f"{_slug(country)}_tzsam_scores.npz")
        np.savez(out, tz_s=tz_s, tz_xy=tz_xy, site_s=inv_s, rand_s=rand_s)
        print(f"  saved {out}", flush=True)

        rows.append(dict(country=country, aef_year=AEF_YEAR,
                         n_inv=len(inv_s), roc_inv=round(roc_inv, 4),
                         n_tz=len(tz_s), roc_tz=round(roc_tz, 4),
                         n_comb=len(comb), roc_comb=round(roc_comb, 4),
                         n_rand=len(rand_s), model=MODEL,
                         min_constructed_year=MIN_CONSTRUCTED_YEAR, min_nn_m=MIN_NN_M,
                         timestamp=datetime.now().isoformat(timespec="seconds")))

    # append to CSV
    if rows:
        csv_path = os.path.join(OUT_DIR, "tzsam_validation.csv")
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {csv_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TZ-SAM external forward-validation of RF-R")
    ap.add_argument("countries", nargs="*", help="country names (TZ-SAM spelling)")
    ap.add_argument("--all", action="store_true", help="run all 13 validation countries")
    a = ap.parse_args()
    cs = VAL_COUNTRIES if a.all else a.countries
    if not cs:
        ap.error("give country names or --all")
    run(cs)

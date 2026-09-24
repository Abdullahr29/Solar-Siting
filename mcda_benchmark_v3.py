#!/usr/bin/env python
"""MCDA benchmark v3 — literature-sourced weights, WLC + TOPSIS, full criteria set, v2/R model.

Reproduces two published solar-siting MCDA configurations and scores them against the SAME
forward-install positives vs random-land negatives the RF is judged on, so Fig A can report
RF ROC vs a *published* MCDA band (not our invented weights).

Sources (see Solar-Siting/MCDA_SOURCES.md):
  - Richards et al. 2025, Table 5 (AHP, CR<0.1): econ-heavy weights.
  - Chen & Jong 2024, Table 6 (fuzzy median): flat weights + population density.
Aggregators: WLC (weighted linear sum) and TOPSIS (fixed ideal=1 / anti-ideal=0 so each point
scores independently -> ROC-compatible). Normalisation: country-relative min-max (2-98 pct of
random land). Land-cover run BOTH ways (excluded / weighted). Protected areas -> exclusion.

Also recomputes, per country, the FULL permutation-importance vector of the same MCDA criteria
against the R suitability score (the "learned importance vs expert weights" comparison), which
paper_v3 only saved as a PNG.

Run:  python Solar-Siting/mcda_benchmark_v3.py "Greece" --year 2021
"""
import argparse, csv, os, sys, time, json
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
REFERENCE_MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"   # v2/R candidate
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
INV_NAME = {"United States": "United States of America"}
OUT_CSV = "Solar-Siting/artifacts/paper_v3/mcda_v3_results.csv"
IMP_DIR = "Solar-Siting/artifacts/paper_v3/mcda_v3_importance"

# criterion name -> (cached/sampled column key, direction: b=benefit higher-better / c=cost)
DIRN = {
    "ghi": ("gsa_ghi", "b"), "pvout": ("gsa_pvout", "b"), "gti": ("gsa_gti", "b"),
    "temp": ("gsa_temp", "c"), "slope": ("slope", "c"), "elevation": ("elevation", "b"),
    "aspect": ("equatorwardness", "b"), "road": ("road_dist_m", "c"),
    "grid": ("grid_dist_m", "c"), "popdens": ("pop_density", "b"),
}
# Published weight vectors, renormalised over the criteria we can supply (MCDA_SOURCES.md sec 5).
WEIGHTS = {
    "richards": {"road": 0.5401, "grid": 0.1392, "slope": 0.0605, "elevation": 0.0225,
                 "ghi": 0.0352, "pvout": 0.0253, "gti": 0.0169, "aspect": 0.0759, "temp": 0.0844},
    "chen": {"ghi": 0.1924, "temp": 0.1701, "slope": 0.1576, "elevation": 0.1594,
             "grid": 0.1661, "popdens": 0.1544},
}
LAND_SUIT = {10: 0.3, 20: 0.8, 30: 0.9, 40: 0.6, 50: 0.1, 60: 1.0, 70: 0.2,
             80: 0.0, 90: 0.0, 95: 0.0, 100: 0.5}
EXCLUDE_WC = {80, 90}
GSA_NEED = [("World_GHI_", "gsa_ghi"), ("World_PVOUT_GISdata_LTAy", "gsa_pvout"),
            ("World_GTI_", "gsa_gti"), ("World_TEMP_", "gsa_temp")]


# ---------- aggregation ----------
def norm_benefit(col, lo, hi, direction):
    x = np.clip((np.asarray(col, float) - lo) / (hi - lo + 1e-12), 0, 1)
    return x if direction == "b" else 1.0 - x            # cost -> invert (all benefit-oriented)


def agg_wlc(N, w):
    return sum(w[c] * N[c] for c in w)


def agg_topsis(N, w):
    crits = list(w)
    ws = np.array([w[c] for c in crits])
    X = np.column_stack([N[c] for c in crits]) * ws       # weighted, benefit-oriented in [0,1]
    dpos = np.sqrt(((X - ws) ** 2).sum(axis=1))           # to ideal (w*1)
    dneg = np.sqrt((X ** 2).sum(axis=1))                  # to anti-ideal (0)
    return dneg / (dpos + dneg + 1e-12)


# ---------- GEE sampling ----------
def sample_population(xy, scale=927, chunk=500):
    import ee
    img = (ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
           .filterDate("2019-01-01", "2021-12-31").first())   # GPW density, persons/km^2 (native ~927 m)
    out = np.full(len(xy), np.nan)
    t0 = time.time()
    for s in range(0, len(xy), chunk):
        feats = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"i": int(s + k)})
                 for k, (x, y) in enumerate(xy[s:s + chunk])]
        try:
            res = img.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.first(), scale).getInfo()
            for f in res["features"]:
                p = f["properties"]
                # reduceRegions(Reducer.first()) on a SINGLE-band image names the output "first", not the band
                out[p["i"]] = np.nan if p.get("first") is None else p["first"]
        except Exception as e:
            print(f"  pop batch {s} failed: {str(e)[:80]}", flush=True)
        print(f"  pop {min(s+chunk,len(xy))}/{len(xy)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def sample_full(country, year, xy, model):
    """Sample every MCDA criterion + R score at the given points. Returns dict aligned to the
    points sample_stack actually returned (it drops timeouts), so all layers stay aligned."""
    import ee, geopandas as gpd
    from covariate_sample import build_stack, sample_stack
    from covariate_append_manual import ensure_gsa_tif, sample_raster, nearest_distance, OSM_SHP, CD
    from covariate_aspect import sample_aspect, derive

    stack, geom = build_stack(country, year, model)
    d = sample_stack(stack, xy)                            # canonical points (may be < len(xy))
    lon, lat = d["lon"], d["lat"]
    out = {k: np.asarray(d[k], float) for k in ("score", "slope", "elevation", "worldcover", "in_wdpa")}
    for sub, col in GSA_NEED:
        out[col] = sample_raster(ensure_gsa_tif(sub), lon, lat)
    pad = 0.6
    bbox = (lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)
    g = gpd.read_file(f"{CD}/gridfinder/grid.gpkg", bbox=bbox)
    out["grid_dist_m"] = nearest_distance(lon, lat, g, "grid") if len(g) else np.full(len(lon), np.nan)
    slug = country.lower().replace(" ", "_"); key = slug.replace("_", " ")
    gpkg = f"{CD}/osm/{slug}_roads.gpkg"
    if os.path.exists(gpkg):
        r = gpd.read_file(gpkg, bbox=bbox)
    elif key in OSM_SHP:
        r = gpd.read_file(f"/vsizip/{os.path.abspath(CD)}/osm/{OSM_SHP[key]}/gis_osm_roads_free_1.shp", bbox=bbox)
    else:
        r = None
    out["road_dist_m"] = nearest_distance(lon, lat, r, "roads") if (r is not None and len(r)) else np.full(len(lon), np.nan)
    asp, _ = sample_aspect(lon, lat)
    out["equatorwardness"], _, _ = derive(asp, out["slope"], lat)
    out["pop_density"] = sample_population(np.c_[lon, lat])
    return out


def score_at(country, year, xy, model):
    """R-model P(solar) at given points, aligned to input order via (lon,lat) match."""
    import ee
    from covariate_sample import build_stack, sample_stack
    stack, _ = build_stack(country, year, model)
    d = sample_stack(stack, xy)
    key = {(round(a, 6), round(b, 6)): s for a, b, s in zip(d["lon"], d["lat"], d["score"])}
    return np.array([key.get((round(x, 6), round(y, 6)), np.nan) for x, y in xy], float)


# ---------- main ----------
def land_suit(worldcover):
    wc = np.where(np.isfinite(worldcover), worldcover, -1).astype(int)
    return np.array([LAND_SUIT.get(int(v), 0.3) for v in wc]), wc


def run_country(country, year, model, n_neg_max=20000, seed=42):
    import ee
    ee.Initialize(project="ee-abdullahr-solar")
    from sklearn.metrics import roc_auc_score
    import geopandas as gpd

    slug = country.lower().replace(" ", "_")
    base = f"Solar-Siting/artifacts/covariate/{slug}/{slug}_{year}"
    neg_path = f"{base}_covsample_corrected.npz" if os.path.exists(f"{base}_covsample_corrected.npz") \
        else f"{base}_covsample_full.npz"
    print(f"[{country}] negatives: {os.path.basename(neg_path)}", flush=True)
    neg = {k: v for k, v in np.load(neg_path).items()}          # cached covariates (+ old score, lon/lat)
    neg_xy = np.c_[neg["lon"], neg["lat"]]

    # forward-install positives (installed >= year+2, unseen by imagery)
    inv_name = INV_NAME.get(country, country)
    sites = gpd.read_file(INV, where=f"country = '{inv_name}' AND year >= {year+2}")
    if len(sites) == 0:
        raise SystemExit(f"{country}: no forward-install sites")
    pts = sites.geometry.representative_point().to_crs("EPSG:4326")
    pos_xy = np.c_[pts.x, pts.y]
    rng = np.random.default_rng(seed)
    if len(pos_xy) > n_neg_max:
        pos_xy = pos_xy[rng.choice(len(pos_xy), n_neg_max, replace=False)]
    print(f"[{country}] {len(pos_xy)} positives, {len(neg_xy)} negatives", flush=True)

    # --- sample ---
    print(f"[{country}] sampling positives (GEE+local)...", flush=True)
    P = sample_full(country, year, pos_xy, model)
    print(f"[{country}] re-scoring negatives with R + population...", flush=True)
    neg_score = score_at(country, year, neg_xy, model)
    neg_pop = sample_population(neg_xy)

    # assemble columns for each set; negatives reuse cached covariates
    def col(tab, key, n):
        return np.asarray(tab[key], float) if key in tab else np.full(n, np.nan)
    N = {}
    for name, (k, _) in DIRN.items():
        if k == "pop_density":
            N[name + "_pos"] = P["pop_density"]; N[name + "_neg"] = neg_pop
        else:
            N[name + "_pos"] = col(P, k, len(P["score"]))
            N[name + "_neg"] = col(neg, k, len(neg_xy))
    pos_score, neg_score = P["score"], neg_score
    pos_wc, neg_wc = P["worldcover"], col(neg, "worldcover", len(neg_xy))
    pos_wdpa, neg_wdpa = P["in_wdpa"], col(neg, "in_wdpa", len(neg_xy))

    # stack pos+neg
    cat = lambda a, b: np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    crit_raw = {name: cat(N[name + "_pos"], N[name + "_neg"]) for name in DIRN}
    score = cat(pos_score, neg_score)
    wc = cat(pos_wc, neg_wc)
    wdpa = cat(pos_wdpa, neg_wdpa)
    label = np.r_[np.ones(len(pos_score)), np.zeros(len(neg_score))]
    n_neg = len(neg_score)

    # NO global row-drop: a single flaky layer must never empty the set. Each config is scored on
    # the rows finite for ITS OWN criteria; normalisation ranges come from finite negatives per criterion.
    neg_mask = label == 0
    ranges = {}
    for name, (k, direction) in DIRN.items():
        col = crit_raw[name]; m = neg_mask & np.isfinite(col)
        frac = 100.0 * np.mean(~np.isfinite(col))
        if m.sum() < 100:
            print(f"[{country}] WARN criterion {name}: {frac:.1f}% NaN ({int(m.sum())} finite neg) -> skipped", flush=True)
            ranges[name] = None
        else:
            if frac > 0:
                print(f"[{country}] note {name}: {frac:.1f}% NaN", flush=True)
            ranges[name] = tuple(np.nanpercentile(col[m], [2, 98]))
    N01 = {name: norm_benefit(crit_raw[name], ranges[name][0], ranges[name][1], DIRN[name][1])
           for name in DIRN if ranges[name] is not None}
    land01, _ = land_suit(wc)                                   # already 0..1 benefit
    excl = (wdpa >= 0.5) | np.isin(np.where(np.isfinite(wc), wc, -1).astype(int), list(EXCLUDE_WC))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rf_ok = np.isfinite(score)
    rf_roc = roc_auc_score(label[rf_ok], score[rf_ok])
    print(f"[{country}] RF n_pos={int(label[rf_ok].sum())} n_neg={int((label[rf_ok]==0).sum())} roc={rf_roc:.4f}", flush=True)
    rows = [dict(country=country, year=year, config="rf_R", aggregator="-", land="-",
                 n_pos=int(label[rf_ok].sum()), n_neg=int((label[rf_ok] == 0).sum()),
                 roc=round(rf_roc, 4), model=os.path.basename(model))]

    configs = {"richards": WEIGHTS["richards"], "chen": WEIGHTS["chen"],
               "equal_richards": {c: 1.0 / len(WEIGHTS["richards"]) for c in WEIGHTS["richards"]},
               "equal_chen": {c: 1.0 / len(WEIGHTS["chen"]) for c in WEIGHTS["chen"]}}
    for cfg, w in configs.items():
        crits = [c for c in w if c in N01]
        dropped = [c for c in w if c not in N01]
        if dropped:
            print(f"[{country}] {cfg}: dropping unavailable {dropped}, renormalising over {crits}", flush=True)
        if not crits:
            continue
        for land_mode in ("excl", "weighted"):
            wuse = {c: w[c] for c in crits}
            Nfull = {c: N01[c] for c in crits}
            if land_mode == "weighted":
                wuse["land"] = float(np.mean(list(w.values()))); Nfull["land"] = land01
            ssum = sum(wuse.values()); wuse = {k: v / ssum for k, v in wuse.items()}
            m = np.isfinite(score)
            for c in wuse:
                m &= np.isfinite(Nfull[c])
            if m.sum() < 50 or label[m].sum() < 5:
                print(f"[{country}] {cfg}/{land_mode}: only {int(m.sum())} rows/{int(label[m].sum())} pos -> skip", flush=True)
                continue
            Nsub = {c: Nfull[c][m] for c in wuse}
            for agg_name, agg in (("wlc", agg_wlc), ("topsis", agg_topsis)):
                s = agg(Nsub, wuse).copy()
                s[excl[m]] = 0.0
                rows.append(dict(country=country, year=year, config=cfg, aggregator=agg_name,
                                 land=land_mode, n_pos=int(label[m].sum()), n_neg=int((label[m] == 0).sum()),
                                 roc=round(float(roc_auc_score(label[m], s)), 4),
                                 model=os.path.basename(model)))

    # learned-importance vector: RF regress R-score on the MCDA criteria (negatives), perm importance
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    crit_names = list(DIRN)
    Xn = np.column_stack([crit_raw[c][neg_mask] for c in crit_names])
    yn = score[neg_mask]
    okrf = np.all(np.isfinite(Xn), axis=1) & np.isfinite(yn)
    imp = {}
    if okrf.sum() > 500:
        rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=20,
                                   n_jobs=-1, random_state=seed).fit(Xn[okrf], yn[okrf])
        r2 = rf.score(Xn[okrf], yn[okrf])
        pi = permutation_importance(rf, Xn[okrf], yn[okrf], n_repeats=10, random_state=seed, n_jobs=-1)
        imp = {c: round(float(v), 5) for c, v in zip(crit_names, pi.importances_mean)}
        os.makedirs(IMP_DIR, exist_ok=True)
        json.dump({"country": country, "r2_rf": round(float(r2), 4), "perm_importance": imp,
                   "note": "RF regress R-score on MCDA criteria; compare vs published weights"},
                  open(f"{IMP_DIR}/{slug}_importance.json", "w"), indent=2)
        print(f"[{country}] learned importance (r2={r2:.3f}): " +
              ", ".join(f"{c} {imp[c]:.3f}" for c in sorted(imp, key=imp.get, reverse=True)[:5]), flush=True)

    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[{country}] RF ROC {rf_roc:.3f}  |  MCDA band:", flush=True)
    for r in rows[1:]:
        print(f"    {r['config']:15s} {r['aggregator']:6s} land={r['land']:8s} ROC {r['roc']:.3f}", flush=True)
    print(f"[{country}] appended {len([r for r in rows])} rows -> {OUT_CSV}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--model", default=REFERENCE_MODEL)
    args = ap.parse_args()
    os.chdir(WS)
    # resumable: skip if this country already has rows
    if os.path.exists(OUT_CSV):
        done = {r["country"] for r in csv.DictReader(open(OUT_CSV))}
        if args.country in done:
            print(f"{args.country} already in {OUT_CSV}; skipping."); return
    run_country(args.country, args.year, args.model)


if __name__ == "__main__":
    main()

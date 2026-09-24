#!/usr/bin/env python
"""Combined-set RF-vs-MCDA benchmark with per-point scores saved for paired bootstrap.

Supersedes the year-2021 mcda_benchmark_v3 Fig A: scores RF and every published MCDA config on
the SAME enlarged positive set used everywhere else — PV_Facility forward-installs (>=2021) PLUS
independent TZ-SAM new sites (constructed_after>=2020) — vs random-land negatives, for all 13
validation countries, at a UNIFORM AEF year 2019.

Why 2019: matches the country validation (Fig B) and the TZ-SAM validation exactly, and guarantees
TZ sites (bare in 2019) are pre-solar so the embedding is uncontaminated. PV_Facility positives at
year 2019 => installs>=2021 == the same 'country2021' positive set. So the RF ROCs here should
reproduce the Fig-B / TZ-SAM numbers (built-in sanity check), now with MCDA scored on identical points.

Per country it SAVES per-point arrays (rf score, each MCDA config score, label, source) so the paired
bootstrap (bootstrap_figA.py) runs locally with no GEE / no re-sampling. Reports inventory-only /
TZ-SAM-only / combined ROC for RF and each config.

Negatives: reuse cached covsample criteria where present (time-invariant terrain/climate/road/grid),
re-scoring only RF (year 2019) + population; else generate random-land points and sample_full.

Run:  python Solar-Siting/mcda_bootstrap_v3.py "Greece"
"""
import argparse, csv, glob, os, time
import numpy as np

from mcda_benchmark_v3 import (sample_full, score_at, sample_population, norm_benefit,
                               agg_wlc, agg_topsis, land_suit, DIRN, WEIGHTS,
                               INV, INV_NAME, REFERENCE_MODEL, EXCLUDE_WC, WS)

TZ_NEW = "Data/external/tz_sam_q1_2026/overlap_analysis/tz_new_sites.csv"
OUT_DIR = "Solar-Siting/artifacts/paper_v3/figA_combined"
COV_DIR = "Solar-Siting/artifacts/covariate"
YEAR = 2019
MIN_CA = 2020          # TZ constructed_after >= 2020 -> bare in 2019
MIN_NN = 1000.0        # >= 1 km from any training positive
N_NEG_GEN = 8000       # random-land negatives for countries without a cached covsample
SEED = 42

# criteria columns needed from a cached covsample (all time-invariant except score/pop)
COV_KEYS = ["gsa_ghi", "gsa_pvout", "gsa_gti", "gsa_temp", "slope", "elevation",
            "equatorwardness", "road_dist_m", "grid_dist_m", "worldcover", "in_wdpa"]

TZ_NAME = {"United States": "United States"}   # tz_new_sites country field spelling


def slug(c):
    return c.lower().replace(" ", "_")


def load_tz(country):
    import csv as _csv
    from datetime import datetime
    name = TZ_NAME.get(country, country)
    xy = []
    with open(os.path.join(WS, TZ_NEW), newline="") as f:
        for row in _csv.DictReader(f):
            if row["country"] != name:
                continue
            ca = (row.get("constructed_after") or "").strip()
            if not ca:
                continue
            try:
                if datetime.fromisoformat(ca[:10]).year < MIN_CA:
                    continue
            except ValueError:
                continue
            try:
                if float(row.get("nn_m") or 0) < MIN_NN:
                    continue
                xy.append((float(row["longitude"]), float(row["latitude"])))
            except (ValueError, KeyError):
                continue
    return np.array(xy, float) if xy else np.empty((0, 2))


def _getinfo_retry(obj, tries=6, base=8):
    """getInfo() with exponential backoff — restricted-mode GEE returns 429 under concurrency."""
    import time as _t
    for k in range(tries):
        try:
            return obj.getInfo()
        except Exception as e:
            if k == tries - 1:
                raise
            wait = base * (2 ** k)
            print(f"  getInfo 429/err (retry {k+1}/{tries} in {wait}s): {str(e)[:70]}", flush=True)
            _t.sleep(wait)


def gen_negatives(country, model, n=N_NEG_GEN, seed=SEED):
    """Random land points inside the country (same approach as run_country_rf)."""
    import ee
    from covariate_sample import build_stack
    from shapely.geometry import shape, Point
    _, geom = build_stack(country, YEAR, model)
    shp = shape(_getinfo_retry(geom))
    minx, miny, maxx, maxy = shp.bounds
    rng = np.random.default_rng(seed)
    xy = []
    while len(xy) < n:
        cand = np.c_[rng.uniform(minx, maxx, 6000), rng.uniform(miny, maxy, 6000)]
        xy += [tuple(c) for c in cand if shp.contains(Point(c))]
    return np.array(xy[:n], float)


def negatives(country, model):
    """Return dict of criteria + score + worldcover + in_wdpa for negatives."""
    cov = glob.glob(os.path.join(WS, COV_DIR, slug(country), f"{slug(country)}_*_covsample_corrected.npz")) \
        or glob.glob(os.path.join(WS, COV_DIR, slug(country), f"{slug(country)}_*_covsample_full.npz"))
    if cov:
        d = {k: np.asarray(v, float) for k, v in np.load(cov[0]).items()}
        # subsample cached negatives to N_NEG_GEN so n_neg is UNIFORM across all 13 countries
        # (cached have ~20k, generated have 8k) and to avoid re-scoring 20k under restricted-mode GEE.
        n0 = len(d["lon"])
        if n0 > N_NEG_GEN:
            idx = np.random.default_rng(SEED).choice(n0, N_NEG_GEN, replace=False)
            d = {k: v[idx] for k, v in d.items()}
        print(f"  negatives: cached {os.path.basename(cov[0])} {n0}->{len(d['lon'])} "
              f"(re-score RF@{YEAR} + pop)", flush=True)
        neg_xy = np.c_[d["lon"], d["lat"]]
        out = {k: d[k] for k in COV_KEYS}
        out["score"] = score_at(country, YEAR, neg_xy, model)
        out["pop_density"] = sample_population(neg_xy)
        return out
    print(f"  negatives: none cached -> generating {N_NEG_GEN} random-land + sample_full", flush=True)
    neg_xy = gen_negatives(country, model)
    return sample_full(country, YEAR, neg_xy, model)


def col(tab, key):
    n = len(tab["score"])
    return np.asarray(tab[key], float) if key in tab else np.full(n, np.nan)


def run_country(country, model):
    import ee
    ee.Initialize(project="ee-abdullahr-solar")
    from sklearn.metrics import roc_auc_score
    import geopandas as gpd

    os.makedirs(os.path.join(WS, OUT_DIR), exist_ok=True)
    out_npz = os.path.join(WS, OUT_DIR, f"{slug(country)}_figA.npz")
    if os.path.exists(out_npz):
        print(f"{country}: {out_npz} exists -> skip"); return

    # ---- positives: PV_Facility forward-installs (>=2021) ----
    inv_name = INV_NAME.get(country, country)
    sites = gpd.read_file(os.path.join(WS, INV), where=f"country = '{inv_name}' AND year >= {YEAR+2}")
    if len(sites) == 0:
        raise SystemExit(f"{country}: no PV_Facility installs >= {YEAR+2}")
    pv_xy = np.c_[sites.geometry.representative_point().to_crs("EPSG:4326").x,
                  sites.geometry.representative_point().to_crs("EPSG:4326").y]
    tz_xy = load_tz(country)
    print(f"{country}: {len(pv_xy)} PV_Facility positives, {len(tz_xy)} TZ-SAM positives", flush=True)

    # ---- sample criteria at all three groups ----
    print("  sampling PV_Facility positives...", flush=True)
    PV = sample_full(country, YEAR, pv_xy, model)
    print("  sampling TZ-SAM positives...", flush=True)
    TZ = sample_full(country, YEAR, tz_xy, model) if len(tz_xy) else None
    print("  assembling negatives...", flush=True)
    NG = negatives(country, model)

    # ---- stack: order = [PV, TZ, NEG] ----
    groups = [("pvfac", PV), ("tz", TZ), ("neg", NG)]
    groups = [(nm, g) for nm, g in groups if g is not None]
    n_by = {nm: len(g["score"]) for nm, g in groups}
    source = np.concatenate([[nm] * n_by[nm] for nm, g in groups])
    label = np.concatenate([np.ones(n_by[nm]) if nm != "neg" else np.zeros(n_by[nm]) for nm, g in groups])

    crit_raw = {name: np.concatenate([col(g, DIRN[name][0]) for _, g in groups]) for name in DIRN}
    score = np.concatenate([col(g, "score") for _, g in groups])
    wc = np.concatenate([col(g, "worldcover") for _, g in groups])
    wdpa = np.concatenate([col(g, "in_wdpa") for _, g in groups])

    # ---- normalisation ranges from finite negatives (2-98 pct), per criterion ----
    neg_mask = label == 0
    ranges = {}
    for name, (k, direction) in DIRN.items():
        c = crit_raw[name]; m = neg_mask & np.isfinite(c)
        ranges[name] = None if m.sum() < 100 else tuple(np.nanpercentile(c[m], [2, 98]))
        if ranges[name] is None:
            print(f"  criterion {name}: <100 finite neg -> dropped", flush=True)
    N01 = {name: norm_benefit(crit_raw[name], ranges[name][0], ranges[name][1], DIRN[name][1])
           for name in DIRN if ranges[name] is not None}
    land01, _ = land_suit(wc)
    excl = (wdpa >= 0.5) | np.isin(np.where(np.isfinite(wc), wc, -1).astype(int), list(EXCLUDE_WC))

    # ---- per-point scores: RF + every config ----
    per_point = {"rf": score, "label": label, "source": source}
    configs = {"richards": WEIGHTS["richards"], "chen": WEIGHTS["chen"],
               "equal_richards": {c: 1.0 / len(WEIGHTS["richards"]) for c in WEIGHTS["richards"]},
               "equal_chen": {c: 1.0 / len(WEIGHTS["chen"]) for c in WEIGHTS["chen"]}}
    for cfg, w in configs.items():
        crits = [c for c in w if c in N01]
        if not crits:
            continue
        for land_mode in ("excl", "weighted"):
            wuse = {c: w[c] for c in crits}; Nfull = {c: N01[c] for c in crits}
            if land_mode == "weighted":
                wuse["land"] = float(np.mean(list(w.values()))); Nfull["land"] = land01
            ssum = sum(wuse.values()); wuse = {k: v / ssum for k, v in wuse.items()}
            m = np.isfinite(score).copy()
            for c in wuse:
                m &= np.isfinite(Nfull[c])
            for agg_name, agg in (("wlc", agg_wlc), ("topsis", agg_topsis)):
                s = np.full(len(score), np.nan)
                Nsub = {c: Nfull[c][m] for c in wuse}
                s[m] = agg(Nsub, wuse)
                s[m & excl] = 0.0
                per_point[f"{cfg}__{agg_name}__{land_mode}"] = s

    # also save RAW criteria (namespaced raw__*) so covariate-novelty + site-profiling run locally
    for name in DIRN:
        per_point[f"raw__{name}"] = crit_raw[name]
    per_point["raw__worldcover"] = wc
    per_point["raw__in_wdpa"] = wdpa

    np.savez(out_npz, **per_point)
    print(f"  saved {out_npz}", flush=True)

    # ---- report ROCs (inventory / tzsam / combined) ----
    subsets = {"inventory": source != "tz", "tzsam": source != "pvfac", "combined": np.ones(len(source), bool)}
    rows = []
    cfg_cols = [k for k in per_point if k not in ("rf", "label", "source")
                and not k.startswith("raw__")]
    for setname, smask in subsets.items():
        if setname == "tzsam" and (source == "tz").sum() == 0:
            continue
        def roc(vals):
            ok = smask & np.isfinite(vals)
            if label[ok].sum() < 5 or (label[ok] == 0).sum() < 5:
                return np.nan, 0, 0
            return (roc_auc_score(label[ok], vals[ok]),
                    int(label[ok].sum()), int((label[ok] == 0).sum()))
        r, npos, nneg = roc(score)
        rows.append(dict(country=country, set=setname, method="rf_R", agg="-", land="-",
                         roc=round(r, 4), n_pos=npos, n_neg=nneg))
        print(f"  [{setname}] RF roc={r:.4f} (n_pos={npos}, n_neg={nneg})", flush=True)
        for ck in cfg_cols:
            cfg, agg_name, land_mode = ck.split("__")
            r, npos, nneg = roc(per_point[ck])
            rows.append(dict(country=country, set=setname, method=cfg, agg=agg_name, land=land_mode,
                             roc=round(r, 4), n_pos=npos, n_neg=nneg))

    # per-country CSV (avoids a write race across parallel jobs; concat later if wanted)
    csv_path = os.path.join(WS, OUT_DIR, f"figA_results_{slug(country)}.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {csv_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--model", default=REFERENCE_MODEL)
    a = ap.parse_args()
    os.chdir(WS)
    t0 = time.time()
    run_country(a.country, a.model)
    print(f"{a.country} done in {time.time()-t0:.0f}s", flush=True)

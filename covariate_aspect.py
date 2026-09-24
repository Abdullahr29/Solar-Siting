"""Aspect (slope-face direction) probe for the covariate analysis.

Motivation: the covariate stack used ee.Terrain.SLOPE = steepness only (direction-
agnostic). "Slope is null" therefore says nothing about ASPECT — which compass
direction a slope faces. South-facing (N-hemisphere) / north-facing (S-hemisphere)
slopes receive more insolation, so aspect is the physically-motivated terrain variable
we never measured. This script adds it, as a CONTROLLED addition to the exact same
sampled points + train/test split used by covariate_analysis.py.

What it does, per country (reusing the existing *_covsample_full.npz points):
  1. Samples ee.Terrain.aspect (Copernicus GLO-30 DEM) at the identical (lon,lat) points.
  2. Folds the circular aspect into physical features (raw degrees are NEVER fed to a
     linear/rank model - 350 deg ~ 10 deg):
       equatorwardness   = cos(aspect - equator_azimuth)   in [-1,+1]
                           (N-hemi: +1 = due south / poleward -1 = due north;
                            S-hemi: +1 = due north)         -- the insolation-relevant fold
       eastness          = sin(aspect)                      (morning vs afternoon sun)
       slope_x_equatorward = sin(slope) * equatorwardness   (the equatorward TILT
                           component; = 0 on flat ground, so flat-cell aspect noise
                           cannot leak into the model)
  3. UNIVARIATE: Spearman + top-decile rank-AUC of each aspect feature vs our score, on
     (a) ALL points and (b) the slope>5 deg subset (where aspect is physically meaningful).
     Binned-median plot of equatorwardness.
  4. INCREMENTAL MULTIVARIATE (the money test): on ONE fixed 70/30 split, fit the RF
     residual model twice -- baseline GIS stack, then stack + aspect features -- and
     report d R2_rf and whether the embedding-novelty residual (1 - R2_rf) shrinks.
     Permutation importance of the aspect features inside the augmented model.

Usage (JASMIN sci node, env_solar):
    python Solar-Siting/covariate_aspect.py Greece --year 2021
"""
import argparse, json, os, sys, time
import numpy as np

# reuse the exact covariate bookkeeping from the main analysis so the baseline
# R2_rf reproduces the published residual before we add aspect.
sys.path.insert(0, os.path.join(os.path.expanduser("~/Solar_Workspace"), "Solar-Siting"))
from covariate_analysis import split_covariates, CATEGORICAL, NON_COVARIATE  # noqa: E402

# slope_correct = terrain slope recomputed at native 30 m (the published `slope` column is
# the ~1deg-grid artifact); added alongside the aspect features so the corrected residual
# reflects real terrain and we can finally test whether slope MAGNITUDE is null.
ADDED_FEATURES = ["slope_correct", "equatorwardness", "eastness", "slope_x_equatorward"]
SLOPE_MIN_DEG = 5.0   # aspect is meaningful only on real slopes


def load_full(country, year):
    slug = country.lower().replace(" ", "_")
    path = f"Solar-Siting/artifacts/covariate/{slug}/{slug}_{year}_covsample_full.npz"
    if not os.path.exists(path):
        sys.exit(f"no full table: {path}")
    d = np.load(path)
    return {k: d[k] for k in d.files}, slug, path


def sample_aspect(lon, lat, scale=10, chunk=400, max_retries=4):
    """Sample ee.Terrain.aspect at the given points, keyed on (lon,lat). Mirrors the
    resilient chunk-and-halve fetch in covariate_sample.py."""
    import ee
    coll = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")
    # CRITICAL: .mosaic() drops the projection, so ee.Terrain on it computes at EE's
    # default ~1deg grid -> slopes collapse to ~0 (this is the bug in covariate_sample.py).
    # Restore the tiles' native ~30 m projection before Terrain so slope/aspect are real.
    proj = coll.first().projection()
    dem = coll.mosaic().setDefaultProjection(proj)
    terr = ee.Terrain.products(dem)
    aspect = terr.select("aspect").rename("aspect")
    slope = terr.select("slope").rename("slope_chk")
    img = aspect.addBands(slope)

    def fetch(idx):
        for attempt in range(max_retries):
            try:
                feats = [ee.Feature(ee.Geometry.Point([float(lon[i]), float(lat[i])]),
                                    {"i": int(i)}) for i in idx]
                fc = ee.FeatureCollection(feats)
                return img.reduceRegions(fc, ee.Reducer.first(), scale=scale).getInfo()["features"]
            except ee.ee_exception.EEException as e:
                msg = str(e).lower()
                if "timed out" in msg or "too many" in msg or "computation" in msg:
                    if len(idx) > 50:
                        m = len(idx) // 2
                        return fetch(idx[:m]) + fetch(idx[m:])
                    time.sleep(5 * (attempt + 1)); continue
                raise
        print(f"  WARN dropped {len(idx)} pts", flush=True)
        return []

    n = len(lon)
    asp = np.full(n, np.nan); slp = np.full(n, np.nan)
    t0 = time.time()
    order = list(range(n))
    for k in range(0, n, chunk):
        for ft in fetch(order[k:k + chunk]):
            pr = ft["properties"]; i = pr["i"]
            asp[i] = np.nan if pr.get("aspect") is None else pr["aspect"]
            slp[i] = np.nan if pr.get("slope_chk") is None else pr["slope_chk"]
        print(f"  sampled {min(k+chunk, n)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return asp, slp


def derive(aspect_deg, slope_deg, lat):
    """Circular aspect -> physical features. equator azimuth is 180 (south) in the N
    hemisphere, 0 (north) in the S hemisphere."""
    a = np.deg2rad(aspect_deg)
    equ_az = np.where(lat >= 0, np.pi, 0.0)           # radians
    equatorwardness = np.cos(a - equ_az)              # +1 faces equator, -1 faces pole
    eastness = np.sin(a)                              # +1 east, -1 west
    slope_x = np.sin(np.deg2rad(slope_deg)) * equatorwardness
    return equatorwardness, eastness, slope_x


def uni_stat(x, score, mask):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    m = mask & np.isfinite(x) & np.isfinite(score)
    if m.sum() < 50:
        return None
    hi = (score[m] >= np.nanquantile(score[m], 0.90)).astype(int)
    rho = float(spearmanr(x[m], score[m]).statistic)
    auc = float(roc_auc_score(hi, x[m]))
    return dict(n=int(m.sum()), spearman=round(rho, 4), rank_auc=round(auc, 4),
                discrimination=round(abs(auc - 0.5) * 2, 4))


def build_X(tab, cont, cat):
    """Reproduce covariate_analysis.multivariate's design matrix (drop-NaN, one-hot),
    returning X, y and a keep-mask into the original rows so aspect cols align."""
    import pandas as pd
    raw = {c: tab[c] for c in cont}
    for c in cat:
        raw[c] = tab[c]
    df = pd.DataFrame(raw)
    df["__score"] = tab["score"]
    df = df.replace([np.inf, -np.inf], np.nan)
    keep = df.notna().all(axis=1).values
    df = df[keep]
    y = df.pop("__score").values
    oh = [c for c in cat if c != "in_wdpa"]
    for c in oh:
        df[c] = df[c].round().astype(int)
    df = pd.get_dummies(df, columns=oh, prefix=oh)
    return df, y, keep


def multivariate_ab(tab, cont, cat, aspect_cols, outdir, slug, year):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score

    df_base, y, keep = build_X(tab, cont, cat)
    # aspect features aligned to the SAME kept rows
    asp_df = pd.DataFrame({c: tab[c][keep] for c in aspect_cols})
    df_aug = pd.concat([df_base.reset_index(drop=True), asp_df.reset_index(drop=True)], axis=1)

    def fit(X, feat):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                   n_jobs=-1, random_state=0).fit(Xtr, ytr)
        r2 = float(r2_score(yte, rf.predict(Xte)))
        return rf, r2, Xte, yte

    rf_b, r2_b, _, _ = fit(df_base.values.astype(float), list(df_base.columns))
    rf_a, r2_a, Xte_a, yte_a = fit(df_aug.values.astype(float), list(df_aug.columns))

    imp = permutation_importance(rf_a, Xte_a, yte_a, n_repeats=10, random_state=0, n_jobs=-1)
    feat_a = list(df_aug.columns)
    asp_imp = {c: round(float(imp.importances_mean[feat_a.index(c)]), 5) for c in aspect_cols}
    order = np.argsort(imp.importances_mean)[::-1]
    top = [(feat_a[i], round(float(imp.importances_mean[i]), 4)) for i in order[:12]]

    fig, ax = plt.subplots(figsize=(7, 5))
    k = min(12, len(order)); sel = order[:k][::-1]
    colors = ["#e63946" if feat_a[i] in aspect_cols else "#457b9d" for i in sel]
    ax.barh([feat_a[i] for i in sel], [imp.importances_mean[i] for i in sel], color=colors)
    ax.set_xlabel("permutation importance (drop in R2)")
    ax.set_title(f"{slug} {year}: score explained, +aspect (red)\n"
                 f"R2_rf base={r2_b:.3f} -> +aspect={r2_a:.3f}  "
                 f"residual {1-r2_b:.3f} -> {1-r2_a:.3f}")
    fig.tight_layout(); fig.savefig(f"{outdir}/aspect_importance.png", dpi=120); plt.close(fig)

    return dict(n=int(len(y)),
                r2_rf_baseline=round(r2_b, 4), r2_rf_with_aspect=round(r2_a, 4),
                delta_r2=round(r2_a - r2_b, 4),
                residual_baseline=round(1 - r2_b, 4), residual_with_aspect=round(1 - r2_a, 4),
                aspect_perm_importance=asp_imp, top_importances=top)


def binned_plot(x, score, mask, outdir, slug, year, name):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = mask & np.isfinite(x) & np.isfinite(score)
    if m.sum() < 50:
        return
    xx, ss = x[m], score[m]
    qs = np.quantile(xx, np.linspace(0, 1, 11)); qs[-1] += 1e-9
    idx = np.clip(np.digitize(xx, qs) - 1, 0, 9)
    bins = [b for b in range(10) if (idx == b).any()]
    cent = [np.median(xx[idx == b]) for b in bins]
    med = [np.median(ss[idx == b]) for b in bins]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cent, med, "o-", color="#c1121f")
    ax.axhline(np.median(ss), ls="--", color="grey", lw=1,
               label=f"median score {np.median(ss):.3f}")
    ax.set_xlabel(name); ax.set_ylabel("median RF score")
    ax.set_title(f"{slug} {year}: score vs {name} (slope>{SLOPE_MIN_DEG:g} deg, n={m.sum()})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{outdir}/aspect_{name}.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()
    os.chdir(os.path.expanduser("~/Solar_Workspace"))

    import ee
    ee.Initialize(project="ee-abdullahr-solar")

    tab, slug, path = load_full(args.country, args.year)
    lon, lat, score, slope0 = tab["lon"], tab["lat"], tab["score"], tab["slope"]
    print(f"{args.country}: {len(lon)} points from {os.path.basename(path)}", flush=True)

    asp, slp_chk = sample_aspect(lon, lat, scale=30)   # slope/aspect are native 30 m
    # prefer the freshly-sampled slope for the interaction (guaranteed aligned); fall
    # back to the stored slope where the re-sample dropped.
    slope = np.where(np.isfinite(slp_chk), slp_chk, slope0)
    equ, eas, slx = derive(asp, slope, lat)
    tab["aspect"] = asp
    tab["slope_correct"] = slope
    tab["equatorwardness"] = equ
    tab["eastness"] = eas
    tab["slope_x_equatorward"] = slx

    outdir = f"Solar-Siting/artifacts/figures/covariate/{slug}"
    os.makedirs(outdir, exist_ok=True)

    allmask = np.ones(len(lon), bool)
    slopemask = np.isfinite(slope) & (slope > SLOPE_MIN_DEG)
    uni = {}
    for name, x in [("slope_correct", slope), ("equatorwardness", equ),
                    ("eastness", eas), ("slope_x_equatorward", slx)]:
        uni[name] = dict(all=uni_stat(x, score, allmask),
                         sloped=uni_stat(x, score, slopemask))
    binned_plot(equ, score, slopemask, outdir, slug, args.year, "equatorwardness")

    # baseline covariate set = exactly what covariate_analysis uses (added cols excluded;
    # NB the baseline still carries the ORIGINAL broken `slope` column, so dR2 = the effect
    # of correct terrain (slope_correct + aspect) over the as-published baseline).
    base_cont, base_cat = split_covariates(
        {k: v for k, v in tab.items() if k not in
         ({"aspect"} | set(ADDED_FEATURES))})
    multi = multivariate_ab(tab, base_cont, base_cat, ADDED_FEATURES,
                            outdir, slug, args.year)

    # persist the augmented table + result
    np.savez(path.replace("_covsample_full.npz", "_covsample_aspect.npz"), **tab)
    result = dict(country=args.country, year=args.year, n_points=int(len(lon)),
                  slope_min_deg=SLOPE_MIN_DEG, n_sloped=int(slopemask.sum()),
                  pct_sloped=round(100 * slopemask.mean(), 2),
                  univariate=uni, incremental_multivariate=multi)
    with open(f"{outdir}/{slug}_{args.year}_aspect.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {args.country} {args.year} ASPECT probe ===")
    print(f"points {len(lon)} | sloped(>{SLOPE_MIN_DEG:g}deg) {slopemask.sum()} "
          f"({100*slopemask.mean():.1f}%)")
    print("UNIVARIATE (equatorwardness = faces-equator; slope_correct = native 30 m):")
    for name in ["slope_correct", "equatorwardness", "eastness", "slope_x_equatorward"]:
        a, s = uni[name]["all"], uni[name]["sloped"]
        fa = f"rho {a['spearman']:+.3f} AUC {a['rank_auc']:.3f}" if a else "n/a"
        fs = f"rho {s['spearman']:+.3f} AUC {s['rank_auc']:.3f}" if s else "n/a"
        print(f"  {name:20s} ALL[{fa}]  SLOPED[{fs}]")
    print("INCREMENTAL MULTIVARIATE (does correct terrain shrink the embedding-novelty residual?):")
    print(f"  R2_rf  base {multi['r2_rf_baseline']:.3f} -> +terrain {multi['r2_rf_with_aspect']:.3f}"
          f"  (dR2 {multi['delta_r2']:+.4f})")
    print(f"  residual {multi['residual_baseline']:.3f} -> {multi['residual_with_aspect']:.3f}")
    print(f"  added-feature permutation importance: {multi['aspect_perm_importance']}")
    print(f"\nsaved {slug}_{args.year}_aspect.json + figures to {outdir}")


if __name__ == "__main__":
    main()

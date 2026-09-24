"""Stage 8: recompute the covariate analysis with the NEW RF (embedding-novelty residuals +
which GIS layers explain the score), for the 7 countries that already have covariate samples,
for BOTH schemes. Only the RF `score` column is refreshed (GIS layers are fixed) -> re-score
the stored lon/lat points via GEE, then rerun covariate_analysis.multivariate. Robust + late
(non-blocking): a country/scheme failure is logged, not fatal.
"""
import os, sys, traceback
import numpy as np, joblib

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
import covariate_analysis as CA

YEAR = 2021
COUNTRIES = [("greece", "Greece"), ("chile", "Chile"), ("germany", "Germany"),
             ("united_states", "United States"), ("australia", "Australia"),
             ("china", "China"), ("south_africa", "South Africa")]
BANDS = [f"A{i:02d}" for i in range(64)]
_clf_cache = {}


def score_points(model_path, lon, lat, year):
    import ee
    from geemap import ml
    ee.Initialize(project="ee-abdullahr-solar")
    if model_path not in _clf_cache:
        rf = joblib.load(os.path.join(WS, model_path))
        _clf_cache[model_path] = ml.strings_to_classifier(
            ml.rf_to_strings(rf, BANDS, processes=32, output_mode="PROBABILITY"))
    clf = _clf_cache[model_path]
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{year}-01-01", f"{year+1}-01-01").mosaic().select(BANDS))
    score = img.classify(clf).unmask(-999).rename("score")   # unmask -> length-aligned output
    out = np.full(len(lon), np.nan)
    for i in range(0, len(lon), 1000):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lo), float(la)]))
                                   for lo, la in zip(lon[i:i+1000], lat[i:i+1000])])
        vals = score.reduceRegions(fc, ee.Reducer.first(), scale=10).aggregate_array("first").getInfo()
        v = np.array(vals, dtype=float)
        v[v == -999] = np.nan
        out[i:i+len(v)] = v
        print(f"    scored {min(i+1000, len(lon))}/{len(lon)}", flush=True)
    return out


def main():
    for cdir, slug in COUNTRIES:
        cov = f"Solar-Siting/artifacts/covariate/{cdir}/{cdir}_{YEAR}_covsample_full.npz"
        if not os.path.exists(os.path.join(WS, cov)):
            print(f"skip {slug}: no covsample"); continue
        z = np.load(os.path.join(WS, cov), allow_pickle=True)
        base = {k: z[k] for k in z.files}
        for scheme in ("R", "D"):
            mp = f"Solar-Siting/artifacts/paper_v3/models/rf30_v3_{scheme}.joblib"
            if not os.path.exists(os.path.join(WS, mp)):
                print(f"skip {slug}/{scheme}: no model"); continue
            outdir = f"Solar-Siting/artifacts/paper_v3/figures/covariate/{cdir}_{scheme}"
            os.makedirs(os.path.join(WS, outdir), exist_ok=True)
            if os.path.exists(os.path.join(WS, outdir, "multivariate_importance.png")):
                print(f"skip covariate {slug}/{scheme} (done — resume)"); continue
            try:
                print(f"## covariate {slug} {scheme}: re-scoring {len(base['lon']):,} points", flush=True)
                new_score = score_points(mp, base["lon"], base["lat"], YEAR)
                tab = dict(base); tab["score"] = new_score
                cont, cat = CA.split_covariates(tab)
                res = CA.multivariate(tab, cont, cat, os.path.join(WS, outdir), slug, YEAR)
                R.append_result("covariate", "rf", scheme, cdir, "residual_fraction",
                                res["residual_fraction"], n_pos=res["n"], aef_year=YEAR,
                                figure_path=f"{outdir}/multivariate_importance.png",
                                notes=f"r2_rf={res['r2_rf']} top={res['top_importances'][:3]}")
                print(f"### covariate {slug} {scheme}: residual(novelty)={res['residual_fraction']} "
                      f"r2_rf={res['r2_rf']}", flush=True)
            except Exception:
                traceback.print_exc(); R.log_stage(f"covariate/{cdir}/{scheme}", outdir, "fail")


if __name__ == "__main__":
    main()

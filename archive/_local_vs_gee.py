"""Confirm local sklearn predict_proba == GEE server-side classify score.
Greece validation points already have GEE scores saved in the npz (from stage-2). Re-sample the
AEF(2019) 64-d embeddings at those exact points, predict locally, compare to the saved GEE scores.
Small (221+ pts) so it won't stress the GEE quota."""
import os, numpy as np, joblib
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import ee
ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
D = "Solar-Siting/artifacts/paper_v3/budget_sweep/scores/country"
M = "Solar-Siting/artifacts/paper_v3/budget_sweep/models"

tag = "px_v3_BEST_ne30_mf32_ms0.34_gini"
z = np.load(f"{D}/greece__{tag}_2019_scores.npz")
site_xy, site_gee = z["site_xy"], z["site_s"]          # coords + GEE scores at sites
print(f"greece sites: {len(site_xy)}  (GEE scores loaded)")

img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2019-01-01", "2020-01-01").mosaic().select(BANDS))

# sample the 64-d embedding at each site point (chunked)
def sample_embeddings(xy):
    out = []
    for i in range(0, len(xy), 500):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(p[0]), float(p[1])]))
                                   for p in xy[i:i+500]])
        res = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()
        for ft in res["features"]:
            pr = ft["properties"]
            out.append([pr.get(b, np.nan) for b in BANDS])
        print(f"  sampled {min(i+500,len(xy))}/{len(xy)}", flush=True)
    return np.array(out, float)

X = sample_embeddings(site_xy)
ok = np.isfinite(X).all(1)
print(f"finite embeddings: {ok.sum()}/{len(X)}")

rf = joblib.load(f"{M}/{tag.replace('BEST_', '')}.joblib")  # model file has no 'BEST_' prefix
local = rf.predict_proba(X[ok])[:, 1]
gee = site_gee[ok] if len(site_gee) == len(X) else site_gee[:len(X)][ok]

# NB: GEE scores may be a filtered subset (drops nulls); align by finite mask length if equal
n = min(len(local), len(gee))
local, gee = local[:n], gee[:n]
d = np.abs(local - gee)
print(f"\nn compared: {n}")
print(f"max|local-GEE| = {d.max():.5f}   mean|Δ| = {d.mean():.5f}")
print(f"corr = {np.corrcoef(local, gee)[0,1]:.6f}")
print(f"sample local: {np.round(local[:5],4)}")
print(f"sample GEE  : {np.round(gee[:5],4)}")

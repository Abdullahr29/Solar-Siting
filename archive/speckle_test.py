"""Speckle proof-or-disproof: does ne15 (fewer trees) produce noisier heatmaps than ne30?
Pull a real AEF-2019 raster patch, score every pixel with ne15/20/30, measure spatial roughness
(mean |Δ| between adjacent pixels). If ne15 roughness >> ne30, speckle is real; else it's not."""
import os, numpy as np, joblib
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
import ee; ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
TC = "artifacts/paper_v3/pools_t3/treecount"
# mixed-terrain patch centres; build a G x G point grid at ~10m spacing (0.00009 deg)
CENTRES = {"spain": (-3.89, 38.91), "india": (77.31, 17.21), "germany": (10.91, 52.11)}
G = 60; STEP = 0.00009
img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2019-01-01","2020-01-01").mosaic().select(BANDS))
rfs = {f"ne{ne}": joblib.load(f"{TC}/px_v3_ne{ne}_mf32.joblib") for ne in [15,20,30]}

def rough(a):
    return (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())/2

def sample_grid(clon, clat):
    lons = clon + (np.arange(G)-G/2)*STEP; lats = clat + (np.arange(G)-G/2)*STEP
    pts = [(lo, la) for la in lats for lo in lons]     # row-major (lat outer)
    out = np.full((len(pts),64), np.nan, np.float32)
    for i in range(0,len(pts),1000):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a),float(b)])) for a,b in pts[i:i+1000]])
        res = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()["features"]
        out[i:i+1000] = [[f["properties"].get(b,np.nan) for b in BANDS] for f in res]
    return out

print(f"{'patch':>8} " + " ".join(f"{k+'_rough':>11}" for k in rfs) + "   ne15/ne30  score_std")
for name,(clon,clat) in CENTRES.items():
    A = sample_grid(clon, clat); fin = np.isfinite(A).all(1)
    out = {}; std = {}
    for k, rf in rfs.items():
        s = np.full(len(A), np.nan); s[fin] = rf.predict_proba(A[fin])[:,1]
        grid = np.nan_to_num(s.reshape(G, G)); out[k] = rough(grid); std[k] = np.nanstd(s)
    ratio = out["ne15"]/out["ne30"] if out["ne30"] else float("nan")
    print(f"{name:>8} " + " ".join(f"{out[k]:>11.5f}" for k in rfs) + f"   {ratio:.3f}x   n15={std['ne15']:.3f} n30={std['ne30']:.3f}")
print("\nratio ~1.0 => no meaningful speckle from fewer trees; >>1 => ne15 noticeably noisier")

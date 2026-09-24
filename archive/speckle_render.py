"""Render ne15/ne20/ne30 suitability heatmaps for a few AEF-2019 patches, side by side,
so the speckle difference is visible. Saves PNGs under artifacts/paper_v3/pools_t3/speckle/."""
import os, numpy as np, joblib
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
import ee; ee.Initialize(project="ee-abdullahr-solar")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BANDS = [f"A{i:02d}" for i in range(64)]
TC = "artifacts/paper_v3/pools_t3/treecount"
OUT = "artifacts/paper_v3/pools_t3/speckle"; os.makedirs(OUT, exist_ok=True)
CENTRES = {"spain": (-3.89, 38.91), "india": (77.31, 17.21), "germany": (10.91, 52.11)}
G = 90; STEP = 0.00009                                   # 90x90 px @ ~10m = ~0.8km, enough to see texture
img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2019-01-01","2020-01-01").mosaic().select(BANDS))
rfs = {f"ne{ne}": joblib.load(f"{TC}/px_v3_ne{ne}_mf32.joblib") for ne in [15,20,30]}

def sample_grid(clon, clat):
    lons = clon + (np.arange(G)-G/2)*STEP; lats = clat + (np.arange(G)-G/2)*STEP
    pts = [(lo, la) for la in lats for lo in lons]
    out = np.full((len(pts),64), np.nan, np.float32)
    for i in range(0,len(pts),1000):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a),float(b)])) for a,b in pts[i:i+1000]])
        res = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()["features"]
        out[i:i+1000] = [[f["properties"].get(b,np.nan) for b in BANDS] for f in res]
        print(f"  {min(i+1000,len(pts))}/{len(pts)}", flush=True)
    return out

def rough(a):
    return (np.abs(np.diff(a,axis=1)).mean()+np.abs(np.diff(a,axis=0)).mean())/2

for name,(clon,clat) in CENTRES.items():
    print(f"patch {name}", flush=True)
    A = sample_grid(clon, clat); fin = np.isfinite(A).all(1)
    grids = {}
    for k, rf in rfs.items():
        s = np.full(len(A), np.nan); s[fin] = rf.predict_proba(A[fin])[:,1]
        grids[k] = s.reshape(G, G)
    vmin = np.nanpercentile(np.concatenate([g.ravel() for g in grids.values()]), 2)
    vmax = np.nanpercentile(np.concatenate([g.ravel() for g in grids.values()]), 98)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for j, k in enumerate(["ne15","ne20","ne30"]):
        im = ax[j].imshow(grids[k], cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax[j].set_title(f"{k}  (roughness {rough(np.nan_to_num(grids[k])):.4f})", fontsize=12)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    fig.suptitle(f"{name.title()} — solar-suitability heatmap, {G}x{G}px @10m (same colour scale)", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(f"{OUT}/speckle_{name}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {OUT}/speckle_{name}.png", flush=True)
print("DONE render", flush=True)

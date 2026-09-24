"""Does the ne15-vs-ne30 speckle 'cease to be an issue at scale'? Sample a fine grid, score
ne15/ne30, then block-average to coarser resolutions and measure agreement (correlation + mean
|diff|) + roughness ratio at each. If agreement -> ~1 and diff -> 0 as scale coarsens, the speckle
is a 10m-native artifact that vanishes at country/parcel scale."""
import os, numpy as np, joblib
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
import ee; ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
TC = "artifacts/paper_v3/pools_t3/treecount"
CENTRES = {"spain": (-3.89, 38.91), "india": (77.31, 17.21)}
G = 100; STEP = 0.00009
img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2019-01-01","2020-01-01").mosaic().select(BANDS))
rf15 = joblib.load(f"{TC}/px_v3_ne15_mf32.joblib"); rf30 = joblib.load(f"{TC}/px_v3_ne30_mf32.joblib")

def sample_grid(clon, clat):
    lons = clon + (np.arange(G)-G/2)*STEP; lats = clat + (np.arange(G)-G/2)*STEP
    pts = [(lo, la) for la in lats for lo in lons]; out = np.full((len(pts),64), np.nan, np.float32)
    for i in range(0,len(pts),1000):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a),float(b)])) for a,b in pts[i:i+1000]])
        res = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()["features"]
        out[i:i+1000] = [[f["properties"].get(b,np.nan) for b in BANDS] for f in res]
    return out
def blockmean(a, b):
    H = (a.shape[0]//b)*b; a = a[:H,:H]
    return a.reshape(a.shape[0]//b, b, a.shape[1]//b, b).mean((1,3))
def rough(a): return (np.abs(np.diff(a,axis=1)).mean()+np.abs(np.diff(a,axis=0)).mean())/2

for name,(clon,clat) in CENTRES.items():
    A = sample_grid(clon, clat); fin = np.isfinite(A).all(1)
    s15 = np.full(len(A), np.nan); s30 = np.full(len(A), np.nan)
    s15[fin] = rf15.predict_proba(A[fin])[:,1]; s30[fin] = rf30.predict_proba(A[fin])[:,1]
    g15 = np.nan_to_num(s15.reshape(G,G)); g30 = np.nan_to_num(s30.reshape(G,G))
    print(f"\n=== {name}: ne15 vs ne30 agreement by aggregation scale ===")
    print(f"{'scale':>8} {'corr':>7} {'mean|Δ|':>9} {'ne15_rough':>11} {'ne30_rough':>11} {'ratio':>6}")
    for b, m in [(1,"10m"),(2,"20m"),(3,"30m"),(5,"50m"),(10,"100m")]:
        a15, a30 = blockmean(g15,b), blockmean(g30,b)
        corr = np.corrcoef(a15.ravel(), a30.ravel())[0,1]
        md = np.abs(a15-a30).mean(); r15, r30 = rough(a15), rough(a30)
        print(f"{m:>8} {corr:>7.4f} {md:>9.4f} {r15:>11.4f} {r30:>11.4f} {r15/r30 if r30 else 0:>6.2f}")
print("\ncorr->1 & mean|Δ|->0 & ratio->1 as scale coarsens => speckle is native-10m only, gone at scale")

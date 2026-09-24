"""Export a GLOBAL equirectangular texture of the 100m suitability asset (magma,
ocean/no-data transparent), for warping into an orthographic 'suitability globe'.
Runs on sci-vm-02 (GEE). -> artifacts/paper_v3/figures/fig1/suitability_equirect.png
"""
import os, io, numpy as np, requests
from PIL import Image
import ee

ASSET = "projects/ee-abdullahr-solar/assets/solar_suitability_2025_R_100m"
OUT   = "/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting/artifacts/paper_v3/figures/fig1"
os.makedirs(OUT, exist_ok=True)
MAGMA = ["000004","2c115f","721f81","b73779","f1605d","feb078","fcfdbf"]

ee.Initialize(project="ee-abdullahr-solar")
raw = ee.Image(ASSET).select("suitability")     # int16 x10000, nodata -1
img = raw.updateMask(raw.gte(0))                 # mask ocean / no-AEF -> transparent in PNG
world = ee.Geometry.Rectangle([-180, -85, 180, 85], proj="EPSG:4326", geodesic=False)

# quick coarse global 2/98 stretch (in raw x10000 units)
try:
    pct = img.reduceRegion(ee.Reducer.percentile([2, 98]), world, scale=25000,
                           maxPixels=1e10, bestEffort=True).getInfo()
    lo, hi = pct.get("suitability_p2"), pct.get("suitability_p98")
    lo = 1000 if lo is None else lo; hi = 7000 if hi is None else hi
except Exception as e:
    print("pct failed:", str(e)[:80]); lo, hi = 1000, 7000
print("stretch (x1e4 units):", lo, hi, flush=True)

def fetch(dim):
    url = img.getThumbURL({"region": world, "dimensions": f"{dim}x{dim//2}",
                           "crs": "EPSG:4326", "min": float(lo), "max": float(hi),
                           "palette": MAGMA, "format": "png"})
    return np.array(Image.open(io.BytesIO(requests.get(url, timeout=900).content)))

arr = None
for dim in (4096, 3072, 2048):
    try:
        print("fetching global texture", dim, flush=True); arr = fetch(dim); break
    except Exception as e:
        print(f"  {dim} failed: {str(e)[:90]}", flush=True)
if arr is None:
    raise SystemExit("all texture fetches failed")
print("texture", arr.shape, "has_alpha", arr.shape[-1] == 4, flush=True)
Image.fromarray(arr).save(f"{OUT}/suitability_equirect.png")
print("wrote suitability_equirect.png", flush=True)

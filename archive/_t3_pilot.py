"""PILOT the t-3 chain on a few chips (minimal GEE), no 40GB master load:
 1. LOCAL: replicate extract_pixels_master._sol selection for a chip -> kept pixels' (row,col)
    -> lon/lat via geotransform, + t-2 embedding from the chip.
 2. VERIFY: finite-pixel count vs pixel_census_v3.csv (n_masked).
 3. GEE (small): sample AEF(year-1)=t-3 at those coords; confirm finite + differs from t-2.
"""
import os, numpy as np, pandas as pd, rasterio
from rasterio.transform import xy as rc_to_xy
from rasterio.warp import transform as warp_transform
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import ee
ee.Initialize(project="ee-abdullahr-solar")
PCH = "Data/aef_solar_chips/positives/chips"
MASKS = "Data/aef_solar_chips/positives/masks_v2"
BANDS = [f"A{i:02d}" for i in range(64)]
census = pd.read_csv("Solar-Siting/artifacts/stats/pixel_census_v3.csv").set_index("chip")["n_masked"]

def local_extract(chip):
    with rasterio.open(os.path.join(MASKS, chip)) as s:
        m = s.read(1).astype(bool)
    with rasterio.open(os.path.join(PCH, chip)) as s:
        a = s.read(); T = s.transform; crs = s.crs
    if not m.any(): return None
    rows, cols = np.where(m)
    x = a[:, m].T.astype(np.float32)
    fin = np.isfinite(x).all(axis=1)
    rows, cols, x = rows[fin], cols[fin], x[fin]
    xs, ys = rc_to_xy(T, rows, cols, offset="center")          # chip CRS coords
    lon, lat = warp_transform(crs, "EPSG:4326", list(xs), list(ys))  # -> lon/lat
    return np.array(lon), np.array(lat), x

# pick 2 canonical chips (from census index) with AEF year >= 2018 that exist on disk
def yr_of(c):
    try: return int(str(c).split("_")[3])
    except Exception: return -1
cand = []
for c in census.index:
    if yr_of(c) >= 2018 and os.path.exists(os.path.join(MASKS, c)) and os.path.exists(os.path.join(PCH, c)):
        cand.append(c)
    if len(cand) == 2: break
print("pilot chips:", cand, flush=True)

for chip in cand:
    r = local_extract(chip)
    if r is None: print(f"{chip}: empty"); continue
    lon, lat, x_t2 = r
    yr = int(chip.split("_")[3])
    ncen = int(census.get(chip, -1))
    print(f"\n{chip} AEFyr={yr} install={yr+2}: finite px={len(x_t2)} census_n_masked={ncen}", flush=True)
    # subsample to <=40 points for the pilot GEE call
    k = min(40, len(lon)); sel = np.linspace(0, len(lon)-1, k).astype(int)
    lon_s, lat_s, x2_s = lon[sel], lat[sel], x_t2[sel]
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{yr-1}-01-01", f"{yr}-01-01").mosaic().select(BANDS))   # t-3 = AEF year-1
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a), float(b)]))
                               for a, b in zip(lon_s, lat_s)])
    res = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()
    x3 = np.array([[f["properties"].get(b, np.nan) for b in BANDS] for f in res["features"]], float)
    fin = np.isfinite(x3).all(1)
    print(f"  t-3 sampled {len(x3)} pts, finite {fin.sum()}/{len(x3)}", flush=True)
    if fin.sum():
        d = np.abs(x3[fin] - x2_s[fin]).mean()
        cos = (x3[fin]*x2_s[fin]).sum(1) / (np.linalg.norm(x3[fin],axis=1)*np.linalg.norm(x2_s[fin],axis=1)+1e-9)
        print(f"  mean|t3 - t2| per dim = {d:.4f}   mean cos(t3,t2) = {cos.mean():.4f}  (differs but similar => plausible)", flush=True)
print("\nPILOT DONE", flush=True)

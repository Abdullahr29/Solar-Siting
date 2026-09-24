"""Generate the Fig-1 tile assets for one site (China POS_CHN_12282_2018_1).
Runs on sci-vm-02 (env_solar, GEE works). Everything is rendered on the chip's
NATIVE UTM 256x256 grid so mask / AEF / heatmap / S2 align pixel-for-pixel.
Into artifacts/paper_v3/figures/fig1/:
  s2_pre.png            natural-colour S2 BEFORE install (bare land), no overlay
  aef_falsecolor.png    AlphaEarth false colour R=A12 G=A23 B=A52 (single image)
  rf_heatmap.png        RF P(solar) magma heatmap, no overlay
  rf_heatmap_outlined.png   heatmap + fine footprint outline (no fill)
  s2_post_outlined.png  natural-colour S2 AFTER install + fine footprint outline (no fill)
  fig1_meta.json
"""
import os, io, json, time, numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy.ndimage import binary_erosion, binary_dilation
import joblib, requests
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS   = "/gws/ssde/j25b/gbov/abdullah_solar"
SITE = "POS_CHN_12282_2018_1"
CHIP = f"{WS}/Data/aef_solar_chips/positives/chips/{SITE}.tif"
MASK = f"{WS}/Data/aef_solar_chips/positives/masks_v2/{SITE}.tif"   # v2 = rebuilt (real footprint)
MODEL= f"{WS}/Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
OUT  = f"{WS}/Solar-Siting/artifacts/paper_v3/figures/fig1"
os.makedirs(OUT, exist_ok=True)
DIM = 256
OUTLINE = (34, 211, 238)          # cyan on BOTH output tiles so the reader links them
EMB_YEAR, POST_YEAR = 2018, 2025  # AEF/before = install-2; after = recent (clear panels)

def pct(a, lo=2, hi=98):
    a = a.astype(np.float32)
    v0, v1 = np.nanpercentile(a, lo), np.nanpercentile(a, hi)
    return np.clip((a - v0) / max(v1 - v0, 1e-6), 0, 1)

def save_rgb(a01, path):
    Image.fromarray((a01 * 255).astype(np.uint8), "RGB").save(path)
    print("wrote", os.path.basename(path), a01.shape[:2], flush=True)

def outline_of(mask_bool):
    edge = mask_bool ^ binary_erosion(mask_bool, iterations=1)
    return binary_dilation(edge, iterations=1)          # ~1.5-2 px, fine

def overlay(a01, edge, path):
    out = (a01 * 255).astype(np.uint8).copy(); out[edge] = OUTLINE
    Image.fromarray(out, "RGB").save(path)
    print("wrote", os.path.basename(path), "outline px", int(edge.sum()), flush=True)

# ---- 64-band AEF chip + native UTM georef ----
with rasterio.open(CHIP) as ds:
    emb = ds.read().astype(np.float32)
    crs = ds.crs; b = ds.bounds
    ll = transform_bounds(crs, "EPSG:4326", *b)
lon, lat = (ll[0] + ll[2]) / 2, (ll[1] + ll[3]) / 2
C, H, W = emb.shape
crs_str = crs.to_string()
print(f"chip {C}x{H}x{W} crs {crs_str} centre {lon:.4f},{lat:.4f}", flush=True)

# AEF false colour R=A12 G=A23 B=A52
save_rgb(np.stack([pct(emb[12]), pct(emb[23]), pct(emb[52])], -1), f"{OUT}/aef_falsecolor.png")

# RF P(solar) heatmap (same UTM grid)
rf = joblib.load(MODEL)
prob = rf.predict_proba(emb.reshape(C, H * W).T)[:, 1].reshape(H, W)
magma = plt.get_cmap("magma")(prob)[..., :3]
save_rgb(magma, f"{OUT}/rf_heatmap.png")
print(f"prob {prob.min():.3f}-{prob.max():.3f} mean {prob.mean():.3f}", flush=True)

# footprint (masks_v2)
with rasterio.open(MASK) as ds:
    mask = ds.read(1) > 0
print("footprint frac", round(float(mask.mean()), 4), flush=True)
edge = outline_of(mask)

# ---- S2 pre/post rendered in the SAME UTM grid via GEE ----
import ee
ee.Initialize(project="ee-abdullahr-solar")
region = ee.Geometry.Rectangle([float(b.left), float(b.bottom), float(b.right), float(b.top)],
                               proj=crs_str, geodesic=False)

def scl_mask(img):
    s = img.select("SCL")
    keep = s.neq(3).And(s.neq(8)).And(s.neq(9)).And(s.neq(10)).And(s.neq(11))
    return img.updateMask(keep)

def fetch(url):
    return np.array(Image.open(io.BytesIO(requests.get(url, timeout=600).content)))[..., :3]

def s2_utm(year):
    thumb = {"region": region, "dimensions": DIM, "crs": crs_str,
             "min": 0, "max": 3000, "format": "png"}
    # (1) SR seasonal SCL-masked median
    sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(f"{year}-01-01", f"{year}-12-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)).map(scl_mask))
    for col in (sr,):
        try:
            img = col.median().select(["B4", "B3", "B2"])
            return fetch(img.getThumbURL(thumb))
        except Exception as e:
            print("  SR fail", str(e)[:80], flush=True)
    # (2) TOA fallback
    toa = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(region)
           .filterDate(f"{year}-01-01", f"{year}-12-31")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    img = toa.median().select(["B4", "B3", "B2"])
    return fetch(img.getThumbURL(thumb))

print("fetching S2 pre (UTM)...", flush=True); pre = pct(s2_utm(EMB_YEAR))
print("fetching S2 post (UTM)...", flush=True); post = pct(s2_utm(POST_YEAR))
save_rgb(pre, f"{OUT}/s2_pre.png")

# outlines (all on identical 256 UTM grid — pixel-perfect)
overlay(magma, edge, f"{OUT}/rf_heatmap_outlined.png")
overlay(post,  edge, f"{OUT}/s2_post_outlined.png")

json.dump({"site": SITE, "lon": round(float(lon), 4), "lat": round(float(lat), 4),
           "crs": crs_str, "before_year": EMB_YEAR, "post_year": POST_YEAR,
           "install_year": 2020, "footprint_frac": round(float(mask.mean()), 4)},
          open(f"{OUT}/fig1_meta.json", "w"), indent=2)
print("DONE fig1 tiles ->", OUT, flush=True)

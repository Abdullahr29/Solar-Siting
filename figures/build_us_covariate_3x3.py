"""US 3x3 covariate figure (appendix, covariate-novelty section).
Row 1 (resource):        our RF suitability (deployed rf30_v3_R, AEF 2021) | GHI | PVOUT
Row 2 (terrain/land):    elevation | slope | ESA WorldCover land cover
Row 3 (infra/climate):   grid distance (Gridfinder) | road distance (OSM major hw) | air temperature
All nine panels are covariates actually used in our analysis (App C stack). CONUS-clipped, 300 dpi.
-> artifacts/paper_v3/figures/fig_us_covariate_3x3.{png,pdf}
"""
import os, io, time, warnings; warnings.filterwarnings("ignore")
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm, Normalize
import rasterio
from rasterio.features import rasterize, geometry_mask
from scipy.ndimage import distance_transform_edt
import pyogrio, requests
from PIL import Image
import ee

ASSET = "projects/ee-abdullahr-solar/assets/solar_suitability_2025_R_100m"  # deployed R model, 100m
CD = "Solar-Siting/covariate_data"
FIG = "Solar-Siting/artifacts/paper_v3/figures"
YEAR = 2021
EXT = [-125.0, -66.5, 24.5, 49.5]                 # CONUS bbox: minx,maxx,miny,maxy
DIM = 900
minx, maxx, miny, maxy = EXT
midlat = (miny + maxy) / 2
ASPECT = 1 / np.cos(np.deg2rad(midlat))

WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
WC_COLORS = ["006400","ffbb22","ffff4c","f096ff","fa0000","b4b4b4","f0f0f0","0064c8","0096a0","00cf75","fae6a0"]
WC_NAMES = ["tree","shrub","grass","crop","built","bare","snow","water","wetland","mangrove","moss"]

ee.Initialize(project="ee-abdullahr-solar")
print("EE init; loading 100m suitability asset...", flush=True)
lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
us_geom = lsib.filter(ee.Filter.eq("country_na", "United States")).geometry()
_raw = ee.Image(ASSET).select("suitability")          # int16 x10000, nodata -1
score_img = _raw.updateMask(_raw.gte(0))              # mask ocean/no-AEF; raw scale 0..10000
wc_img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
conus = ee.Geometry.Rectangle([minx, miny, maxx, maxy])   # NB: xmin,ymin,xmax,ymax (not EXT order)
geom = us_geom.intersection(conus, 1000)          # drop Alaska/Hawaii
from shapely.geometry import shape
shp = shape(geom.getInfo())

def ee_thumb(img, vis, dim=DIM):
    for att in range(5):
        try:
            url = img.clip(geom).getThumbURL({"region": conus, "dimensions": dim, **vis})
            return np.array(Image.open(io.BytesIO(requests.get(url, timeout=600).content)))
        except Exception as e:
            if att == 4: raise
            print(f"    thumb retry {att}: {str(e)[:70]}", flush=True); time.sleep(15 * (att + 1))

def conus_mask(shape_hw):
    h, w = shape_hw
    tr = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
    return geometry_mask([shp.__geo_interface__], shape_hw, tr, invert=True)  # True=inside

def gsa_window(tif, out=DIM, derive_slope=False):
    from rasterio.windows import from_bounds
    with rasterio.open(tif) as ds:
        win = from_bounds(minx, miny, maxx, maxy, ds.transform)
        h = out; w = int(out * (maxx - minx) / (maxy - miny))
        arr = ds.read(1, window=win, out_shape=(h, w), boundless=True).astype(float)
        nod = ds.nodata
    if nod is not None: arr[arr == nod] = np.nan
    arr[arr < -1e30] = np.nan
    if derive_slope:
        land = np.where(arr > 0, arr, 0.0)
        dy = (maxy - miny) / h * 111320.0
        dx = (maxx - minx) / w * 111320.0 * np.cos(np.deg2rad(midlat))
        gy, gx = np.gradient(land, dy, dx)
        arr = np.degrees(np.arctan(np.hypot(gx, gy)))
    inside = conus_mask(arr.shape)
    arr[~inside] = np.nan
    if not derive_slope:
        arr[np.where(np.isfinite(arr), arr, 1) <= 0] = np.nan
    return arr

def distance_km(gpkg, H=900, ntile=6):
    """Tiled memory-safe rasterize of line features to the CONUS grid, then Euclidean distance (km).
    Reads geometry-only in lon strips (pyogrio bbox) and burns each strip into the shared mask."""
    W = int(H * (maxx - minx) / (maxy - miny))
    tr = rasterio.transform.from_bounds(minx, miny, maxx, maxy, W, H)
    mask = np.zeros((H, W), "uint8")
    edges = np.linspace(minx, maxx, ntile + 1)
    for k in range(ntile):
        gdf = pyogrio.read_dataframe(gpkg, bbox=(edges[k], miny, edges[k + 1], maxy),
                                     columns=[], read_geometry=True)
        geoms = [(g, 1) for g in gdf.geometry.values if g is not None and not g.is_empty]
        if geoms:
            rasterize(geoms, out=mask, transform=tr, default_value=1)
        print(f"    tile {k+1}/{ntile}: {len(geoms)} lines", flush=True)
    dy = (maxy - miny) / H * 111.32
    dx = (maxx - minx) / W * 111.32 * np.cos(np.deg2rad(midlat))
    d = distance_transform_edt(mask == 0, sampling=[dy, dx])
    d[~conus_mask((H, W))] = np.nan
    return d

# ---------- gather panels (cached so later cosmetic tweaks are instant) ----------
CACHE = f"{FIG}/_us_cov_panels.npz"
if os.path.exists(CACHE) and os.environ.get("FORCE", "0") != "1":
    z = np.load(CACHE, allow_pickle=True)
    score_im, wc_im = z["score_im"], z["wc_im"]
    ghi, pvout, ele, slope, temp = z["ghi"], z["pvout"], z["ele"], z["slope"], z["temp"]
    grid_d, road_d = z["grid_d"], z["road_d"]; slo, shi = float(z["slo"]), float(z["shi"])
    print("loaded panels from cache", flush=True)
else:
    print("EE: score percentile stretch + thumbnails...", flush=True)
    try:
        pct = score_img.reduceRegion(ee.Reducer.percentile([2, 98]), conus, scale=5000,
                                     maxPixels=1e10, bestEffort=True).getInfo()
        slo, shi = pct.get("suitability_p2") or 0.1, pct.get("suitability_p98") or 0.9
    except Exception as e:
        print("  percentile failed, using 0.1/0.9:", str(e)[:70], flush=True); slo, shi = 0.1, 0.9
    magma = ["000004","2c115f","721f81","b73779","f1605d","feb078","fcfdbf"]
    score_im = ee_thumb(score_img, {"min": float(slo), "max": float(shi), "palette": magma})
    wc = wc_img.remap(WC_CLASSES, list(range(len(WC_CLASSES))))
    wc_im = ee_thumb(wc, {"min": 0, "max": len(WC_CLASSES) - 1, "palette": WC_COLORS})
    print("  score stretch", round(float(slo), 3), round(float(shi), 3), flush=True)

    print("GSA windows (ghi/pvout/ele/slope/temp)...", flush=True)
    ghi = gsa_window(f"{CD}/gsa/tif/GHI.tif")
    pvout = gsa_window(f"{CD}/gsa/tif/PVOUT.tif")
    ele = gsa_window(f"{CD}/gsa/tif/ELE.tif")
    slope = gsa_window(f"{CD}/gsa/tif/ELE.tif", derive_slope=True)
    temp = gsa_window(f"{CD}/gsa/tif/TEMP.tif")

    print("grid distance (gridfinder, bbox)...", flush=True)
    grid_d = distance_km(f"{CD}/gridfinder/grid.gpkg")
    print(f"  grid_dist km med={np.nanmedian(grid_d):.1f}", flush=True)
    print("road distance (OSM US major highways)...", flush=True)
    road_d = distance_km(f"{CD}/osm/united_states_roads.gpkg")
    print(f"  road_dist km med={np.nanmedian(road_d):.1f}", flush=True)
    np.savez(CACHE, score_im=score_im, wc_im=wc_im, ghi=ghi, pvout=pvout, ele=ele,
             slope=slope, temp=temp, grid_d=grid_d, road_d=road_d, slo=slo, shi=shi)
    print("cached panels", flush=True)

# ---------- plot ----------
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(3, 3, figsize=(15.5, 9.6))
imk = dict(extent=EXT, origin="upper", aspect=ASPECT)

def rgb(a, im, title):
    a.imshow(im, **imk); a.set_title(title, fontsize=10.5); a.set_xticks([]); a.set_yticks([])

def raster(a, arr, title, cmap, vmin=None, vmax=None, norm=None, clab=""):
    if norm is not None:
        m = a.imshow(arr, cmap=cmap, norm=norm, **imk)
    else:
        m = a.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, **imk)
    a.set_title(title, fontsize=10.5); a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(m, ax=a, fraction=0.030, pad=0.012); cb.ax.tick_params(labelsize=7)
    if clab: cb.set_label(clab, fontsize=7)

rgb(ax[0, 0], score_im, "Our RF suitability (AEF 2025)")
_sm = plt.cm.ScalarMappable(cmap="magma", norm=Normalize(vmin=float(slo)/1e4, vmax=float(shi)/1e4))
_cb = fig.colorbar(_sm, ax=ax[0, 0], fraction=0.030, pad=0.012)
_cb.set_label("$P$(solar)", fontsize=7); _cb.ax.tick_params(labelsize=7)
raster(ax[0, 1], ghi, "GHI irradiance", "inferno", clab="kWh/m$^2$/day")
raster(ax[0, 2], pvout, "PVOUT yield", "inferno", clab="kWh/kWp/day")
raster(ax[1, 0], ele, "Elevation", "terrain", vmin=0, vmax=np.nanpercentile(ele, 99), clab="m")
raster(ax[1, 1], slope, "Slope", "cividis", vmin=0, vmax=np.nanpercentile(slope, 98), clab="deg")
rgb(ax[1, 2], wc_im, "ESA WorldCover land cover")
ax[1, 2].legend([plt.Rectangle((0, 0), 1, 1, color="#" + c) for c in WC_COLORS], WC_NAMES,
                fontsize=6, loc="lower right", ncol=2, framealpha=0.75)
raster(ax[2, 0], grid_d, "Grid distance (Gridfinder)", "viridis",
       vmin=0, vmax=np.nanpercentile(grid_d, 92), clab="km")
raster(ax[2, 1], road_d, "Road distance (OSM major highways)", "viridis",
       vmin=0, vmax=np.nanpercentile(road_d, 92), clab="km")
raster(ax[2, 2], temp, "Air temperature", "RdYlBu_r", clab="°C")

fig.tight_layout()
fig.savefig(f"{FIG}/fig_us_covariate_3x3.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/fig_us_covariate_3x3.pdf", dpi=300, bbox_inches="tight")
print("saved fig_us_covariate_3x3", flush=True)

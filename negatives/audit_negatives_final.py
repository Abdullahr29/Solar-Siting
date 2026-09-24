"""Contamination audit of the FINAL on-disk negative set (13,444 chips).

Rule: every negative chip must be >=5 km from ANY known solar site, across ALL time,
from our inventory OR TZ-SAM (analysis + raw). Audits the actual imagery on disk (each
chip's geotransform center + footprint), not the vector bookkeeping.

Distance is true-metre great-circle via an ECEF (EPSG:4978) KD-tree over all solar points
(chord~=arc to <1 mm at 5 km). Reports center-distance (how the set was defined) AND
footprint-edge clearance (center_dist - chip half-diagonal), plus any polygon that actually
contains a chip center.
"""
import os, glob, time
import numpy as np
import geopandas as gpd
import rasterio
from pyproj import Transformer, CRS
from scipy.spatial import cKDTree

ROOT = os.path.expanduser("~/Solar_Workspace"); os.chdir(ROOT)
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
TZ = "Data/external/tz_sam_q1_2026/TZ-SAM Q1-2026 CC-BY-NC"
NEG_DIR = "Data/aef_solar_chips/negatives/chips"
OUT = "Solar-Siting/artifacts/stats/negative_final_audit_20260809.csv"
BUF = 5000.0

def log(*a): print(*a, flush=True);

def rep_pts_ll(path, layer=None):
    g = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    g = g[g.geometry.notna()]
    c = g.geometry.representative_point()
    return np.c_[c.x.values, c.y.values]

t0 = time.time()
log("loading solar sources (inventory + TZ-SAM analysis + raw)...")
inv = rep_pts_ll(INV)
tza = rep_pts_ll(f"{TZ}/2026-Q1_analysis_polygons.gpkg")
tzr = rep_pts_ll(f"{TZ}/2026-Q1_raw_polygons.gpkg")
solar_ll = np.vstack([inv, tza, tzr])
log(f"  inventory={len(inv):,}  tz_analysis={len(tza):,}  tz_raw={len(tzr):,}  total={len(solar_ll):,}")

# ECEF for true-metre nearest neighbour
to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
sx, sy, sz = to_ecef.transform(solar_ll[:, 0], solar_ll[:, 1], np.zeros(len(solar_ll)))
tree = cKDTree(np.c_[sx, sy, sz])
log(f"  ECEF KD-tree built ({time.time()-t0:.0f}s)")

chips = sorted(glob.glob(f"{NEG_DIR}/*.tif"))
log(f"auditing {len(chips):,} negative chips...")

_tf_cache = {}
def tf_for(crs):
    k = crs.to_string()
    if k not in _tf_cache:
        _tf_cache[k] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return _tf_cache[k]

rows, bad = [], 0
for i, f in enumerate(chips):
    try:
        with rasterio.open(f) as ds:
            cx, cy = ds.transform * (ds.width / 2.0, ds.height / 2.0)
            resx, resy = abs(ds.transform.a), abs(ds.transform.e)
            half_diag = np.hypot(ds.width * resx / 2.0, ds.height * resy / 2.0)
            lon, lat = tf_for(ds.crs).transform(cx, cy)
        ex, ey, ez = to_ecef.transform(lon, lat, 0.0)
        d, _ = tree.query([ex, ey, ez], k=1)  # chord ~= arc at this scale
        rows.append((os.path.basename(f), lon, lat, half_diag, d, d - half_diag))
    except Exception as e:
        bad += 1
        log(f"  ERR {os.path.basename(f)}: {e}")
    if (i + 1) % 2000 == 0:
        log(f"  {i+1:,}/{len(chips):,} ({time.time()-t0:.0f}s)")

import pandas as pd
df = pd.DataFrame(rows, columns=["chip", "lon", "lat", "half_diag_m", "center_dist_m", "footprint_dist_m"])
df["iso"] = df["chip"].str.split("_").str[1]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.sort_values("center_dist_m").to_csv(OUT, index=False)

log("\n" + "=" * 60)
log(f"FINAL NEGATIVE CONTAMINATION AUDIT  ({len(df):,} chips, read-errors={bad})")
log("=" * 60)
log(f"center-distance to nearest solar (any source, any year):")
log(f"  min      = {df.center_dist_m.min():,.0f} m")
log(f"  p0.1     = {df.center_dist_m.quantile(0.001):,.0f} m")
log(f"  p1       = {df.center_dist_m.quantile(0.01):,.0f} m")
log(f"  median   = {df.center_dist_m.median():,.0f} m")
cviol = df[df.center_dist_m < BUF]
fviol = df[df.footprint_dist_m < BUF]
contain = df[df.center_dist_m < df.half_diag_m]  # solar inside the chip footprint
log(f"\nWITHIN 5 km (center)        : {len(cviol)}  ({100*len(cviol)/len(df):.3f}%)")
log(f"WITHIN 5 km (footprint edge): {len(fviol)}  ({100*len(fviol)/len(df):.3f}%)")
log(f"solar inside chip footprint : {len(contain)}")
if len(cviol):
    log("\n--- center within 5 km (CONTAMINATION) ---")
    for r in cviol.itertuples():
        log(f"  {r.chip:32s} {r.iso}  center={r.center_dist_m:,.0f} m  footprint={r.footprint_dist_m:,.0f} m")
elif len(fviol):
    log("\nNo center within 5 km. Closest footprint edges (all still >0, but <5 km from a chip corner):")
    for r in fviol.sort_values("footprint_dist_m").head(15).itertuples():
        log(f"  {r.chip:32s} {r.iso}  center={r.center_dist_m:,.0f} m  footprint_edge={r.footprint_dist_m:,.0f} m")
log(f"\nwrote {OUT}")
log(f"total time {time.time()-t0:.0f}s")

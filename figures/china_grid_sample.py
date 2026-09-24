"""Dense regular-grid sampling over China for a high-res, aligned RF-vs-MCDA map.

Samples a ~0.1 deg (~11 km) land grid with the RF score + the Richards-MCDA criteria (skips population,
which Richards doesn't use), using the SAME machinery as the benchmark (GSA rasters, local-CRS road/grid
distance, aspect). Because RF and MCDA come from the SAME grid cells, the render can build RF, MCDA and a
clean per-cell disagreement panel at identical resolution. Saves china_grid.npz for local rendering.
"""
import os, time
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
OUT = "Solar-Siting/artifacts/paper_v3/figures/china_grid.npz"
YEAR = 2019
STEP = 0.1
CD = "Solar-Siting/covariate_data"

from covariate_sample import build_stack
from covariate_append_manual import ensure_gsa_tif, sample_raster


def gi(obj, tries=6, base=8):
    import time as _t
    for k in range(tries):
        try:
            return obj.getInfo()
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  retry {k+1} in {base*2**k}s: {str(e)[:60]}", flush=True); _t.sleep(base * 2 ** k)


def dist_km(lon, lat, gpkg, bbox=None):
    import geopandas as gpd
    from shapely import STRtree
    g = gpd.read_file(os.path.join(WS, gpkg), bbox=bbox).to_crs("EPSG:6933")
    if len(g) == 0:
        return np.full(len(lon), np.nan)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs("EPSG:6933")
    tree = STRtree(g.geometry.values); idx = tree.nearest(pts.values)
    return np.array([pts.values[i].distance(g.geometry.values[j]) for i, j in enumerate(idx)])  # metres (EPSG:6933)


def main():
    os.chdir(WS)
    import ee
    from shapely.geometry import shape
    from shapely.prepared import prep
    from shapely.geometry import Point
    from covariate_aspect import sample_aspect, derive
    ee.Initialize(project="ee-abdullahr-solar")

    stack, geom = build_stack("China", YEAR, MODEL)
    shp = shape(geom.getInfo()); pgeom = prep(shp)
    minx, miny, maxx, maxy = shp.bounds
    xs = np.arange(minx, maxx, STEP); ys = np.arange(miny, maxy, STEP)
    gx, gy = np.meshgrid(xs, ys)
    cand = np.c_[gx.ravel(), gy.ravel()]
    keep = np.array([pgeom.contains(Point(x, y)) for x, y in cand])
    xy = cand[keep]
    print(f"grid {len(xs)}x{len(ys)} -> {len(xy)} land cells", flush=True)

    bands = [f"A{i:02d}" for i in range(64)]
    # multiband GEE sample: score + slope + elevation + worldcover + in_wdpa
    out = {k: [] for k in ("lon", "lat", "score", "slope", "elevation", "worldcover", "in_wdpa")}
    t0 = time.time()
    for i in range(0, len(xy), 1000):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]))
                                   for x, y in xy[i:i+1000]])
        res = gi(stack.reduceRegions(fc, ee.Reducer.first(), scale=10))
        for feat, (x, y) in zip(res["features"], xy[i:i+1000]):
            p = feat["properties"]
            if p.get("score") is None:
                continue
            out["lon"].append(x); out["lat"].append(y)
            out["score"].append(p.get("score")); out["slope"].append(p.get("slope"))
            out["elevation"].append(p.get("elevation")); out["worldcover"].append(p.get("worldcover"))
            out["in_wdpa"].append(p.get("in_wdpa", 0))
        print(f"  gee {min(i+1000,len(xy))}/{len(xy)} ({time.time()-t0:.0f}s)", flush=True)
    d = {k: np.array(v, float) for k, v in out.items()}
    lon, lat = d["lon"], d["lat"]
    print(f"kept {len(lon)} finite-score cells", flush=True)

    # GSA climate (local rasters)
    for sub, col in [("World_GHI_", "gsa_ghi"), ("World_PVOUT_GISdata_LTAy", "gsa_pvout"),
                     ("World_GTI_", "gsa_gti"), ("World_TEMP_", "gsa_temp")]:
        d[col] = sample_raster(ensure_gsa_tif(sub), lon, lat)
        print(f"  gsa {col} done", flush=True)
    # aspect -> equatorwardness (GEE)
    print("  sampling aspect...", flush=True)
    asp, _ = sample_aspect(lon, lat)
    d["equatorwardness"], _, _ = derive(asp, d["slope"], lat)
    # road + grid distance (local)
    pad = 0.5; bbox = (lon.min()-pad, lat.min()-pad, lon.max()+pad, lat.max()+pad)
    print("  road distance...", flush=True)
    d["road_dist_m"] = dist_km(lon, lat, f"{CD}/osm/china_roads.gpkg", bbox)
    print("  grid distance...", flush=True)
    d["grid_dist_m"] = dist_km(lon, lat, f"{CD}/gridfinder/grid.gpkg", bbox)

    np.savez(os.path.join(WS, OUT), **d)
    print(f"saved {OUT} ({len(lon)} cells)", flush=True)


if __name__ == "__main__":
    main()

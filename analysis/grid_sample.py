"""Dense regular-grid sampling over a country for a high-res, aligned RF-vs-MCDA map.

Samples a ~0.05 deg (~5.5 km) land grid with the RF score + all MCDA criteria, using the SAME machinery
as the benchmark (GSA rasters, local-CRS road/grid distance, aspect). Aspect is folded into the single
multiband GEE call (setDefaultProjection DEM), and population is sampled only with --pop (Chen config).
Because RF and every criterion come from the SAME cells, the render builds RF, MCDA and a per-cell
disagreement panel at identical resolution. Saves {slug}_grid.npz.

Usage: python grid_sample.py "China"          # Richards config (no population)
       python grid_sample.py "Colombia" --pop # Chen config (needs population)
"""
import os, sys, time
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
YEAR = 2019
STEP = 0.025
CHUNK = 2000
CD = "Solar-Siting/covariate_data"
COUNTRY = [a for a in sys.argv[1:] if not a.startswith("--")][0]
SLUG = COUNTRY.lower().replace(" ", "_")
POP = "--pop" in sys.argv
OUT = f"Solar-Siting/artifacts/paper_v3/figures/{SLUG}_grid.npz"

from covariate_sample import build_stack
from covariate_append_manual import ensure_gsa_tif, sample_raster, OSM_SHP
from covariate_aspect import derive


def gi(obj, tries=6, base=8):
    for k in range(tries):
        try:
            return obj.getInfo()
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  retry {k+1} in {base*2**k}s: {str(e)[:60]}", flush=True); time.sleep(base * 2 ** k)


def dist_m(lon, lat, gpkg, bbox=None):
    import geopandas as gpd
    from shapely import STRtree
    g = gpd.read_file(os.path.join(WS, gpkg) if not gpkg.startswith("/vsizip") else gpkg, bbox=bbox).to_crs("EPSG:6933")
    if len(g) == 0:
        return np.full(len(lon), np.nan)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs("EPSG:6933")
    from shapely import distance as _sd
    tree = STRtree(g.geometry.values); idx = tree.nearest(pts.values)
    return _sd(pts.values, g.geometry.values[idx])  # vectorised (fast for millions of points)


def main():
    os.chdir(WS)
    import ee
    from shapely.geometry import shape, Point
    from shapely.prepared import prep
    ee.Initialize(project="ee-abdullahr-solar")

    stack, geom = build_stack(COUNTRY, YEAR, MODEL)   # used only for its cheap terrain/land bands
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().setDefaultProjection("EPSG:4326", None, 30)
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    # RF panel = the PRECOMPUTED 100 m released suitability product (cheap read; purely visual) — do NOT
    # re-classify. Only the MCDA criteria are computed. All sampled bands are cheap reads -> fast (~min not hrs).
    suit = ee.Image("projects/ee-abdullahr-solar/assets/solar_suitability_2025_R_100m").select("suitability").rename("score")
    stack = suit.addBands([stack.select(["slope", "elevation", "worldcover", "in_wdpa"]), aspect])

    shp = shape(geom.getInfo()); pgeom = prep(shp)
    minx, miny, maxx, maxy = shp.bounds
    xs = np.arange(minx, maxx, STEP); ys = np.arange(miny, maxy, STEP)
    gx, gy = np.meshgrid(xs, ys)
    cand = np.c_[gx.ravel(), gy.ravel()]
    xy = cand[np.array([pgeom.contains(Point(x, y)) for x, y in cand])]
    print(f"grid {len(xs)}x{len(ys)} @ {STEP} deg -> {len(xy)} land cells (pop={POP})", flush=True)

    keys = ["score", "slope", "elevation", "worldcover", "in_wdpa", "aspect"]
    CKPT = os.path.join(WS, OUT.replace(".npz", "_ckpt.npz"))
    if os.path.exists(CKPT):
        z = np.load(CKPT)
        out = {k: list(z[k]) for k in keys + ["lon", "lat"]}
        start_i = int(z["next_i"])
        print(f"RESUME from batch i={start_i}, {len(out['lon'])} pts already sampled", flush=True)
    else:
        out = {k: [] for k in keys + ["lon", "lat"]}; start_i = 0
    t0 = time.time()
    for i in range(start_i, len(xy), CHUNK):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]))
                                   for x, y in xy[i:i+CHUNK]])
        res = gi(stack.reduceRegions(fc, ee.Reducer.first(), scale=10))
        for feat, (x, y) in zip(res["features"], xy[i:i+CHUNK]):
            p = feat["properties"]
            if p.get("score") is None:
                continue
            out["lon"].append(x); out["lat"].append(y)
            for k in keys:
                out[k].append(p.get(k, np.nan))
        if (i // CHUNK) % 10 == 0:
            np.savez(CKPT, next_i=i + CHUNK, **{k: np.array(v) for k, v in out.items()})
        print(f"  gee {min(i+CHUNK,len(xy))}/{len(xy)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(CKPT, next_i=len(xy), **{k: np.array(v) for k, v in out.items()})  # GEE phase done
    d = {k: np.array(v, float) for k, v in out.items()}
    lon, lat = d["lon"], d["lat"]
    print(f"kept {len(lon)} cells", flush=True)

    for sub, col in [("World_GHI_", "gsa_ghi"), ("World_PVOUT_GISdata_LTAy", "gsa_pvout"),
                     ("World_GTI_", "gsa_gti"), ("World_TEMP_", "gsa_temp")]:
        d[col] = sample_raster(ensure_gsa_tif(sub), lon, lat)
    d["equatorwardness"], _, _ = derive(d["aspect"], d["slope"], lat)
    pad = 0.5; bbox = (lon.min()-pad, lat.min()-pad, lon.max()+pad, lat.max()+pad)
    print("  road distance...", flush=True)
    key = SLUG.replace("_", " ")
    rg = f"{CD}/osm/{SLUG}_roads.gpkg"
    if os.path.exists(os.path.join(WS, rg)):
        d["road_dist_m"] = dist_m(lon, lat, rg, bbox)
    elif key in OSM_SHP:
        d["road_dist_m"] = dist_m(lon, lat, f"/vsizip/{os.path.abspath(os.path.join(WS, CD))}/osm/{OSM_SHP[key]}/gis_osm_roads_free_1.shp", bbox)
    else:
        d["road_dist_m"] = np.full(len(lon), np.nan)
    print("  grid distance...", flush=True)
    d["grid_dist_m"] = dist_m(lon, lat, f"{CD}/gridfinder/grid.gpkg", bbox)
    if POP:
        from mcda_benchmark_v3 import sample_population
        print("  population...", flush=True)
        d["pop_density"] = sample_population(np.c_[lon, lat])

    np.savez(os.path.join(WS, OUT), step=STEP, **d)
    print(f"saved {OUT} ({len(lon)} cells)", flush=True)


if __name__ == "__main__":
    main()

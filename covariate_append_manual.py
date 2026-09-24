"""Append manually-transferred covariates to a sampled point table, keyed on (lon,lat).

Runs offline on JASMIN after the ~40 GB transfer. Samples the Global Solar Atlas
irradiance rasters (rasterio, native EPSG:4326 — no reprojection) and, optionally,
distance-to-grid (Gridfinder) and distance-to-road (OSM) at the same points the
GEE-native pass used. Writes an augmented *_covsample_full.npz.

Covariate completeness caveats (report these, per the plan):
  * OSM road distance ~ distance to where OSM mappers were active; density varies by
    country (China notably sparse). We log road count / km of network per country.
  * Gridfinder MV grid is a MODEL (predicted from night-lights + roads), not survey
    truth, so grid distance carries model error where the grid is inferred.

Usage (JASMIN sci node, env_solar; set PROJ_DATA for GDAL reprojection):
  export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
  python Solar-Siting/covariate_append_manual.py Greece --year 2021 --gsa --grid --roads
"""
import argparse, glob, os, zipfile
import numpy as np

CD = "Solar-Siting/covariate_data"

# GSA: match on a substring of the zip name -> output column. LTAm (monthly) kept
# separate; the annual LTAy PVOUT is the one we use as the irradiance-yield covariate.
GSA_LAYERS = [
    ("World_GHI_",  "gsa_ghi"),
    ("World_DNI_",  "gsa_dni"),
    ("World_DIF_",  "gsa_dif"),
    ("World_GTI_",  "gsa_gti"),
    ("World_PVOUT_GISdata_LTAy", "gsa_pvout"),
    ("World_OPTA_", "gsa_opta"),
    ("World_TEMP_", "gsa_temp"),
    ("World_ELE_",  "gsa_ele"),
]

OSM_SHP = {  # Geofabrik shapefiles (roads read directly via /vsizip)
    "greece": "greece-260719-free.shp.zip",
    "chile": "chile-260719-free.shp.zip",
    "south africa": "south-africa-latest-free.shp.zip",
    "spain": "spain-latest-free.shp.zip",
    "india": "india-latest-free.shp.zip",
    "colombia": "colombia-latest-free.shp.zip",
    "philippines": "philippines-latest-free.shp.zip",
    "malaysia": "malaysia-latest-free.shp.zip",   # geofabrik malaysia-singapore-brunei, bbox-clipped to MYS
}
OSM_PBF = {  # .osm.pbf (roads read via pyrosm, major-highway filter to bound memory)
    "germany": "germany-latest.osm.pbf", "united states": "us-latest.osm.pbf",
    "china": "china-latest.osm.pbf", "australia": "australia-latest.osm.pbf",
}
# major highway classes only: keeps distance-to-transport meaningful and bounds the
# feature count on huge countries (US "driving" incl. residential would be ~30M lines).
MAJOR_HW = ["motorway", "trunk", "primary", "secondary", "tertiary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link"]


def roads_from_pbf(fp, bbox):
    """Major-road lines within bbox from a .osm.pbf, as a GeoDataFrame (pyrosm)."""
    from pyrosm import OSM
    osm = OSM(fp, bounding_box=list(bbox))
    r = osm.get_network(network_type="driving",
                        custom_filter={"highway": MAJOR_HW})
    return r


def ensure_gsa_tif(zip_sub):
    """Extract the single .tif from a GSA zip (once) and return its path."""
    zips = glob.glob(f"{CD}/gsa/{zip_sub}*.zip")
    if not zips:
        return None
    z = zips[0]
    outdir = f"{CD}/gsa/tif"
    os.makedirs(outdir, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        member = next((m for m in zf.namelist() if m.lower().endswith(".tif")), None)
        if member is None:
            return None
        target = f"{outdir}/{os.path.basename(member)}"
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            with zf.open(member) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)
    return target


def sample_raster(path, lon, lat):
    import rasterio
    with rasterio.open(path) as ds:
        nod = ds.nodata
        vals = np.array([v[0] for v in ds.sample(np.c_[lon, lat])], dtype=float)
    if nod is not None:
        vals[vals == nod] = np.nan
    vals[vals < -1e30] = np.nan
    return vals


def utm_epsg(lon, lat):
    zone = int((np.nanmean(lon) + 180) // 6) + 1
    return (32600 if np.nanmean(lat) >= 0 else 32700) + zone


def nearest_distance(lon, lat, gdf, label):
    """Distance (m) from each point to the nearest geometry in gdf, via UTM + sjoin_nearest."""
    import geopandas as gpd
    from shapely.geometry import Point
    epsg = utm_epsg(lon, lat)
    print(f"  {label}: {len(gdf)} features in bbox; UTM EPSG:{epsg}", flush=True)
    pts = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([Point(x, y) for x, y in zip(lon, lat)], crs=4326).to_crs(epsg)
    ).reset_index(drop=True)
    lines = gpd.GeoDataFrame(geometry=gdf.to_crs(epsg).geometry.values).reset_index(drop=True)
    j = gpd.sjoin_nearest(pts, lines, distance_col="__d")
    j = j[~j.index.duplicated(keep="first")].sort_index()   # drop tie duplicates
    return j["__d"].values.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--gsa", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--roads", action="store_true")
    args = ap.parse_args()
    os.chdir(os.path.expanduser("~/Solar_Workspace"))

    slug = args.country.lower().replace(" ", "_")
    base = f"Solar-Siting/artifacts/covariate/{slug}/{slug}_{args.year}"
    # incremental: start from the fullest table so a later --roads pass keeps prior columns
    src = f"{base}_covsample_full.npz" if os.path.exists(f"{base}_covsample_full.npz") \
        else f"{base}_covsample.npz"
    tab = {k: v for k, v in np.load(src).items()}
    lon, lat = tab["lon"], tab["lat"]
    print(f"{args.country}: {len(lon)} points loaded from {os.path.basename(src)}", flush=True)

    if args.gsa:
        for sub, col in GSA_LAYERS:
            tif = ensure_gsa_tif(sub)
            if tif is None:
                print(f"  GSA {col}: zip not present yet -> skip", flush=True)
                continue
            tab[col] = sample_raster(tif, lon, lat)
            v = tab[col][np.isfinite(tab[col])]
            print(f"  GSA {col:11s} n={len(v):6d} med={np.median(v):.4g}", flush=True)

    if args.grid:
        import geopandas as gpd
        pad = 0.6
        bbox = (lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)
        g = gpd.read_file(f"{CD}/gridfinder/grid.gpkg", bbox=bbox)
        if len(g):
            tab["grid_dist_m"] = nearest_distance(lon, lat, g, "gridfinder MV grid")
            v = tab["grid_dist_m"]
            print(f"  grid_dist_m med={np.median(v):.0f} m", flush=True)
        else:
            print("  gridfinder: no grid features in bbox -> skip", flush=True)

    if args.roads:
        # best-effort: a roads failure (missing file, pbf OOM/timeout) must not lose the
        # rest of the country's covariates -> wrap and continue.
        try:
            import geopandas as gpd
            key = slug.replace("_", " ")
            pad = 0.6
            bbox = (lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)
            gpkg = f"{CD}/osm/{slug}_roads.gpkg"          # pre-extracted (ogr2ogr) major roads
            if os.path.exists(gpkg):
                r = gpd.read_file(gpkg, bbox=bbox)
                src = "OSM gpkg (major roads, ogr2ogr)"
            elif key in OSM_SHP:
                vp = f"/vsizip/{os.path.abspath(CD)}/osm/{OSM_SHP[key]}/gis_osm_roads_free_1.shp"
                r = gpd.read_file(vp, bbox=bbox)
                src = "OSM shp (all roads)"
            elif key in OSM_PBF:
                r = roads_from_pbf(f"{CD}/osm/{OSM_PBF[key]}", bbox)
                src = "OSM pbf (major roads, pyrosm)"
            else:
                r = None; src = "no OSM file"
            if r is not None and len(r):
                tab["road_dist_m"] = nearest_distance(lon, lat, r, src)
                tab["road_n_bbox"] = np.full(len(lon), len(r))   # completeness diagnostic
                print(f"  road_dist_m med={np.median(tab['road_dist_m']):.0f} m "
                      f"| {len(r)} {src} features in bbox", flush=True)
            else:
                print(f"  roads: no features for {slug} ({src}) -> skipped", flush=True)
        except Exception as e:
            print(f"  roads: FAILED for {slug} ({type(e).__name__}: {e}) -> skipped", flush=True)

    out = f"{base}_covsample_full.npz"
    np.savez(out, **tab)
    print(f"\nsaved {out} with columns {list(tab.keys())}", flush=True)


if __name__ == "__main__":
    main()

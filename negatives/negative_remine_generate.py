"""Generate NEW negative sites for the re-mine (4,242 across 22 deficit countries).

Distances and spacing are computed in a PER-COUNTRY azimuthal-equidistant CRS (true
metres), not a global equal-area grid, so the >=5 km solar buffer and inter-negative
spacing are accurate. Export bbox = 2700 m real square (built from metres-per-degree at
the site latitude) -> ~270 px UTM export -> center-cropped to 256 px downstream.

Each site is: >=5 km from ANY solar (inventory + TZ-SAM analysis + raw), spread from other
negatives (Poisson-disk, adaptive), inside the country and within ~100 km of that country's
positives, on valid AlphaEarth data for a random year 2017-2022. Output = a GeoJSON bundle
in the schema export_negatives.py expects.
"""
import os, json, sys, time, math
import numpy as np, pandas as pd, geopandas as gpd
import shapely
from shapely import contains_xy
from shapely.geometry import MultiPoint
from scipy.spatial import cKDTree
from pyproj import Transformer, CRS
import ee

ROOT = os.path.expanduser("~/Solar_Workspace"); os.chdir(ROOT)
WGS = "EPSG:4326"
NE = "Data/external/naturalearth/ne_50m_admin_0_countries.shp"
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
TZ = "Data/external/tz_sam_q1_2026/TZ-SAM Q1-2026 CC-BY-NC"
GPKG = "Solar-Siting/global_solar_ml_pipeline.gpkg"
MAN = "Data/external/tz_sam_q1_2026/negative_contamination/remine_manifest_13450.csv"
OUT_BUNDLE = "Solar-Siting/artifacts/splits/negatives_remine_bundle.geojson"
OUT_CSV = "Solar-Siting/artifacts/splits/negatives_remine_sites.csv"
BBOX_HALF = 1350.0            # -> 2700 m real square -> ~270 px
SOLAR_BUF = 5000.0
SEP_LEVELS = [12000, 10500, 9000, 7500, 6000]
POS_REGION_BUF = 100000.0
YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
ID_BASE = 100000
EE_PROJECT = "ee-abdullahr-solar"
OVERACCEPT = 1.12
rng = np.random.default_rng(42)


def log(*a): print(*a, flush=True)


def poly_pts_ll(path, layer=None):
    """Solar centroids as lon/lat (WGS84)."""
    g = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    g = g[g.geometry.notna()]
    c = g.geometry.representative_point()
    return np.c_[c.x.values, c.y.values]


def aeqd_crs(clon, clat):
    return CRS.from_proj4(f"+proj=aeqd +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m +no_defs")


def main():
    os.makedirs(os.path.dirname(OUT_BUNDLE), exist_ok=True)
    man = pd.read_csv(MAN, index_col=0)["add_new"].astype(int)
    log(f"targets: {len(man)} countries, {int(man.sum())} new negatives")

    log("loading solar (lon/lat) ...")
    solar_ll = np.vstack([poly_pts_ll(INV),
                          poly_pts_ll(f"{TZ}/2026-Q1_analysis_polygons.gpkg"),
                          poly_pts_ll(f"{TZ}/2026-Q1_raw_polygons.gpkg")])
    log(f"  solar points: {len(solar_ll):,}")

    tdn = gpd.read_file(GPKG, layer="training_data_negatives")
    exneg_ll = np.c_[tdn.longitude.values, tdn.latitude.values]

    ne = gpd.read_file(NE)[["ISO_A3_EH", "ADMIN", "geometry"]]
    tds = gpd.read_file(GPKG, layer="training_data_solar")
    tds = tds[tds.latitude.notna()].copy()
    pos = gpd.GeoDataFrame(tds, geometry=gpd.points_from_xy(tds.longitude, tds.latitude), crs=WGS)
    pos = gpd.sjoin(pos, ne[["ISO_A3_EH", "geometry"]], predicate="within", how="left")
    pos["iso"] = pos["ISO_A3_EH"]

    rows, summ = [], []
    for iso, need in man.sort_values(ascending=False).items():
        need = int(need)
        row = ne[ne["ISO_A3_EH"] == iso]
        if not len(row):
            log(f"!! {iso}: no NE polygon, SKIP"); continue
        cname = row.iloc[0]["ADMIN"]
        cpoly_ll = row.geometry.union_all()
        clon, clat = cpoly_ll.centroid.x, cpoly_ll.centroid.y
        acrs = aeqd_crs(clon, clat)
        T_ll2a = Transformer.from_crs(WGS, acrs, always_xy=True)
        T_a2ll = Transformer.from_crs(acrs, WGS, always_xy=True)

        cpoly = gpd.GeoSeries([cpoly_ll], crs=WGS).to_crs(acrs).iloc[0]     # country in AEQD
        # positive-region clip (AEQD)
        pp = pos[pos["iso"] == iso]
        if len(pp) >= 1:
            px, py = T_ll2a.transform(pp.geometry.x.values, pp.geometry.y.values)
            if len(pp) >= 3:
                region = MultiPoint(np.c_[px, py]).convex_hull.buffer(POS_REGION_BUF)
            else:
                region = MultiPoint(np.c_[px, py]).buffer(POS_REGION_BUF)
            samp = cpoly.intersection(region); clip = "pos"
        else:
            samp = cpoly; clip = "none"
        if samp.is_empty:
            samp = cpoly; clip = "empty->full"
        minx, miny, maxx, maxy = samp.bounds

        # solar + existing-neg subset near country (bbox filter in lon/lat, then project)
        lo, la = solar_ll[:, 0], solar_ll[:, 1]
        bx0, by0, bx1, by1 = cpoly_ll.bounds
        m = (lo >= bx0-0.2) & (lo <= bx1+0.2) & (la >= by0-0.2) & (la <= by1+0.2)
        if m.any():
            sx, sy = T_ll2a.transform(solar_ll[m, 0], solar_ll[m, 1])
            solar_tree = cKDTree(np.c_[sx, sy])
        else:
            solar_tree = None
        me = (exneg_ll[:,0] >= bx0-0.3) & (exneg_ll[:,0] <= bx1+0.3) & \
             (exneg_ll[:,1] >= by0-0.3) & (exneg_ll[:,1] <= by1+0.3)
        if me.any():
            ex, ey = T_ll2a.transform(exneg_ll[me, 0], exneg_ll[me, 1])
            seed = np.c_[ex, ey]
        else:
            seed = np.empty((0, 2))

        # candidate pool: in sampling region, >=5km from solar
        want_pool = max(need * 40, 3000); pool = []; tries = 0
        while sum(len(p) for p in pool) < want_pool and tries < 400:
            xs = rng.uniform(minx, maxx, 20000); ys = rng.uniform(miny, maxy, 20000)
            mm = contains_xy(samp, xs, ys); xs, ys = xs[mm], ys[mm]
            if len(xs):
                if solar_tree is not None:
                    d, _ = solar_tree.query(np.c_[xs, ys], k=1); k = d >= SOLAR_BUF
                    xs, ys = xs[k], ys[k]
                if len(xs): pool.append(np.c_[xs, ys])
            tries += 1
        cand = np.vstack(pool) if pool else np.empty((0, 2))

        need_over = int(np.ceil(need * OVERACCEPT)); accepted = []; used_sep = None
        for sep in SEP_LEVELS:
            grid = {}; cell = sep
            def gk(x, y): return (int(x // cell), int(y // cell))
            def ok(x, y):
                k = gk(x, y)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for (px_, py_) in grid.get((k[0]+dx, k[1]+dy), ()):
                            if (px_-x)**2 + (py_-y)**2 < sep*sep: return False
                return True
            def put(x, y): grid.setdefault(gk(x, y), []).append((x, y))
            for (x, y) in seed: put(x, y)
            acc = []
            for i in rng.permutation(len(cand)):
                x, y = cand[i]
                if ok(x, y):
                    put(x, y); acc.append((x, y))
                    if len(acc) >= need_over: break
            if len(acc) >= need_over or sep == SEP_LEVELS[-1]:
                accepted, used_sep = acc, sep
                if len(acc) >= need_over: break
        accepted = np.array(accepted) if len(accepted) else np.empty((0, 2))
        # AEQD centres -> lon/lat
        if len(accepted):
            alon, alat = T_a2ll.transform(accepted[:, 0], accepted[:, 1])
        else:
            alon, alat = np.array([]), np.array([])
        yrs = rng.choice(YEARS, size=len(accepted))
        summ.append((iso, need, len(cand), len(accepted), used_sep, clip))
        log(f"  {iso}: need {need} pool {len(cand)} accepted {len(accepted)} sep {used_sep} clip {clip}")
        for lon, lat, yr in zip(alon, alat, yrs):
            rows.append({"iso": iso, "country": cname, "lon": float(lon), "lat": float(lat), "year": int(yr)})

    sites = pd.DataFrame(rows)
    log(f"\naccepted total (pre-validity): {len(sites)}")

    log("checking AlphaEarth validity in EE ...")
    ee.Initialize(project=EE_PROJECT)
    aef = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    sites["valid"] = False
    for yr in YEARS:
        idx = sites.index[sites["year"] == yr].tolist()
        if not idx: continue
        mask = aef.filterDate(f"{yr}-01-01", f"{yr}-12-31").mosaic().select(0).mask()
        for s in range(0, len(idx), 500):
            chunk = idx[s:s+500]
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(sites.loc[i,"lon"]),
                     float(sites.loc[i,"lat"])]), {"i": int(i)}) for i in chunk])
            for attempt in range(4):
                try:
                    res = mask.reduceRegions(fc, ee.Reducer.first(), scale=10, tileScale=16).getInfo(); break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(8)
            for f in res["features"]:
                v = f["properties"].get("first", None)
                sites.loc[f["properties"]["i"], "valid"] = (v is not None and v >= 0.5)
    log(f"valid: {int(sites['valid'].sum())}/{len(sites)}")

    sites = sites[sites["valid"]].copy()
    final = pd.concat([sites[sites["iso"] == iso].head(int(n)) for iso, n in man.items()]).reset_index(drop=True)
    short = {iso: int(n) - int((final["iso"] == iso).sum()) for iso, n in man.items()
             if int(n) - int((final["iso"] == iso).sum()) > 0}

    feats, counter = [], {}
    for r in final.itertuples():
        counter[r.iso] = counter.get(r.iso, 0) + 1
        name = f"NEG_{r.iso}_{ID_BASE + counter[r.iso]}_{r.year}"
        dlat = BBOX_HALF / 111320.0
        dlon = BBOX_HALF / (111320.0 * math.cos(math.radians(r.lat)))
        b = [r.lon - dlon, r.lat - dlat, r.lon + dlon, r.lat + dlat]
        feats.append({"type": "Feature",
            "properties": {"old_filename": name + ".tif", "new_filename": name + ".tif",
                           "country": r.country, "target_year": int(r.year), "bounds": b},
            "geometry": {"type": "Polygon", "coordinates": [[[b[0],b[1]],[b[2],b[1]],
                          [b[2],b[3]],[b[0],b[3]],[b[0],b[1]]]]}})
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT_BUNDLE, "w"))
    final.to_csv(OUT_CSV, index=False)
    pd.DataFrame(summ, columns=["iso","need","pool","accepted","sep","clip"]).to_csv(
        "Solar-Siting/artifacts/splits/negatives_remine_summary.csv", index=False)
    log(f"\nBUNDLE: {len(feats)} features -> {OUT_BUNDLE}")
    log(f"shortfalls: {short if short else 'NONE'}")


if __name__ == "__main__":
    main()

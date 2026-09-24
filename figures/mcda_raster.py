"""High-res LOCAL raster MCDA (1 km) + RF-vs-MCDA figure — accurate: equal-area distances, tiled 1 km RF, GPW pop.

MCDA on a 0.01 deg (~1 km) country grid:
  * climate GHI/PVOUT/GTI/temp/elevation : GSA GeoTIFFs (native ~1 km) reprojected (exact).
  * slope / equatorwardness              : from GSA elevation (numpy gradient; ~1 km terrain, minor weight — noted caveat).
  * road / grid distance                 : rasterize in EPSG:6933 (equal-area) -> scipy distance transform -> exact metres.
  * population (Chen)                     : GPW pulled from GEE at 1 km (getDownloadURL).
RF panel = 100 m placeholder asset pulled at 1 km via TILED getDownloadURL (int16, no re-classify). Normalised
country-relative (2-98 pct), weighted (Richards/Chen), WLC. 3 panels: RF | MCDA | disagreement.

Usage: python mcda_raster.py China   |   python mcda_raster.py Colombia --pop
"""
import os, sys, tempfile
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
CD = "Solar-Siting/covariate_data"
STEP = 0.01
COUNTRY = [a for a in sys.argv[1:] if not a.startswith("--")][0]
SLUG = COUNTRY.lower().replace(" ", "_")
POP = "--pop" in sys.argv
OUTDIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figures")
from mcda_benchmark_v3 import WEIGHTS, DIRN, norm_benefit, agg_wlc, EXCLUDE_WC
from covariate_append_manual import ensure_gsa_tif, OSM_SHP

META = {"china": dict(cfg="richards", label="Richards-AHP (road-weighted)", rf=0.872, mcda=0.615),
        "colombia": dict(cfg="chen", label="Chen (GHI+population)", rf=0.937, mcda=0.905)}


def target_grid(te):
    import rasterio
    minx, miny, maxx, maxy = te
    W = int(round((maxx - minx) / STEP)); H = int(round((maxy - miny) / STEP))
    return H, W, rasterio.transform.from_bounds(minx, miny, maxx, maxy, W, H)


def warp_arr(src, H, W, tr):
    import rasterio
    from rasterio.warp import reproject, Resampling
    out = np.full((H, W), np.nan, dtype="float32")
    with rasterio.open(src) as s:
        reproject(rasterio.band(s, 1), out, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear, dst_nodata=np.nan)
    return out


def dist_m_raster(vec_path, te, H, W, tr):
    """Distance-to-feature in EXACT metres: rasterize + distance transform in equal-area EPSG:6933, warp back."""
    import geopandas as gpd, rasterio
    from rasterio import features
    from rasterio.warp import reproject, Resampling
    from scipy.ndimage import distance_transform_edt
    from pyproj import Transformer
    g = gpd.read_file(vec_path, bbox=(te[0]-0.6, te[1]-0.6, te[2]+0.6, te[3]+0.6))
    if len(g) == 0:
        return np.full((H, W), np.nan)
    g = g.to_crs("EPSG:6933")
    t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    xs, ys = t.transform([te[0], te[2], te[0], te[2]], [te[1], te[1], te[3], te[3]])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    res = STEP * 111000
    W6 = int((maxx - minx) / res) + 1; H6 = int((maxy - miny) / res) + 1
    tr6 = rasterio.transform.from_bounds(minx, miny, maxx, maxy, W6, H6)
    burn = features.rasterize(((geom, 1) for geom in g.geometry), out_shape=(H6, W6), transform=tr6,
                              fill=0, all_touched=True).astype(bool)
    dist6 = (distance_transform_edt(~burn) * res).astype("float32")   # exact metres in equal-area
    out = np.full((H, W), np.nan, dtype="float32")
    reproject(dist6, out, src_transform=tr6, src_crs="EPSG:6933", dst_transform=tr, dst_crs="EPSG:4326",
              resampling=Resampling.bilinear, dst_nodata=np.nan)
    return out


def pull_asset(img, te, H, W, tr, ee, requests, rasterio, tmp, name, resamp="bilinear", scale=None):
    """Pull a GEE image to the 1 km display grid via getDownloadURL, tiling until each request fits the cap."""
    from rasterio.warp import reproject, Resampling
    sc = scale or STEP * 111000
    minx, miny, maxx, maxy = te
    for n in (1, 2, 3, 4):
        acc = np.full((H, W), np.nan, dtype="float32"); ok = True
        xs = np.linspace(minx, maxx, n + 1); ys = np.linspace(miny, maxy, n + 1)
        try:
            for i in range(n):
                for j in range(n):
                    reg = ee.Geometry.Rectangle([float(xs[i]), float(ys[j]), float(xs[i+1]), float(ys[j+1])], None, False)
                    url = img.getDownloadURL({"region": reg, "scale": sc, "crs": "EPSG:4326", "format": "GEO_TIFF"})
                    p = f"{tmp}/{name}_{i}_{j}.tif"; open(p, "wb").write(requests.get(url, timeout=600).content)
                    with rasterio.open(p) as s:
                        a = s.read(1).astype("float32")
                        if s.nodata is not None:
                            a[a == s.nodata] = np.nan
                        tile = np.full((H, W), np.nan, dtype="float32")
                        reproject(a, tile, src_transform=s.transform, src_crs="EPSG:4326", dst_transform=tr,
                                  dst_crs="EPSG:4326", resampling=getattr(Resampling, resamp), dst_nodata=np.nan)
                        acc = np.where(np.isfinite(tile), tile, acc)
            print(f"  pulled {name} at 1 km in {n}x{n} tiles", flush=True); return acc
        except Exception as e:
            print(f"  {name} {n}x{n} failed: {str(e)[:60]}", flush=True); ok = False
    return np.full((H, W), np.nan, dtype="float32")


def main():
    os.chdir(WS)
    import ee, requests, rasterio
    from rasterio import features
    from shapely.geometry import shape
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata
    ee.Initialize(project="ee-abdullahr-solar")

    geom = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", COUNTRY)).geometry()
    shp = shape(geom.getInfo())
    te = tuple(round(v, 2) for v in shp.bounds)
    H, W, tr = target_grid(te); minx, miny, maxx, maxy = te; ext = [minx, maxx, miny, maxy]
    tmp = tempfile.mkdtemp()
    print(f"{COUNTRY} grid {H}x{W} @ {STEP} deg", flush=True)

    crit = {}
    for sub, name in [("World_GHI_", "gsa_ghi"), ("World_PVOUT_GISdata_LTAy", "gsa_pvout"),
                      ("World_GTI_", "gsa_gti"), ("World_TEMP_", "gsa_temp"), ("World_ELE_", "elevation")]:
        src = ensure_gsa_tif(sub); src = src if os.path.isabs(src) else os.path.join(WS, src)
        crit[name] = warp_arr(src, H, W, tr)
    ele = crit["elevation"]; px = STEP * 111000
    gy, gx = np.gradient(np.nan_to_num(ele), px)
    crit["slope"] = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(-gx, gy)) % 360
    crit["equatorwardness"] = (np.cos(np.radians(aspect - 180)) + 1) / 2
    print("  climate/terrain ok", flush=True)

    print("  road distance (equal-area)...", flush=True)
    key = SLUG.replace("_", " ")
    rgpkg = os.path.join(WS, f"{CD}/osm/{SLUG}_roads.gpkg")
    if os.path.exists(rgpkg):
        crit["road_dist_m"] = dist_m_raster(rgpkg, te, H, W, tr)
    elif key in OSM_SHP:
        crit["road_dist_m"] = dist_m_raster(f"/vsizip/{os.path.abspath(os.path.join(WS, CD))}/osm/{OSM_SHP[key]}/gis_osm_roads_free_1.shp", te, H, W, tr)
    print("  grid distance (equal-area)...", flush=True)
    crit["grid_dist_m"] = dist_m_raster(os.path.join(WS, f"{CD}/gridfinder/grid.gpkg"), te, H, W, tr)

    m = META.get(SLUG, dict(cfg="richards", label="Richards", rf=0, mcda=0))
    weights = dict(WEIGHTS[m["cfg"]])
    if m["cfg"] == "chen":
        gpw = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density").filterDate("2019-01-01", "2021-12-31").first().clip(geom)
        crit["pop_density"] = pull_asset(gpw, te, H, W, tr, ee, requests, rasterio, tmp, "pop")

    mask = features.rasterize([(shp, 1)], out_shape=(H, W), transform=tr, fill=0).astype(bool)
    N01 = {}
    for cname, (col, direction) in DIRN.items():
        if col not in crit:
            continue
        a = crit[col]; v = a[mask & np.isfinite(a)]
        if v.size < 100:
            continue
        lo, hi = np.percentile(v, [2, 98]); N01[cname] = norm_benefit(a, lo, hi, direction)
    w = {c: weights[c] for c in weights if c in N01}; ssum = sum(w.values()); w = {k: v/ssum for k, v in w.items()}
    print(f"  {m['cfg']} criteria: {dict((k, round(v,3)) for k,v in w.items())}", flush=True)
    mcda = np.where(mask, agg_wlc({c: N01[c] for c in w}, w), np.nan)

    # exclusion EXACTLY as the benchmark: water/wetland (WorldCover EXCLUDE_WC) OR protected (WDPA)
    wc = pull_asset(ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(geom),
                    te, H, W, tr, ee, requests, rasterio, tmp, "wc", resamp="nearest")
    wdpa_fc = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(geom)   # only this country's PAs
    # protected areas are large -> pull at a coarser 3 km scale so the paint is tractable over big countries
    wdpa = pull_asset(ee.Image().byte().paint(wdpa_fc, 1).unmask(0).clip(geom),
                      te, H, W, tr, ee, requests, rasterio, tmp, "wdpa", resamp="nearest", scale=3000)
    if not np.isfinite(wdpa).any():
        print("  WARN: WDPA pull failed -> proceeding with water/wetland exclusion only", flush=True)
    excl = (np.isin(np.nan_to_num(wc, nan=-1).astype(int), list(EXCLUDE_WC)) | (np.nan_to_num(wdpa) >= 0.5)) & mask
    if excl.any():
        mcda[excl] = np.nanmin(mcda[mask & np.isfinite(mcda)])
    print(f"  excluded {int(excl.sum())} px (water/wetland {sorted(EXCLUDE_WC)} + WDPA protected)", flush=True)

    suit = ee.Image("projects/ee-abdullahr-solar/assets/solar_suitability_2025_R_100m").select("suitability").clip(geom)
    rf_rs = pull_asset(suit, te, H, W, tr, ee, requests, rasterio, tmp, "rf")
    rf_rs = np.where(mask, rf_rs, np.nan)

    def pctl(a):
        o = np.full(a.shape, np.nan); f = np.isfinite(a); o[f] = rankdata(a[f]) / f.sum(); return o
    mc_p, rf_p = pctl(mcda), pctl(rf_rs); diff = rf_p - mc_p

    from mcda_map_render import all_sites
    our_xy, tz_xy = all_sites(COUNTRY); site_xy = np.vstack([our_xy, tz_xy]) if len(our_xy)+len(tz_xy) else np.empty((0, 2))
    ntot = len(site_xy)
    disp = site_xy[np.random.default_rng(0).choice(ntot, 5000, replace=False)] if ntot > 5000 else site_xy
    # cache computed arrays for cheap re-rendering (no ~10 min recompute)
    np.savez(f"{OUTDIR}/{SLUG}_raster_data.npz", rf_p=rf_p, mc_p=mc_p, diff=diff, ext=np.array(ext),
             site_xy=site_xy, ntot=ntot, rf_roc=m["rf"], mcda_roc=m["mcda"], label=m["label"])

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(rf_p, extent=ext, cmap="magma", vmin=0, vmax=1)
    ax[0].scatter(disp[:, 0], disp[:, 1], s=1.2, c="cyan", alpha=0.35, linewidths=0,
                  label=f"solar sites (n={ntot:,}; {len(disp):,} shown)")
    ax[0].legend(loc="lower left", fontsize=8); ax[0].set_title(f"RF suitability — 100 m placeholder (ROC {m['rf']:.3f})")
    im1 = ax[1].imshow(mc_p, extent=ext, cmap="magma", vmin=0, vmax=1)
    ax[1].scatter(disp[:, 0], disp[:, 1], s=1.2, c="cyan", alpha=0.35, linewidths=0)
    ax[1].set_title(f"MCDA — {m['label']}, WLC, 1 km LOCAL raster (ROC {m['mcda']:.3f})")
    plt.colorbar(im1, ax=ax[1], fraction=0.03)
    im2 = ax[2].imshow(diff, extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[2].set_title("Disagreement RF − MCDA (red=RF-high/MCDA-low; blue=MCDA-high/RF-low)")
    plt.colorbar(im2, ax=ax[2], fraction=0.03)
    fig.suptitle(f"{COUNTRY} — RF (100 m placeholder) vs MCDA (1 km local raster, equal-area distances)", fontsize=13)
    fig.tight_layout()
    out = f"{OUTDIR}/{SLUG}_rf_vs_mcda_raster.png"
    fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out} ({int(mask.sum())} land px)", flush=True)


if __name__ == "__main__":
    main()

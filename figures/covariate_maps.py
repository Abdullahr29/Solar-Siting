"""Side-by-side spatial maps: our RF suitability raster vs the physical covariates.

Renders a co-registered multi-panel figure over a country so the overlap between our
AlphaEarth suitability score and the standard GIS layers (irradiance, yield, elevation,
slope, land cover) can be read by eye. Score / slope / land cover come straight from
Earth Engine as dense thumbnails; GHI / PVOUT / elevation come from the local Global
Solar Atlas GeoTIFFs (windowed to the country bbox).

Usage (JASMIN sci node, env_solar):
    python Solar-Siting/covariate_maps.py Greece --year 2021
"""
import argparse, io, os
import numpy as np

REFERENCE_MODEL = "Solar-Siting/artifacts/models/rf30_final.joblib"
CD = "Solar-Siting/covariate_data"

# ESA WorldCover official class -> colour (remapped to 0..10 for the palette)
WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
WC_COLORS = ["006400", "ffbb22", "ffff4c", "f096ff", "fa0000", "b4b4b4",
             "f0f0f0", "0064c8", "0096a0", "00cf75", "fae6a0"]
WC_NAMES = ["tree", "shrub", "grass", "crop", "built", "bare",
            "snow", "water", "wetland", "mangrove", "moss"]


def ee_thumb(img, geom, dim, vis):
    import requests
    from PIL import Image
    url = img.clip(geom).getThumbURL({"region": geom.bounds(), "dimensions": dim, **vis})
    return np.array(Image.open(io.BytesIO(requests.get(url, timeout=600).content)))


def _greece_mask(ext, shape_hw, shp):
    import rasterio
    from rasterio.features import geometry_mask
    minx, maxx, miny, maxy = ext
    h, w = shape_hw
    tr = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
    return geometry_mask([shp.__geo_interface__], shape_hw, tr, invert=True)  # True = inside


def gsa_window(tif, ext, shp, out=700, derive_slope=False):
    import rasterio
    from rasterio.windows import from_bounds
    minx, maxx, miny, maxy = ext
    with rasterio.open(tif) as ds:
        win = from_bounds(minx, miny, maxx, maxy, ds.transform)
        h = out
        w = int(out * (maxx - minx) / (maxy - miny))
        arr = ds.read(1, window=win, out_shape=(h, w), boundless=True).astype(float)
        nod = ds.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    arr[arr < -1e30] = np.nan
    if derive_slope:                       # slope (deg) from the elevation window, native res
        land = np.where(arr > 0, arr, 0.0)
        dy = (maxy - miny) / h * 111320.0
        dx = (maxx - minx) / w * 111320.0 * np.cos(np.deg2rad((miny + maxy) / 2))
        gy, gx = np.gradient(land, dy, dx)
        arr = np.degrees(np.arctan(np.hypot(gx, gy)))
    sea = ~np.isfinite(arr)
    inside = _greece_mask(ext, arr.shape, shp)
    arr[~inside] = np.nan                   # clip to Greece polygon
    if not derive_slope:
        arr[np.where(np.isfinite(arr), arr, 1) <= 0] = np.nan   # drop sea/bathymetry
    else:
        arr[sea] = np.nan
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--dim", type=int, default=700)
    args = ap.parse_args()
    os.chdir(os.path.expanduser("~/Solar_Workspace"))

    import ee, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    ee.Initialize(project="ee-abdullahr-solar")

    from shapely.geometry import shape
    from covariate_sample import build_stack
    stack, geom = build_stack(args.country, args.year, REFERENCE_MODEL)
    shp = shape(geom.getInfo())

    b = geom.bounds().coordinates().getInfo()[0]
    xs, ys = [c[0] for c in b], [c[1] for c in b]
    ext = [min(xs), max(xs), min(ys), max(ys)]           # minx,maxx,miny,maxy
    midlat = (ext[2] + ext[3]) / 2
    aspect = 1 / np.cos(np.deg2rad(midlat))

    slug = args.country.lower().replace(" ", "_")
    d = np.load(f"Solar-Siting/artifacts/covariate/{slug}/{slug}_{args.year}_covsample_full.npz")
    s = d["score"][np.isfinite(d["score"])]
    slo, shi = np.percentile(s, [2, 98])

    magma = ["000004", "2c115f", "721f81", "b73779", "f1605d", "feb078", "fcfdbf"]

    print("rendering EE thumbnails...", flush=True)
    score_im = ee_thumb(stack.select("score"), geom, args.dim,
                        {"min": float(slo), "max": float(shi), "palette": magma})
    wc = stack.select("worldcover").remap(WC_CLASSES, list(range(len(WC_CLASSES))))
    wc_im = ee_thumb(wc, geom, args.dim, {"min": 0, "max": len(WC_CLASSES) - 1, "palette": WC_COLORS})

    print("reading GSA windows...", flush=True)
    ghi = gsa_window(f"{CD}/gsa/tif/GHI.tif", ext, shp, args.dim)
    pvout = gsa_window(f"{CD}/gsa/tif/PVOUT.tif", ext, shp, args.dim)
    ele = gsa_window(f"{CD}/gsa/tif/ELE.tif", ext, shp, args.dim)
    slope = gsa_window(f"{CD}/gsa/tif/ELE.tif", ext, shp, args.dim, derive_slope=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    imk = dict(extent=ext, origin="upper", aspect=aspect)

    axes[0, 0].imshow(score_im, **imk)
    axes[0, 0].set_title(f"OUR RF suitability score (AEF {args.year})", fontweight="bold")

    im = axes[0, 1].imshow(ghi, cmap="inferno", **imk)
    axes[0, 1].set_title("GSA GHI irradiance (kWh/m²/day)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(pvout, cmap="inferno", **imk)
    axes[0, 2].set_title("GSA PVOUT yield (kWh/kWp/day)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(ele, cmap="terrain", vmin=0, vmax=np.nanpercentile(ele, 99), **imk)
    axes[1, 0].set_title("Elevation (m, GSA — land only)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(slope, cmap="cividis", vmin=0, vmax=np.nanpercentile(slope, 98), **imk)
    axes[1, 1].set_title("Slope (deg, derived from DEM)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(wc_im, **imk)
    axes[1, 2].set_title("ESA WorldCover 2021 land cover")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#" + c) for c in WC_COLORS]
    axes[1, 2].legend(handles, WC_NAMES, fontsize=6, loc="lower left", ncol=2, framealpha=0.7)

    for ax in axes.ravel():
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.suptitle(f"{args.country} {args.year}: AlphaEarth suitability vs physical covariates",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = f"Solar-Siting/artifacts/figures/covariate/{slug}/{slug}_{args.year}_maps.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()

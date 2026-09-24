"""Clean paper version of the India percentile-normalisation weak-country figure (magma throughout).
Top row: rasterised RF-suitability maps (GEE thumbnails, magma palette) on the absolute 0-1 scale
(dim) and the within-country percentile stretch (brightened) with solar sites overlaid. Bottom row:
raw and percentile score histograms (magma). Panel labels a-d; descriptive text lives in the LaTeX
caption, so titles are minimal. Map thumbnails are cached so re-runs need no GEE.
-> figures/fig_weakcountry_map.{pdf,png}
"""
import io, os
import numpy as np

WS = "/gws/ssde/j25b/gbov/abdullah_solar"
MODEL = f"{WS}/Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
SCORES = f"{WS}/Solar-Siting/artifacts/paper_v3/scores/country/india_R_2019_scores.npz"
FIG = f"{WS}/Solar-Siting/artifacts/paper_v3/figures"
CACHE = f"{FIG}/weak_country_v3/_india_thumb_cache.npz"
YEAR = 2019
# magma control points as hex (matches cm.magma sampling used elsewhere)
PALETTE = ["000004", "2c115f", "721f81", "b73779", "f1605d", "feb078", "fcfdbf"]


def auc(pos, neg):
    ns = np.sort(neg); lo = np.searchsorted(ns, pos, "left"); hi = np.searchsorted(ns, pos, "right")
    return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))


def pctl(s, ref):
    rs = np.sort(ref); lo = np.searchsorted(rs, s, "left"); hi = np.searchsorted(rs, s, "right")
    return (lo + 0.5 * (hi - lo)) / len(rs)


def get_thumbs(lo, hi):
    """Return (im_abs, im_str, ext) from cache, else fetch from GEE and cache."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print("loaded thumbnails from cache", flush=True)
        return z["im_abs"], z["im_str"], z["ext"]
    import ee, joblib, requests
    from PIL import Image
    from geemap import ml
    ee.Initialize(project="ee-abdullahr-solar")
    bands = [f"A{i:02d}" for i in range(64)]
    rf = joblib.load(MODEL)
    clf = ml.strings_to_classifier(ml.rf_to_strings(rf, bands, processes=16, output_mode="PROBABILITY"))
    geom = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "India")).geometry()
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01").mosaic().select(bands))
    score = img.classify(clf).clip(geom).rename("score")

    def thumb(vmin, vmax):
        for dim in (1400, 1000, 600):
            try:
                url = score.getThumbURL({"region": geom.bounds(), "dimensions": dim,
                                         "min": float(vmin), "max": float(vmax), "palette": PALETTE})
                return np.array(Image.open(io.BytesIO(requests.get(url, timeout=600).content)))
            except Exception as e:
                print(f"  thumb {dim}px failed: {str(e)[:90]}", flush=True)
        return None

    print("fetching absolute-scale thumbnail...", flush=True); im_abs = thumb(0.0, 1.0)
    print("fetching percentile-stretch thumbnail...", flush=True); im_str = thumb(lo, hi)
    b = geom.bounds().getInfo()["coordinates"][0]
    xs = [p[0] for p in b]; ys = [p[1] for p in b]
    ext = np.array([min(xs), max(xs), min(ys), max(ys)])
    np.savez(CACHE, im_abs=im_abs, im_str=im_str, ext=ext)
    print("cached thumbnails", flush=True)
    return im_abs, im_str, ext


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import to_hex

    d = np.load(SCORES)
    site_s, rand_s = d["site_s"].astype(float), d["rand_s"].astype(float)
    site_xy = d["site_xy"]
    lo, hi = np.percentile(rand_s, [2, 98])
    a_raw = auc(site_s, rand_s)
    site_p, rand_p = pctl(site_s, rand_s), pctl(rand_s, rand_s)
    im_abs, im_str, ext = get_thumbs(lo, hi)

    C_NEG, C_POS = to_hex(cm.magma(0.30)), to_hex(cm.magma(0.78))
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.linewidth": 0.7})
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.9))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.15, wspace=0.10)

    def lab(a, t):
        a.text(0.012, 0.985, t, transform=a.transAxes, fontsize=14, fontweight="bold",
               va="top", ha="left", color="black",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # (a) absolute-scale map
    if im_abs is not None:
        ax[0].imshow(im_abs, extent=ext, aspect="auto")
    ax[0].set_title("Absolute scale (0–1)", fontsize=10.5); ax[0].axis("off"); lab(ax[0], "a")

    # (b) percentile-stretch map + validation sites (Greece/China marker style)
    if im_str is not None:
        ax[1].imshow(im_str, extent=ext, aspect="auto")
    ax[1].scatter(site_xy[:, 0], site_xy[:, 1], s=6, c="#2b8cff", edgecolors="none", alpha=0.9,
                  label=f"validation sites (n={len(site_xy)})")
    ax[1].legend(loc="lower right", fontsize=8, framealpha=0.9, markerscale=1.8, handletextpad=0.3)
    ax[1].set_title("Within-country percentile", fontsize=10.5); ax[1].axis("off"); lab(ax[1], "b")

    # (c) score histogram (colour labels top-right)
    b = np.linspace(0, 1, 45)
    a2 = ax[2]
    a2.hist(rand_s, b, density=True, color=C_NEG, alpha=0.85, label="random land")
    a2.hist(site_s, b, density=True, color=C_POS, alpha=0.80, label="solar sites")
    a2.axvline(np.median(rand_s), ls="--", c=C_NEG, lw=1.1)
    a2.axvline(np.median(site_s), ls="--", c=C_POS, lw=1.1)
    a2.set_title(f"Score distribution (ROC-AUC {a_raw:.3f})", fontsize=10.5)
    a2.set_xlabel("RF $P$(solar)"); a2.set_yticks([]); a2.legend(fontsize=8, loc="upper right")
    for sp in ("top", "right"): a2.spines[sp].set_visible(False)
    lab(a2, "c")

    fig.savefig(f"{FIG}/fig_weakcountry_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_weakcountry_map.pdf", dpi=300, bbox_inches="tight")
    print(f"saved fig_weakcountry_map | India AUC {a_raw:.3f} | sites {len(site_xy)}", flush=True)


if __name__ == "__main__":
    main()

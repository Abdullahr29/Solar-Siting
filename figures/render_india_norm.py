"""India normalisation demo in the run_country_rf HEATMAP style (getThumbURL thumbnails, not rasters).

Two thumbnails of the SAME classified image: (1) absolute colour scale 0-1 -> India looks dim (its land
scores squish low); (2) country-relative stretch to India's 2-98 pct -> brightened, structure revealed.
Plus the raw and country-percentile histograms (solar vs non-solar) from the cached point scores. Shows
the map brightens and the distributions separate on a readable axis, while AUC is unchanged (rank-preserving).
"""
import io, os
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
SCORES = "Solar-Siting/artifacts/paper_v3/scores/country/india_R_2019_scores.npz"
OUT = "Solar-Siting/artifacts/paper_v3/figures/weak_country_v3/india_normalization_heatmap.png"
YEAR = 2019
PALETTE = ["000004", "2c115f", "721f81", "b73779", "f1605d", "feb078", "fcfdbf"]


def auc(pos, neg):
    ns = np.sort(neg); lo = np.searchsorted(ns, pos, "left"); hi = np.searchsorted(ns, pos, "right")
    return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))


def pctl(s, ref):
    rs = np.sort(ref); lo = np.searchsorted(rs, s, "left"); hi = np.searchsorted(rs, s, "right")
    return (lo + 0.5 * (hi - lo)) / len(rs)


def thumb(score, geom, vmin, vmax):
    import requests
    from PIL import Image
    for dim in (1200, 800, 500):
        try:
            url = score.getThumbURL({"region": geom.bounds(), "dimensions": dim,
                                     "min": float(vmin), "max": float(vmax), "palette": PALETTE})
            png = requests.get(url, timeout=600).content
            return np.array(Image.open(io.BytesIO(png)))
        except Exception as e:
            print(f"  thumb {dim}px failed: {str(e)[:80]}", flush=True)
    return None


def main():
    os.chdir(WS)
    import ee, joblib
    from geemap import ml
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ee.Initialize(project="ee-abdullahr-solar")
    bands = [f"A{i:02d}" for i in range(64)]

    d = np.load(SCORES)
    site_s, rand_s = d["site_s"].astype(float), d["rand_s"].astype(float)
    site_xy, rand_xy, ext = d["site_xy"], d["rand_xy"], d["ext"]
    lo, hi = np.percentile(rand_s, [2, 98])
    a_raw = auc(site_s, rand_s)
    site_p, rand_p = pctl(site_s, rand_s), pctl(rand_s, rand_s)

    rf = joblib.load(MODEL)
    trees = ml.rf_to_strings(rf, bands, processes=32, output_mode="PROBABILITY")
    clf = ml.strings_to_classifier(trees)
    print("classifier injected", flush=True)
    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    geom = lsib.filter(ee.Filter.eq("country_na", "India")).geometry()
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01").mosaic().select(bands))
    score = img.classify(clf).clip(geom).rename("score")

    print("rendering absolute-scale thumbnail (0-1)...", flush=True)
    im_abs = thumb(score, geom, 0.0, 1.0)
    print("rendering country-stretched thumbnail (2-98 pct)...", flush=True)
    im_str = thumb(score, geom, lo, hi)

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    if im_abs is not None:
        ax[0, 0].imshow(im_abs, extent=ext, aspect="auto")
    ax[0, 0].set_title("MAP — absolute scale 0-1 (India looks dim: land scores squish low)")
    ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])
    if im_str is not None:
        ax[0, 1].imshow(im_str, extent=ext, aspect="auto")
    ax[0, 1].scatter(site_xy[:, 0], site_xy[:, 1], s=5, c="cyan", alpha=0.5, label=f"solar sites (n={len(site_xy)})")
    ax[0, 1].legend(loc="lower right", fontsize=8)
    ax[0, 1].set_title(f"MAP — country-relative stretch to India's 2-98 pct ({lo:.2f}-{hi:.2f}): brightened")
    ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])

    b = np.linspace(0, 1, 45)
    ax[1, 0].hist(rand_s, b, density=True, alpha=0.55, color="#999", label="non-solar (random land)")
    ax[1, 0].hist(site_s, b, density=True, alpha=0.55, color="#e63946", label="solar sites")
    ax[1, 0].axvline(np.median(rand_s), ls="--", c="#555"); ax[1, 0].axvline(np.median(site_s), ls="--", c="#c1121f")
    ax[1, 0].set_title(f"RAW score histogram (AUC={a_raw:.3f})"); ax[1, 0].set_xlabel("RF probability"); ax[1, 0].legend(fontsize=9)
    ax[1, 1].hist(rand_p, b, density=True, alpha=0.55, color="#999", label="non-solar")
    ax[1, 1].hist(site_p, b, density=True, alpha=0.55, color="#e63946", label="solar sites")
    ax[1, 1].set_title(f"COUNTRY-PERCENTILE histogram (AUC={auc(site_p,rand_p):.3f}, unchanged) — sites median {np.median(site_p)*100:.0f}th pct")
    ax[1, 1].set_xlabel("within-country percentile"); ax[1, 1].legend(fontsize=9)
    fig.suptitle("India — percentile normalisation: map brightens & distributions separate on a readable axis; discrimination (AUC) unchanged", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

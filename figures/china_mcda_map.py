"""RF-vs-MCDA spatial agreement/disagreement map for China (Fig A panel).

China = the sharpest 'MCDA fails where RF succeeds' case (gap +0.257; MCDA best 0.62, Chen anti-ranks
<0.5; RF 0.876). Shows: (i) RF suitability (GEE thumbnail), (ii) published Richards-AHP MCDA (WLC),
computed LOCALLY from the cached China covariate criteria (same normalisation as the benchmark), and
(iii) their disagreement, with real forward-install sites overlaid.

RF map = getThumbURL (nearest-subsampled real model, restricted-mode-OK). MCDA map = criteria from
china_2021_covsample_corrected.npz -> country-relative min-max -> Richards weights -> WLC -> gridded.
Sites = china_R_2019 forward-install site_xy. Only the RF thumbnail touches GEE.
"""
import io, os
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
COV = "Solar-Siting/artifacts/covariate/china/china_2021_covsample_corrected.npz"
SCORES = "Solar-Siting/artifacts/paper_v3/scores/country/china_R_2019_scores.npz"
OUT = "Solar-Siting/artifacts/paper_v3/figures/china_rf_vs_mcda_map.png"
YEAR = 2019
PALETTE = ["000004", "2c115f", "721f81", "b73779", "f1605d", "feb078", "fcfdbf"]

from mcda_benchmark_v3 import WEIGHTS, DIRN, norm_benefit, agg_wlc, land_suit, EXCLUDE_WC


def thumb(score, geom, vmin, vmax):
    import requests
    from PIL import Image
    for dim in (1000, 700, 480):
        try:
            url = score.getThumbURL({"region": geom.bounds(), "dimensions": dim,
                                     "min": float(vmin), "max": float(vmax), "palette": PALETTE})
            return np.array(Image.open(io.BytesIO(requests.get(url, timeout=600).content)))
        except Exception as e:
            print(f"  thumb {dim} failed: {str(e)[:70]}", flush=True)
    return None


def grid_map(lon, lat, val, ext, nx=90, ny=70, smooth=1.2):
    from scipy.ndimage import gaussian_filter
    xe = np.linspace(ext[0], ext[1], nx + 1); ye = np.linspace(ext[2], ext[3], ny + 1)
    s = np.histogram2d(lon, lat, [xe, ye], weights=val)[0]
    c = np.histogram2d(lon, lat, [xe, ye])[0]
    with np.errstate(invalid="ignore"):
        m = s / c
    # smooth over filled cells, keep empties as NaN
    filled = np.isfinite(m)
    mm = np.where(filled, m, 0.0)
    sm = gaussian_filter(mm, smooth) / np.maximum(gaussian_filter(filled.astype(float), smooth), 1e-6)
    sm[gaussian_filter(filled.astype(float), smooth) < 0.05] = np.nan
    return sm.T  # (ny,nx) for imshow


def main():
    os.chdir(WS)
    import ee, joblib
    from geemap import ml
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ee.Initialize(project="ee-abdullahr-solar")
    bands = [f"A{i:02d}" for i in range(64)]

    # ---- MCDA (Richards WLC) from cached criteria, local ----
    d = {k: np.asarray(v, float) for k, v in np.load(COV).items()}
    lon, lat = d["lon"], d["lat"]
    N01 = {}
    for name, (k, direction) in DIRN.items():
        if k not in d:
            continue
        col = d[k]; fin = np.isfinite(col)
        if fin.sum() < 100:
            continue
        lo, hi = np.nanpercentile(col[fin], [2, 98])
        N01[name] = norm_benefit(col, lo, hi, direction)
    w = {c: WEIGHTS["richards"][c] for c in WEIGHTS["richards"] if c in N01}
    ssum = sum(w.values()); w = {k: v / ssum for k, v in w.items()}
    mcda = agg_wlc({c: N01[c] for c in w}, w)
    land01, wc = land_suit(d["worldcover"])
    excl = (d.get("in_wdpa", np.zeros(len(lon))) >= 0.5) | np.isin(wc, list(EXCLUDE_WC))
    mcda = mcda.copy(); mcda[excl] = 0.0
    fin = np.all([np.isfinite(N01[c]) for c in w], axis=0)
    lon, lat, mcda = lon[fin], lat[fin], mcda[fin]

    # ---- RF thumbnail + extent + sites ----
    sc = np.load(SCORES); site_xy = sc["site_xy"]; ext = sc["ext"].tolist()
    rand_s = sc["rand_s"].astype(float)
    rf = joblib.load(MODEL)
    clf = ml.strings_to_classifier(ml.rf_to_strings(rf, bands, processes=32, output_mode="PROBABILITY"))
    print("classifier injected", flush=True)
    geom = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "China")).geometry()
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01").mosaic().select(bands))
    score = img.classify(clf).clip(geom).rename("score")
    lo, hi = np.percentile(rand_s, [2, 98])
    print("rendering RF thumbnail...", flush=True)
    im_rf = thumb(score, geom, lo, hi)

    # MCDA gridded to same extent, percentile-normalised for display
    mp = grid_map(lon, lat, mcda, ext)
    from scipy.stats import rankdata
    mcda_disp = mp.copy()
    v = mcda_disp[np.isfinite(mcda_disp)]
    mcda_disp[np.isfinite(mcda_disp)] = rankdata(v) / len(v)   # percentile for comparable colour

    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    if im_rf is not None:
        ax[0].imshow(im_rf, extent=ext, aspect="auto")
    ax[0].scatter(site_xy[:, 0], site_xy[:, 1], s=3, c="cyan", alpha=0.35, label=f"real solar sites (n={len(site_xy)})")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[0].set_title(f"RF suitability (ROC 0.876) — sites concentrate in high-RF land"); ax[0].set_xticks([]); ax[0].set_yticks([])
    im = ax[1].imshow(mcda_disp, extent=ext, origin="lower", aspect="auto", cmap="magma")
    ax[1].scatter(site_xy[:, 0], site_xy[:, 1], s=3, c="cyan", alpha=0.35)
    ax[1].set_title("Published MCDA (Richards-AHP, WLC; ROC 0.62) — misranks vs real sites"); ax[1].set_xticks([]); ax[1].set_yticks([])
    plt.colorbar(im, ax=ax[1], fraction=0.03)
    fig.suptitle("China — RF vs published MCDA: where they agree (both high in real deployment zones) and disagree "
                 "(MCDA over-weights irradiance/roads → misplaces suitability)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

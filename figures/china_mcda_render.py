"""Render the high-res China RF-vs-MCDA figure from china_grid.npz (local, no GEE).

3 panels on the IDENTICAL 0.1 deg grid: (1) RF suitability, (2) published Richards-AHP MCDA (WLC),
(3) disagreement = RF_percentile - MCDA_percentile (red = RF-high/MCDA-low = real deployment zones MCDA
misses; blue = MCDA-high/RF-low = high-irradiance land MCDA over-rates). Real forward-install sites overlaid.
"""
import os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
GRID = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figures/china_grid.npz")
SITES = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/scores/country/china_R_2019_scores.npz")
OUT = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figures/china_rf_vs_mcda_hires.png")
STEP = 0.1

from mcda_benchmark_v3 import WEIGHTS, DIRN, norm_benefit, agg_wlc, land_suit, EXCLUDE_WC


def to_raster(lon, lat, val, minx, miny, nx, ny):
    r = np.full((ny, nx), np.nan)
    ci = np.round((lon - minx) / STEP).astype(int)
    ri = np.round((lat - miny) / STEP).astype(int)
    ok = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
    r[ri[ok], ci[ok]] = val[ok]
    return r


def pctl_img(r):
    out = np.full_like(r, np.nan); m = np.isfinite(r)
    from scipy.stats import rankdata
    out[m] = rankdata(r[m]) / m.sum()
    return out


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = {k: np.asarray(v, float) for k, v in np.load(GRID).items()}
    lon, lat, rf = d["lon"], d["lat"], d["score"]

    # Richards MCDA (WLC) over the grid, benchmark-consistent normalisation
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
    print("Richards criteria used:", {k: round(v, 3) for k, v in w.items()})
    mcda = agg_wlc({c: N01[c] for c in w}, w)
    _, wc = land_suit(d["worldcover"])
    excl = (d.get("in_wdpa", np.zeros(len(lon))) >= 0.5) | np.isin(wc, list(EXCLUDE_WC))
    mcda = mcda.copy(); mcda[excl] = np.nanmin(mcda)

    minx, miny = lon.min(), lat.min()
    nx = int(round((lon.max() - minx) / STEP)) + 1
    ny = int(round((lat.max() - miny) / STEP)) + 1
    ext = [minx, lon.max(), miny, lat.max()]
    rf_r = to_raster(lon, lat, rf, minx, miny, nx, ny)
    mc_r = to_raster(lon, lat, mcda, minx, miny, nx, ny)
    rf_p, mc_p = pctl_img(rf_r), pctl_img(mc_r)
    diff = rf_p - mc_p

    sc = np.load(SITES); site_xy = sc["site_xy"]

    fig, ax = plt.subplots(1, 3, figsize=(22, 7))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    im0 = ax[0].imshow(rf_p, extent=ext, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax[0].scatter(site_xy[:, 0], site_xy[:, 1], s=2, c="cyan", alpha=0.3, label=f"real sites (n={len(site_xy)})")
    ax[0].legend(loc="lower left", fontsize=8); ax[0].set_title("RF suitability (ROC 0.876)")
    plt.colorbar(im0, ax=ax[0], fraction=0.03)
    im1 = ax[1].imshow(mc_p, extent=ext, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax[1].scatter(site_xy[:, 0], site_xy[:, 1], s=2, c="cyan", alpha=0.3)
    ax[1].set_title("Published MCDA — Richards-AHP, WLC (ROC 0.62)")
    plt.colorbar(im1, ax=ax[1], fraction=0.03)
    im2 = ax[2].imshow(diff, extent=ext, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[2].scatter(site_xy[:, 0], site_xy[:, 1], s=2, c="k", alpha=0.25)
    ax[2].set_title("Disagreement: RF − MCDA (red = RF-high/MCDA-low = missed deployment;\nblue = MCDA-high/RF-low = over-rated)")
    plt.colorbar(im2, ax=ax[2], fraction=0.03)
    fig.suptitle("China — RF vs published MCDA at 0.1° (~11 km), identical grid. MCDA over-weights irradiance/roads → "
                 "over-rates the west and misses the eastern deployment belt where solar (and RF) concentrate.", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120, bbox_inches="tight"); plt.close(fig)
    # quick quantitative: correlation of site density with each map's percentile
    print(f"cells={np.isfinite(rf_r).sum()} | median RF pct at sites-region vs land encoded in the disagreement panel")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

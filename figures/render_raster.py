"""Re-render the RF-vs-MCDA raster figure from cached arrays ({slug}_raster_data.npz) — instant, no recompute.

Usage: python render_raster.py China
"""
import os, sys
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
SLUG = sys.argv[1].lower().replace(" ", "_")
NSHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 2000   # sites shown in the overlay
D = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figures")


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(f"{D}/{SLUG}_raster_data.npz", allow_pickle=True)
    rf_p, mc_p, diff = z["rf_p"], z["mc_p"], z["diff"]
    ext = list(z["ext"]); site_xy = z["site_xy"]; ntot = int(z["ntot"])
    rf_roc, mcda_roc, label = float(z["rf_roc"]), float(z["mcda_roc"]), str(z["label"])
    disp = site_xy[np.random.default_rng(0).choice(ntot, NSHOW, replace=False)] if ntot > NSHOW else site_xy

    asp = 1.0 / np.cos(np.radians((ext[2] + ext[3]) / 2))   # latitude-corrected geographic aspect
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(rf_p, extent=ext, cmap="magma", vmin=0, vmax=1, aspect=asp)
    ax[0].scatter(disp[:, 0], disp[:, 1], s=1.2, c="cyan", alpha=0.35, linewidths=0,
                  label=f"solar sites (n={ntot:,}; {len(disp):,} shown)")
    ax[0].legend(loc="lower left", fontsize=8); ax[0].set_title(f"RF suitability — 100 m placeholder (ROC {rf_roc:.3f})")
    im1 = ax[1].imshow(mc_p, extent=ext, cmap="magma", vmin=0, vmax=1, aspect=asp)
    ax[1].scatter(disp[:, 0], disp[:, 1], s=1.2, c="cyan", alpha=0.35, linewidths=0)
    ax[1].set_title(f"MCDA — {label}, WLC, 1 km LOCAL raster (ROC {mcda_roc:.3f})")
    plt.colorbar(im1, ax=ax[1], fraction=0.03)
    im2 = ax[2].imshow(diff, extent=ext, cmap="RdBu_r", vmin=-1, vmax=1, aspect=asp)
    ax[2].set_title("Disagreement RF − MCDA (red=RF-high/MCDA-low; blue=MCDA-high/RF-low)")
    plt.colorbar(im2, ax=ax[2], fraction=0.03)
    fig.suptitle(f"{sys.argv[1]} — RF (100 m placeholder) vs MCDA (1 km local raster, equal-area distances)", fontsize=13)
    fig.tight_layout()
    out = f"{D}/{SLUG}_rf_vs_mcda_raster.png"
    fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"re-rendered {out}")


if __name__ == "__main__":
    main()

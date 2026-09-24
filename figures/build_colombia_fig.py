"""Colombia 1x3 RF-vs-MCDA figure (appendix) -- a mirror of the China figure (Fig 3):
(a) RF suitability + validation sites | (b) published MCDA | (c) RF - MCDA disagreement.
All three from figures/colombia_raster_data.npz (no GEE). Same design as build_china_fig.py:
left-side colorbar ticks/labels, bold top-left panel letters, magma / RdBu_r, 300 dpi.
Colombia has few sites (240) so every validation site is shown (no thinning).
-> figures/fig_colombia_mcda.{png,pdf}
"""
import os, numpy as np
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = "Solar-Siting/artifacts/paper_v3/figures"

z = np.load(f"{FIG}/colombia_raster_data.npz", allow_pickle=True)
rf_p, mc_p, diff, ext = z["rf_p"], z["mc_p"], z["diff"], z["ext"]
site_xy = z["site_xy"]; rf_roc = float(z["rf_roc"]); mcda_roc = float(z["mcda_roc"])
H, W = rf_p.shape
minx, maxx, miny, maxy = [float(v) for v in ext]
print(f"colombia grid {H}x{W} ext {ext} rf_roc {rf_roc:.3f} mcda_roc {mcda_roc:.3f} sites {len(site_xy)}", flush=True)

ds = 2
sl = (slice(None, None, ds), slice(None, None, ds))
rf_d, mc_d, df_d = rf_p[sl], mc_p[sl], diff[sl]
dots = site_xy                                    # few sites -> show all
aspect = 1.0 / np.cos(np.radians((miny + maxy) / 2))
EXT = [minx, maxx, miny, maxy]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.5))
fig.subplots_adjust(left=0.008, right=0.992, top=0.90, bottom=0.02, wspace=0.07)

def panel(ax, img, title, cmap, vmin, vmax, clab, lab, dots=None, ticks=None):
    im = ax.imshow(img, extent=EXT, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect(aspect); ax.axis("off"); ax.set_title(title, fontsize=10, pad=3)
    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], s=3.0, c="#2b8cff", edgecolors="none", alpha=0.8,
                   label=f"validation sites (n={len(dots)})")
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, markerscale=3.5,
                  handletextpad=0.3, borderpad=0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.012)
    cb.set_label(clab, fontsize=7); cb.ax.tick_params(labelsize=6)  # ticks/label on the right (default)
    if ticks is not None:
        cb.set_ticks(ticks)
    ax.text(0.01, 0.99, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

panel(axs[0], rf_d, f"RF suitability  (ROC-AUC {rf_roc:.2f})", "magma", 0, 1, "$P$(solar)", "a",
      dots=dots, ticks=[0, 0.5, 1])
panel(axs[1], mc_d, f"Published MCDA  (ROC-AUC {mcda_roc:.2f})", "magma", 0, 1, "score", "b", ticks=[0, 0.5, 1])
lim = float(np.nanpercentile(np.abs(df_d), 96))
panel(axs[2], df_d, "RF $-$ MCDA (disagreement)", "RdBu_r", -lim, lim, "$\\Delta$", "c",
      ticks=[-round(lim, 1), 0, round(lim, 1)])

fig.savefig(f"{FIG}/fig_colombia_mcda.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/fig_colombia_mcda.pdf", dpi=300, bbox_inches="tight")
print("saved fig_colombia_mcda (1x3)", flush=True)

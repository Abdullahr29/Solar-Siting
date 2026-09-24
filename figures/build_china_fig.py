"""China 1x3 results figure (wide-and-short, PVOUT dropped from the narrative):
(a) RF suitability + 5k sites | (b) published MCDA | (c) RF - MCDA disagreement.
All three from figures/china_raster_data.npz (no GEE / no PVOUT warp needed).
-> figures/fig_china_mcda.{png,pdf}
"""
import os, numpy as np
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = "Solar-Siting/artifacts/paper_v3/figures"

z = np.load(f"{FIG}/china_raster_data.npz", allow_pickle=True)
rf_p, mc_p, diff, ext = z["rf_p"], z["mc_p"], z["diff"], z["ext"]
site_xy = z["site_xy"]; rf_roc = float(z["rf_roc"]); mcda_roc = float(z["mcda_roc"])
H, W = rf_p.shape
minx, maxx, miny, maxy = [float(v) for v in ext]
print(f"china grid {H}x{W} ext {ext} rf_roc {rf_roc:.3f} mcda_roc {mcda_roc:.3f} sites {len(site_xy)}", flush=True)

ds = 2                                        # finer downsample -> crisper small panels
sl = (slice(None, None, ds), slice(None, None, ds))
rf_d, mc_d, df_d = rf_p[sl], mc_p[sl], diff[sl]
rng = np.random.default_rng(0)
# density-preserving thinning: subsample weighted by (local density)^-alpha so the displayed
# density is proportional to (true density)^(1-alpha). alpha=0.5 -> displayed ~ sqrt(true):
# keeps the real east-dense / west-sparse gradient while de-saturating the central-east blob.
def thin_preserve(xy, cell=0.5, target=2000, alpha=0.5):
    from collections import Counter
    kx = np.floor(xy[:, 0] / cell).astype(int); ky = np.floor(xy[:, 1] / cell).astype(int)
    keys = list(zip(kx.tolist(), ky.tolist()))
    cnt = Counter(keys)
    c = np.array([cnt[k] for k in keys], float)
    p = c ** (-alpha); p /= p.sum()
    n = min(target, len(xy))
    return xy[rng.choice(len(xy), size=n, replace=False, p=p)]
dots = thin_preserve(site_xy)
print(f"density-preserving overlay dots: {len(dots)} (from {len(site_xy)})", flush=True)
aspect = 1.0 / np.cos(np.radians((miny + maxy) / 2))
EXT = [minx, maxx, miny, maxy]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.5))
fig.subplots_adjust(left=0.008, right=0.992, top=0.90, bottom=0.02, wspace=0.07)

def panel(ax, img, title, cmap, vmin, vmax, clab, lab, dots=None, ticks=None):
    im = ax.imshow(img, extent=EXT, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect(aspect); ax.axis("off"); ax.set_title(title, fontsize=10, pad=3)
    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], s=1.4, c="#2b8cff", edgecolors="none", alpha=0.7,
                   label=f"validation sites (n={len(dots)})")
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, markerscale=6,
                  handletextpad=0.3, borderpad=0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.012)
    cb.set_label(clab, fontsize=7); cb.ax.tick_params(labelsize=6)
    cb.ax.yaxis.set_ticks_position("left"); cb.ax.yaxis.set_label_position("left")  # ticks/label on left of bar
    if ticks is not None:
        cb.set_ticks(ticks)
    ax.text(0.01, 0.99, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

panel(axs[0], rf_d, f"RF suitability  (ROC-AUC {rf_roc:.2f})", "magma", 0, 1, "$P$(solar)", "a",
      dots=dots, ticks=[0, 0.5, 1])
panel(axs[1], mc_d, f"Published MCDA  (ROC-AUC {mcda_roc:.2f})", "magma", 0, 1, "score", "b", ticks=[0, 0.5, 1])
lim = float(np.nanpercentile(np.abs(df_d), 96))
panel(axs[2], df_d, "RF $-$ MCDA (disagreement)", "RdBu_r", -lim, lim, "$\\Delta$", "c",
      ticks=[-round(lim, 1), 0, round(lim, 1)])

fig.savefig(f"{FIG}/fig_china_mcda.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/fig_china_mcda.pdf", dpi=300, bbox_inches="tight")
print("saved fig_china_mcda (1x3)", flush=True)

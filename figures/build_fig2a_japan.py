"""Paper Fig 2(a): 2x2 chip panel for the Japan site POS_JPN_15723_2017_3.
Same layout/style as the original (Sentinel-2 before/after, AEF embedding 2yr before install,
RF P(solar) linear magma) -- NO nonlinear panel. Reuses the cached Sentinel-2 composite from the
candidate render (no GEE). Saves fig2a_japan.{png,pdf}.
"""
import os, warnings, numpy as np; warnings.filterwarnings("ignore")
os.chdir("/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting")
import rasterio, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA

CH = "../Data/aef_solar_chips/positives/chips"; MK = "../Data/aef_solar_chips/positives/masks_v2"
RF = "artifacts/paper_v3/models/rf30_v3_R.joblib"
CACHE = "artifacts/paper_v3/figures/fig2a_candidates"
OUT = "artifacts/paper_v3/figures/fig2a_japan"
CHIP = "POS_JPN_15723_2017_3.tif"
EMB = int(CHIP.split("_")[3])

with rasterio.open(f"{MK}/{CHIP}") as s: mask = s.read(1)
with rasterio.open(f"{CH}/{CHIP}") as s: aef = s.read().astype(np.float32)
C, H, W = aef.shape
flat = aef.reshape(C, -1).T; fin = np.isfinite(flat).all(1)
rgb = np.full((H*W, 3), np.nan, np.float32)
rgb[fin] = PCA(3, random_state=0).fit_transform(flat[fin])
rgb = rgb.reshape(H, W, 3)
for i in range(3):
    lo, hi = np.nanpercentile(rgb[..., i], [2, 98]); rgb[..., i] = np.clip((rgb[..., i]-lo)/(hi-lo+1e-9), 0, 1)
aef_rgb = np.nan_to_num(rgb)
rf = joblib.load(RF)
pred = np.full(H*W, np.nan, np.float32); pred[fin] = rf.predict_proba(flat[fin])[:, 1]
pred = pred.reshape(H, W)

z = np.load(f"{CACHE}/_s2_{CHIP}.npz", allow_pickle=True)
s2_pre = z["pre"]; s2_post = z["post"]
s2_pre = None if s2_pre.dtype == object else s2_pre
s2_post = None if s2_post.dtype == object else s2_post
print(f"{CHIP}: frac {(mask>0).mean():.2f} P_in {np.nanmean(pred[mask>0]):.2f} "
      f"P_bg {np.nanmean(pred[mask==0]):.2f} | S2 pre {'ok' if s2_pre is not None else 'NA'}", flush=True)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})
fig, axs = plt.subplots(2, 2, figsize=(5.3, 5.5))
fig.subplots_adjust(left=0.004, right=0.996, top=0.945, bottom=0.006, wspace=0.03, hspace=0.10)

def show(ax, img, title, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    mh, mw = mask.shape
    ax.contour(np.linspace(0, img.shape[1]-1, mw), np.linspace(0, img.shape[0]-1, mh),
               mask > 0, levels=[0.5], colors="cyan", linewidths=1.0)
    ax.set_title(title, fontsize=9.5, pad=3); ax.set_xticks([]); ax.set_yticks([])
    return im

show(axs[0, 0], s2_pre, f"Sentinel-2 · {EMB} (before)")
show(axs[0, 1], s2_post, f"Sentinel-2 · {EMB+3} (after)")
show(axs[1, 0], aef_rgb, f"AlphaEarth · {EMB}")
imP = show(axs[1, 1], pred, "RF $P$(solar)", cmap="magma", vmin=0, vmax=1)
# vertical colorbar along the right edge (low->high bottom->top), so it never overlaps a footprint
cax = axs[1, 1].inset_axes([0.905, 0.10, 0.035, 0.80])
cb = fig.colorbar(imP, cax=cax, orientation="vertical")
cb.set_ticks([0, 0.5, 1])
cax.yaxis.set_ticks_position("left"); cax.yaxis.set_label_position("left")
cb.ax.tick_params(labelsize=7, colors="white", pad=1, length=2)
for t in cb.ax.get_yticklabels():
    t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="black")])
cb.outline.set_edgecolor("white"); cb.outline.set_linewidth(0.6)
# bold panel label "a" at top-left, on the tile-title line in the left margin (style matches Figs 3/4)
fig.text(0.018, 0.96, "a", fontsize=13, fontweight="bold", va="center", ha="left")
fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}.pdf", dpi=300, bbox_inches="tight")
print(f"saved {OUT}.png/.pdf", flush=True)

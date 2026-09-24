"""Appendix figure: 3x5 grid of additional site-scale examples.
Rows = 3 candidate sites (AUS, USA, PHL); columns = Sentinel-2 before, Sentinel-2 after,
AlphaEarth embedding (2yr before install), RF P(solar) linear, RF P(solar) nonlinear (power gamma).
Reuses cached Sentinel-2 (no GEE). Column headers on the top row, site labels down the left.
Saves fig_appendix_3x5.{png,pdf}.
"""
import os, warnings, numpy as np; warnings.filterwarnings("ignore")
os.chdir("/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting")
import rasterio, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from sklearn.decomposition import PCA

CH = "../Data/aef_solar_chips/positives/chips"; MK = "../Data/aef_solar_chips/positives/masks_v2"
RF = "artifacts/paper_v3/models/rf30_v3_R.joblib"
CACHE = "artifacts/paper_v3/figures/fig2a_candidates"
OUT = "artifacts/paper_v3/figures/fig_appendix_3x5"
GAMMA = 2.2
ROWS = [  # (chip, country label)
    ("POS_AUS_15366_2018_2.tif", "Australia"),
    ("POS_USA_1174_2022_4.tif",  "United States"),
    ("POS_PHL_14037_2019_2.tif", "Philippines"),
]
rf = joblib.load(RF)

def load(chip):
    with rasterio.open(f"{MK}/{chip}") as s: mask = s.read(1)
    with rasterio.open(f"{CH}/{chip}") as s: aef = s.read().astype(np.float32)
    C, H, W = aef.shape
    flat = aef.reshape(C, -1).T; fin = np.isfinite(flat).all(1)
    rgb = np.full((H*W, 3), np.nan, np.float32)
    rgb[fin] = PCA(3, random_state=0).fit_transform(flat[fin]); rgb = rgb.reshape(H, W, 3)
    for i in range(3):
        lo, hi = np.nanpercentile(rgb[..., i], [2, 98]); rgb[..., i] = np.clip((rgb[..., i]-lo)/(hi-lo+1e-9), 0, 1)
    aef_rgb = np.nan_to_num(rgb)
    pred = np.full(H*W, np.nan, np.float32); pred[fin] = rf.predict_proba(flat[fin])[:, 1]
    pred = pred.reshape(H, W)
    z = np.load(f"{CACHE}/_s2_{chip}.npz", allow_pickle=True)
    pre = z["pre"]; post = z["post"]
    pre = None if pre.dtype == object else pre; post = None if post.dtype == object else post
    return mask, aef_rgb, pred, pre, post

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})
nrow, ncol = len(ROWS), 5
fig, axs = plt.subplots(nrow, ncol, figsize=(12.6, 2.55*nrow))
fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.06)

def show(ax, img, mask, cmap=None, vmin=None, vmax=None, norm=None):
    if img is None:
        ax.text(0.5, 0.5, "S2 n/a", ha="center", va="center", fontsize=9); ax.set_facecolor("0.9")
    else:
        if norm is not None: ax.imshow(img, cmap=cmap, norm=norm, aspect="auto")
        else: ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        mh, mw = mask.shape
        ax.contour(np.linspace(0, (img.shape[1] if img.ndim > 1 else mw)-1, mw),
                   np.linspace(0, (img.shape[0] if img.ndim > 1 else mh)-1, mh),
                   mask > 0, levels=[0.5], colors="cyan", linewidths=0.9)
    ax.set_xticks([]); ax.set_yticks([])

COLTITLES = ["Sentinel-2 (before)", "Sentinel-2 (after)", "AlphaEarth", "RF $P$(solar)",
             f"RF $P$(solar) nonlinear ($\\gamma$={GAMMA:g})"]
for r, (chip, country) in enumerate(ROWS):
    mask, aef_rgb, pred, pre, post = load(chip)
    emb = int(chip.split("_")[3])
    show(axs[r, 0], pre, mask)
    show(axs[r, 1], post, mask)
    show(axs[r, 2], aef_rgb, mask)
    show(axs[r, 3], pred, mask, cmap="magma", vmin=0, vmax=1)
    imP = show(axs[r, 4], pred, mask, cmap="magma", norm=PowerNorm(gamma=GAMMA, vmin=0, vmax=1))
    axs[r, 0].set_ylabel(f"{country}\n(install {emb+2}; imagery {emb}/{emb+3})",
                         fontsize=8.7, labelpad=4)
    print(f"{country} {chip}: P_in {np.nanmean(pred[mask>0]):.2f} P_bg {np.nanmean(pred[mask==0]):.2f} "
          f"S2 {'ok' if pre is not None else 'NA'}", flush=True)
for c in range(ncol):
    axs[0, c].set_title(COLTITLES[c], fontsize=9.3, pad=4)
fig.savefig(f"{OUT}.png", dpi=170, bbox_inches="tight")
fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
print(f"saved {OUT}.png/.pdf", flush=True)

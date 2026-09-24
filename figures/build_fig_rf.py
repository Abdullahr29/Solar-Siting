"""RF validation figure (paper).
LEFT box (A): 2x2 chip tiles for one solar site -- Sentinel-2 before / after (panels built),
AlphaEarth embedding, RF P(solar); footprint outlined on each.
RIGHT (B): large geographically-proportioned Greece suitability map + post-2021 forward sites,
with a histogram inset (magma-scheme colours) bottom-right.
Saves fig_rf_multipanel.{png,pdf}.
"""
import os, warnings, numpy as np
warnings.filterwarnings("ignore")
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import rasterio, rasterio.warp, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
import matplotlib.patheffects as pe
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import pystac_client, planetary_computer, odc.stac

CH = "Data/aef_solar_chips/positives/chips"; MK = "Data/aef_solar_chips/positives/masks_v2"
RF = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
CTRY = "Solar-Siting/artifacts/paper_v3/scores/country"
OUT = "Solar-Siting/artifacts/paper_v3/figures/fig_rf_multipanel"
CHIP = os.environ.get("CHIP", "POS_USA_489_2019_20.tif")
EMB = int(CHIP.split("_")[3])

# ---- chip data ----
with rasterio.open(f"{MK}/{CHIP}") as s: mask = s.read(1)
with rasterio.open(f"{CH}/{CHIP}") as s:
    aef = s.read().astype(np.float32); bounds, crs = s.bounds, s.crs
C, H, W = aef.shape
flat = aef.reshape(C, -1).T; fin = np.isfinite(flat).all(1)
rgb = np.full((H * W, 3), np.nan, np.float32)
rgb[fin] = PCA(3, random_state=0).fit_transform(flat[fin])
rgb = rgb.reshape(H, W, 3)
for i in range(3):
    lo, hi = np.nanpercentile(rgb[..., i], [2, 98]); rgb[..., i] = np.clip((rgb[..., i] - lo) / (hi - lo + 1e-9), 0, 1)
aef_rgb = np.nan_to_num(rgb)
rf = joblib.load(RF)
pred = np.full(H * W, np.nan, np.float32); pred[fin] = rf.predict_proba(flat[fin])[:, 1]
pred = pred.reshape(H, W)
print(f"{CHIP}: frac {(mask>0).mean():.2f} pred[mask] {np.nanmean(pred[mask>0]):.2f} bg {np.nanmean(pred[mask==0]):.2f}", flush=True)

# ---- Sentinel-2 before (EMB) + after (EMB+3) via Google Earth Engine (cached) ----
import time, io, requests
lo_lon, lo_lat, hi_lon, hi_lat = rasterio.warp.transform_bounds(crs, "EPSG:4326", *bounds)
S2CACHE = f"Solar-Siting/artifacts/paper_v3/figures/_s2cache_{CHIP}.npz"
if os.path.exists(S2CACHE):
    c = np.load(S2CACHE); s2_pre, s2_post = c["pre"], c["post"]
    print(f"S2 from cache {S2CACHE}", flush=True)
else:
    import ee
    ee.Initialize(project="ee-abdullahr-solar")
    region = ee.Geometry.Rectangle([float(lo_lon), float(lo_lat), float(hi_lon), float(hi_lat)])
    def s2(yr):
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
               .filterDate(f"{yr}-05-01", f"{yr}-09-30")
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
        img = col.median().select(["B4", "B3", "B2"])
        for att in range(4):
            try:
                url = img.getThumbURL({"region": region, "dimensions": 320, "min": 0, "max": 3000, "format": "png"})
                a = np.array(Image.open(io.BytesIO(requests.get(url, timeout=180).content))).astype(np.float32)
                return np.clip(a[:, :, :3] / 255.0, 0, 1)
            except Exception as e:
                print(f"  S2 {yr} attempt {att} failed: {str(e)[:80]}; retrying", flush=True); time.sleep(10)
        raise RuntimeError(f"S2 {yr} failed")
    s2_pre = s2(EMB); s2_post = s2(EMB + 3)
    np.savez(S2CACHE, pre=s2_pre, post=s2_post)
print(f"S2 pre {EMB} {s2_pre.shape}, post {EMB+3} {s2_post.shape}", flush=True)

# ---- Greece country: inventory forward (>=2021) + independent TZ-SAM = all validation sites ----
nat = np.array(Image.open(f"{CTRY}/greece_R_2019.png"))
z = np.load(f"{CTRY}/greece_R_2019_scores.npz")
site_xy, site_s, rand_s, ext = z["site_xy"], z["site_s"], z["rand_s"], z["ext"]
tz = np.load("Solar-Siting/artifacts/paper_v3/tzsam/greece_tzsam_scores.npz")
all_xy = np.vstack([site_xy, tz["tz_xy"]]); all_s = np.concatenate([site_s, tz["tz_s"]])
roc = roc_auc_score(np.r_[np.ones(len(all_s)), np.zeros(len(rand_s))], np.r_[all_s, rand_s])
print(f"Greece sites: {len(site_s)} inventory + {len(tz['tz_s'])} TZ-SAM = {len(all_s)}; combined ROC {roc:.3f}", flush=True)
C_NEG, C_POS = cm.magma(0.20), cm.magma(0.82)  # heatmap-scheme low/high

# ---- figures (two separate square panels, stitched in LaTeX) ----
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})

# ===== Figure 1: 2x2 chip panel (tiles fill the space) =====
fig1, axs = plt.subplots(2, 2, figsize=(5.3, 5.5))
fig1.subplots_adjust(left=0.004, right=0.996, top=0.945, bottom=0.006, wspace=0.03, hspace=0.10)

def show(ax, img, title, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    mh, mw = mask.shape
    ax.contour(np.linspace(0, img.shape[1] - 1, mw), np.linspace(0, img.shape[0] - 1, mh),
               mask > 0, levels=[0.5], colors="cyan", linewidths=1.0)
    ax.set_title(title, fontsize=9.5, pad=3); ax.set_xticks([]); ax.set_yticks([])
    return im

show(axs[0, 0], s2_pre, f"Sentinel-2 · {EMB} (before)")
show(axs[0, 1], s2_post, f"Sentinel-2 · {EMB + 3} (after)")
show(axs[1, 0], aef_rgb, f"AlphaEarth · {EMB}")
imP = show(axs[1, 1], pred, "RF $P$(solar)", cmap="magma", vmin=0, vmax=1)
# horizontal colorbar inside the RF tile, along the bottom
cax = axs[1, 1].inset_axes([0.10, 0.075, 0.80, 0.05])
cb = fig1.colorbar(imP, cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=7, colors="white", pad=1, length=2)
cb.outline.set_edgecolor("white"); cb.outline.set_linewidth(0.6)
fig1.savefig(f"{OUT}_chips.png", dpi=200, bbox_inches="tight")
fig1.savefig(f"{OUT}_chips.pdf", bbox_inches="tight"); plt.close(fig1)
print(f"saved {OUT}_chips", flush=True)

# ===== Figure 2: Greece suitability map (square) =====
fig2, axM = plt.subplots(figsize=(5.5, 5.5))
fig2.subplots_adjust(left=0.008, right=0.992, top=0.95, bottom=0.008)
axM.imshow(nat, extent=[ext[0], ext[1], ext[2], ext[3]], origin="upper")
axM.set_aspect(1.0 / np.cos(np.radians((ext[2] + ext[3]) / 2)))
axM.scatter(all_xy[:, 0], all_xy[:, 1], s=4, c="#2b8cff", edgecolors="none", alpha=0.9,
            label=f"validation sites (n={len(all_s)})")
axM.axis("off")
# bold panel label "b" in the top-right corner (style matches Figs 3/4)
axM.text(0.01, 0.99, "b", transform=axM.transAxes, fontsize=13, fontweight="bold",
         va="top", ha="left", color="black",
         path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
axM.legend(loc="lower right", fontsize=9, framealpha=0.9, markerscale=2.5, handletextpad=0.3)
# histogram inset, bottom-left
axH = inset_axes(axM, width="40%", height="30%", loc="lower left", borderpad=0.6)
bins = np.linspace(0, 1, 40)
axH.hist(rand_s, bins=bins, density=True, color=to_hex(C_NEG), alpha=0.9, label="random land")
axH.hist(all_s, bins=bins, density=True, color=to_hex(C_POS), alpha=0.85, label="future solar")
axH.set_xlabel("RF $P$(solar)", fontsize=6.5, labelpad=1); axH.set_yticks([])
axH.tick_params(axis="x", labelsize=6, pad=1)
axH.legend(fontsize=5.5, loc="upper left", handlelength=0.9, handletextpad=0.4,
           framealpha=0.85, borderpad=0.3, labelspacing=0.25)
axH.text(0.5, 0.93, f"ROC {roc:.3f}", transform=axH.transAxes, ha="center", va="top", fontsize=6.5)
for sp in ("top", "right"): axH.spines[sp].set_visible(False)
axH.patch.set_alpha(0.9)
fig2.savefig(f"{OUT}_greece.png", dpi=300, bbox_inches="tight")
fig2.savefig(f"{OUT}_greece.pdf", dpi=300, bbox_inches="tight"); plt.close(fig2)
print(f"saved {OUT}_greece", flush=True)

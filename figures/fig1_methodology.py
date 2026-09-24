"""Fig 1 methodology-schematic DRAFTS from a real AEF chip.

The engine: a solar site -> (2 yr before) 64-band AlphaEarth embedding -> RF -> per-pixel P(new solar).
Builds panels from a real positive chip (64-band AEF + masks_v2) + the deployed RF-R, in 2 layouts.

Usage: python fig1_methodology.py POS_IND_10860_2018_24.tif
"""
import os, sys
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
CHIP = sys.argv[1]
CHIPD = "Data/aef_solar_chips/positives/chips"
MASKD = "Data/aef_solar_chips/positives/masks"
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
OUTDIR = "Solar-Siting/artifacts/paper_v3/figures/fig1"


def norm(a):
    a = a.astype(float)
    lo, hi = np.nanpercentile(a, [2, 98])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    os.chdir(WS)
    import rasterio, joblib
    from sklearn.decomposition import PCA
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow

    with rasterio.open(f"{CHIPD}/{CHIP}") as s:
        emb = s.read().astype(np.float32)          # (64, H, W)
    with rasterio.open(f"{MASKD}/{CHIP}") as s:
        mask = s.read(1)
    C, H, Wd = emb.shape
    X = emb.reshape(C, -1).T                        # (H*W, 64)

    # PCA -> RGB false colour of the 64-band embedding
    pcs = PCA(3, random_state=0).fit_transform(X)
    rgb = np.stack([norm(pcs[:, i].reshape(H, Wd)) for i in range(3)], -1)

    # RF probability heatmap
    rf = joblib.load(MODEL)
    prob = rf.predict_proba(X)[:, 1].reshape(H, Wd)

    # per-pixel 64-vector (a solar pixel) for the "blocky vector" motif
    ys, xs = np.where(mask > 0)
    if len(ys):
        vec = emb[:, ys[len(ys)//2], xs[len(xs)//2]]
    else:
        vec = emb[:, H//2, Wd//2]

    os.makedirs(OUTDIR, exist_ok=True)
    stem = CHIP.replace(".tif", "")

    # ---------- Layout A: horizontal pipeline ----------
    figA, ax = plt.subplots(1, 5, figsize=(22, 5), gridspec_kw={"width_ratios": [1, 0.5, 1, 1, 1]})
    ax[0].imshow(rgb); ax[0].set_title("① AlphaEarth embedding\n(64-band, PCA→RGB), 2 yr pre-install", fontsize=11)
    # vector motif
    ax[1].imshow(vec.reshape(-1, 1), aspect="auto", cmap="viridis")
    ax[1].set_title("② per-pixel\n64-vector", fontsize=11); ax[1].set_xticks([])
    ax[1].set_yticks([0, 63]); ax[1].set_ylabel("64 dims")
    ax[2].text(0.5, 0.5, "③\nRandom Forest\n(learned\ndiscriminator)", ha="center", va="center",
               fontsize=13, bbox=dict(boxstyle="round", fc="#e9c46a")); ax[2].axis("off")
    im = ax[3].imshow(prob, cmap="magma", vmin=0, vmax=1); ax[3].set_title("④ P(new solar)\nper-pixel suitability", fontsize=11)
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    ax[4].imshow(mask, cmap="Greens"); ax[4].set_title("⑤ ground truth\n(masks_v2 install)", fontsize=11)
    for a in [ax[0], ax[3], ax[4]]:
        a.set_xticks([]); a.set_yticks([])
    figA.suptitle("The engine — from AlphaEarth embedding to per-pixel solar suitability", fontsize=14)
    figA.tight_layout(); figA.savefig(f"{OUTDIR}/{stem}_A_pipeline.png", dpi=130, bbox_inches="tight"); plt.close(figA)

    # ---------- Layout B: PCA + channel grid + outputs ----------
    figB = plt.figure(figsize=(18, 9))
    gs = figB.add_gridspec(2, 4)
    a0 = figB.add_subplot(gs[0, 0]); a0.imshow(rgb); a0.set_title("AlphaEarth embedding (PCA→RGB)")
    a1 = figB.add_subplot(gs[0, 1]); im1 = a1.imshow(prob, cmap="magma", vmin=0, vmax=1); a1.set_title("RF P(new solar)")
    plt.colorbar(im1, ax=a1, fraction=0.046)
    a2 = figB.add_subplot(gs[0, 2]); a2.imshow(mask, cmap="Greens"); a2.set_title("masks_v2 ground truth")
    a3 = figB.add_subplot(gs[0, 3]); a3.imshow(vec.reshape(-1, 1), aspect="auto", cmap="viridis")
    a3.set_title("one pixel's\n64-vector"); a3.set_xticks([]); a3.set_yticks([0, 63])
    # channel grid: 12 example embedding channels
    for i, ch in enumerate(np.linspace(0, C - 1, 12).astype(int)):
        aa = figB.add_subplot(gs[1, i // 3].get_subplotspec()) if False else None
    # simpler: a 3x4 inset of channels in the bottom row spanning all 4 cols
    for j, ch in enumerate(np.linspace(0, C - 1, 8).astype(int)):
        axc = figB.add_axes([0.06 + 0.115 * j, 0.06, 0.10, 0.36])
        axc.imshow(norm(emb[ch]), cmap="gray"); axc.set_title(f"band {ch}", fontsize=8)
        axc.set_xticks([]); axc.set_yticks([])
    for a in [a0, a1, a2]:
        a.set_xticks([]); a.set_yticks([])
    figB.suptitle("AlphaEarth 64-band embedding → RF → suitability (8 of 64 bands shown below)", fontsize=14)
    figB.savefig(f"{OUTDIR}/{stem}_B_grid.png", dpi=130, bbox_inches="tight"); plt.close(figB)

    print(f"saved {OUTDIR}/{stem}_A_pipeline.png + _B_grid.png  (solar frac {float((mask>0).mean()):.2f})")


if __name__ == "__main__":
    main()

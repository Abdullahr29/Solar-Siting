"""Capacity + generation coverage figure (results).
(a) cumulative capacity & generation captured vs fraction of most-suitable land (Lorenz-style);
(b) per-capacity-size-bin site-vs-random ROC (the large-farm limitation).
-> figures/fig_coverage.{png,pdf}
"""
import os, numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
P3 = "Solar-Siting/artifacts/paper_v3"
FIG = f"{P3}/figures"

cap = pd.read_csv(f"{P3}/figA_combined/capacity_coverage.csv")   # capacity = TZ-SAM (set-specific)
gen = np.load(f"{P3}/generation/gen_join.npz", allow_pickle=True)
gx, gc = gen["Xf"] * 100, gen["capt"] * 100                       # generation = GPVF (set-specific)
sb = pd.read_csv(f"{P3}/figA_combined/capacity_sizebin_roc.csv")  # size_mw, n, roc, ci_lo, ci_hi

# --- sites-captured curve over the COMBINED validation set (inventory + TZ-SAM), global-threshold ---
# (the sites line is the general validation metric, so it pools both inventories for homogeneity;
#  capacity/generation stay on their set-specific sources above.)
import glob as _glob
_S, _L = [], []
for f in _glob.glob(f"{P3}/scores/country/*_R_2019_scores.npz"):
    c = os.path.basename(f).split("_R_2019")[0]
    z = np.load(f); _L += list(z["rand_s"].astype(float)); _S += list(z["site_s"].astype(float))
    tf = f"{P3}/tzsam/{c}_tzsam_scores.npz"
    if os.path.exists(tf): _S += list(np.load(tf)["tz_s"].astype(float))
_S, _L = np.array(_S), np.array(_L)
site_x = np.arange(1, 101)
site_y = np.array([100 * (_S >= np.percentile(_L, 100 - p)).mean() for p in site_x])
print(f"combined sites: n={len(_S)}  top10={np.interp(10,site_x,site_y):.1f}%  top20={np.interp(20,site_x,site_y):.1f}%", flush=True)

C_SITE, C_CAP, C_GEN = cm.magma(0.30), cm.magma(0.55), cm.magma(0.80)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.7})
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.7))
fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.15, wspace=0.28)

# --- (a) coverage curves ---
a.plot([0, 100], [0, 100], ls="--", c="0.6", lw=1, label="random land")
a.plot(site_x, site_y, c=C_SITE, lw=2, label="solar sites")
a.plot(cap["top_pct_land"], cap["pct_capacity_captured"], c=C_CAP, lw=2, label="installed capacity")
a.plot(gx, gc, c=C_GEN, lw=2, label="estimated generation")
for x0 in (10, 20):
    syi = np.interp(x0, site_x, site_y)
    cyi = np.interp(x0, cap["top_pct_land"], cap["pct_capacity_captured"])
    gyi = np.interp(x0, gx, gc)
    a.plot([x0, x0], [0, max(syi, cyi, gyi)], c="0.8", lw=0.6, zorder=0)   # vertical guide at 10/20%
    for yv, cc in ((syi, C_SITE), (cyi, C_CAP), (gyi, C_GEN)):             # faint dashed curve-coloured drop to y-axis
        a.plot([0, x0], [yv, yv], c=cc, lw=0.6, alpha=0.4, ls=(0, (4, 3)), zorder=0)
    a.scatter([x0, x0, x0], [syi, cyi, gyi], c=[C_SITE, C_CAP, C_GEN], s=14, zorder=5)
a.set_xlim(0, 100); a.set_ylim(0, 100)
a.set_xlabel("most-suitable land (%)"); a.set_ylabel("cumulative share captured (%)")
a.set_yticks(range(10, 100, 20), minor=True)                              # ticks at odd 10s (10,30,50,70,90)
a.tick_params(axis="y", which="minor", length=3.5)
a.legend(fontsize=7.5, loc="lower right")
a.text(-0.14, 1.02, "a", transform=a.transAxes, fontsize=13, fontweight="bold")

# --- (b) size-bin ROC ---
y = np.arange(len(sb))
b.barh(y, sb["roc"], color=cm.magma(np.linspace(0.3, 0.8, len(sb))),
       xerr=[sb["roc"] - sb["ci_lo"], sb["ci_hi"] - sb["roc"]], error_kw=dict(lw=0.8, capsize=2))
for i, r in sb.iterrows():
    b.text(r["ci_hi"] + 0.012, i, f"{r['roc']:.2f}", va="center", fontsize=7)  # clear of the error-bar cap
b.set_yticks(y); b.set_yticklabels([f"{s} MW\n(n={n})" for s, n in zip(sb["size_mw"], sb["n"])], fontsize=7.5)
b.set_xlim(0.5, 1.0); b.axvline(0.5, c="0.6", lw=0.5)
b.set_xlabel("site-vs-random ROC-AUC")
b.text(-0.22, 1.02, "b", transform=b.transAxes, fontsize=13, fontweight="bold")

fig.savefig(f"{FIG}/fig_coverage.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/fig_coverage.pdf", dpi=300, bbox_inches="tight")
print("saved fig_coverage | cap@10/20:",
      round(float(np.interp(10, cap['top_pct_land'], cap['pct_capacity_captured'])), 1),
      round(float(np.interp(20, cap['top_pct_land'], cap['pct_capacity_captured'])), 1),
      "| gen@10/20:", round(float(np.interp(10, gx, gc)), 1), round(float(np.interp(20, gx, gc)), 1), flush=True)

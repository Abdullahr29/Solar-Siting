"""Appendix weak-country figure (magma scheme): percentile-normalisation on the dimmest country
(India) lifts readability without changing skill. (a) raw score histogram; (b) raw suitability map;
(c) country-percentile map. -> figures/fig_weakcountry.{png,pdf}
"""
import os, numpy as np
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from sklearn.metrics import roc_auc_score
CO = "Solar-Siting/artifacts/paper_v3/scores/country"
FIG = "Solar-Siting/artifacts/paper_v3/figures"

z = np.load(f"{CO}/india_R_2019_scores.npz")
site_s, rand_s = z["site_s"], z["rand_s"]
rand_xy = z["rand_xy"]
auc = roc_auc_score(np.r_[np.ones(len(site_s)), np.zeros(len(rand_s))], np.r_[site_s, rand_s])
# country percentile of each random-land point (rank within random land)
order = rand_s.argsort(); pct = np.empty_like(rand_s); pct[order] = np.linspace(0, 1, len(rand_s))
C_NEG, C_POS = cm.magma(0.20), cm.magma(0.82)
aspect = 1.0 / np.cos(np.radians(float(rand_xy[:, 1].mean())))

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.7})
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.13, wspace=0.18)

# (a) raw histogram
a = axs[0]; bins = np.linspace(0, max(rand_s.max(), site_s.max()), 40)
a.hist(rand_s, bins=bins, density=True, color=to_hex(C_NEG), alpha=0.9, label="random land")
a.hist(site_s, bins=bins, density=True, color=to_hex(C_POS), alpha=0.85, label="solar sites")
a.set_xlabel("RF $P$(solar), raw"); a.set_yticks([]); a.set_title(f"Raw scores (ROC-AUC {auc:.3f})", fontsize=9.5)
a.legend(fontsize=7.5, loc="upper center")
for sp in ("top", "right"): a.spines[sp].set_visible(False)
a.text(-0.05, 1.04, "a", transform=a.transAxes, fontsize=13, fontweight="bold")

def mapp(ax, c, title, lab):
    ax.scatter(rand_xy[:, 0], rand_xy[:, 1], c=c, cmap="magma", s=6, vmin=0, vmax=1, edgecolors="none")
    ax.set_aspect(aspect); ax.axis("off"); ax.set_title(title, fontsize=9.5)
    ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")

# scale raw to 0-1 by its own 2-98 for display parity
lo, hi = np.percentile(rand_s, [2, 98]); raw01 = np.clip((rand_s - lo) / (hi - lo + 1e-9), 0, 1)
mapp(axs[1], raw01, "Raw suitability (compressed)", "b")
mapp(axs[2], pct, "Country-percentile (readable)", "c")
fig.savefig(f"{FIG}/fig_weakcountry.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{FIG}/fig_weakcountry.pdf", bbox_inches="tight")
print(f"saved fig_weakcountry; India AUC {auc:.3f}", flush=True)

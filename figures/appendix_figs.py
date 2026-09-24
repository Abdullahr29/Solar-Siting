"""Appendix figures: (1) 13-country forward ROC curves; (2) leakage-controlled ablations
(LOCO spatial + temporal holdouts) vs the in-sample forward, per country. All recomputed from
the saved score npz for consistency. -> figures/appendix_roc_curves.{png,pdf}, appendix_ablations.{png,pdf}
"""
import os, numpy as np
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, roc_auc_score
CO = "Solar-Siting/artifacts/paper_v3/scores/country"
LO = "Solar-Siting/artifacts/paper_v3/scores/loco"
TE = "Solar-Siting/artifacts/paper_v3/scores/temporal"
FIG = "Solar-Siting/artifacts/paper_v3/figures"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})

def roc_xy(path):
    z = np.load(path); s, r = z["site_s"], z["rand_s"]
    y = np.r_[np.ones(len(s)), np.zeros(len(r))]; x = np.r_[s, r]
    fpr, tpr, _ = roc_curve(y, x)
    return fpr, tpr, roc_auc_score(y, x)

# ---------- (1) 13-country forward ROC curves ----------
COUNTRIES = [("greece", "GRC"), ("germany", "DEU"), ("china", "CHN"), ("spain", "ESP"),
             ("poland", "POL"), ("japan", "JPN"), ("india", "IND"), ("united_states", "USA"),
             ("south_africa", "ZAF"), ("colombia", "COL"), ("philippines", "PHL"),
             ("malaysia", "MYS"), ("chile", "CHL")]
fwd = {}
fig, ax = plt.subplots(figsize=(5.0, 5.0))
colors = cm.turbo(np.linspace(0.05, 0.95, len(COUNTRIES)))
order = []
for (name, iso) in COUNTRIES:
    fpr, tpr, a = roc_xy(f"{CO}/{name}_R_2019_scores.npz"); fwd[iso] = a
    order.append((a, iso, fpr, tpr))
for (a, iso, fpr, tpr), c in zip(sorted(order, reverse=True), colors):
    ax.plot(fpr, tpr, color=c, lw=1.3, label=f"{iso}  {a:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.6)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("Per-country forward ROC (inventory installs $\\geq$2021)", fontsize=9)
ax.legend(fontsize=6.5, ncol=2, loc="lower right", title="ROC-AUC", title_fontsize=7)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(f"{FIG}/appendix_roc_curves.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/appendix_roc_curves.pdf", dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved appendix_roc_curves", flush=True)

# ---------- (2) ablations: in-sample forward vs held-out ----------
LOCO = ["DEU", "ESP", "POL", "CHN", "IND", "ZAF", "CHL", "PHL"]   # have both forward + LOCO
TEMP = ["CHN", "USA", "IND", "ESP"]                                 # have both forward + temporal
loco_roc = {iso: roc_xy(f"{LO}/{iso}_R_2019_scores.npz")[2] for iso in LOCO if os.path.exists(f"{LO}/{iso}_R_2019_scores.npz")}
temp_roc = {iso: roc_xy(f"{TE}/{iso}_R_2022_scores.npz")[2] for iso in TEMP if os.path.exists(f"{TE}/{iso}_R_2022_scores.npz")}

fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
def paired(ax, isos, held, title, held_label):
    isos = [i for i in isos if i in held]
    isos = sorted(isos, key=lambda i: fwd[i])
    y = np.arange(len(isos)); h = 0.38
    ax.barh(y + h/2, [fwd[i] for i in isos], height=h, color=cm.magma(0.75), label="full model")
    ax.barh(y - h/2, [held[i] for i in isos], height=h, color=cm.magma(0.30), label=held_label)
    for j, i in enumerate(isos):
        ax.text(fwd[i] + .005, y[j] + h/2, f"{fwd[i]:.2f}", va="center", fontsize=6)
        ax.text(held[i] + .005, y[j] - h/2, f"{held[i]:.2f}", va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels(isos); ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, c="grey", lw=.5); ax.set_xlabel("site-vs-random ROC-AUC")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=5.5, loc="lower right", handlelength=1.0, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.3, framealpha=0.9)
paired(axs[0], LOCO, loco_roc, "Spatial hold-out (leave-one-country-out)", "country ablation")
paired(axs[1], TEMP, temp_roc, "Temporal hold-out (train $\\leq$2021, validate 2024)", "temporal ablation")
fig.tight_layout()
fig.savefig(f"{FIG}/appendix_ablations.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/appendix_ablations.pdf", dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved appendix_ablations", flush=True)
print("LOCO:", {k: round(v, 3) for k, v in loco_roc.items()})
print("TEMP:", {k: round(v, 3) for k, v in temp_roc.items()})

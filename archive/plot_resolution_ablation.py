"""Closing figure for the Greece sampling-resolution ablation (embedding models).

One self-describing panel: RF suitability ROC vs sampling resolution, three aggregation
methods, native-10 m baseline drawn as the reference. Data are the two same-baseline
runs (fine 10-100 m + the 300 m point), n_rand=3000, AEF 2021, forward installs >=2023.
Writes a consolidated CSV + PNG to artifacts/{results,figures/resolution/greece}/.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.expanduser("~/Solar_Workspace"))

# res(m): (A_raw, A_renorm, B_scoremean)   [R=10 = native single-pixel baseline]
DATA = {
    10:  (0.8732, 0.8732, 0.8732),
    30:  (0.8713, 0.8715, 0.8721),
    50:  (0.8662, 0.8674, 0.8702),
    70:  (0.8639, 0.8643, 0.8688),
    100: (0.8620, 0.8617, 0.8647),
    300: (0.8447, 0.8463, 0.8481),
}
BASE = DATA[10][0]
res = np.array(sorted(DATA))
a_raw = np.array([DATA[r][0] for r in res])
a_ren = np.array([DATA[r][1] for r in res])
b_sm  = np.array([DATA[r][2] for r in res])

# consolidated results CSV
os.makedirs("Solar-Siting/artifacts/results", exist_ok=True)
with open("Solar-Siting/artifacts/results/resolution_ablation_greece_full.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["resolution_m", "A_raw", "A_renorm", "B_scoremean", "delta_vs_native"])
    for r in res:
        w.writerow([r, DATA[r][0], DATA[r][1], DATA[r][2], round(DATA[r][2] - BASE, 4)])

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.2, 5.2))

ax.axhline(BASE, ls="--", lw=1.3, color="#555", zorder=1)
ax.annotate(f"native 10 m baseline = {BASE:.3f}", (res[-1], BASE), xytext=(0, 6),
            textcoords="offset points", ha="right", va="bottom", fontsize=9, color="#555")

ax.plot(res, b_sm,  "-o", color="#2a78d6", lw=2, ms=6, label="B: score-mean (classify 10 m, then avg score)")
ax.plot(res, a_ren, "-s", color="#eda100", lw=2, ms=6, label="A: mean-embedding, renormalised")
ax.plot(res, a_raw, "-^", color="#c98500", lw=1.8, ms=6, label="A: mean-embedding, raw")

ax.scatter([10], [BASE], s=130, facecolor="white", edgecolor="#111", zorder=5, linewidth=1.6)
ax.annotate("optimum", (10, BASE), xytext=(8, -16), textcoords="offset points", fontsize=9)

ax.set_xscale("log")
ax.set_xticks(res); ax.set_xticklabels([str(r) for r in res])
ax.set_xlabel("sampling resolution — embedding aggregation cell (m, log scale)")
ax.set_ylabel("suitability ROC (forward installs ≥ 2023 vs random-land)")
ax.set_title("Greece resolution ablation — native 10 m is optimal; coarsening only degrades\n"
             "AEF 2021, rf30_final, n_rand=3000  (embedding models, Model 1–2 close-out)",
             fontsize=11)
ax.legend(loc="lower left", fontsize=9, frameon=False)
ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
ax.set_ylim(0.84, 0.877)

outdir = "Solar-Siting/artifacts/figures/resolution/greece"
os.makedirs(outdir, exist_ok=True)
out = f"{outdir}/roc_vs_resolution_greece.png"
fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print("wrote Solar-Siting/artifacts/results/resolution_ablation_greece_full.csv")

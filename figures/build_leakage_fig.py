"""Summarise the -2 vs -3 AEF-lag leakage test in one wide 2-panel figure + a table.
Numbers are final (artifacts/paper_v3/temporal_probe/big/leakage_summary.txt).

Panel A: deep-lookback -- ROC vs lead-time on FIXED AEF 2017 (predict installs 2..7yr forward).
         Flat = pre-existing-land signal; a climb toward short lead would mean near-install leakage.
Panel B: matched-land discrimination -- status-quo (-2 on -2) vs honest (-3 on -3), for both the
         training-site holdout (2x2) and the REAL held-out 2024 installs (referee). Shows the -3
         model matches -2 on matched land (leakage premium ~1 ROC pt / ~0 on real installs).

Saves -> artifacts/paper_v3/figures/leakage_lag/{fig_leakage_lag.png,.pdf, leakage_results.csv}
"""
import os, csv, warnings; warnings.filterwarnings("ignore")
os.chdir(os.path.expanduser("~/jasmin-mount/Solar-Siting"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts/paper_v3/figures/leakage_lag"
os.makedirs(OUT, exist_ok=True)

# ---- final numbers ----
leads = [2, 3, 4, 5, 6, 7]
deep = {
    "RF trained $-3$ (leak-reduced)": [0.9369, 0.9349, 0.9338, 0.9380, 0.9178, 0.8793],
    "RF trained $-2$ (status quo)":   [0.9347, 0.9278, 0.9265, 0.9347, 0.9101, 0.8715],
    "Deployed RF (full pool)":        [0.9567, 0.9527, 0.9515, 0.9499, 0.9320, 0.8988],
}
deep_overall = {"RF trained $-3$ (leak-reduced)": 0.923, "RF trained $-2$ (status quo)": 0.918,
                "Deployed RF (full pool)": 0.940}
col = {"RF trained $-3$ (leak-reduced)": "#1a9850", "RF trained $-2$ (status quo)": "#d73027",
       "Deployed RF (full pool)": "#4575b4"}

# matched-land: (status-quo -2on-2, honest -3on-3)
holdout = {"$-2$ on $-2$ land": 0.8942, "$-3$ on $-3$ land": 0.8850}   # training-site 2x2 diagonal
referee = {"$-2$ on $-2$ land": 0.9141, "$-3$ on $-3$ land": 0.9137}   # real held-out 2024 installs

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.7})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.15, wspace=0.26)

# ===== Panel A: deep-lookback ROC vs lead-time =====
for name, ys in deep.items():
    axA.plot(leads, ys, "-o", color=col[name], lw=2.0, ms=5, label=f"{name}  (all {deep_overall[name]:.3f})")
axA.set_xlabel("lead time: years from AEF-2017 imagery to installation")
axA.set_ylabel("ROC AUC")
axA.set_title("(a) Deep-lookback on fixed AEF 2017 — leak-free forward prediction", fontsize=9.5, loc="left")
axA.set_ylim(0.85, 0.97); axA.set_xticks(leads)
axA.grid(alpha=0.25, lw=0.6)
axA.legend(fontsize=7.6, loc="lower left", framealpha=0.9)
axA.annotate("imagery predates installation by up to 7 yr —\ndiscrimination holds ⇒ pre-existing-land signal,\nnot a construction detector",
             xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom", fontsize=7.0,
             color="0.25", linespacing=1.25)

# ===== Panel B: matched-land status-quo vs honest =====
groups = ["Training-site\nhold-out (2×2)", "Real held-out\n2024 installs (referee)"]
x = np.arange(len(groups)); w = 0.34
v2 = [holdout["$-2$ on $-2$ land"], referee["$-2$ on $-2$ land"]]
v3 = [holdout["$-3$ on $-3$ land"], referee["$-3$ on $-3$ land"]]
b2 = axB.bar(x - w/2, v2, w, color="#d73027", label="$-2$ on $-2$ land (status quo)")
b3 = axB.bar(x + w/2, v3, w, color="#1a9850", label="$-3$ on $-3$ land (honest)")
for bars in (b2, b3):
    for b in bars:
        axB.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f"{b.get_height():.3f}",
                 ha="center", va="bottom", fontsize=7.3)
for i in range(len(groups)):
    d = v2[i]-v3[i]
    axB.text(x[i], min(v2[i], v3[i])-0.012, f"Δ={d:+.3f}", ha="center", va="top", fontsize=7.5, color="0.2")
axB.set_xticks(x); axB.set_xticklabels(groups, fontsize=8.3)
axB.set_ylabel("ROC AUC on matched pre-install land")
axB.set_ylim(0.86, 0.93)
axB.set_title("(b) Retraining at $-3$ costs ~0 on matched land", fontsize=9.5, loc="left")
axB.legend(fontsize=7.6, loc="upper left", framealpha=0.9)
axB.grid(axis="y", alpha=0.25, lw=0.6)

fig.suptitle("AEF sampling lag: is the RF a suitability model or a construction detector?  "
             "→ leakage premium ≈ 0.9 ROC pt; signal is durable pre-existing land",
             fontsize=10.5, y=0.985)
fig.savefig(f"{OUT}/fig_leakage_lag.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{OUT}/fig_leakage_lag.pdf", bbox_inches="tight")
print(f"saved {OUT}/fig_leakage_lag.png/.pdf")

# ---- results table ----
with open(f"{OUT}/leakage_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["block", "metric", "value"])
    w.writerow(["matched_2x2_trainhold", "ROC(-2 on -2)", 0.8942])
    w.writerow(["matched_2x2_trainhold", "ROC(-2 on -3)", 0.8765])
    w.writerow(["matched_2x2_trainhold", "ROC(-3 on -2)", 0.8896])
    w.writerow(["matched_2x2_trainhold", "ROC(-3 on -3)", 0.8850])
    w.writerow(["matched_2x2_trainhold", "leakage_premium(-2on-2 minus -3on-3)", 0.0092])
    w.writerow(["referee_real2024", "rf_m2 on t-2 land(2022)", 0.9141])
    w.writerow(["referee_real2024", "rf_m2 on t-3 land(2021)", 0.9083])
    w.writerow(["referee_real2024", "rf_m3 on t-2 land(2022)", 0.9039])
    w.writerow(["referee_real2024", "rf_m3 on t-3 land(2021)", 0.9137])
    w.writerow(["referee_real2024", "matched-diagonal gap (m2@t2 - m3@t3)", round(0.9141-0.9137, 4)])
    for name, ys in deep.items():
        tag = "rf_m3" if "-3" in name else ("rf_m2" if "-2" in name else "deployed")
        for L, y in zip(leads, ys):
            w.writerow([f"deep_lookback_AEF2017_{tag}", f"lead{L}", y])
        w.writerow([f"deep_lookback_AEF2017_{tag}", "overall", deep_overall[name]])
print(f"saved {OUT}/leakage_results.csv")

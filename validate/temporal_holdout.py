"""Temporal holdout (task 1b): honest future-forecast vs leaky in-sample.

Retrains RF-30 on installs <=2023 ONLY (extract_pixels --max-year 2021, i.e. AEF imagery year
<=2021, since install = AEF year + 2), then validates against 2024 installs on the AEF-2022 map
(2-year pre-install lead the model was trained with). Compares, per country:
  * honest : rf30_temporal (never saw any 2024 site)         scoring 2024 installs
  * leaky  : rf30_final     (trained ON those 2024 sites)     scoring the SAME 2024 installs
gap = leaky - honest = how much "trained through the present" overstates true forecast skill.

Tested where 2024 data is thick enough: China (5,916 sites) and USA (1,156). Single-year test:
the inventory stops at 2024. Outputs -> artifacts/figures/temporal/<tag>_2024/ + results/temporal.csv.
"""
import os, sys, csv, json, subprocess
import numpy as np
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, "Solar-Siting")
PY = sys.executable

POOL = "Solar-Siting/artifacts/stats/px_temporal_le2021.npz"
TEMPORAL = "Solar-Siting/artifacts/models/rf30_temporal_le2021.joblib"
FINAL = "Solar-Siting/artifacts/models/rf30_final.joblib"
os.makedirs("Solar-Siting/artifacts/results", exist_ok=True)

COUNTRIES = [dict(lsib="China", inv="China", tag="china"),
             dict(lsib="United States", inv="United States of America", tag="usa")]

def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd).returncode: sys.exit(f"FAILED: {' '.join(cmd)}")

# ---- stage 1+2: temporal pool (installs <=2023) + train, as subprocesses ----
if not os.path.exists(POOL):
    sh([PY, "Solar-Siting/extract_pixels.py", "--max-year", "2021", "--out", POOL])
else:
    print(f"pool exists, reuse: {POOL}", flush=True)
if not os.path.exists(TEMPORAL):
    sh([PY, "Solar-Siting/train_rf.py", "--pool", POOL, "--out", TEMPORAL])
else:
    print(f"model exists, reuse: {TEMPORAL}", flush=True)

from run_country_rf import run_country

def roc(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp/tp[-1]]; fpr = np.r_[0, fp/fp[-1]]
    return fpr, tpr, float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2))
def topk(pos, neg):
    pct = np.array([(neg < s).mean() for s in pos])
    return [(pct>=.8).mean()*100, (pct>=.9).mean()*100, (pct>=.95).mean()*100]

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

csvp = "Solar-Siting/artifacts/results/temporal.csv"
rows = list(csv.DictReader(open(csvp))) if os.path.exists(csvp) else []

for C in COUNTRIES:
    tag, lsib, inv = C["tag"], C["lsib"], C["inv"]
    FIG = f"Solar-Siting/artifacts/figures/temporal/{tag}_2024"
    os.makedirs(FIG, exist_ok=True)
    print(f"\n########## TEMPORAL {lsib} (2024 installs, AEF-2022 map) ##########", flush=True)
    m_h = run_country(lsib, year=2022, min_install=2024, model_path=TEMPORAL,
                      out_prefix=f"{FIG}/{tag}_honest_temporal_2022", inv_country=inv)
    m_l = run_country(lsib, year=2022, min_install=2024, model_path=FINAL,
                      out_prefix=f"{FIG}/{tag}_leaky_final_2022", inv_country=inv)
    H = np.load(f"{FIG}/{tag}_honest_temporal_2022_scores.npz")
    L = np.load(f"{FIG}/{tag}_leaky_final_2022_scores.npz")
    fh, th, ah = roc(H["site_s"], H["rand_s"]); fl, tl, al = roc(L["site_s"], L["rand_s"])
    htk, ltk = topk(H["site_s"], H["rand_s"]), topk(L["site_s"], L["rand_s"])

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fl, tl, c="C0", ls="--", lw=2, label=f"leaky: rf30_final (saw 2024)  ROC {al:.3f}")
    ax.plot(fh, th, c="C3", lw=2, label=f"honest: rf30_temporal (<=2023)  ROC {ah:.3f}")
    ax.plot([0,1],[0,1], c="k", lw=.6, alpha=.4)
    ax.set_xlabel(f"false positive rate (random {lsib} land)")
    ax.set_ylabel(f"true positive rate ({lsib} sites installed 2024)")
    ax.set_title(f"{lsib} temporal holdout — predict 2024 from AEF-2022 (n={len(H['site_s']):,})\n"
                 f"temporal leakage = {al-ah:+.3f} ROC")
    ax.legend(loc="lower right", fontsize=9); fig.tight_layout()
    fig.savefig(f"{FIG}/{tag}_temporal_roc_compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5)); x = np.arange(3); w = 0.38
    ax.bar(x-w/2, ltk, w, color="C0", label="leaky: rf30_final")
    ax.bar(x+w/2, htk, w, color="C3", label="honest: rf30_temporal")
    for i,(a,b) in enumerate(zip(ltk, htk)):
        ax.text(i-w/2, a+.5, f"{a:.0f}", ha="center", fontsize=8); ax.text(i+w/2, b+.5, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["top 20%","top 10%","top 5%"])
    ax.set_ylabel(f"% of {lsib} 2024 sites captured")
    ax.set_title(f"{lsib} temporal holdout — top-k% recall (predict 2024)\nhonest forecast vs leaky in-sample")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(f"{FIG}/{tag}_temporal_topk.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    row = dict(date="2026-07-16", experiment="temporal", country=lsib, map_year=2022, install_year=2024,
               model_honest="rf30_temporal_le2021", model_leaky="rf30_final",
               roc_honest=round(ah,4), roc_leaky=round(al,4), roc_leakage_gap=round(al-ah,4),
               top20_honest=round(htk[0],1), top10_honest=round(htk[1],1), top5_honest=round(htk[2],1),
               top20_leaky=round(ltk[0],1), top10_leaky=round(ltk[1],1), top5_leaky=round(ltk[2],1),
               n_sites=len(H["site_s"]))
    rows = [r for r in rows if not (r.get("experiment")=="temporal" and r.get("country")==lsib)]
    rows.append(row)
    print(f"\n=== {lsib} TEMPORAL: honest {ah:.4f} vs leaky {al:.4f}  gap {al-ah:+.4f} (n={len(H['site_s'])}) ===", flush=True)

with open(csvp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[-1].keys())); w.writeheader(); w.writerows(rows)
print("\n==================== TEMPORAL HOLDOUT DONE ====================")
for r in rows:
    if r.get("experiment")=="temporal":
        print(f"{r['country']:15s} honest {r['roc_honest']} | leaky {r['roc_leaky']} | gap {r['roc_leakage_gap']} | n={r['n_sites']}")
print("ALL DONE", flush=True)

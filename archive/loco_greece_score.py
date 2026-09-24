"""Greece LOCO validation + organised plots.

Scores the Greece-excluded RF (rf30_loco_grc) on Greek sites and compares to the frozen
reference (rf30_final, ROC 0.884 post-2021). Runs BOTH validation regimes Abdullah asked for:
  * post-2021  -> honest forecast (sites installed after the 2019 imagery), comparable to baseline
  * all-years  -> diagnostic upper bound (includes solar already visible in the 2019 map)

Everything lands under artifacts/figures/loco/greece/ + a row in artifacts/results/ablations.csv,
per the output-layout convention.
"""
import os, sys, csv, json, time
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
os.chdir(WS)
sys.path.insert(0, "Solar-Siting")
from run_country_rf import run_country

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

FIGDIR = "Solar-Siting/artifacts/figures/loco/greece"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs("Solar-Siting/artifacts/results", exist_ok=True)

LOCO = "Solar-Siting/artifacts/models/rf30_loco_grc.joblib"
REF = "Solar-Siting/artifacts/models/rf30_final.joblib"
FROZEN_REF_SCORES = "Solar-Siting/artifacts/country/greece_rf_2019_scores.npz"  # rf30_final post-2021

runs = {}  # tag -> (metrics dict, scores npz path)

def do(tag, model, min_install):
    prefix = f"{FIGDIR}/greece_{tag}_2019"
    print(f"\n########## RUN {tag}  model={os.path.basename(model)}  min_install={min_install} ##########", flush=True)
    m = run_country("Greece", year=2019, min_install=min_install, model_path=model,
                    out_prefix=prefix)
    runs[tag] = (m, f"{prefix}_scores.npz")
    return m

# 1) LOCO, honest forecast (post-2021) -- the headline number
do("loco_post2021", LOCO, 2021)
# 2) LOCO, all-years diagnostic
do("loco_allyears", LOCO, 2000)
# 3) reference, all-years -- baseline for the all-years comparison
do("ref_allyears", REF, 2000)

# frozen reference post-2021 already computed -- reuse it rather than re-score
zf = np.load(FROZEN_REF_SCORES)
ref_post = dict(site_s=zf["site_s"], rand_s=zf["rand_s"])
y = np.r_[np.ones(len(ref_post["site_s"])), np.zeros(len(ref_post["rand_s"]))]
ref_post_roc = roc_auc_score(y, np.r_[ref_post["site_s"], ref_post["rand_s"]])
print(f"\nfrozen reference post-2021 ROC (reused): {ref_post_roc:.3f}", flush=True)

def load(tag):
    z = np.load(runs[tag][1]); return z["site_s"], z["rand_s"]

def roc_of(site_s, rand_s):
    yy = np.r_[np.ones(len(site_s)), np.zeros(len(rand_s))]
    xx = np.r_[site_s, rand_s]
    fpr, tpr, _ = roc_curve(yy, xx)
    return fpr, tpr, roc_auc_score(yy, xx)

def topk(site_s, rand_s):
    pct = np.array([(rand_s < s).mean() for s in site_s])
    return [ (pct>=.8).mean()*100, (pct>=.9).mean()*100, (pct>=.95).mean()*100 ]

# ---- COMPARISON FIGURE 1: ROC overlay, both regimes, LOCO vs reference ----
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for j, (regime, loco_tag, ref_pair) in enumerate([
        ("post-2021 (honest forecast)", "loco_post2021", ("frozen ref", ref_post["site_s"], ref_post["rand_s"], ref_post_roc)),
        ("all-years (diagnostic)",      "loco_allyears", None)]):
    ls, rs = load(loco_tag)
    f1, t1, r1 = roc_of(ls, rs)
    ax[j].plot(f1, t1, lw=2, c="C3", label=f"LOCO (Greece dropped)  ROC {r1:.3f}")
    if ref_pair is None:
        rls, rrs = load("ref_allyears")
        f0, t0, r0 = roc_of(rls, rrs)
    else:
        _, rls, rrs, r0 = ref_pair
        f0, t0, _ = roc_of(rls, rrs)
    ax[j].plot(f0, t0, lw=2, c="C0", ls="--", label=f"reference (rf30_final)  ROC {r0:.3f}")
    ax[j].plot([0,1],[0,1], c="k", lw=.6, alpha=.4)
    ax[j].set_xlabel("false positive rate (random Greek land)")
    ax[j].set_ylabel("true positive rate (Greek sites)")
    ax[j].set_title(f"Greece {regime}\nAEF 2019 map | gap = leakage/memorisation premium")
    ax[j].legend(loc="lower right", fontsize=9)
fig.suptitle("Greece LOCO — RF trained WITHOUT any Greek chips, scored on Greece", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/greece_loco_roc_compare.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---- COMPARISON FIGURE 2: top-k% recall bars, LOCO vs reference, post-2021 ----
ls, rs = load("loco_post2021")
loco_tk = topk(ls, rs)
ref_tk = topk(ref_post["site_s"], ref_post["rand_s"])
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(3); w = 0.38
ax.bar(x-w/2, ref_tk, w, color="C0", label="reference (rf30_final)")
ax.bar(x+w/2, loco_tk, w, color="C3", label="LOCO (Greece dropped)")
for i,(a,b) in enumerate(zip(ref_tk, loco_tk)):
    ax.text(i-w/2, a+0.5, f"{a:.0f}", ha="center", fontsize=8)
    ax.text(i+w/2, b+0.5, f"{b:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["top 20%","top 10%","top 5%"])
ax.set_ylabel("% of Greek sites (installed >=2021) captured")
ax.set_title("Greece LOCO — top-k% recall (honest forecast)\nhow much the Greek premium was worth")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/greece_loco_topk_compare.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---- RESULTS CSV ROW ----
row = dict(
    date="2026-07-16", experiment="LOCO", country="Greece", model="rf30_loco_grc",
    baseline_model="rf30_final",
    roc_post2021=round(runs["loco_post2021"][0]["ROC"],4), roc_baseline_post2021=round(ref_post_roc,4),
    roc_gap_post2021=round(runs["loco_post2021"][0]["ROC"]-ref_post_roc,4),
    top20_post2021=round(loco_tk[0],1), top10_post2021=round(loco_tk[1],1), top5_post2021=round(loco_tk[2],1),
    roc_allyears_loco=round(runs["loco_allyears"][0]["ROC"],4),
    roc_allyears_ref=round(runs["ref_allyears"][0]["ROC"],4),
    n_sites_post2021=runs["loco_post2021"][0]["n_sites"],
    n_sites_allyears=runs["loco_allyears"][0]["n_sites"],
)
csvp = "Solar-Siting/artifacts/results/ablations.csv"
newfile = not os.path.exists(csvp)
with open(csvp, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if newfile: w.writeheader()
    w.writerow(row)

json.dump({k:v[0] for k,v in runs.items()}, open(f"{FIGDIR}/metrics.json","w"), indent=2, default=str)
print("\n==================== GREECE LOCO SUMMARY ====================")
print(f"POST-2021 (headline):  LOCO {row['roc_post2021']}  vs  reference {row['roc_baseline_post2021']}"
      f"   gap {row['roc_gap_post2021']:+.4f}")
print(f"  top 20/10/5%:  LOCO {loco_tk[0]:.1f}/{loco_tk[1]:.1f}/{loco_tk[2]:.1f}"
      f"   ref {ref_tk[0]:.1f}/{ref_tk[1]:.1f}/{ref_tk[2]:.1f}")
print(f"ALL-YEARS (diag):      LOCO {row['roc_allyears_loco']}  vs  reference {row['roc_allyears_ref']}")
print(f"figures -> {FIGDIR}")
print("ALL DONE", flush=True)

"""Resume Greece LOCO: only the all-years diagnostic (post-2021 already done + plotted).

Runs the two remaining GEE scorings (LOCO all-years, reference all-years), then builds the
combined 2-panel ROC figure and fills the all-years columns of results/ablations.csv.
The server was culled mid-run earlier; post-2021 outputs already exist on disk and are reused.
"""
import os, sys, csv, json
import numpy as np
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, "Solar-Siting")
from run_country_rf import run_country
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "Solar-Siting/artifacts/figures/loco/greece"
LOCO = "Solar-Siting/artifacts/models/rf30_loco_grc.joblib"
REF = "Solar-Siting/artifacts/models/rf30_final.joblib"

def do(tag, model):
    pre = f"{FIG}/greece_{tag}_2019"
    print(f"\n##### {tag}  {os.path.basename(model)}  all-years #####", flush=True)
    return run_country("Greece", year=2019, min_install=2000, model_path=model, out_prefix=pre), f"{pre}_scores.npz"

m_loco, p_loco = do("loco_allyears", LOCO)
m_ref, p_ref = do("ref_allyears", REF)

def roc(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp/tp[-1]]; fpr = np.r_[0, fp/fp[-1]]
    return fpr, tpr, float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2))

# post-2021 (already on disk)
lp = np.load(f"{FIG}/greece_loco_post2021_2019_scores.npz")
rp = np.load("Solar-Siting/artifacts/country/greece_rf_2019_scores.npz")
la = np.load(p_loco); ra = np.load(p_ref)

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for j,(title, L, R) in enumerate([
        ("post-2021 (honest forecast, n=%d)" % len(lp["site_s"]), lp, rp),
        ("all-years (diagnostic, n=%d)" % len(la["site_s"]), la, ra)]):
    f1,t1,a1 = roc(L["site_s"], L["rand_s"])
    f0,t0,a0 = roc(R["site_s"], R["rand_s"])
    ax[j].plot(f0,t0, c="C0", ls="--", lw=2, label=f"reference (rf30_final)  ROC {a0:.3f}")
    ax[j].plot(f1,t1, c="C3", lw=2, label=f"LOCO (Greece dropped)  ROC {a1:.3f}")
    ax[j].plot([0,1],[0,1], c="k", lw=.6, alpha=.4)
    ax[j].set_xlabel("false positive rate (random Greek land)")
    ax[j].set_ylabel("true positive rate (Greek sites)")
    ax[j].set_title(f"Greece {title}\ngap = {a1-a0:+.3f} ROC")
    ax[j].legend(loc="lower right", fontsize=9)
fig.suptitle("Greece LOCO — RF trained WITHOUT any Greek chips, scored on Greece (AEF 2019)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIG}/greece_loco_roc_compare.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# update the CSV all-years columns (rewrite the single LOCO/Greece row)
csvp = "Solar-Siting/artifacts/results/ablations.csv"
rows = list(csv.DictReader(open(csvp)))
for r in rows:
    if r["experiment"] == "LOCO" and r["country"] == "Greece" and r["model"] == "rf30_loco_grc":
        r["roc_allyears_loco"] = round(m_loco["ROC"],4)
        r["roc_allyears_ref"] = round(m_ref["ROC"],4)
        r["n_sites_allyears"] = m_loco["n_sites"]
with open(csvp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

json.dump(dict(loco_post2021=None, loco_allyears=m_loco, ref_allyears=m_ref),
          open(f"{FIG}/metrics_allyears.json","w"), indent=2, default=str)
print("\n==================== GREECE LOCO all-years ====================")
print(f"ALL-YEARS:  LOCO {m_loco['ROC']:.4f}  vs  reference {m_ref['ROC']:.4f}   gap {m_loco['ROC']-m_ref['ROC']:+.4f}")
print(f"(n_sites all-years = {m_loco['n_sites']})")
print(f"combined figure -> {FIG}/greece_loco_roc_compare.png")
print("ALL DONE", flush=True)

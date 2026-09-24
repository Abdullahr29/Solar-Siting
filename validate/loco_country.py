"""Reusable leave-one-country-out (LOCO) driver: extract -> train -> score -> plots -> CSV.

    python Solar-Siting/loco_country.py <Country> <ISO3>
    e.g.  python Solar-Siting/loco_country.py Spain ESP

Differs from the frozen reference (rf30_final) by EXACTLY one flag: --exclude-iso3 <ISO3>.
Scores the country in both regimes Abdullah asked for:
  * post-2021 : honest forecast (sites installed after the 2019 imagery)  [headline]
  * all-years : diagnostic upper bound (includes solar already visible in the map)
Outputs land under artifacts/figures/loco/<country>/ + a row in artifacts/results/ablations.csv.
Reference country-runs are reused from artifacts/country/ if present (Greece/Germany/China were
pre-frozen); otherwise they are scored here too.
"""
import os, sys, csv, json, subprocess
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, "Solar-Siting")

country, iso3 = sys.argv[1], sys.argv[2]
cl = country.lower().replace(" ", "_")
FIG = f"Solar-Siting/artifacts/figures/loco/{cl}"
os.makedirs(FIG, exist_ok=True)
os.makedirs("Solar-Siting/artifacts/results", exist_ok=True)
PY = sys.executable
POOL = f"Solar-Siting/artifacts/stats/px_loco_{iso3.lower()}.npz"
LOCO = f"Solar-Siting/artifacts/models/rf30_loco_{iso3.lower()}.joblib"
REF = "Solar-Siting/artifacts/models/rf30_final.joblib"
FROZEN_REF_POST = f"Solar-Siting/artifacts/country/{cl}_rf_2019_scores.npz"  # reference post-2021, if pre-frozen

def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode: sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")

# ---- stage 1+2: extract the Greece-excluded pool and train, as subprocesses (clean memory) ----
if not os.path.exists(POOL):
    sh([PY, "Solar-Siting/extract_pixels.py", "--exclude-iso3", iso3, "--out", POOL])
else:
    print(f"pool exists, reuse: {POOL}", flush=True)
if not os.path.exists(LOCO):
    sh([PY, "Solar-Siting/train_rf.py", "--pool", POOL, "--out", LOCO])
else:
    print(f"model exists, reuse: {LOCO}", flush=True)

# ---- stage 3: country scorings ----
from run_country_rf import run_country
runs = {}
def score(tag, model, min_install, out):
    print(f"\n##### {tag}  {os.path.basename(model)}  min_install={min_install} #####", flush=True)
    m = run_country(country, year=2019, min_install=min_install, model_path=model, out_prefix=out)
    runs[tag] = (m, f"{out}_scores.npz")
    return m

score("loco_post2021", LOCO, 2021, f"{FIG}/{cl}_loco_post2021_2019")
score("loco_allyears", LOCO, 2000, f"{FIG}/{cl}_loco_allyears_2019")
score("ref_allyears", REF, 2000, f"{FIG}/{cl}_ref_allyears_2019")
if os.path.exists(FROZEN_REF_POST):
    print(f"reusing pre-frozen reference post-2021: {FROZEN_REF_POST}", flush=True)
    runs["ref_post2021"] = (None, FROZEN_REF_POST)
else:
    score("ref_post2021", REF, 2021, f"{FIG}/{cl}_ref_post2021_2019")

# ---- metrics helpers ----
def roc(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp/tp[-1]]; fpr = np.r_[0, fp/fp[-1]]
    return fpr, tpr, float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2))
def topk(pos, neg):
    pct = np.array([(neg < s).mean() for s in pos])
    return [(pct>=.8).mean()*100, (pct>=.9).mean()*100, (pct>=.95).mean()*100]
def load(tag):
    z = np.load(runs[tag][1]); return z["site_s"], z["rand_s"]

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- FIG 1: 2-panel ROC overlay (post-2021 | all-years), LOCO vs reference ----
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for j,(title, lt, rt) in enumerate([
        ("post-2021 (honest forecast)", "loco_post2021", "ref_post2021"),
        ("all-years (diagnostic)", "loco_allyears", "ref_allyears")]):
    ls, lr = load(lt); rs, rr = load(rt)
    f1,t1,a1 = roc(ls, lr); f0,t0,a0 = roc(rs, rr)
    ax[j].plot(f0,t0, c="C0", ls="--", lw=2, label=f"reference (rf30_final)  ROC {a0:.3f}")
    ax[j].plot(f1,t1, c="C3", lw=2, label=f"LOCO ({country} dropped)  ROC {a1:.3f}")
    ax[j].plot([0,1],[0,1], c="k", lw=.6, alpha=.4)
    ax[j].set_xlabel(f"false positive rate (random {country} land)")
    ax[j].set_ylabel(f"true positive rate ({country} sites)")
    ax[j].set_title(f"{country} {title} (n={len(ls):,})\ngap = {a1-a0:+.3f} ROC")
    ax[j].legend(loc="lower right", fontsize=9)
fig.suptitle(f"{country} LOCO — RF trained WITHOUT any {country} chips, scored on {country} (AEF 2019)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIG}/{cl}_loco_roc_compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# ---- FIG 2: top-k% bars, post-2021 ----
ls, lr = load("loco_post2021"); rs, rr = load("ref_post2021")
ltk, rtk = topk(ls, lr), topk(rs, rr)
fig, ax = plt.subplots(figsize=(7, 5)); x = np.arange(3); w = 0.38
ax.bar(x-w/2, rtk, w, color="C0", label="reference (rf30_final)")
ax.bar(x+w/2, ltk, w, color="C3", label=f"LOCO ({country} dropped)")
for i,(a,b) in enumerate(zip(rtk, ltk)):
    ax.text(i-w/2, a+.5, f"{a:.0f}", ha="center", fontsize=8); ax.text(i+w/2, b+.5, f"{b:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["top 20%","top 10%","top 5%"])
ax.set_ylabel(f"% of {country} sites (installed >=2021) captured")
ax.set_title(f"{country} LOCO — top-k% recall (honest forecast)\nhow much the {country} premium was worth")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f"{FIG}/{cl}_loco_topk_post2021.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# ---- CSV row + metrics json ----
_,_,lp_roc = roc(*load("loco_post2021")); _,_,rp_roc = roc(*load("ref_post2021"))
_,_,la_roc = roc(*load("loco_allyears")); _,_,ra_roc = roc(*load("ref_allyears"))
row = dict(date="2026-07-16", experiment="LOCO", country=country, model=f"rf30_loco_{iso3.lower()}",
           baseline_model="rf30_final",
           roc_post2021=round(lp_roc,4), roc_baseline_post2021=round(rp_roc,4), roc_gap_post2021=round(lp_roc-rp_roc,4),
           top20_post2021=round(ltk[0],1), top10_post2021=round(ltk[1],1), top5_post2021=round(ltk[2],1),
           roc_allyears_loco=round(la_roc,4), roc_allyears_ref=round(ra_roc,4),
           n_sites_post2021=len(load("loco_post2021")[0]), n_sites_allyears=len(load("loco_allyears")[0]))
csvp = "Solar-Siting/artifacts/results/ablations.csv"
rows = list(csv.DictReader(open(csvp))) if os.path.exists(csvp) else []
rows = [r for r in rows if not (r["experiment"]=="LOCO" and r["country"]==country)]  # replace any prior row
rows.append(row)
with open(csvp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerows(rows)
json.dump({k:v[0] for k,v in runs.items() if v[0]}, open(f"{FIG}/metrics.json","w"), indent=2, default=str)

print(f"\n==================== {country.upper()} LOCO SUMMARY ====================")
print(f"POST-2021 (headline):  LOCO {lp_roc:.4f}  vs  reference {rp_roc:.4f}   gap {lp_roc-rp_roc:+.4f}")
print(f"  top 20/10/5%:  LOCO {ltk[0]:.1f}/{ltk[1]:.1f}/{ltk[2]:.1f}   ref {rtk[0]:.1f}/{rtk[1]:.1f}/{rtk[2]:.1f}")
print(f"ALL-YEARS (diag):      LOCO {la_roc:.4f}  vs  reference {ra_roc:.4f}   gap {la_roc-ra_roc:+.4f}")
print(f"figures -> {FIG}")
print("ALL DONE", flush=True)

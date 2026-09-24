"""Weak-country analysis (LOCAL, no GEE).

Question: for the shipped R model, is a country's validation weakness explained by
(a) how much training data it had, or (b) how 'novel' its geography is to the embedding
(the covariate residual_fraction = share of the score physical layers can't explain)?

Inputs (all already on disk):
  - artifacts/paper_v3/results_master.csv   -> per-country RF roc (scheme R)
  - artifacts/results/covariate_summary.csv -> per-country r2_rf, residual_fraction, importances
  - training positive volume per country     -> best-effort from splits / pools

Outputs -> artifacts/paper_v3/weak_country/:
  weak_country_summary.csv, weak_country_scatter.png, correlations.txt
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = os.path.expanduser("~/Solar_Workspace")
ART = f"{WS}/Solar-Siting/artifacts"
OUT = f"{ART}/paper_v3/weak_country"
os.makedirs(OUT, exist_ok=True)

# ---- 1. per-country R-model ROC ----
rm = pd.read_csv(f"{ART}/paper_v3/results_master.csv")
iso = rm[(rm.method == "rf") & (rm.metric == "roc") & (rm.scheme == "R")
         & rm.scope.str.fullmatch(r"[A-Z]{3}", na=False)].copy()
# a country can have several validation experiments (LOCO / held-out / forward).
# report the FULL spread and, as the headline, the *hardest* (lowest) roc per country
# -> that is the honest 'weakest forward' number we publish.
g = iso.groupby("scope").agg(
    roc_min=("value", "min"), roc_max=("value", "max"),
    roc_mean=("value", "mean"), n_val=("value", "size"),
    val_npos=("n_pos", "max")).reset_index().rename(columns={"scope": "iso3"})

# ---- 2. covariate residual_fraction (embedding novelty) ----
name2iso = {"Australia": "AUS", "Chile": "CHL", "China": "CHN", "Germany": "DEU",
            "Greece": "GRC", "South Africa": "ZAF", "United States": "USA"}
cov = pd.read_csv(f"{ART}/results/covariate_summary.csv")
cov["iso3"] = cov.country.map(name2iso)
cov = cov[["iso3", "r2_rf", "residual_fraction", "top3_importances",
           "gsa_ghi_rankauc", "grid_dist_m_rankauc", "slope_rankauc", "landcover_eta2"]]

# ---- 3. training positive volume per country (best-effort) ----
def training_counts():
    # try the paper_v3 pools first, then the split tables; count POS_<CCC>_ prefixes.
    from collections import Counter
    cand = (glob.glob(f"{ART}/paper_v3/pools/*.csv") +
            glob.glob(f"{ART}/splits/*.csv"))
    for f in cand:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        col = next((c for c in df.columns if df[c].astype(str).str.startswith("POS_").any()), None)
        if col is None:
            # maybe an explicit country/iso column + a split=='train' filter
            iso_col = next((c for c in df.columns if c.lower() in ("iso3", "country_code", "ccc", "iso")), None)
            if iso_col is None:
                continue
            sub = df
            if "split" in df.columns:
                sub = df[df.split.astype(str).str.contains("train", case=False, na=False)]
            cc = Counter(sub[iso_col].astype(str))
            if cc:
                return pd.DataFrame({"iso3": list(cc), "train_pos": list(cc.values())}), f
            continue
        cc = Counter(df[col].astype(str).str.split("_").str[1])
        if cc:
            return pd.DataFrame({"iso3": list(cc), "train_pos": list(cc.values())}), f
    return None, None

tc, tc_src = training_counts()

# ---- merge ----
m = g.merge(cov, on="iso3", how="left")
if tc is not None:
    m = m.merge(tc, on="iso3", how="left")
    vol_col = "train_pos"
else:
    m["train_pos"] = np.nan
    vol_col = "val_npos"   # fallback proxy
m = m.sort_values("roc_min").reset_index(drop=True)
m.to_csv(f"{OUT}/weak_country_summary.csv", index=False)

# ---- correlations ----
lines = []
lines.append(f"training-count source: {tc_src}")
lines.append(f"volume column used for corr: {vol_col}\n")
def corr(a, b, lab):
    d = m[[a, b]].dropna()
    if len(d) < 4:
        lines.append(f"{lab}: n={len(d)} too few"); return
    from scipy.stats import spearmanr, pearsonr
    rs, ps = spearmanr(d[a], d[b]); rp, pp = pearsonr(d[a], d[b])
    lines.append(f"{lab}: n={len(d)}  spearman={rs:+.3f} (p={ps:.3f})  pearson={rp:+.3f} (p={pp:.3f})")
corr("roc_min", vol_col, "ROC(min) vs training volume")
corr("roc_min", "residual_fraction", "ROC(min) vs residual_fraction (embedding novelty)")
corr("roc_min", "r2_rf", "ROC(min) vs covariate r2_rf")
open(f"{OUT}/correlations.txt", "w").write("\n".join(lines))
print("\n".join(lines))

# ---- figure: ROC vs training volume, sized/coloured by residual_fraction ----
fig, ax = plt.subplots(figsize=(9, 6))
d = m.dropna(subset=[vol_col])
sc = ax.scatter(d[vol_col], d.roc_min, c=d.residual_fraction, s=90,
                cmap="magma", edgecolor="k", vmin=0.1, vmax=0.35, zorder=3)
for _, r in d.iterrows():
    ax.annotate(r.iso3, (r[vol_col], r.roc_min), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel(f"training positives per country ({vol_col}, log)")
ax.set_ylabel("validation ROC (hardest tier per country)")
ax.axhline(0.8, color="grey", ls="--", lw=.8)
ax.set_title("Weak-country analysis: R model — data volume vs geographic novelty")
cb = fig.colorbar(sc); cb.set_label("residual_fraction (embedding novelty; higher = harder geography)")
fig.tight_layout(); fig.savefig(f"{OUT}/weak_country_scatter.png", dpi=140)
print("\nwrote", OUT)
print(m[["iso3", "roc_min", "roc_max", "n_val", vol_col, "residual_fraction", "top3_importances"]].to_string(index=False))

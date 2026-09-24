"""Appendix covariate permutation-importance heatmap (redone on the paper_v3 RF), from
covariate_novelty/novelty_v3.csv. Covariates (rows) x countries (cols); WorldCover classes summed
into one 'land cover' row; bottom labels carry the embedding-novelty residual per country.
-> figures/appendix_importance.{png,pdf}
"""
import os, json, numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
CN = "Solar-Siting/artifacts/paper_v3/figA_combined/covariate_novelty"
FIG = "Solar-Siting/artifacts/paper_v3/figures"

NAME = {"temp": "air temperature", "slope": "slope", "elevation": "elevation", "ghi": "GHI (irradiance)",
        "gti": "GTI (tilted irrad.)", "dni": "DNI", "dif": "DIF (diffuse)", "pvout": "PV yield (PVOUT)",
        "road": "road distance", "grid": "grid distance", "popdens": "population density",
        "in_wdpa": "protected area", "opta": "opt. tilt angle", "equatorwardness": "aspect (equatorward)"}
DISP = {"south_africa": "South Africa", "united_states": "United States", "china": "China",
        "germany": "Germany", "chile": "Chile", "greece": "Greece", "india": "India", "japan": "Japan",
        "spain": "Spain", "poland": "Poland", "colombia": "Colombia", "philippines": "Philippines",
        "malaysia": "Malaysia"}

df = pd.read_csv(f"{CN}/novelty_v3.csv").sort_values("residual_novelty", ascending=False)
mats = {}
for _, r in df.iterrows():
    d = {}
    for feat, imp in json.loads(r["top_importances"]):
        key = "land cover (WorldCover)" if feat.startswith("wc") else NAME.get(feat, feat)
        d[key] = d.get(key, 0.0) + float(imp)
    mats[r["country"]] = d
countries = list(df["country"])
feats = sorted({f for d in mats.values() for f in d}, key=lambda f: -sum(mats[c].get(f, 0) for c in countries))
M = np.array([[mats[c].get(f, 0.0) for c in countries] for f in feats])

fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(countries)), 0.5 * len(feats) + 1))
im = ax.imshow(M, cmap="BuGn", aspect="auto", vmin=0, vmax=np.percentile(M, 98))
resid = df.set_index("country")["residual_novelty"]
ax.set_xticks(range(len(countries)))
ax.set_xticklabels([f"{DISP.get(c, c)}  (resid {resid.loc[c]:.2f})" for c in countries],
                   fontsize=8, rotation=40, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8.5)
for i in range(len(feats)):
    for j in range(len(countries)):
        v = M[i, j]
        if v >= 0.005:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > np.percentile(M, 98) * 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("permutation importance", fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.set_title("Per-country covariate permutation importance (RF-score model) and embedding-novelty residual", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{FIG}/appendix_importance.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{FIG}/appendix_importance.pdf", bbox_inches="tight")
print("saved appendix_importance; feats:", feats, flush=True)

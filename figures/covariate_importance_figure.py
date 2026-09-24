"""Consolidated feature-importance figure across all 7 countries.

One heatmap: covariate (rows) x country (cols), cell = RF permutation importance from
the multivariate model that regresses our AlphaEarth suitability score on the GIS
covariate stack. Importance is computed in full here (not the stored top-15) and the
one-hot worldcover_*/dw_label_* dummies are summed back to their parent covariate so
each land-cover source is a single row. Lets the cross-country importance trends read
at a glance.

Sequential single-hue ramp (magnitude); values annotated so exact numbers are readable
regardless of saturation. Rows ordered by mean importance across countries; columns by
each country's embedding-novelty residual.

Usage (JASMIN sci node, env_solar):
    python Solar-Siting/covariate_importance_figure.py
"""
import os, json
import numpy as np

os.chdir(os.path.expanduser("~/Solar_Workspace"))
import sys
sys.path.insert(0, os.path.join(os.path.expanduser("~/Solar_Workspace"), "Solar-Siting"))
from covariate_analysis import load_table, split_covariates, CATEGORICAL, NON_COVARIATE  # noqa

COUNTRIES = ["Greece", "Chile", "Germany", "Australia", "China", "United States", "South Africa"]
YEAR = 2021

PRETTY = {
    "slope": "slope", "equatorwardness": "aspect (equatorward)",
    "slope_x_equatorward": "slope × aspect", "eastness": "aspect (eastward)",
    "elevation": "elevation",
    "gsa_ghi": "GHI (irradiance)", "gsa_pvout": "PV yield (PVOUT)", "gsa_gti": "GTI (tilted irrad.)",
    "gsa_dni": "DNI", "gsa_dif": "DIF (diffuse)", "gsa_temp": "air temperature",
    "gsa_opta": "opt. tilt angle", "gsa_ele": "GSA elevation",
    "grid_dist_m": "grid distance", "road_dist_m": "road distance",
    "worldcover": "land cover (WorldCover)", "dw_label": "land cover (DynamicWorld)",
    "in_wdpa": "protected area",
}


def full_importance(country):
    """Refit the exact covariate_analysis multivariate RF and return {parent_cov: perm_imp}."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    import pandas as pd

    tab, slug = load_table(country, YEAR)
    cont, cat = split_covariates(tab)
    raw = {c: tab[c] for c in cont}
    for c in cat:
        raw[c] = tab[c]
    df = pd.DataFrame(raw)
    df["__score"] = tab["score"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df.pop("__score").values
    oh = [c for c in cat if c != "in_wdpa"]
    for c in oh:
        df[c] = df[c].round().astype(int)
    df = pd.get_dummies(df, columns=oh, prefix=oh)
    X = df.values.astype(float)
    feat = list(df.columns)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                               n_jobs=-1, random_state=0).fit(Xtr, ytr)
    imp = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0, n_jobs=-1)

    # sum one-hot dummies back to their parent covariate (worldcover_10 -> worldcover)
    agg = {}
    for f, v in zip(feat, imp.importances_mean):
        parent = f.rsplit("_", 1)[0] if (f.startswith("worldcover_") or f.startswith("dw_label_")) else f
        agg[parent] = agg.get(parent, 0.0) + float(v)
    r2 = None
    jp = f"Solar-Siting/artifacts/figures/covariate/{slug}/{slug}_{YEAR}_analysis.json"
    if os.path.exists(jp):
        r2 = json.load(open(jp))["multivariate"]["residual_fraction"]
    print(f"  {country:14s} done ({len(feat)} feats -> {len(agg)} covariates)", flush=True)
    return agg, r2


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    data, resid = {}, {}
    for c in COUNTRIES:
        data[c], resid[c] = full_importance(c)

    # rows: all covariates ordered by mean importance across countries (descending)
    all_cov = set().union(*[d.keys() for d in data.values()])
    rows = sorted(all_cov, key=lambda k: -np.mean([data[c].get(k, 0.0) for c in COUNTRIES]))

    # columns ordered by each country's embedding-novelty residual (descending)
    cols = sorted(COUNTRIES, key=lambda c: -(resid[c] if resid[c] is not None else 0))

    M = np.array([[max(data[c].get(r, 0.0), 0.0) for c in cols] for r in rows])
    vmax = np.percentile(M, 99)  # robust cap so one large cell doesn't wash the ramp out

    # sequential single-hue ramp: near-white -> deep teal (magnitude, light->dark)
    cmap = LinearSegmentedColormap.from_list(
        "teal_seq", ["#f6f8f8", "#cfe3e1", "#8fc3bf", "#4f9d97", "#227e77", "#0d5751", "#08322e"])

    fig_h = 0.42 * len(rows) + 1.5
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c}\nresid {resid[c]:.3f}" if resid[c] is not None else c
                        for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([PRETTY.get(r, r) for r in rows], fontsize=9)

    # 2px surface gap between cells (white gridlines), recessive frame
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # annotate exact importance; hide clutter below 0.01. ink color flips on dark cells.
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if v < 0.01:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else "#1a1a1a")

    ax.set_title("What explains the AlphaEarth suitability score — GIS covariate importance by country\n"
                 "permutation importance (drop in R²) from the multivariate random forest",
                 fontsize=10.5, pad=18)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("permutation importance", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    outdir = "Solar-Siting/artifacts/figures/covariate/_summary"
    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/feature_importance_by_country.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")

    import csv
    with open(f"{outdir}/feature_importance_by_country.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["covariate"] + cols)
        for i, r in enumerate(rows):
            w.writerow([PRETTY.get(r, r)] + [f"{M[i, j]:.4f}" for j in range(len(cols))])
    print(f"saved {outdir}/feature_importance_by_country.csv")


if __name__ == "__main__":
    main()

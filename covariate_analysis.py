"""Covariate-analysis ANALYSIS pass (offline, on the sampled point table).

Answers two questions about the RF suitability score, per country:

  Univariate  — how strongly does each GIS covariate alone track our score?
                Spearman rho + a ranking AUC (does the covariate rank the top-decile
                high-score pixels?) for continuous covariates; score-by-class summaries
                for categorical ones.

  Multivariate — the headline. Regress our score on the FULL covariate stack:
                R2_linear (ridge) and R2_rf (random forest, nonlinear ceiling), on a
                held-out split. 1 - R2 is the RESIDUAL: the fraction of our score's
                variance that the standard GIS covariates do NOT explain = the
                AlphaEarth-embedding novelty. High R2 (~0.95) => the embedding just
                re-derives a GIS map; moderate R2 (~0.6-0.8) => it carries extra signal.
                Plus RF + permutation importances = which covariates drive the overlap.

Caveats surfaced deliberately: pixels are spatially autocorrelated, so p-values are
over-confident -> we report effect sizes / AUC / R2, not significance stars. A random
train/test split still lets neighbouring pixels leak across the split, so R2_rf is an
UPPER bound on covariate-explainable variance; a spatial-block CV (future) would lower it.

Usage (JASMIN sci node, env_solar):
    python Solar-Siting/covariate_analysis.py Greece --year 2021
"""
import argparse, json, os
import numpy as np

WORLDCOVER = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Crop", 50: "Built",
              60: "Bare", 70: "Snow/ice", 80: "Water", 90: "Wetland",
              95: "Mangrove", 100: "Moss"}
DW = {0: "water", 1: "trees", 2: "grass", 3: "flooded", 4: "crops",
      5: "shrub", 6: "built", 7: "bare", 8: "snow"}
CATEGORICAL = ["worldcover", "dw_label", "in_wdpa"]
NON_COVARIATE = {"lon", "lat", "score", "road_n_bbox"}   # keys that are not model inputs


def load_table(country, year):
    """Load the fullest available table (manual covariates appended if present).

    Preference order: the CORRECTED table (native-30 m slope + folded aspect features;
    see covariate_corrected.py) > the manual-appended full table > the GEE-native table.
    """
    slug = country.lower().replace(" ", "_")
    base = f"Solar-Siting/artifacts/covariate/{slug}/{slug}_{year}"
    for suffix in ("_covsample_corrected.npz", "_covsample_full.npz", "_covsample.npz"):
        if os.path.exists(f"{base}{suffix}"):
            path = f"{base}{suffix}"
            break
    d = np.load(path)
    print(f"loaded {os.path.basename(path)}", flush=True)
    return {k: d[k] for k in d.files}, slug


def split_covariates(tab):
    """Continuous = every numeric covariate that isn't categorical or bookkeeping."""
    cat = [c for c in CATEGORICAL if c in tab]
    cont = [c for c in tab if c not in NON_COVARIATE and c not in CATEGORICAL]
    return cont, cat


def univariate(tab, cont, cat, outdir, slug, year):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    score = tab["score"]
    hi = (score >= np.nanquantile(score, 0.90)).astype(int)   # top-decile membership
    rows = []

    for c in cont:
        x = tab[c]
        m = np.isfinite(x) & np.isfinite(score)
        if m.sum() < 50:
            continue
        rho = spearmanr(x[m], score[m]).statistic
        auc = roc_auc_score(hi[m], x[m])            # >0.5 higher cov => higher score
        rows.append(dict(covariate=c, kind="continuous", n=int(m.sum()),
                         spearman=round(float(rho), 4),
                         rank_auc=round(float(auc), 4),
                         discrimination=round(abs(auc - 0.5) * 2, 4)))
        # binned-median plot
        fig, ax = plt.subplots(figsize=(6, 4))
        qs = np.quantile(x[m], np.linspace(0, 1, 11))
        qs[-1] += 1e-9
        idx = np.clip(np.digitize(x[m], qs) - 1, 0, 9)
        bins = [b for b in range(10) if (idx == b).any()]   # skip empty bins (tied values)
        cent = [np.median(x[m][idx == b]) for b in bins]
        med = [np.median(score[m][idx == b]) for b in bins]
        ax.plot(cent, med, "o-", color="#c1121f")
        ax.axhline(np.median(score[m]), ls="--", color="grey", lw=1,
                   label=f"country median score {np.median(score[m]):.3f}")
        ax.set_xlabel(c); ax.set_ylabel("median RF score")
        ax.set_title(f"{slug} {year}: score vs {c}\nSpearman {rho:+.3f} | rank-AUC {auc:.3f}")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(f"{outdir}/uni_{c}.png", dpi=110); plt.close(fig)

    for c in cat:
        x = tab[c]
        m = np.isfinite(x) & np.isfinite(score)
        if m.sum() < 50:
            continue
        xi = x[m].astype(int); s = score[m]
        classes = sorted(np.unique(xi))
        lut = WORLDCOVER if c == "worldcover" else (DW if c == "dw_label" else {0: "no", 1: "yes"})
        stats = {int(k): dict(label=lut.get(int(k), str(k)), n=int((xi == k).sum()),
                              med_score=round(float(np.median(s[xi == k])), 4),
                              mean_score=round(float(s[xi == k].mean()), 4))
                 for k in classes}
        # eta^2: between-class variance fraction
        grand = s.mean()
        ss_tot = ((s - grand) ** 2).sum()
        ss_bet = sum((xi == k).sum() * (s[xi == k].mean() - grand) ** 2 for k in classes)
        eta2 = float(ss_bet / ss_tot) if ss_tot > 0 else 0.0
        extra = {}
        if c == "in_wdpa":
            from sklearn.metrics import roc_auc_score as _auc
            extra["auc"] = round(float(_auc(xi, s)), 4)   # does inside-PA predict low score?
        rows.append(dict(covariate=c, kind="categorical", n=int(m.sum()),
                         eta2=round(eta2, 4), by_class=stats, **extra))
        # boxplot by class
        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.9), 4))
        ax.boxplot([s[xi == k] for k in classes], showfliers=False,
                   tick_labels=[f"{lut.get(int(k), k)}\n(n={(xi==k).sum()})" for k in classes])
        ax.axhline(np.median(s), ls="--", color="grey", lw=1)
        ax.set_ylabel("RF score"); ax.set_title(f"{slug} {year}: score by {c} (eta2={eta2:.3f})")
        plt.xticks(rotation=45, ha="right", fontsize=8); fig.tight_layout()
        fig.savefig(f"{outdir}/uni_{c}.png", dpi=110); plt.close(fig)

    return rows


def multivariate(tab, cont, cat, outdir, slug, year):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score

    score = tab["score"]
    raw = {c: tab[c] for c in cont}
    for c in cat:
        raw[c] = tab[c]
    df = pd.DataFrame(raw)
    df["__score"] = score
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df.pop("__score").values
    # one-hot the multi-class categoricals (in_wdpa stays a 0/1 column)
    oh_cols = [c for c in cat if c != "in_wdpa"]
    for c in oh_cols:
        df[c] = df[c].round().astype(int)
    df = pd.get_dummies(df, columns=oh_cols, prefix=oh_cols)
    X = df.values.astype(float)
    feat = list(df.columns)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    r2_lin = r2_score(yte, ridge.predict(Xte))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                               n_jobs=-1, random_state=0).fit(Xtr, ytr)
    r2_rf = r2_score(yte, rf.predict(Xte))

    imp = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1]
    top = [(feat[i], round(float(imp.importances_mean[i]), 4)) for i in order[:15]]

    fig, ax = plt.subplots(figsize=(7, 5))
    k = min(12, len(order)); sel = order[:k][::-1]
    ax.barh([feat[i] for i in sel], [imp.importances_mean[i] for i in sel], color="#457b9d")
    ax.set_xlabel("permutation importance (drop in R2)")
    ax.set_title(f"{slug} {year}: what explains our RF score\n"
                 f"R2_rf={r2_rf:.3f}  R2_lin={r2_lin:.3f}  residual={1-r2_rf:.3f}")
    fig.tight_layout(); fig.savefig(f"{outdir}/multivariate_importance.png", dpi=120); plt.close(fig)

    return dict(n=int(len(y)), n_features=len(feat),
                r2_linear=round(float(r2_lin), 4), r2_rf=round(float(r2_rf), 4),
                residual_fraction=round(float(1 - r2_rf), 4), top_importances=top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()
    os.chdir(os.path.expanduser("~/Solar_Workspace"))

    tab, slug = load_table(args.country, args.year)
    cont, cat = split_covariates(tab)
    outdir = f"Solar-Siting/artifacts/figures/covariate/{slug}"
    os.makedirs(outdir, exist_ok=True)

    uni = univariate(tab, cont, cat, outdir, slug, args.year)
    multi = multivariate(tab, cont, cat, outdir, slug, args.year)

    result = dict(country=args.country, year=args.year,
                  n_points=int(len(tab["score"])),
                  continuous=cont, categorical=cat,
                  univariate=uni, multivariate=multi)
    with open(f"{outdir}/{slug}_{args.year}_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"=== {args.country} {args.year} covariate analysis (GEE-native covariates) ===")
    print(f"n={result['n_points']}  |  MULTIVARIATE: R2_rf={multi['r2_rf']:.3f} "
          f"R2_lin={multi['r2_linear']:.3f}  RESIDUAL={multi['residual_fraction']:.3f}")
    print("  (residual = variance in our score NOT explained by GIS covariates = embedding novelty)")
    print("UNIVARIATE:")
    for r in uni:
        if r["kind"] == "continuous":
            print(f"  {r['covariate']:11s} Spearman {r['spearman']:+.3f}  rank-AUC {r['rank_auc']:.3f}")
        else:
            print(f"  {r['covariate']:11s} eta2 {r['eta2']:.3f}" +
                  (f"  auc {r.get('auc')}" if 'auc' in r else ""))
    print("TOP covariate importances:", multi["top_importances"][:6])
    print(f"\nsaved figures + {slug}_{args.year}_analysis.json to {outdir}")


if __name__ == "__main__":
    main()

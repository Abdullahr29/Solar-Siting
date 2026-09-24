"""Covariate-novelty analysis, uniform across all 13 countries, from the Fig-A per-point npz.

Same method as covariate_analysis.multivariate: over the random-land NEGATIVES, regress the RF
suitability score on the GIS covariates (RF regressor + ridge) -> R2_rf (nonlinear ceiling),
R2_lin, residual = 1 - R2_rf = "embedding novelty" (score variance NOT explained by covariates),
plus permutation importance of the covariates. Uses the SAME MCDA-criteria feature set for every
country (more comparable than mixing full-feature vs subset countries).

Local: reads artifacts/paper_v3/figA_combined/<slug>_figA.npz (needs raw__* criteria). No GEE.
Out -> artifacts/paper_v3/figA_combined/covariate_novelty/{novelty_v3.csv, <slug>_importance.png}
"""
import csv, glob, json, os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
DIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figA_combined")
OUT = os.path.join(DIR, "covariate_novelty")
CONT = ["ghi", "pvout", "gti", "temp", "slope", "elevation", "equatorwardness",
        "road", "grid", "popdens"]   # raw__<name>


def main():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for npz in sorted(glob.glob(os.path.join(DIR, "*_figA.npz"))):
        slug = os.path.basename(npz).replace("_figA.npz", "")
        d = np.load(npz, allow_pickle=True)
        if "raw__ghi" not in d.files:
            print(f"skip {slug}: npz has no raw__ criteria"); continue
        neg = d["label"].astype(float) == 0
        y = d["rf"].astype(float)
        feats, cols = [], []
        for c in CONT:
            k = f"raw__{c}"
            if k in d.files:
                feats.append(d[k].astype(float)); cols.append(c)
        # worldcover as one-hot (coarse classes), in_wdpa as binary
        wc = d["raw__worldcover"].astype(float) if "raw__worldcover" in d.files else None
        X = np.column_stack(feats)
        extra_names = []
        if wc is not None:
            for cls in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]:
                X = np.column_stack([X, (wc == cls).astype(float)]); extra_names.append(f"wc{cls}")
        if "raw__in_wdpa" in d.files:
            X = np.column_stack([X, d["raw__in_wdpa"].astype(float)]); extra_names.append("in_wdpa")
        names = cols + extra_names

        m = neg & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if m.sum() < 500:
            print(f"skip {slug}: only {int(m.sum())} finite negatives"); continue
        Xm, ym = X[m], y[m]
        Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=0)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                   random_state=0).fit(Xtr, ytr)
        r2_rf = rf.score(Xte, yte)
        sc = StandardScaler().fit(Xtr)
        lin = Ridge().fit(sc.transform(Xtr), ytr)
        r2_lin = lin.score(sc.transform(Xte), yte)
        pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0, n_jobs=-1)
        order = np.argsort(pi.importances_mean)[::-1]
        top = [(names[i], round(float(pi.importances_mean[i]), 4)) for i in order[:12]]
        resid = 1 - r2_rf
        print(f"{slug:14} novelty(resid)={resid:.3f}  R2_rf={r2_rf:.3f}  R2_lin={r2_lin:.3f}  "
              f"n_neg={int(m.sum())}  top={top[:3]}", flush=True)
        rows.append(dict(country=slug, residual_novelty=round(resid, 4), r2_rf=round(r2_rf, 4),
                         r2_lin=round(r2_lin, 4), n_neg=int(m.sum()),
                         top_importances=json.dumps(top)))
        sel = order[:12][::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([names[i] for i in sel], [pi.importances_mean[i] for i in sel], color="#457b9d")
        ax.set_xlabel("permutation importance (drop in R2)")
        ax.set_title(f"{slug}: covariates -> RF score  (R2_rf={r2_rf:.2f}, novelty={resid:.2f})")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{slug}_importance.png"), dpi=120)
        plt.close(fig)

    if rows:
        with open(os.path.join(OUT, "novelty_v3.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {os.path.join(OUT, 'novelty_v3.csv')} ({len(rows)} countries)")


if __name__ == "__main__":
    main()

"""Train the GEE-deployable RF on a v3 pool — with optional de-confounding (Model D) and
in-memory LOCO / temporal filters (no re-extraction). Evaluates on the held-out test pool
and logs px ROC/PR into results_master.csv.

Model R : full pool, class_weight="balanced".
Model D : + per-country sample_weight so each country's NEGATIVE mass == its POSITIVE mass
          (both classes are per-chip-capped, so this matches by chip count, not farm area).
          Weights clipped to [0.05, 20] so no country is fully dropped.
LOCO    : --exclude-iso3 GRC  (drops that country's rows from the loaded pool)
temporal: --max-year 2021     (drops rows with AEF year > 2021 == installs > 2023)

    python train_rf_v3.py --pool px_v3_train.npz --test px_v3_test.npz \
        --out models/rf30_v3_R.joblib --scheme R --scope test-pixels --method rf
"""
import argparse, os, sys, time
import joblib, numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
BANDS = [f"A{i:02d}" for i in range(64)]


def compute_deconf_weight(y, iso, K):
    """sample_weight matching negative country-mass to positive country-mass."""
    pos, neg = y == 1, y == 0
    pc = np.bincount(iso[pos], minlength=K).astype(float)
    nc = np.bincount(iso[neg], minlength=K).astype(float)
    p, n = pc / max(pc.sum(), 1), nc / max(nc.sum(), 1)
    w = np.where(n > 0, p / np.where(n > 0, n, 1.0), 0.0)
    w = np.clip(w, 0.05, 20.0)
    sw = np.ones(len(y), np.float32)
    sw[neg] = w[iso[neg]].astype(np.float32)
    return sw


def load_pool(path, exclude_iso3=None, max_year=None, deconf=False, seed=0):
    z = np.load(path)
    names = list(z["iso_names"])
    filt = bool(exclude_iso3) or (max_year is not None)
    ms = np.ones(z["iso_sol"].shape[0], bool)
    mn = np.ones(z["iso_neg"].shape[0], bool)
    if exclude_iso3:
        if exclude_iso3 not in names:
            sys.exit(f"exclude_iso3={exclude_iso3} not in pool countries")
        ec = names.index(exclude_iso3)
        ms &= z["iso_sol"] != ec
        mn &= z["iso_neg"] != ec
    if max_year is not None:
        ms &= z["yr_sol"] <= max_year
        mn &= z["yr_neg"] <= max_year
    ns, nn = int(ms.sum()), int(mn.sum())
    X = np.empty((ns + nn, 64), np.float32)
    if filt:
        X[:ns] = z["X_sol"][ms]
        X[ns:] = z["X_neg"][mn]
    else:                                   # reference path: no bool-index temp
        X[:ns] = z["X_sol"]
        X[ns:] = z["X_neg"]
    y = np.zeros(ns + nn, np.float32); y[:ns] = 1.0
    iso = np.empty(ns + nn, np.int32)
    iso[:ns] = z["iso_sol"][ms] if filt else z["iso_sol"]
    iso[ns:] = z["iso_neg"][mn] if filt else z["iso_neg"]
    del z
    sw = compute_deconf_weight(y, iso, len(names)) if deconf else None
    # identical seed => identical permutation; shuffle X, y, sw in lockstep
    for arr in (X, y) + ((sw,) if sw is not None else ()):
        np.random.default_rng(seed).shuffle(arr)
    print(f"pool: {len(X):,} rows ({ns:,} sol / {nn:,} neg), pos share {ns/len(X):.3f}, "
          f"{X.nbytes/1e9:.2f} GB | deconf={deconf} exclude={exclude_iso3} max_year={max_year}",
          flush=True)
    return X, y, sw, ns, nn


def evaluate(rf, test_path):
    from sklearn.metrics import roc_auc_score, average_precision_score
    z = np.load(test_path)
    Xt = np.concatenate([z["X_sol"], z["X_neg"]])
    yt = np.r_[np.ones(z["X_sol"].shape[0]), np.zeros(z["X_neg"].shape[0])]
    keep = np.isfinite(Xt).all(axis=1)
    Xt, yt = Xt[keep], yt[keep]
    p = rf.predict_proba(Xt)[:, 1]
    roc, pr = roc_auc_score(yt, p), average_precision_score(yt, p)
    print(f"  [test] px ROC {roc:.4f} | PR {pr:.4f} (chance {yt.mean():.3f}) on {len(yt):,} px", flush=True)
    return roc, pr


def main(a):
    os.chdir(WS)
    from geemap import ml
    from sklearn.ensemble import RandomForestClassifier
    import results_util_v3 as R

    X, y, sw, ns, nn = load_pool(a.pool, a.exclude_iso3, a.max_year, a.deconf)
    min_leaf = a.min_leaf
    while True:
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=a.trees, max_depth=a.max_depth,
                                    min_samples_leaf=min_leaf, n_jobs=a.jobs, max_samples=0.34,
                                    class_weight="balanced", random_state=0).fit(X, y, sample_weight=sw)
        trees = ml.rf_to_strings(rf, BANDS, processes=32, output_mode="PROBABILITY")
        mb = sum(len(s) for s in trees) / 1e6
        print(f"min_samples_leaf={min_leaf}: {mb:.1f} MB tree strings ({time.time()-t0:.0f}s)", flush=True)
        if mb <= a.mb_cap:
            break
        min_leaf = int(min_leaf * mb / (a.mb_cap - 0.5))
    os.makedirs(os.path.dirname(os.path.join(WS, a.out)), exist_ok=True)
    joblib.dump(rf, os.path.join(WS, a.out))
    print(f"saved {a.out} ({mb:.1f} MB, min_samples_leaf={min_leaf})", flush=True)

    if a.test:
        roc, pr = evaluate(rf, a.test)
        R.append_result(a.experiment, a.method, a.scheme, a.scope, "roc", roc,
                        n_pos=ns, n_neg=nn, model_path=a.out, notes=f"min_leaf={min_leaf}")
        R.append_result(a.experiment, a.method, a.scheme, a.scope, "pr", pr,
                        n_pos=ns, n_neg=nn, model_path=a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True, help="path relative to workspace root")
    ap.add_argument("--test", default="Solar-Siting/artifacts/paper_v3/pools/px_v3_test.npz")
    ap.add_argument("--deconf", action="store_true")
    ap.add_argument("--exclude-iso3", default=None)
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument("--trees", type=int, default=30)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-leaf", type=int, default=1500)
    ap.add_argument("--mb-cap", type=float, default=9.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--experiment", default="train")
    ap.add_argument("--method", default="rf")
    ap.add_argument("--scheme", default="R")
    ap.add_argument("--scope", default="test-pixels")
    main(ap.parse_args())

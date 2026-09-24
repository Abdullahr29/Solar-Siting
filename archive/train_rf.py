"""Train the GEE-deployable random forest on a pixel pool and score it on held-out pixels.

Pairs with extract_pixels.py: that builds a pool (optionally with an ablation filter),
this fits the RF on it. Reference model and every ablation model go through this same
code path, so they differ ONLY by the pool they were given.

WHY A RANDOM FOREST, AND WHY THIS SHAPE
---------------------------------------
The scorer has to run inside Earth Engine, because AlphaEarth embeddings only exist there
and a planner cannot download a country of imagery. GEE can host a tree ensemble directly:
`geemap.ml.rf_to_strings` serialises the sklearn forest into text, and
`ee.Classifier.decisionTreeEnsemble` parses it back. But the request has a ~10 MB cap, so
the forest must stay SMALL — hence 30 trees and the size loop below.

The bake-off (notebook S15) put held-out pixel ROC at: linear probe 0.864, QDA 0.811,
RFF-512 0.897, RF 0.900, HistGB 0.914, MLP 0.917. The MLP/HistGB win but are not
GEE-native. RF-30 is the best thing that runs *inside* GEE, which is what makes it the
rapid country-scale tool.

TWO TRAPS BAKED IN HERE
-----------------------
  * NEVER size the forest with `max_leaf_nodes`. It switches sklearn to best-first tree
    building, and geemap serialises those trees into strings GEE cannot parse
    ("Error parsing line N: expected 8, got 3"). Size with min_samples_leaf / max_depth
    only — the loop below grows min_samples_leaf until the strings fit.
  * The kernel cgroup is ~25.8 GB and the pool is ~7.6 GB. Loading it naively
    (concatenate + fancy-index shuffle) peaks ~3x and gets OOM-killed. We preallocate and
    shuffle X and y in place with the same seed.

USAGE
    python Solar-Siting/train_rf.py --pool artifacts/stats/px_v2_reference.npz \
                                    --out artifacts/models/rf30_v2.joblib
"""
import argparse
import os
import time

import joblib
import numpy as np

WORKSPACE = os.path.expanduser("~/Solar_Workspace")
BANDS = [f"A{i:02d}" for i in range(64)]
DEFAULT_TEST = "Solar-Siting/artifacts/stats/emb_disc_testsamples.npz"


def load_pool(path, seed=0):
    """Load X, y from a pool npz without ever holding the data twice."""
    z = np.load(path)
    n_s = z["X_sol"].shape[0]
    n_n = z["X_neg"].shape[0]
    X = np.empty((n_s + n_n, 64), np.float32)
    X[:n_s] = z["X_sol"]
    X[n_s:] = z["X_neg"]
    del z
    y = np.zeros(len(X), np.float32)
    y[:n_s] = 1.0
    # identical seed => identical permutation, so rows and labels stay paired without
    # ever building an index array the size of the pool
    np.random.default_rng(seed).shuffle(X)
    np.random.default_rng(seed).shuffle(y)
    print(f"pool: {len(X):,} rows ({n_s:,} solar / {n_n:,} negative), "
          f"positive share {n_s/len(X):.3f}, {X.nbytes/1e9:.2f} GB")
    return X, y


def evaluate(rf, test_path, tag):
    from sklearn.metrics import roc_auc_score, average_precision_score
    z = np.load(test_path)
    Xs, Xn = z["X_sol"], z["X_neg"]
    Xt = np.concatenate([Xs, Xn])
    yt = np.r_[np.ones(len(Xs)), np.zeros(len(Xn))]
    keep = np.isfinite(Xt).all(axis=1)
    Xt, yt = Xt[keep], yt[keep]
    p = rf.predict_proba(Xt)[:, 1]
    roc = roc_auc_score(yt, p)
    pr = average_precision_score(yt, p)
    print(f"  [{tag}] px ROC {roc:.4f} | PR {pr:.4f} (chance {yt.mean():.3f}) "
          f"on {len(yt):,} px")
    return roc, pr


def main(pool, out, test, n_trees=30, max_depth=12, min_leaf=1500, mb_cap=9.0, jobs=6):
    os.chdir(WORKSPACE)
    from geemap import ml
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_pool(pool)

    while True:
        t0 = time.time()
        # n_jobs is deliberately low: each running tree needs its own workspace, and the
        # cgroup cannot hold 32 of them over a 7.6 GB pool. max_samples caps each tree's
        # bootstrap at ~1/3 of the rows. class_weight="balanced" (NOT balanced_subsample,
        # which recomputes weights per tree and costs memory for no benefit here).
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                    min_samples_leaf=min_leaf, n_jobs=jobs,
                                    max_samples=0.34, class_weight="balanced",
                                    random_state=0).fit(X, y)
        trees = ml.rf_to_strings(rf, BANDS, processes=32, output_mode="PROBABILITY")
        mb = sum(len(s) for s in trees) / 1e6
        print(f"min_samples_leaf={min_leaf}: {mb:.1f} MB of tree strings "
              f"({time.time()-t0:.0f}s)", flush=True)
        if mb <= mb_cap:
            break
        min_leaf = int(min_leaf * mb / (mb_cap - 0.5))
        print(f"  too big for the GEE request cap -> retry with min_samples_leaf={min_leaf}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(rf, out)
    print(f"\nsaved {out} ({mb:.1f} MB as GEE tree strings, "
          f"min_samples_leaf={min_leaf})")

    print("\n=== held-out evaluation ===")
    evaluate(rf, test, os.path.basename(test))
    return rf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--trees", type=int, default=30)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-leaf", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    main(a.pool, a.out, a.test, a.trees, a.max_depth, a.min_leaf, jobs=a.jobs)

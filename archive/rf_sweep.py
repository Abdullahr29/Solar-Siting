"""Stage 1 of the RF hyperparameter sweep — the GEE-DEPLOYABLE (<=9 MB) frontier.

Motivation: the deployed RF (rf30_v3_R) was NEVER hyperparameter-searched; its knobs
(30 trees, depth 12, min_samples_leaf=1500) were set to make rf_to_strings fit under the
Earth-Engine ~9 MB export budget, not for accuracy. This sweep finds the best RF that still
deploys to GEE, judged (Stage 2) on the honest country-forward metric.

One SLURM array task == one config. Each task:
  * loads the full train pool, filters non-finite rows,
  * splits 85/15 (fixed shuffle from load_pool, so the val slice is identical across tasks),
  * fits the RF on 85%, measures the GEE rf_to_strings size (hard <=9 MB gate) and a held-out
    15% pixel ROC/PR (shortlisting metric only — NOT a reported number),
  * writes artifacts/paper_v3/sweep/config_{id:02d}.json and, if feasible, the model.

Config id 0 == the incumbent's hyperparameters, so it is re-measured in this same harness and
becomes the fair within-sweep baseline (85%-trained, like every candidate).

Selection here is pixel-val only; the honest country / temporal / TZ-SAM metrics are reserved
for Stage 2 / Stage 3 and never used to pick hyperparameters.
"""
import os, sys, json, time, itertools
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
from train_rf_v3 import load_pool, BANDS  # noqa: E402

POOL = "Solar-Siting/artifacts/paper_v3/pools/px_v3_train.npz"
OUT = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/sweep")
MB_CAP = 9.0
VAL_FRAC = 0.15
N_CONFIGS = 40


def make_configs():
    """Deterministic config list. Index 0 is the incumbent; the rest are a fixed randomized
    draw over the feasible-ish grid (kept shared between all stages by importing this fn)."""
    incumbent = dict(n_estimators=30, max_depth=12, min_samples_leaf=1500,
                     max_features="sqrt", max_samples=0.34)
    grid = dict(
        n_estimators=[30, 45, 60],
        max_depth=[10, 12, 16, 20],
        min_samples_leaf=[500, 800, 1500, 3000],
        max_features=["sqrt", 0.25, 0.5],
        max_samples=[0.34, 0.5],
    )
    keys = list(grid)
    allc = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    def k(c): return tuple(c[x] for x in keys)
    uniq = {k(c): c for c in allc}
    inc = uniq.pop(k(incumbent), incumbent)
    rest = list(uniq.values())
    np.random.default_rng(20260825).shuffle(rest)
    return [inc] + rest[:N_CONFIGS - 1]


CONFIGS = make_configs()


def main(cid, smoke=False):
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    cfg = CONFIGS[cid]
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from geemap import ml

    X, y, _, ns, nn = load_pool(POOL)
    keep = np.isfinite(X).all(1)
    X, y = X[keep], y[keep]
    if smoke:
        idx = np.random.default_rng(0).choice(len(X), 200_000, replace=False)
        X, y = X[idx], y[idx]
    nval = int(len(X) * VAL_FRAC)
    Xtr, ytr, Xva, yva = X[:-nval], y[:-nval], X[-nval:], y[-nval:]

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
        max_samples=cfg["max_samples"], class_weight="balanced",
        n_jobs=ncpu, random_state=0).fit(Xtr, ytr)
    fit_s = time.time() - t0

    p = rf.predict_proba(Xva)[:, 1]
    roc = float(roc_auc_score(yva, p))
    pr = float(average_precision_score(yva, p))

    t1 = time.time()
    trees = ml.rf_to_strings(rf, BANDS, processes=ncpu, output_mode="PROBABILITY")
    mb = sum(len(s) for s in trees) / 1e6
    feasible = mb <= MB_CAP
    mean_leaves = int(np.mean([t.get_n_leaves() for t in rf.estimators_]))

    res = dict(config_id=cid, is_incumbent=(cid == 0), **cfg,
               val_roc=round(roc, 5), val_pr=round(pr, 5), size_mb=round(mb, 2),
               feasible=bool(feasible), mean_leaves=mean_leaves,
               fit_s=round(fit_s), strings_s=round(time.time() - t1),
               n_train=int(len(Xtr)), n_val=int(len(Xva)))
    mpath = ""
    if feasible and not smoke:
        import joblib
        mpath = f"Solar-Siting/artifacts/paper_v3/sweep/models/rf_cfg{cid:02d}.joblib"
        joblib.dump(rf, os.path.join(WS, mpath))
    res["model_path"] = mpath

    if not smoke:
        with open(os.path.join(OUT, f"config_{cid:02d}.json"), "w") as f:
            json.dump(res, f, indent=2)
    print("RESULT", json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.config_id, a.smoke)

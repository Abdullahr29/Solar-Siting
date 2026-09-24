"""Stage 7: which of the 64 AlphaEarth dimensions drive the RF (permutation importance on the
held-out test pool), for BOTH schemes. Local + cheap; no GIS/MCDA dependency.
"""
import os, sys
import numpy as np, joblib

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score

TEST = "Solar-Siting/artifacts/paper_v3/pools/px_v3_test.npz"
FIGDIR = "Solar-Siting/artifacts/paper_v3/figures"
N_SUB = 200_000


def main():
    z = np.load(TEST)
    Xt = np.concatenate([z["X_sol"], z["X_neg"]])
    yt = np.r_[np.ones(z["X_sol"].shape[0]), np.zeros(z["X_neg"].shape[0])]
    keep = np.isfinite(Xt).all(1); Xt, yt = Xt[keep], yt[keep]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yt), min(N_SUB, len(yt)), replace=False)
    Xt, yt = Xt[idx], yt[idx]
    scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    for scheme in ("R", "D"):
        mp = f"Solar-Siting/artifacts/paper_v3/models/rf30_v3_{scheme}.joblib"
        if not os.path.exists(os.path.join(WS, mp)):
            print(f"skip {scheme}: no model"); continue
        rf = joblib.load(os.path.join(WS, mp))
        imp = permutation_importance(rf, Xt, yt, scoring=scorer, n_repeats=5,
                                     random_state=0, n_jobs=6)
        mean = imp.importances_mean
        order = np.argsort(mean)[::-1]
        out_npz = f"Solar-Siting/artifacts/paper_v3/scores/emb_importance_{scheme}.npz"
        np.savez(os.path.join(WS, out_npz), importances_mean=mean, importances_std=imp.importances_std)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(64), mean[np.argsort(np.arange(64))], color="#4C78A8")
        ax.set_xlabel("AlphaEarth embedding dimension"); ax.set_ylabel("perm. importance (ΔROC)")
        ax.set_title(f"Embedding-dimension importance — RF-{scheme} (top dim A{order[0]:02d})")
        out_fig = f"{FIGDIR}/emb_importance_{scheme}.png"
        fig.savefig(os.path.join(WS, out_fig), dpi=120, bbox_inches="tight"); plt.close(fig)
        R.append_result("importance", "rf", scheme, "test-pixels", "perm_top1", float(mean.max()),
                        scores_path=out_npz, figure_path=out_fig,
                        notes=f"top dims A{order[0]:02d},A{order[1]:02d},A{order[2]:02d}")
        print(f"### importance {scheme}: top dims {[f'A{d:02d}' for d in order[:5]]}", flush=True)


if __name__ == "__main__":
    main()

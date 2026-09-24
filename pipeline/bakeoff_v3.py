"""8-method extraction bake-off on masks_v2, all on a COMMON balanced subsample, all scored
on the SAME held-out test pool -> a fair method ranking (Fig D). Methods:
  model1 (mean-embedding prototype, cosine) | probe (logistic) | qda | rff-512 | rf |
  histgb | mlp | maxent
RF row uses the deployed full-pool RF-R (rf30_v3_R) for the honest deployment number AND a
subsample-trained RF for apples-to-apples fairness; both logged. Everything -> results_master.
"""
import os, sys, time
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
from sklearn.metrics import roc_auc_score, average_precision_score

TRAIN = "Solar-Siting/artifacts/paper_v3/pools/px_v3_train.npz"
TEST = "Solar-Siting/artifacts/paper_v3/pools/px_v3_test.npz"
RF_R = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
FIG = "Solar-Siting/artifacts/paper_v3/figures/bakeoff/bakeoff_roc.png"
N_PER_CLASS = 700_000     # 1.4M rows: plenty for a fair method ranking, caps RFF transform RAM
SEED = 0


def balanced(Xs, Xn, n, seed):
    rng = np.random.default_rng(seed)
    si = rng.choice(len(Xs), min(n, len(Xs)), replace=False)
    ni = rng.choice(len(Xn), min(n, len(Xn)), replace=False)
    X = np.vstack([Xs[si], Xn[ni]]).astype(np.float32)
    y = np.r_[np.ones(len(si)), np.zeros(len(ni))]
    p = rng.permutation(len(y))
    return X[p], y[p]


def load_test():
    z = np.load(TEST)
    Xt = np.concatenate([z["X_sol"], z["X_neg"]])
    yt = np.r_[np.ones(z["X_sol"].shape[0]), np.zeros(z["X_neg"].shape[0])]
    keep = np.isfinite(Xt).all(axis=1)
    return Xt[keep], yt[keep]


def score(name, proba, yt, notes=""):
    roc, pr = roc_auc_score(yt, proba), average_precision_score(yt, proba)
    print(f"  {name:10s} ROC {roc:.4f} | PR {pr:.4f}  {notes}", flush=True)
    R.append_result("bakeoff", name, "R", "test-pixels", "roc", roc, notes=notes)
    R.append_result("bakeoff", name, "R", "test-pixels", "pr", pr, notes=notes)
    return roc


def main():
    zt = np.load(TRAIN)
    Xs, Xn = zt["X_sol"], zt["X_neg"]
    print(f"train pool {len(Xs):,} sol / {len(Xn):,} neg -> subsample {N_PER_CLASS:,}/class", flush=True)
    X, y = balanced(Xs, Xn, N_PER_CLASS, SEED)
    Xt, yt = load_test()
    print(f"test pool {len(yt):,} px (chance {yt.mean():.3f})", flush=True)
    rocs = {}

    # model1: unit-norm prototype of solar pixels; cosine similarity (normalise prototype, not inputs)
    proto = Xs[np.random.default_rng(SEED).choice(len(Xs), min(2_000_000, len(Xs)), replace=False)].mean(0)
    proto /= (np.linalg.norm(proto) + 1e-9)
    rocs["model1"] = score("model1", Xt @ proto, yt, "mean-embedding cosine")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    Xs_, Xt_ = sc.transform(X), sc.transform(Xt)
    lr = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(Xs_, y)
    rocs["probe"] = score("probe", lr.predict_proba(Xt_)[:, 1], yt, "logistic")

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    rocs["qda"] = score("qda", qda.predict_proba(Xt)[:, 1], yt)

    from sklearn.kernel_approximation import RBFSampler
    rbf = RBFSampler(n_components=512, gamma=1.0 / X.shape[1], random_state=SEED).fit(Xs_)
    lr2 = LogisticRegression(max_iter=300, class_weight="balanced").fit(rbf.transform(Xs_), y)
    rocs["rff"] = score("rff", lr2.predict_proba(rbf.transform(Xt_))[:, 1], yt, "rff-512")

    import joblib
    if os.path.exists(RF_R):
        rf = joblib.load(RF_R)
        rocs["rf"] = score("rf", rf.predict_proba(Xt)[:, 1], yt, "full-pool RF-R (deployed)")
    from sklearn.ensemble import RandomForestClassifier
    rfb = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=50,
                                 n_jobs=8, class_weight="balanced", random_state=0).fit(X, y)
    score("rf_sub", rfb.predict_proba(Xt)[:, 1], yt, "subsample RF (fairness ref)")

    from sklearn.ensemble import HistGradientBoostingClassifier
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                          class_weight="balanced", random_state=0).fit(X, y)
    rocs["histgb"] = score("histgb", hgb.predict_proba(Xt)[:, 1], yt)

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60, early_stopping=True,
                        random_state=0).fit(Xs_, y)
    rocs["mlp"] = score("mlp", mlp.predict_proba(Xt_)[:, 1], yt)

    # maxent: proper elapid if available, else linear+quadratic L1 logistic (IWLR-equivalent)
    try:
        from elapid import MaxentModel
        me = MaxentModel().fit(X, y)
        pme = me.predict(Xt)
        note = "elapid maxent"
    except Exception as e:
        print(f"  (elapid unavailable: {e}; using linear+quadratic L1 logistic, IWLR-equivalent)", flush=True)
        Xq = np.hstack([Xs_, Xs_ ** 2]); Xtq = np.hstack([Xt_, Xt_ ** 2])
        me = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=200,
                                class_weight="balanced").fit(Xq, y)
        pme = me.predict_proba(Xtq)[:, 1]
        note = "linear+quadratic L1 logistic (IWLR-equivalent)"
    rocs["maxent"] = score("maxent", pme, yt, note)

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = sorted(rocs, key=rocs.get)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(order, [rocs[m] for m in order], color="#4C78A8")
    ax.axvline(rocs.get("rf", 0.9), ls="--", c="k", lw=1, label="deployed RF-R")
    for i, m in enumerate(order):
        ax.text(rocs[m] + 0.002, i, f"{rocs[m]:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("held-out pixel ROC-AUC (masks_v2 test pool)")
    ax.set_title("Extraction-method bake-off on AlphaEarth embeddings (masks_v2)")
    ax.legend(loc="lower right", fontsize=8)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {FIG}", flush=True)


if __name__ == "__main__":
    main()

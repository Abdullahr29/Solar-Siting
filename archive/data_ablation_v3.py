"""Data-quality ablation: does masks_v2 + the decontaminated/rebalanced negatives improve the
RF vs the OLD data (v1 masks + old negatives)? Same recipe, same referee.
  * pixel level  : OLD rf30_final vs NEW rf30_v3_R on the SAME masks_v2 test pool (px_v3_test).
                   (New was trained on these corrected labels, so this is the 'predicts corrected
                    ground truth better' check; the deployment number below is the honest one.)
  * deployment   : OLD rf30_final run through the SAME country forward validation (dedup) as the
                   new model -> old-vs-new forward ROC. New numbers come from the country stage.
"""
import os, sys, traceback
import numpy as np, pandas as pd

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
from run_country_rf import run_country
from sklearn.metrics import roc_auc_score, average_precision_score

OLD = "Solar-Siting/artifacts/models/rf30_final.joblib"
NEW = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
TEST = "Solar-Siting/artifacts/paper_v3/pools/px_v3_test.npz"
CORE = [("Greece", "GRC", None), ("Germany", "DEU", None), ("China", "CHN", None),
        ("Spain", "ESP", None), ("Poland", "POL", None), ("Japan", "JPN", None),
        ("India", "IND", None), ("United States", "USA", "United States of America")]


def px_compare():
    import joblib
    z = np.load(TEST)
    Xt = np.concatenate([z["X_sol"], z["X_neg"]])
    yt = np.r_[np.ones(z["X_sol"].shape[0]), np.zeros(z["X_neg"].shape[0])]
    keep = np.isfinite(Xt).all(1); Xt, yt = Xt[keep], yt[keep]
    res = {}
    for tag, path in [("old-data", OLD), ("new-data", NEW)]:
        if not os.path.exists(os.path.join(WS, path)):
            print(f"skip px {tag}: missing {path}"); continue
        rf = joblib.load(os.path.join(WS, path))
        p = rf.predict_proba(Xt)[:, 1]
        roc, pr = roc_auc_score(yt, p), average_precision_score(yt, p)
        res[tag] = (roc, pr)
        R.append_result("data_ablation", "rf", tag, "test-pixels", "roc", roc, model_path=path)
        R.append_result("data_ablation", "rf", tag, "test-pixels", "pr", pr, model_path=path)
        print(f"  px {tag}: ROC {roc:.4f} PR {pr:.4f}", flush=True)
    if "old-data" in res and "new-data" in res:
        d = res["new-data"][0] - res["old-data"][0]
        print(f"### DATA IMPROVEMENT (px ROC, masks_v2 test): new-old = {d:+.4f} "
              f"({res['old-data'][0]:.4f} -> {res['new-data'][0]:.4f})", flush=True)
    return res


def country_old():
    for lsib, iso3, inv in CORE:
        prefix = f"Solar-Siting/artifacts/paper_v3/scores/country/{lsib.lower().replace(' ', '_')}_OLD_2019"
        try:
            m = run_country(lsib, year=2019, min_install=2021, model_path=OLD,
                            out_prefix=prefix, inv_country=inv)
            R.append_result("data_ablation", "rf", "old-data", iso3, "roc", m["ROC"],
                            n_pos=m["n_sites"], model_path=OLD, aef_year=2019,
                            scores_path=prefix + "_scores.npz")
            print(f"### old-data country {lsib}: ROC {m['ROC']:.3f}", flush=True)
        except Exception:
            traceback.print_exc(); R.log_stage(f"data_ablation/{iso3}", prefix, "fail")


def figure():
    """Old-vs-new bars: px + per-country (new-data country ROC from the country2021 stage)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m = pd.read_csv(os.path.join(WS, "Solar-Siting/artifacts/paper_v3/results_master.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    old = m[(m.experiment == "data_ablation") & (m.scheme == "old-data") &
            (m.metric == "roc") & (m.scope != "test-pixels")].set_index("scope").value
    new = m[(m.experiment == "country2021") & (m.scheme == "R") &
            (m.metric == "roc")].set_index("scope").value
    common = [c for c in new.index if c in old.index]
    if common:
        x = np.arange(len(common)); w = 0.38
        ax.bar(x - w/2, [old[c] for c in common], w, label="old data (v1 masks + old neg)", color="#B0B0B0")
        ax.bar(x + w/2, [new[c] for c in common], w, label="new data (masks_v2 + decontam neg)", color="#4C78A8")
        ax.set_xticks(x); ax.set_xticklabels(common, rotation=45); ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("forward country ROC (installs >= 2021, dedup)")
        ax.set_title("Data-quality improvement: old vs new training data (same RF recipe)")
        ax.legend()
        out = "Solar-Siting/artifacts/paper_v3/figures/data_ablation_old_vs_new.png"
        fig.savefig(os.path.join(WS, out), dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    px_compare()
    country_old()
    figure()

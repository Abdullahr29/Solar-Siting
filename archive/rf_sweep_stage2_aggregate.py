"""Stage 2 aggregator / decision. Recomputes country-forward ROC from every sweep scores npz
(config x country), builds the config x country table, ranks configs by site-count-weighted mean
ROC (robust to the tiny-n countries), and prints the winner vs the incumbent (config 0).

Site-weighted mean is the primary rank (weights each country by its validation-site count);
plain 13-country mean and a robust-8 mean (drops ZAF/COL/PHL/MYS/CHL) are shown alongside.
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

WS = os.path.expanduser("~/Solar_Workspace")
SW = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/sweep")
SWSCORES = os.path.join(SW, "scores")
ROBUST_DROP = {"south_africa", "colombia", "philippines", "malaysia", "chile"}


def roc_from_npz(path):
    z = np.load(path)
    s, r = z["site_s"], z["rand_s"]
    y = np.r_[np.ones(len(s)), np.zeros(len(r))]
    return roc_auc_score(y, np.r_[s, r]), len(s)


def main():
    with open(os.path.join(SW, "stage2_shortlist.json")) as f:
        shortlist = {int(c["config_id"]): c for c in json.load(f)}
    rows = []
    for f in sorted(glob.glob(os.path.join(SWSCORES, "cfg*_scores.npz"))):
        m = re.match(r"cfg(\d+)_(.+)_(\d{4})_scores\.npz", os.path.basename(f))
        if not m:
            continue
        cid, country = int(m.group(1)), m.group(2)
        try:
            roc, n = roc_from_npz(f)
        except Exception as e:
            print(f"bad npz {f}: {e}"); continue
        rows.append(dict(config_id=cid, country=country, roc=roc, n=n))
    if not rows:
        sys.exit("no sweep score npz found — did Stage 2 run?")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SW, "sweep_stage2_country.csv"), index=False)

    def summarize(g):
        w = np.average(g.roc, weights=g.n)
        plain = g.roc.mean()
        rob = g[~g.country.isin(ROBUST_DROP)].roc.mean()
        return pd.Series({"weighted_roc": w, "mean_roc": plain, "robust8_roc": rob,
                          "n_countries": g.country.nunique()})
    summ = df.groupby("config_id").apply(summarize).reset_index()

    def label(cid):
        c = shortlist.get(cid, {})
        return (f"trees={c.get('n_estimators')},d={c.get('max_depth')},"
                f"leaf={c.get('min_samples_leaf')},feat={c.get('max_features')},"
                f"samp={c.get('max_samples')},{c.get('size_mb')}MB"
                + (" [INCUMBENT]" if cid == 0 else ""))
    summ["config"] = summ.config_id.map(label)
    summ = summ.sort_values("weighted_roc", ascending=False)

    pd.set_option("display.width", 200)
    print("\n=== Stage 2: country-forward validation of the shortlist ===")
    print(summ[["config_id", "weighted_roc", "mean_roc", "robust8_roc", "n_countries", "config"]]
          .to_string(index=False))

    inc = summ[summ.config_id == 0]
    best = summ.iloc[0]
    inc_w = float(inc.weighted_roc.iloc[0]) if len(inc) else float("nan")
    print(f"\nincumbent (id0) weighted ROC: {inc_w:.4f}")
    print(f"best config id{int(best.config_id)} weighted ROC: {best.weighted_roc:.4f} "
          f"(delta {best.weighted_roc - inc_w:+.4f})")
    if int(best.config_id) == 0:
        print("=> incumbent is the best GEE-deployable config. Change nothing; report as tuned.")
    else:
        print(f"=> config id{int(best.config_id)} beats the incumbent by "
              f"{best.weighted_roc - inc_w:+.4f} weighted ROC. Candidate to retrain on the FULL "
              f"pool and adopt (re-export global raster) IF the gain is worth it.")
    print(f"\nwrote {SW}/sweep_stage2_country.csv")


if __name__ == "__main__":
    main()

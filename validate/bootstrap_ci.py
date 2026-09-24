"""Bootstrap 95% CIs for country ROC-AUC — inventory, TZ-SAM, and combined positive sets.

Purely local: reads cached score arrays, no GEE, no model. ROC-AUC computed rank-wise
(Mann-Whitney U), so no sklearn needed. For each country and each positive set we resample
positives AND negatives with replacement (n_boot times) and take the 2.5/97.5 percentiles.

Sets per country:
  * inventory : site_s  vs rand_s   (our supervisor's PV inventory, installs >= 2021)
  * tzsam     : tz_s    vs rand_s   (independent TZ-SAM new sites, constructed_after >= 2020)
  * combined  : [site_s, tz_s] vs rand_s
All three share the SAME cached RF-R random-land negatives, so CIs are directly comparable.

Out -> Solar-Siting/artifacts/paper_v3/tzsam/bootstrap_ci.csv
"""
import csv, os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
COUNTRY_DIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/scores/country")
TZ_DIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/tzsam")
OUT = os.path.join(TZ_DIR, "bootstrap_ci.csv")

COUNTRIES = ["Greece", "Germany", "China", "Spain", "Poland", "Japan", "India",
             "United States", "South Africa", "Colombia", "Philippines",
             "Malaysia", "Chile"]
N_BOOT = 2000
SEED = 42


def slug(c):
    return c.lower().replace(" ", "_")


def auc(pos, neg, neg_sorted=None):
    """Rank-based ROC-AUC = P(pos > neg) + 0.5 P(pos == neg)."""
    ns = neg_sorted if neg_sorted is not None else np.sort(neg)
    lo = np.searchsorted(ns, pos, side="left")
    hi = np.searchsorted(ns, pos, side="right")
    return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))


def boot_ci(pos, neg, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    ns = np.sort(neg)
    point = auc(pos, neg, ns)
    np_, nn = len(pos), len(neg)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        p = pos[rng.integers(0, np_, np_)]
        n = neg[rng.integers(0, nn, nn)]
        vals[b] = auc(p, n)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi, vals.std()


def main():
    rows = []
    for c in COUNTRIES:
        cp = os.path.join(COUNTRY_DIR, f"{slug(c)}_R_2019_scores.npz")
        tp = os.path.join(TZ_DIR, f"{slug(c)}_tzsam_scores.npz")
        if not os.path.exists(cp):
            print(f"skip {c}: no country npz"); continue
        d = np.load(cp)
        site_s = d["site_s"].astype(float)
        rand_s = d["rand_s"].astype(float)
        tz_s = None
        if os.path.exists(tp):
            tz_s = np.load(tp)["tz_s"].astype(float)

        sets = {"inventory": site_s}
        if tz_s is not None and len(tz_s):
            sets["tzsam"] = tz_s
            sets["combined"] = np.r_[site_s, tz_s]

        for name, pos in sets.items():
            pt, lo, hi, sd = boot_ci(pos, rand_s)
            print(f"{c:14} {name:9} ROC {pt:.4f}  95%CI [{lo:.4f}, {hi:.4f}]  "
                  f"(n_pos={len(pos)}, n_neg={len(rand_s)}, SE={sd:.4f})", flush=True)
            rows.append(dict(country=c, set=name, roc=round(pt, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             se=round(sd, 4), n_pos=len(pos), n_neg=len(rand_s),
                             n_boot=N_BOOT))

    os.makedirs(TZ_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

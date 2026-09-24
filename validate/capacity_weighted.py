"""Capacity-weighted validation: does RF-R discriminate solar CAPACITY, not just site count?

Reuses the TZ-SAM validation npz (tz_s scores + tz_xy coords + rand_s negatives), joins each site's
capacity_mw from tz_new_sites.csv, and computes a capacity-WEIGHTED ROC (each positive weighted by its
installed MW) vs the unweighted ROC. If they match, the model ranks where the GIGAWATTS go, not just
where many small farms cluster — answering "are you just catching tiny sites?".

Local, no GEE. Out -> artifacts/paper_v3/figA_combined/capacity_weighted.csv
"""
import csv, glob, os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
TZDIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/tzsam")
TZ_NEW = os.path.join(WS, "Data/external/tz_sam_q1_2026/overlap_analysis/tz_new_sites.csv")
OUT = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figA_combined/capacity_weighted.csv")


def cap_lookup():
    d = {}
    with open(TZ_NEW, newline="") as f:
        for row in csv.DictReader(f):
            try:
                d[(round(float(row["longitude"]), 5), round(float(row["latitude"]), 5))] = float(row["capacity_mw"])
            except (ValueError, KeyError):
                pass
    return d


def wauc(pos, neg, w):
    """Weighted rank-AUC: sum_i w_i * P(neg < pos_i) / sum_i w_i."""
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, "left"); hi = np.searchsorted(ns, pos, "right")
    frac = (lo + 0.5 * (hi - lo)) / len(ns)
    return float(np.sum(w * frac) / np.sum(w)), float(np.mean(frac))  # weighted, unweighted


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    cap = cap_lookup()
    rows = []
    allpos, allneg, allw = [], [], []
    for f in sorted(glob.glob(os.path.join(TZDIR, "*_tzsam_scores.npz"))):
        c = os.path.basename(f).replace("_tzsam_scores.npz", "")
        d = np.load(f)
        tz_s, tz_xy, rand_s = d["tz_s"].astype(float), d["tz_xy"], d["rand_s"].astype(float)
        w = np.array([cap.get((round(float(x), 5), round(float(y), 5)), np.nan) for x, y in tz_xy])
        m = np.isfinite(tz_s) & np.isfinite(w) & (w > 0)
        if m.sum() < 20:
            print(f"skip {c}: only {int(m.sum())} matched capacities"); continue
        wa, ua = wauc(tz_s[m], rand_s, w[m])
        # direct correlation: do higher-scored sites have higher capacity? (log-capacity, since heavy-tailed)
        rho = spearman(tz_s[m], np.log(w[m]))
        print(f"{c:14} n={int(m.sum()):5d}  ROC unweighted={ua:.4f}  cap-weighted={wa:.4f}  "
              f"score~logcap rho={rho:+.3f}  (medcap={np.median(w[m]):.1f}MW, tot={w[m].sum()/1000:.1f}GW)")
        rows.append(dict(country=c, n=int(m.sum()), roc_unweighted=round(ua, 4),
                         roc_capacity_weighted=round(wa, 4), delta=round(wa - ua, 4),
                         score_logcap_spearman=round(rho, 4), total_gw=round(w[m].sum() / 1000, 2)))
        allpos += list(tz_s[m]); allneg += list(rand_s); allw += list(w[m])
    # global pooled (pool all countries; negatives already country-specific — pool them too)
    allpos, allneg, allw = np.array(allpos), np.array(allneg), np.array(allw)
    wa, ua = wauc(allpos, allneg, allw)
    rho = spearman(allpos, np.log(allw))
    print(f"\nGLOBAL pooled: ROC unweighted={ua:.4f}  capacity-weighted={wa:.4f}  "
          f"score~logcap rho={rho:+.3f}  (n={len(allpos)}, totcap={allw.sum()/1000:.0f}GW)")
    rows.append(dict(country="GLOBAL_pooled", n=len(allpos), roc_unweighted=round(ua, 4),
                     roc_capacity_weighted=round(wa, 4), delta=round(wa - ua, 4),
                     score_logcap_spearman=round(rho, 4), total_gw=round(allw.sum() / 1000, 1)))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

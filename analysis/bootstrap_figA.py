"""Paired bootstrap for RF vs MCDA (Fig A) from saved per-point scores.

For each country and each set (inventory / tzsam / combined), reads the per-point npz written by
mcda_bootstrap_v3.py, picks the best MCDA config by point estimate, then resamples the SAME points
with replacement (2000x) and recomputes BOTH RF and that config's ROC on each resample -> a
distribution of the GAP (RF - MCDA). Reports RF ROC [CI], best-MCDA ROC [CI], gap [CI], and whether
the gap CI excludes 0 (i.e. RF significantly beats the strongest published MCDA).

Purely local: no GEE, no model. Out -> artifacts/paper_v3/figA_combined/figA_bootstrap.csv
"""
import csv, glob, os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
DIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figA_combined")
N_BOOT = 2000
SEED = 42


def auc(pos, neg, neg_sorted=None):
    ns = neg_sorted if neg_sorted is not None else np.sort(neg)
    lo = np.searchsorted(ns, pos, side="left")
    hi = np.searchsorted(ns, pos, side="right")
    return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))


def main():
    rows = []
    for npz in sorted(glob.glob(os.path.join(DIR, "*_figA.npz"))):
        country = os.path.basename(npz).replace("_figA.npz", "")
        d = np.load(npz, allow_pickle=True)
        rf = d["rf"].astype(float); label = d["label"].astype(float)
        source = d["source"].astype(str)
        cfg_cols = [k for k in d.files if k not in ("rf", "label", "source")
                    and not k.startswith("raw__")]

        subsets = {"inventory": source != "tz", "tzsam": source != "pvfac",
                   "combined": np.ones(len(source), bool)}
        for setname, smask in subsets.items():
            if setname == "tzsam" and (source == "tz").sum() == 0:
                continue
            # best MCDA config by point estimate on this set
            best, best_roc = None, -1
            for ck in cfg_cols:
                v = d[ck].astype(float)
                ok = smask & np.isfinite(v)
                pos, neg = v[ok & (label == 1)], v[ok & (label == 0)]
                if len(pos) < 5 or len(neg) < 5:
                    continue
                r = auc(pos, neg)
                if r > best_roc:
                    best_roc, best = r, ck
            if best is None:
                continue
            mcda = d[best].astype(float)
            # rows finite for BOTH rf and best mcda, within subset
            ok = smask & np.isfinite(rf) & np.isfinite(mcda)
            y = label[ok]; rfv = rf[ok]; mcv = mcda[ok]
            pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
            rf_pt = auc(rfv[pos_idx], rfv[neg_idx])
            mc_pt = auc(mcv[pos_idx], mcv[neg_idx])

            rng = np.random.default_rng(SEED)
            gaps = np.empty(N_BOOT); rfb = np.empty(N_BOOT); mcb = np.empty(N_BOOT)
            for b in range(N_BOOT):
                pi = pos_idx[rng.integers(0, len(pos_idx), len(pos_idx))]
                ni = neg_idx[rng.integers(0, len(neg_idx), len(neg_idx))]
                rr = auc(rfv[pi], rfv[ni]); mm = auc(mcv[pi], mcv[ni])
                rfb[b] = rr; mcb[b] = mm; gaps[b] = rr - mm
            g_lo, g_hi = np.percentile(gaps, [2.5, 97.5])
            sig = "yes" if g_lo > 0 else "no"
            rf_lo, rf_hi = np.percentile(rfb, [2.5, 97.5])
            mc_lo, mc_hi = np.percentile(mcb, [2.5, 97.5])
            print(f"{country:14} {setname:9} RF {rf_pt:.3f}[{rf_lo:.3f},{rf_hi:.3f}]  "
                  f"MCDA {mc_pt:.3f}[{mc_lo:.3f},{mc_hi:.3f}] ({best})  "
                  f"gap {rf_pt-mc_pt:+.3f}[{g_lo:+.3f},{g_hi:+.3f}] sig={sig}", flush=True)
            rows.append(dict(country=country, set=setname,
                             rf_roc=round(rf_pt, 4), rf_lo=round(rf_lo, 4), rf_hi=round(rf_hi, 4),
                             best_mcda=best, mcda_roc=round(mc_pt, 4),
                             mcda_lo=round(mc_lo, 4), mcda_hi=round(mc_hi, 4),
                             gap=round(rf_pt - mc_pt, 4), gap_lo=round(g_lo, 4),
                             gap_hi=round(g_hi, 4), gap_sig=sig,
                             n_pos=len(pos_idx), n_neg=len(neg_idx)))
    if rows:
        out = os.path.join(DIR, "figA_bootstrap.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

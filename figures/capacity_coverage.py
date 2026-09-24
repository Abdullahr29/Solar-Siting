"""Capacity coverage/gains curve (a) + per-size-bin ROC (b) — the capacity contribution.

(a) Coverage: what % of the world's installed solar CAPACITY (GW) falls in the model's top-X% of
    land by RF score, vs % of SITES (the gap = capacity concentrates less than count: mega-farms on
    lower-scored land). A capacity-weighted gains curve.
(b) Per-size-bin ROC: does site-vs-land discrimination hold across the plant-size spectrum
    (<1, 1-5, 5-20, 20-100, 100+ MW)? Bootstrap 95% CI per bin.

Local, no GEE. Reuses TZ-SAM validation scores (tz_s + tz_xy) + capacity_mw joined from tz_new_sites.
Out -> artifacts/paper_v3/figA_combined/{capacity_coverage.csv, capacity_sizebin_roc.csv} + figure.
"""
import csv, glob, os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
TZDIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/tzsam")
TZ_NEW = os.path.join(WS, "Data/external/tz_sam_q1_2026/overlap_analysis/tz_new_sites.csv")
FIGDIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figures")
OUTDIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figA_combined")


def auc(pos, neg, ns=None):
    ns = np.sort(neg) if ns is None else ns
    lo = np.searchsorted(ns, pos, "left"); hi = np.searchsorted(ns, pos, "right")
    return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))


def boot_ci(pos, neg, n=1000, seed=0):
    rng = np.random.default_rng(seed); ns = np.sort(neg); v = np.empty(n)
    for b in range(n):
        p = pos[rng.integers(0, len(pos), len(pos))]
        nn = neg[rng.integers(0, len(neg), len(neg))]
        v[b] = auc(p, nn)
    return np.percentile(v, [2.5, 97.5])


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cap = {}
    for row in csv.DictReader(open(TZ_NEW)):
        try:
            cap[(round(float(row["longitude"]), 5), round(float(row["latitude"]), 5))] = float(row["capacity_mw"])
        except (ValueError, KeyError):
            pass
    S, W, L = [], [], []
    for f in glob.glob(os.path.join(TZDIR, "*_tzsam_scores.npz")):
        d = np.load(f); ts = d["tz_s"].astype(float); xy = d["tz_xy"]; rs = d["rand_s"].astype(float)
        w = np.array([cap.get((round(float(x), 5), round(float(y), 5)), np.nan) for x, y in xy])
        m = np.isfinite(ts) & np.isfinite(w) & (w > 0)
        S += list(ts[m]); W += list(w[m]); L += list(rs)
    S, W, L = np.array(S), np.array(W), np.array(L)
    print(f"pooled sites n={len(S)}  totcap={W.sum()/1000:.0f}GW  land n={len(L)}")

    # (a) coverage curve
    pcts = np.arange(1, 101)
    cap_capt, site_capt = [], []
    for pct in pcts:
        thr = np.percentile(L, 100 - pct)
        cap_capt.append(100 * W[S >= thr].sum() / W.sum())
        site_capt.append(100 * (S >= thr).mean())
    cap_capt, site_capt = np.array(cap_capt), np.array(site_capt)
    with open(os.path.join(OUTDIR, "capacity_coverage.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["top_pct_land", "pct_capacity_captured", "pct_sites_captured"])
        for i, p in enumerate(pcts):
            w.writerow([p, round(cap_capt[i], 2), round(site_capt[i], 2)])
    for p in [10, 20, 30]:
        i = p - 1
        print(f"  top {p}% land -> {cap_capt[i]:.0f}% of GW ({site_capt[i]:.0f}% of sites), "
              f"concentration {cap_capt[i]/p:.1f}x")

    # (b) per-size-bin ROC
    bins = [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1e9)]
    lbl = ["<1", "1-5", "5-20", "20-100", "100+"]
    rows, rocs, los, his, ns = [], [], [], [], []
    nsort = np.sort(L)
    for (lo, hi), name in zip(bins, lbl):
        sub = S[(W >= lo) & (W < hi)]
        r = auc(sub, L, nsort); ci = boot_ci(sub, L)
        rocs.append(r); los.append(ci[0]); his.append(ci[1]); ns.append(len(sub))
        rows.append(dict(size_mw=name, n=len(sub), roc=round(r, 4), ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4)))
        print(f"  {name:>6} MW: n={len(sub):5d}  ROC={r:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
    with open(os.path.join(OUTDIR, "capacity_sizebin_roc.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # figure: coverage curve + size-bin ROC
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].plot(pcts, cap_capt, color="#e63946", lw=2.5, label="installed capacity (GW)")
    ax[0].plot(pcts, site_capt, color="#457b9d", lw=2, ls="-", label="site count")
    ax[0].plot(pcts, pcts, color="#999", lw=1, ls="--", label="random (no skill)")
    ax[0].fill_between(pcts, pcts, cap_capt, color="#e63946", alpha=0.08)
    for p in [20]:
        ax[0].axvline(p, color="k", lw=0.7, ls=":")
        ax[0].annotate(f"top {p}% land\n{cap_capt[p-1]:.0f}% of GW", (p, cap_capt[p-1]),
                       xytext=(p+6, cap_capt[p-1]-14), fontsize=9,
                       arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].set_xlabel("top X% of land by RF suitability"); ax[0].set_ylabel("% captured")
    ax[0].set_title("Capacity coverage: top-ranked land captures most installed GW\n(capacity lags site-count: mega-farms on lower-scored land)")
    ax[0].legend(fontsize=9); ax[0].set_xlim(0, 100); ax[0].set_ylim(0, 100)

    x = np.arange(len(lbl))
    ax[1].bar(x, rocs, yerr=[np.array(rocs)-np.array(los), np.array(his)-np.array(rocs)],
              color="#2a9d8f", capsize=4)
    for i, n in enumerate(ns):
        ax[1].text(i, rocs[i]+0.02, f"n={n}", ha="center", fontsize=8)
    ax[1].axhline(0.5, color="#999", ls="--", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lbl); ax[1].set_ylim(0.5, 1.0)
    ax[1].set_xlabel("plant capacity (MW)"); ax[1].set_ylabel("site-vs-land ROC")
    ax[1].set_title("Discrimination across the size spectrum\n(holds for small-mid; weaker for the largest farms)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "capacity_coverage.png"), dpi=130)
    plt.close(fig)
    print(f"\nsaved {FIGDIR}/capacity_coverage.png + 2 csvs")


if __name__ == "__main__":
    main()

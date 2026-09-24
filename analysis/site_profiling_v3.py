"""Site-level covariate profiling from the Fig-A per-point npz — enlarged PV_Facility+TZ-SAM sites.

For each country and criterion: median at positives (combined, and PV_Facility vs TZ-SAM separately)
vs random-land negatives, and the single-criterion discrimination AUC (how well that one GIS
covariate alone separates real sites from random land). Answers "what land characteristics do
solar sites share" on the 3-6x larger, independent-augmented positive set.

Local: reads artifacts/paper_v3/figA_combined/<slug>_figA.npz (raw__* criteria). No GEE.
Out -> artifacts/paper_v3/figA_combined/site_profile/{site_profile_v3.csv, <slug>_profile.png}
"""
import csv, glob, os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
DIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figA_combined")
OUT = os.path.join(DIR, "site_profile")
CRIT = ["ghi", "pvout", "gti", "temp", "slope", "elevation", "equatorwardness",
        "road", "grid", "popdens"]
PANEL = ["ghi", "slope", "road", "grid", "elevation", "popdens"]   # for the figure


def auc(pos, neg):
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, side="left"); hi = np.searchsorted(ns, pos, side="right")
    return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for npz in sorted(glob.glob(os.path.join(DIR, "*_figA.npz"))):
        slug = os.path.basename(npz).replace("_figA.npz", "")
        d = np.load(npz, allow_pickle=True)
        if "raw__ghi" not in d.files:
            print(f"skip {slug}: no raw__ criteria"); continue
        src = d["source"].astype(str); label = d["label"].astype(float)
        pos_all = label == 1; neg = label == 0
        for c in CRIT:
            k = f"raw__{c}"
            if k not in d.files:
                continue
            v = d[k].astype(float)
            fn = np.isfinite(v)
            pn = v[neg & fn]
            if len(pn) < 50:
                continue
            def stat(mask):
                x = v[mask & fn]
                if len(x) < 5:
                    return np.nan, np.nan, 0
                return float(np.median(x)), (auc(x, pn) if len(pn) else np.nan), len(x)
            m_all, a_all, n_all = stat(pos_all)
            m_pv, a_pv, n_pv = stat((src == "pvfac"))
            m_tz, a_tz, n_tz = stat((src == "tz"))
            rows.append(dict(country=slug, criterion=c, median_neg=round(float(np.median(pn)), 3),
                             median_pos=round(m_all, 3), auc_pos=round(a_all, 4), n_pos=n_all,
                             median_pvfac=round(m_pv, 3) if n_pv else "", auc_pvfac=round(a_pv, 4) if n_pv else "",
                             median_tzsam=round(m_tz, 3) if n_tz else "", auc_tzsam=round(a_tz, 4) if n_tz else ""))

        # figure: pos vs neg distributions for key criteria
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, c in zip(axes.ravel(), PANEL):
            k = f"raw__{c}"
            if k not in d.files:
                ax.set_visible(False); continue
            v = d[k].astype(float); fn = np.isfinite(v)
            pv = v[pos_all & fn]; nv = v[neg & fn]
            if len(pv) < 5 or len(nv) < 5:
                ax.set_visible(False); continue
            lo, hi = np.nanpercentile(np.r_[pv, nv], [1, 99])
            bins = np.linspace(lo, hi, 40)
            ax.hist(nv, bins=bins, density=True, alpha=0.5, label="random land", color="#999999")
            ax.hist(pv, bins=bins, density=True, alpha=0.5, label="solar sites", color="#e63946")
            ax.set_title(f"{c} (AUC={auc(pv, nv):.2f})"); ax.set_yticks([])
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(f"{slug}: GIS profile of solar sites vs random land "
                     f"(n_pos={int(pos_all.sum())}, n_neg={int(neg.sum())})")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{slug}_profile.png"), dpi=110)
        plt.close(fig)
        print(f"{slug}: profiled {int(pos_all.sum())} sites vs {int(neg.sum())} random", flush=True)

    if rows:
        with open(os.path.join(OUT, "site_profile_v3.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {os.path.join(OUT, 'site_profile_v3.csv')} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

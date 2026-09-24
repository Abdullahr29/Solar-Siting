"""Weak-country wrap-up on the COMBINED (PV_Facility + TZ-SAM) data + a percentile-normalisation demo.

Three short outputs (a few lines in the paper, not a full figure):
  1) ranking: combined RF ROC per country (from figA_bootstrap.csv) -> who is weak.
  2) ROC vs training volume (difficulty, not data-starvation) — Spearman on combined numbers.
  3) NORMALISATION DEMO on the dimmest country (India): raw vs country-percentile scores, showing
     the map "brightens" and the solar/non-solar histograms, WITH the honest note that percentile
     rescaling is rank-preserving (AUC identical) — it aids readability, it does not add skill.

Local: reads artifacts/paper_v3/figA_combined/figA_bootstrap.csv and scores/country/*_R_2019_scores.npz.
Out -> artifacts/paper_v3/figures/weak_country_v3/
"""
import csv, os
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
FIGA = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figA_combined")
COUNTRY = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/scores/country")
OUT = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figures/weak_country_v3")
DEMO_COUNTRY = "india"   # dimmest: lowest site-score distribution of the 13


def auc(pos, neg):
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, "left"); hi = np.searchsorted(ns, pos, "right")
    return (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(ns))


def pctl(scores, ref):
    """percentile of each score within the country's random-land distribution `ref` (0..1)."""
    rs = np.sort(ref)
    lo = np.searchsorted(rs, scores, "left"); hi = np.searchsorted(rs, scores, "right")
    return (lo + 0.5 * (hi - lo)) / len(rs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    os.makedirs(OUT, exist_ok=True)

    # 1) ranking (combined) + val n_pos
    rank = {}
    for r in csv.DictReader(open(os.path.join(FIGA, "figA_bootstrap.csv"))):
        if r["set"] != "combined":
            continue
        rank[r["country"]] = (float(r["rf_roc"]), float(r["rf_lo"]), float(r["rf_hi"]), int(r["n_pos"]))
    order = sorted(rank, key=lambda c: rank[c][0])
    print("=== combined RF ROC ranking (weakest first) ===")
    rows = []
    for c in order:
        roc, lo, hi, npos = rank[c]
        print(f"  {c:14} ROC {roc:.3f} [{lo:.3f},{hi:.3f}]  n_pos={npos}")
        rows.append(dict(country=c, roc=roc, ci_lo=lo, ci_hi=hi, n_pos=npos))
    with open(os.path.join(OUT, "weak_country_v3_ranking.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # 2) ROC vs "how much solar" (n_pos as volume proxy) — difficulty not starvation
    cs = [c for c in order]
    rocs = np.array([rank[c][0] for c in cs]); vols = np.array([rank[c][3] for c in cs], float)
    rho = spearman(vols, rocs)
    print(f"\nROC vs n_pos (volume proxy): Spearman={rho:+.3f}, n={len(cs)}")
    print("  (negative/flat => more solar does NOT mean higher ROC; weakness tracks difficulty)")

    # 3) NORMALISATION DEMO on the dimmest country
    d = np.load(os.path.join(COUNTRY, f"{DEMO_COUNTRY}_R_2019_scores.npz"))
    site_s, rand_s = d["site_s"].astype(float), d["rand_s"].astype(float)
    site_xy, rand_xy = d["site_xy"], d["rand_xy"]
    a_raw = auc(site_s, rand_s)
    site_p, rand_p = pctl(site_s, rand_s), pctl(rand_s, rand_s)
    a_pct = auc(site_p, rand_p)
    print(f"\n[{DEMO_COUNTRY}] raw score range: sites {site_s.min():.2f}-{site_s.max():.2f} "
          f"(med {np.median(site_s):.2f}) | land med {np.median(rand_s):.2f}")
    print(f"[{DEMO_COUNTRY}] AUC raw={a_raw:.4f}  AUC percentile={a_pct:.4f}  (identical: rank-preserving)")
    print(f"[{DEMO_COUNTRY}] median site percentile = {np.median(site_p)*100:.0f}th of national land")

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # (a) raw histograms
    b = np.linspace(min(rand_s.min(), site_s.min()), max(rand_s.max(), site_s.max()), 40)
    ax[0, 0].hist(rand_s, b, density=True, alpha=0.55, color="#999", label="non-solar (random land)")
    ax[0, 0].hist(site_s, b, density=True, alpha=0.55, color="#e63946", label="solar sites")
    ax[0, 0].set_title(f"RAW score — {DEMO_COUNTRY.title()} (AUC={a_raw:.3f})")
    ax[0, 0].set_xlabel("RF suitability (raw)"); ax[0, 0].legend(fontsize=9); ax[0, 0].set_yticks([])
    # (b) percentile histograms
    bp = np.linspace(0, 1, 40)
    ax[0, 1].hist(rand_p, bp, density=True, alpha=0.55, color="#999", label="non-solar")
    ax[0, 1].hist(site_p, bp, density=True, alpha=0.55, color="#e63946", label="solar sites")
    ax[0, 1].set_title(f"COUNTRY-PERCENTILE score (AUC={a_pct:.3f}, identical)")
    ax[0, 1].set_xlabel("within-country percentile"); ax[0, 1].legend(fontsize=9); ax[0, 1].set_yticks([])
    # (c) raw map, (d) percentile map — same points, same color scale 0..1
    sc0 = ax[1, 0].scatter(rand_xy[:, 0], rand_xy[:, 1], c=rand_s, s=3, cmap="viridis", vmin=0, vmax=1)
    ax[1, 0].set_title(f"MAP raw (land scores squished at {np.median(rand_s):.2f})"); ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
    plt.colorbar(sc0, ax=ax[1, 0], fraction=0.046)
    sc1 = ax[1, 1].scatter(rand_xy[:, 0], rand_xy[:, 1], c=rand_p, s=3, cmap="viridis", vmin=0, vmax=1)
    ax[1, 1].set_title("MAP percentile (brightened, full 0-1 range)"); ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
    plt.colorbar(sc1, ax=ax[1, 1], fraction=0.046)
    fig.suptitle(f"Percentile normalisation on a dim country ({DEMO_COUNTRY.title()}): readability up, skill unchanged (AUC {a_raw:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{DEMO_COUNTRY}_normalization_demo.png"), dpi=120)
    plt.close(fig)
    print(f"\nsaved {OUT}/{DEMO_COUNTRY}_normalization_demo.png + ranking csv")


if __name__ == "__main__":
    main()

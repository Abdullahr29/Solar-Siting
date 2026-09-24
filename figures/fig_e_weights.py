"""Fig E — published expert MCDA weights vs the EMPIRICAL importance of each criterion.

Empirical importance = how well each single GIS criterion actually separates real solar sites from
random land, averaged over the 13 countries: |2*AUC - 1| from site_profile_v3.csv (0 = no signal,
1 = perfect), then normalised to a share. Compared against the published weight shares (Richards
2025, Chen 2024). The story: experts put large weight on solar resource (GHI/PVOUT/GTI), but the
real discriminative signal is dominated by infrastructure/terrain (road, grid, slope, popdens).

Local: reads figA_combined/site_profile/site_profile_v3.csv. Out -> figures/fig_e_expert_vs_learned.png (+ csv).
"""
import csv, os
from collections import defaultdict
import numpy as np

WS = os.path.expanduser(os.environ.get("WS", "."))
SP = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figA_combined/site_profile/site_profile_v3.csv")
OUTDIR = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/figures")

# published weights (from mcda_benchmark_v3.WEIGHTS; equatorwardness==aspect)
RICHARDS = {"road": 0.5401, "grid": 0.1392, "slope": 0.0605, "elevation": 0.0225, "ghi": 0.0352,
            "pvout": 0.0253, "gti": 0.0169, "equatorwardness": 0.0759, "temp": 0.0844}
CHEN = {"ghi": 0.1924, "temp": 0.1701, "slope": 0.1576, "elevation": 0.1594, "grid": 0.1661,
        "popdens": 0.1544}
LABEL = {"ghi": "GHI", "pvout": "PVOUT", "gti": "GTI", "temp": "temp", "slope": "slope",
         "elevation": "elev", "equatorwardness": "aspect", "road": "road", "grid": "grid",
         "popdens": "popdens"}
RESOURCE = {"ghi", "pvout", "gti"}   # solar-resource criteria (highlight)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # empirical: mean |2*AUC-1| across countries per criterion
    disc = defaultdict(list)
    for r in csv.DictReader(open(SP)):
        try:
            disc[r["criterion"]].append(abs(2 * float(r["auc_pos"]) - 1))
        except ValueError:
            pass
    emp = {c: float(np.mean(v)) for c, v in disc.items()}
    crits = [c for c in LABEL if c in emp]
    # normalise each series to a share over the common criteria
    def share(d):
        tot = sum(d.get(c, 0.0) for c in crits) or 1.0
        return {c: d.get(c, 0.0) / tot for c in crits}
    emp_s, ric_s, chen_s = share(emp), share(RICHARDS), share(CHEN)
    order = sorted(crits, key=lambda c: emp_s[c], reverse=True)

    rows = [dict(criterion=c, empirical_disc=round(emp[c], 4), emp_share=round(emp_s[c], 4),
                 richards_share=round(ric_s[c], 4), chen_share=round(chen_s[c], 4)) for c in order]
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "fig_e_expert_vs_learned.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['criterion']:15} emp_share={r['emp_share']:.3f}  richards={r['richards_share']:.3f}  chen={r['chen_share']:.3f}")

    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w, [emp_s[c] for c in order], w, label="empirical (real-site discrimination)", color="#2a9d8f")
    ax.bar(x, [ric_s[c] for c in order], w, label="Richards 2025 (expert weight)", color="#e9c46a")
    ax.bar(x + w, [chen_s[c] for c in order], w, label="Chen 2024 (expert weight)", color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in order])
    for i, c in enumerate(order):
        if c in RESOURCE:
            ax.get_xticklabels()[i].set_color("#c1121f"); ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_ylabel("share of importance / weight")
    ax.set_title("Expert MCDA weights vs empirical site-discrimination (mean over 13 countries)\n"
                 "solar-resource criteria (red) carry expert weight but little real discriminative signal")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig_e_expert_vs_learned.png"), dpi=130)
    plt.close(fig)
    print(f"\nsaved {OUTDIR}/fig_e_expert_vs_learned.png (+ csv)")


if __name__ == "__main__":
    main()

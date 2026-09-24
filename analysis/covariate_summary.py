"""Roll up every country's covariate-analysis JSON into one tidy CSV for a morning read.

Reads artifacts/covariate/<slug>/<slug>_<year>_analysis.json for all countries and writes
artifacts/results/covariate_summary.csv — one row per country with the residual-R² headline,
the strongest covariates, and per-covariate Spearman/rank-AUC for the key physical layers.

Usage: python Solar-Siting/covariate_summary.py [--year 2021]
"""
import argparse, csv, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()
    os.chdir(os.path.expanduser("~/Solar_Workspace"))

    files = sorted(glob.glob(f"Solar-Siting/artifacts/figures/covariate/*/*_{args.year}_analysis.json"))
    keycols = ["elevation", "slope", "gsa_ghi", "gsa_pvout", "gsa_dni",
               "grid_dist_m", "road_dist_m"]
    rows = []
    for f in files:
        j = json.load(open(f))
        uni = {r["covariate"]: r for r in j["univariate"]}
        m = j["multivariate"]
        row = {
            "country": j["country"], "year": j["year"], "n_points": j["n_points"],
            "n_features": m["n_features"], "r2_rf": m["r2_rf"], "r2_linear": m["r2_linear"],
            "residual_fraction": m["residual_fraction"],
            "top3_importances": "; ".join(f"{n}={v}" for n, v in m["top_importances"][:3]),
        }
        for c in keycols:
            r = uni.get(c, {})
            row[f"{c}_spearman"] = r.get("spearman", "")
            row[f"{c}_rankauc"] = r.get("rank_auc", "")
        # which categoricals had the strongest effect
        cats = [(r["covariate"], r["eta2"]) for r in j["univariate"] if r["kind"] == "categorical"]
        row["landcover_eta2"] = "; ".join(f"{n}={v}" for n, v in cats)
        rows.append(row)

    if not rows:
        print("no analysis JSONs found yet"); return
    out = "Solar-Siting/artifacts/results/covariate_summary.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = list(rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out} ({len(rows)} countries)")
    for r in rows:
        print(f"  {r['country']:15s} residual={r['residual_fraction']}  R2_rf={r['r2_rf']}  "
              f"top: {r['top3_importances']}")


if __name__ == "__main__":
    main()

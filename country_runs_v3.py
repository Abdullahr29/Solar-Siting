"""Stage 4: country forward validation (installs >= 2021), leakage-deduped, for BOTH schemes.
Decides R vs D on the honest metric. One row per (country, scheme, metric) -> results_master;
per-country R-vs-D bar figure. Robust: a single country/scheme failure is logged, not fatal.
"""
import os, re, sys, json, traceback
import numpy as np, pandas as pd

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
from run_country_rf import run_country

MODELS = {"R": "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib",
          "D": "Solar-Siting/artifacts/paper_v3/models/rf30_v3_D.joblib"}
YEAR, MIN_INSTALL = 2019, 2021
CLUSTER_RE = re.compile(r"^POS_[A-Z]{2,4}_(\d+)_")
# (LSIB name, iso3, inventory-name override)
COUNTRIES = [("Greece", "GRC", None), ("Germany", "DEU", None), ("China", "CHN", None),
             ("Spain", "ESP", None), ("Poland", "POL", None), ("Japan", "JPN", None),
             ("India", "IND", None), ("United States", "USA", "United States of America"),
             ("South Africa", "ZAF", None), ("Colombia", "COL", None),
             ("Philippines", "PHL", None), ("Malaysia", "MYS", None), ("Chile", "CHL", None)]


def training_pv_ids():
    """PV_IDs that were in the TRAIN split (to drop from validation). Defensive: [] on failure."""
    try:
        import geopandas as gpd
        sp = pd.read_csv("Solar-Siting/artifacts/splits/emb_search_split_v2.csv")
        train_clusters = set()
        for b in sp[sp.split == "train"].base.astype(str):
            m = CLUSTER_RE.match(b + "_")
            if m:
                train_clusters.add(int(m.group(1)))
        tds = gpd.read_file("Solar-Siting/global_solar_ml_pipeline.gpkg", layer="training_data_solar")
        idcol = next((c for c in ("cluster_id", "id", "cluster") if c in tds.columns), None)
        oscol = next((c for c in tds.columns if "original_site" in c.lower() or "site_id" in c.lower()), None)
        if idcol is None or oscol is None:
            print(f"  dedup: cannot find cluster/site-id cols in training_data_solar {list(tds.columns)}")
            return set()
        pv = set()
        for cid, ids in zip(tds[idcol], tds[oscol]):
            if int(cid) not in train_clusters:
                continue
            if isinstance(ids, str):
                pv |= {int(x) for x in re.findall(r"\d+", ids)}
            elif ids is not None and not (isinstance(ids, float) and np.isnan(ids)):
                pv.add(int(ids))
        print(f"  dedup set: {len(pv):,} training PV_IDs across {len(train_clusters):,} train clusters")
        return pv
    except Exception as e:
        print(f"  dedup unavailable ({e}); country runs will NOT dedup"); return set()


def main():
    pv = training_pv_ids()
    rows = []
    for lsib, iso3, invname in COUNTRIES:
        for scheme, model in MODELS.items():
            if not os.path.exists(model):
                print(f"skip {lsib}/{scheme}: model missing {model}"); continue
            prefix = f"Solar-Siting/artifacts/paper_v3/scores/country/{lsib.lower().replace(' ', '_')}_{scheme}_{YEAR}"
            if os.path.exists(os.path.join(WS, prefix + "_scores.npz")):
                print(f"skip {lsib}/{scheme} (scores exist — resume)"); continue
            try:
                m = run_country(lsib, year=YEAR, min_install=MIN_INSTALL, model_path=model,
                                out_prefix=prefix, inv_country=invname,
                                exclude_pv_ids=(pv or None))
                for metric, val in [("roc", m["ROC"]), ("pr", m["PR"]),
                                    ("topk05", m["top5"]), ("topk10", m["top10"])]:
                    R.append_result("country2021", "rf", scheme, iso3, metric, val,
                                    n_pos=m["n_sites"], model_path=model,
                                    scores_path=prefix + "_scores.npz",
                                    figure_path=prefix + "_fig.png", aef_year=YEAR,
                                    notes="dedup" if pv else "no-dedup")
                rows.append(dict(country=iso3, scheme=scheme, roc=m["ROC"], top5=m["top5"], n=m["n_sites"]))
                R.log_stage(f"country/{iso3}/{scheme}", prefix, "ok", {"roc": m["ROC"]})
                print(f"### {lsib} {scheme}: ROC {m['ROC']:.3f} top5 {m['top5']:.1f}% n={m['n_sites']}", flush=True)
            except SystemExit as e:
                print(f"FAIL {lsib}/{scheme}: {e}"); R.log_stage(f"country/{iso3}/{scheme}", prefix, f"fail:{e}")
            except Exception:
                traceback.print_exc(); R.log_stage(f"country/{iso3}/{scheme}", prefix, "fail:exc")

    # build the summary + figure from the master index (complete even after a resume)
    mmp = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/results_master.csv")
    if not os.path.exists(mmp):
        return
    df = pd.read_csv(mmp)
    df = df[(df.experiment == "country2021") & (df.metric == "roc")]
    if not len(df):
        return
    df.to_csv("Solar-Siting/artifacts/paper_v3/scores/country/country_summary.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    piv = df.pivot_table(index="scope", columns="scheme", values="value")
    piv = piv.sort_values(piv.columns[0])
    ax = piv.plot(kind="barh", figsize=(8, 7))
    ax.set_xlim(0.5, 1.0); ax.axvline(0.5, c="grey", lw=.5)
    ax.set_xlabel("forward country ROC (installs >= 2021, deduped)")
    ax.set_title("Per-country forward validation: Model R vs D")
    out = "Solar-Siting/artifacts/paper_v3/figures/country/country_R_vs_D_roc.png"
    ax.get_figure().savefig(os.path.join(WS, out), dpi=120, bbox_inches="tight"); plt.close()
    print(f"saved {out}")
    low = {"ZAF", "COL", "PHL", "MYS", "CHL"}
    for scheme in piv.columns:
        col = piv[scheme].dropna()
        lowvals = piv.loc[[i for i in piv.index if i in low], scheme].dropna()
        print(f"  {scheme}: mean ROC all={col.mean():.3f} | low-data={lowvals.mean():.3f}")


if __name__ == "__main__":
    main()

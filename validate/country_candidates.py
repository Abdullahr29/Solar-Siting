"""S2 step 2: country forward-validation (the honest referee) for the T-sweep candidates.
Runs the SAME GEE server-side pipeline as country_runs_v3.py (AEF 2019, installs >= 2021,
training-PV deduped, 13 countries) for each squeezed candidate model, and tabulates ROC/top-k
against the incumbent baseline rf30_v3_R (cap_sol=120; its country numbers are pulled from
results_master -- identical pipeline, already computed). Resumable: skips (model,country)
pairs whose scores already exist.
"""
import os, sys, json, traceback
import numpy as np, pandas as pd

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
from run_country_rf import run_country
from country_runs_v3 import COUNTRIES, YEAR, MIN_INSTALL, training_pv_ids

MODELS = {  # candidate tag -> squeezed joblib (relative to WS)
    "T10M": "Solar-Siting/artifacts/paper_v3/models/rf30_T10M.joblib",
    "T6M":  "Solar-Siting/artifacts/paper_v3/models/rf30_T6M.joblib",
    "T4M":  "Solar-Siting/artifacts/paper_v3/models/rf30_T4M.joblib",
    "T1M":  "Solar-Siting/artifacts/paper_v3/models/rf30_T1M.joblib",
}
SCORES = "Solar-Siting/artifacts/paper_v3/scores/country"
OUTCSV = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/results/tsweep_country.csv")
os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)


def main():
    pv = training_pv_ids()
    rows = []
    for lsib, iso3, invname in COUNTRIES:
        for tag, model in MODELS.items():
            prefix = f"{SCORES}/{lsib.lower().replace(' ', '_')}_{tag}_{YEAR}"
            spath = os.path.join(WS, prefix + "_scores.npz")
            if os.path.exists(spath):
                # already scored in a prior (possibly timed-out) run -> skip the GEE re-run.
                # The metric is preserved in results_master; final table is assembled from there.
                print(f"[skip cached] {lsib}/{tag}", flush=True)
                continue
            try:
                m = run_country(lsib, year=YEAR, min_install=MIN_INSTALL, model_path=model,
                                out_prefix=prefix, inv_country=invname,
                                exclude_pv_ids=(pv or None))
                rows.append(dict(country=iso3, tag=tag, roc=round(m["ROC"], 4),
                                 pr=round(m["PR"], 4), top5=round(m["top5"], 1),
                                 top10=round(m["top10"], 1), n_sites=m["n_sites"]))
                R.append_result("country2021_tsweep", "rf", tag, iso3, "roc", m["ROC"],
                                n_pos=m["n_sites"], model_path=model,
                                scores_path=prefix + "_scores.npz", aef_year=YEAR,
                                notes="tsweep-candidate")
                print(f"### {lsib} {tag}: ROC {m['ROC']:.3f} top5 {m['top5']:.1f}% n={m['n_sites']}",
                      flush=True)
                pd.DataFrame(rows).to_csv(OUTCSV, index=False)   # checkpoint each run
            except SystemExit as e:
                print(f"FAIL {lsib}/{tag}: {e}", flush=True)
            except Exception as e:
                print(f"ERROR {lsib}/{tag}: {e}\n{traceback.format_exc()}", flush=True)
    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print(f"\nwrote {OUTCSV} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()

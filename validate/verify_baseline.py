"""Reproduction check: run the incumbent rf30_v3_R through the SAME driver/path used for the
T-sweep candidates, on the countries where candidates lost (+Greece as a win control), and
compare to the numbers recorded in results_master. If they match, the candidate-vs-incumbent
comparison is sound (no pipeline drift).
"""
import os, sys
import numpy as np, pandas as pd
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
from run_country_rf import run_country
from country_runs_v3 import YEAR, MIN_INSTALL, training_pv_ids

MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
# (LSIB name, iso3, invname, recorded R ROC)
CHECK = [("Greece", "GRC", None, 0.89526), ("Germany", "DEU", None, 0.84994),
         ("China", "CHN", None, 0.87584),
         ("United States", "USA", "United States of America", 0.88334)]
SC = "Solar-Siting/artifacts/paper_v3/scores/country"

pv = training_pv_ids()
print(f"\n{'country':>14} {'recorded':>9} {'reproduced':>11} {'diff':>8}")
for lsib, iso3, invname, rec in CHECK:
    prefix = f"{SC}/{lsib.lower().replace(' ', '_')}_Rverify_{YEAR}"
    m = run_country(lsib, year=YEAR, min_install=MIN_INSTALL, model_path=MODEL,
                    out_prefix=prefix, inv_country=invname, exclude_pv_ids=(pv or None))
    print(f"{iso3:>14} {rec:>9.4f} {m['ROC']:>11.4f} {m['ROC']-rec:>+8.4f}", flush=True)

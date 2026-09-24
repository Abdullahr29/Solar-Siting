"""Stage 2 of the budget sweep: country forward-ROC for the BEST feasible RF per data size.
Reads each `{POOL}_stage1.csv`, picks the top-val_roc feasible config per pool (plus the
in-harness 'incumbent' config as an internal control), and runs the same GEE country pipeline
(13 countries, AEF 2019, installs>=2021, deduped) used everywhere else. Resumable.
"""
import os, sys, glob, traceback
import numpy as np, pandas as pd
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
# FIX: geemap.ml.rf_to_strings mis-serializes best-first (max_leaf_nodes) trees -> GEE
# 'decisionTreeEnsemble: expected 8, got 3'. Patch reorders each tree to pre-order DFS
# before serialization (lossless). Must run before any run_country() call.
import gee_tree_fix; gee_tree_fix.apply()
from run_country_rf import run_country
from country_runs_v3 import COUNTRIES, YEAR, MIN_INSTALL, training_pv_ids

SWEEP = "Solar-Siting/artifacts/paper_v3/budget_sweep"
SC = f"{SWEEP}/scores/country"
os.makedirs(os.path.join(WS, SC), exist_ok=True)
OUTCSV = os.path.join(WS, f"{SWEEP}/budget_country.csv")

# select models: best feasible per pool + each pool's incumbent control
models = {}
for f in sorted(glob.glob(os.path.join(WS, SWEEP, "*_stage1.csv"))):
    d = pd.read_csv(f)
    pool = d.pool.iloc[0]
    best = d[d.feasible].sort_values("val_roc", ascending=False).head(1)
    if len(best):
        models[f"{pool}_BEST_{best.tag.iloc[0]}"] = best.model_path.iloc[0]
    inc = d[d.tag == "incumbent"]
    if len(inc):
        models[f"{pool}_incumbent"] = inc.model_path.iloc[0]
print("models to validate:")
for k, v in models.items():
    print(f"  {k}: {v}")

pv = training_pv_ids()
rows = []
for lsib, iso3, invname in COUNTRIES:
    for tag, model in models.items():
        prefix = f"{SC}/{lsib.lower().replace(' ', '_')}__{tag}_{YEAR}"
        if os.path.exists(os.path.join(WS, prefix + "_scores.npz")):
            print(f"[skip cached] {lsib}/{tag}", flush=True); continue
        try:
            m = run_country(lsib, year=YEAR, min_install=MIN_INSTALL, model_path=model,
                            out_prefix=prefix, inv_country=invname, exclude_pv_ids=(pv or None))
            rows.append(dict(country=iso3, tag=tag, roc=round(m["ROC"], 4), pr=round(m["PR"], 4),
                             top5=round(m["top5"], 1), n_sites=m["n_sites"]))
            print(f"### {lsib} {tag}: ROC {m['ROC']:.3f} n={m['n_sites']}", flush=True)
            pd.DataFrame(rows).to_csv(OUTCSV, index=False)
        except SystemExit as e:
            print(f"FAIL {lsib}/{tag}: {e}", flush=True)
        except Exception as e:
            print(f"ERROR {lsib}/{tag}: {e}\n{traceback.format_exc()}", flush=True)
pd.DataFrame(rows).to_csv(OUTCSV, index=False)
print(f"\nwrote {OUTCSV} ({len(rows)} rows)", flush=True)

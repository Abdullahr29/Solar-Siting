"""Validate gee_tree_fix: (1) reorder preserves predictions exactly; (2) all 7 winners now
round-trip through GEE. Writes results line-buffered to stdout (run with -u)."""
import os, signal, joblib, numpy as np
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import gee_tree_fix
from gee_tree_fix import reorder_tree_preorder
import geemap.ml as gml
gee_tree_fix.apply()                                   # patch rf_to_strings
import ee
ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
M = "Solar-Siting/artifacts/paper_v3/budget_sweep/models"
img = ee.Image.constant([0.05]*64).rename(BANDS)
pt = ee.Geometry.Point([0.0, 0.0])

BEST = ["px_v3_ne30_mf32_ms0.34_gini", "T1000000_ne30_mf32_ms0.34_gini",
        "T2000000_ne30_mf32_ms0.6_gini", "T4000000_ne30_mf32_ms1.0_gini",
        "T6000000_ne30_mf32_ms0.6_entropy", "T10000000_ne30_mf32_ms0.6_entropy",
        "T200000000_ne30_mf32_ms0.6_entropy"]

# (1) prediction-equivalence check on the first model
rf = joblib.load(f"{M}/{BEST[0]}.joblib")
Xr = np.random.default_rng(0).normal(0, 0.1, size=(2000, 64)).astype(np.float32)
p_before = rf.predict_proba(Xr)[:, 1]
for e_i in range(len(rf.estimators_)):
    rf.estimators_[e_i] = reorder_tree_preorder(rf.estimators_[e_i])
p_after = rf.predict_proba(Xr)[:, 1]
print(f"PRED-EQUIV max|Δ|={np.abs(p_before-p_after).max():.2e}  (0 => reorder is lossless)", flush=True)

class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
for name in BEST:
    crit = "entropy" if "entropy" in name else "gini"
    try:
        rf = joblib.load(f"{M}/{name}.joblib")
        trees = gml.rf_to_strings(rf, BANDS, processes=8, output_mode="PROBABILITY")
        clf = gml.strings_to_classifier(trees)
        val = img.classify(clf).rename("score")
        signal.alarm(120)
        r = val.reduceRegion(ee.Reducer.first(), pt, scale=10).getInfo()
        signal.alarm(0)
        print(f"PASS  {name:>34} [{crit}] -> {r}", flush=True)
    except TO:
        print(f"TIMEOUT {name}", flush=True)
    except Exception as e:
        signal.alarm(0)
        print(f"FAIL  {name:>34} [{crit}] -> {str(e).splitlines()[0][:110]}", flush=True)
print("DONE", flush=True)

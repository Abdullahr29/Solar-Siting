"""Round-trip each pool's BEST model through GEE to find which tree-strings actually PARSE.
Classifies a single constant 64-band point -> forces server-side parse of the ensemble.
Fast (no country data). Reports PASS/FAIL per model so we know which winners are deployable.
"""
import os, sys, signal, joblib
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import ee
from geemap import ml
ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
MODELS = "Solar-Siting/artifacts/paper_v3/budget_sweep/models"

# one constant point image with all 64 bands = A00..A63
img = ee.Image.constant([0.05]*64).rename(BANDS)
pt = ee.Geometry.Point([0.0, 0.0])

BEST = {
 "px_v3":      "px_v3_ne30_mf32_ms0.34_gini",
 "T1000000":   "T1000000_ne30_mf32_ms0.34_gini",
 "T2000000":   "T2000000_ne30_mf32_ms0.6_gini",
 "T4000000":   "T4000000_ne30_mf32_ms1.0_gini",
 "T6000000":   "T6000000_ne30_mf32_ms0.6_entropy",
 "T10000000":  "T10000000_ne30_mf32_ms0.6_entropy",
 "T200000000": "T200000000_ne30_mf32_ms0.6_entropy",
}

class TO(Exception): pass
def _to(s,f): raise TO()
signal.signal(signal.SIGALRM, _to)

for pool, name in BEST.items():
    path = f"{MODELS}/{name}.joblib"
    crit = "entropy" if "entropy" in name else "gini"
    try:
        rf = joblib.load(path)
        trees = ml.rf_to_strings(rf, BANDS, processes=8, output_mode="PROBABILITY")
        clf = ml.strings_to_classifier(trees)
        val = img.classify(clf).rename("score")
        signal.alarm(90)
        r = val.reduceRegion(ee.Reducer.first(), pt, scale=10).getInfo()
        signal.alarm(0)
        print(f"PASS  {pool:>11} [{crit:>7}] {name}  -> {r}", flush=True)
    except TO:
        print(f"TIMEOUT {pool:>11} [{crit:>7}] {name}", flush=True)
    except Exception as e:
        signal.alarm(0)
        msg = str(e).replace("\n"," ")[:130]
        print(f"FAIL  {pool:>11} [{crit:>7}] {name}  -> {msg}", flush=True)

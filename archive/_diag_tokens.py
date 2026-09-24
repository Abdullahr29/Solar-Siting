"""Find the malformed tree-string line. Compare a WORKING incumbent vs a FAILING winner.
GEE error = 'decisionTreeEnsemble: Error parsing line 4: expected 8, got 3'.
Pure local (no GEE). Dump line-by-line token counts + flag any line whose token count
is not the modal one, across ALL trees.
"""
import os, joblib, collections
os.chdir(os.path.expanduser("~/Solar_Workspace"))
from geemap import ml
BANDS = [f"A{i:02d}" for i in range(64)]
M = "Solar-Siting/artifacts/paper_v3/budget_sweep/models"

def dump(name, ntrees_scan=None):
    rf = joblib.load(f"{M}/{name}.joblib")
    trees = ml.rf_to_strings(rf, BANDS, processes=8, output_mode="PROBABILITY")
    print(f"\n########## {name}  ({len(trees)} trees) ##########")
    t0 = trees[0].split("\n")
    print(f"tree0 has {len(t0)} lines; first 8 with repr + ntok:")
    for i, ln in enumerate(t0[:8]):
        print(f"  L{i} ntok={len(ln.split()):>2}  {repr(ln)}")
    # token-count histogram + anomalies across all trees
    hist = collections.Counter()
    anomalies = []
    for ti, ts in enumerate(trees if ntrees_scan is None else trees[:ntrees_scan]):
        lines = ts.split("\n")
        for li, ln in enumerate(lines):
            if ln.strip() == "":
                continue
            nt = len(ln.split())
            hist[nt] += 1
            if nt < 5:                       # suspiciously short
                anomalies.append((ti, li, nt, ln))
    print(f"token-count histogram (all trees): {dict(sorted(hist.items()))}")
    print(f"short lines (<5 tok): {len(anomalies)}")
    for a in anomalies[:10]:
        print(f"   tree{a[0]} L{a[1]} ntok={a[2]} {repr(a[3])}")

dump("T10000000_incumbent")                       # WORKS in GEE
dump("px_v3_ne30_mf32_ms0.34_gini")               # FAILS in GEE

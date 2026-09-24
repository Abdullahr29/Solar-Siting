"""Diagnose the GEE 'expected 8, got 3' tree-string parse error.
rf_to_strings is local (sklearn->string), no GEE call needed to inspect the malformed line.
Compare an ENTROPY winner vs a GINI winner; find lines whose token-count != the modal count.
"""
import os, joblib, numpy as np
os.chdir(os.path.expanduser("~/Solar_Workspace"))
from geemap import ml
BANDS = [f"A{i:02d}" for i in range(64)]
MODELS = f"Solar-Siting/artifacts/paper_v3/budget_sweep/models"

def probe(name):
    path = f"{MODELS}/{name}.joblib"
    if not os.path.exists(path):
        print(f"  MISSING {name}"); return
    rf = joblib.load(path)
    trees = ml.rf_to_strings(rf, BANDS, processes=8, output_mode="PROBABILITY")
    print(f"\n=== {name}  ({len(trees)} trees, {sum(len(s) for s in trees)/1e6:.2f} MB) ===")
    # inspect tree 0
    t0 = trees[0].split("\n")
    print(f"tree0: {len(t0)} lines. First 6 lines with token counts:")
    for i, ln in enumerate(t0[:6]):
        print(f"  line{i}: ntok={len(ln.split())}  |{ln}|")
    # scan ALL trees for anomalous lines (very few tokens on a non-empty, non-header line)
    bad = []
    for ti, ts in enumerate(trees):
        for li, ln in enumerate(ts.split("\n")):
            s = ln.split()
            if len(s) and len(s) <= 3 and not ln.startswith("#") and li > 1:
                bad.append((ti, li, len(s), ln[:80]))
    print(f"anomalous short lines (<=3 tok, past header): {len(bad)}")
    for b in bad[:8]:
        print(f"   tree{b[0]} line{b[1]} ntok={b[2]} |{b[3]}|")

for n in ["T10000000_ne30_mf32_ms0.6_entropy",     # the failing one
          "px_v3_ne30_mf32_ms0.34_gini",            # a gini winner
          "T1000000_ne30_mf32_ms0.34_gini"]:        # another gini winner
    probe(n)

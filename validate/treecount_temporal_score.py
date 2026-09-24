"""Temporal-holdout scoring: sample AEF-2022 at the SAME 2024-install validation points used by
the original temporal test (China, USA), then score ne15/20/30 (temporal-trained, never saw 2024)
+ the deployed rf30_temporal (honest baseline) LOCALLY. Honest 2024-forecast ROC.
Run --sample first (GEE, cached), then (after fits) plain to score."""
import os, sys, time, glob, numpy as np, joblib
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
BANDS = [f"A{i:02d}" for i in range(64)]
PRIOR = {"china": "artifacts/figures/temporal/china_2024/china_honest_temporal_2022_scores.npz",
         "usa":   "artifacts/figures/temporal/usa_2024/usa_honest_temporal_2022_scores.npz"}
EMB = "artifacts/paper_v3/pools_t3/temporal_emb"; os.makedirs(EMB, exist_ok=True)
TC = "artifacts/paper_v3/pools_t3/treecount/temporal"

def sample_2022():
    import ee
    ee.Initialize(project="ee-abdullahr-solar")
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate("2022-01-01", "2023-01-01").mosaic().select(BANDS))
    def s(xy, b=1000):
        out = np.full((len(xy),64), np.nan, np.float32)
        for i in range(0,len(xy),b):
            for att in range(4):
                try:
                    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a),float(c)])) for a,c in xy[i:i+b]])
                    res = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()
                    out[i:i+b] = [[f["properties"].get(bn,np.nan) for bn in BANDS] for f in res["features"]]; break
                except Exception as e: print(f"  retry {att}: {str(e)[:60]}",flush=True); time.sleep(20*(att+1))
            print(f"  {min(i+b,len(xy))}/{len(xy)}",flush=True)
        return out
    for c,p in PRIOR.items():
        op = f"{EMB}/{c}_2022emb.npz"
        if os.path.exists(op): print(f"[skip] {c}"); continue
        z = np.load(p); sx, rx = z["site_xy"], z["rand_xy"]
        print(f"{c}: {len(sx)} sites, {len(rx)} rand", flush=True)
        np.savez(op, site=s(sx), rand=s(rx)); print(f"  wrote {op}", flush=True)

def score():
    models = {f"ne{ne}": f"{TC}/temporal_ne{ne}_mf32.joblib" for ne in [15,20,30]}
    models["deployed_rf30_temporal"] = "artifacts/models/rf30_temporal_le2021.joblib"
    rfs = {k: joblib.load(v) for k,v in models.items() if os.path.exists(v)}
    print("scored models:", list(rfs), flush=True)
    import pandas as pd; rows={}
    for c in PRIOR:
        z = np.load(f"{EMB}/{c}_2022emb.npz"); Xs, Xr = z["site"], z["rand"]
        fs = np.isfinite(Xs).all(1); fr = np.isfinite(Xr).all(1); Xs, Xr = Xs[fs], Xr[fr]
        y = np.r_[np.ones(len(Xs)), np.zeros(len(Xr))]
        rows[c] = {"n_2024_sites": len(Xs)}
        for k,rf in rfs.items():
            x = np.r_[rf.predict_proba(Xs)[:,1], rf.predict_proba(Xr)[:,1]]
            rows[c][k] = round(roc_auc_score(y,x),4)
    df = pd.DataFrame(rows).T
    print("\n=== TEMPORAL HOLDOUT: honest 2024-forecast ROC (models never saw 2024) ===")
    print(df.to_string())
    df.to_csv(f"{TC}/temporal_holdout_roc.csv"); print(f"\nwrote {TC}/temporal_holdout_roc.csv",flush=True)

if __name__ == "__main__":
    if "--sample" in sys.argv: sample_2022()
    else: score()

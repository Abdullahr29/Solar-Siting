import time, numpy as np, joblib, os
WS = os.path.expanduser("~/Solar_Workspace")
rf = joblib.load(f"{WS}/Solar-Siting/artifacts/paper_v3/models/asset_R_alldata.joblib")
print("RF:", rf.n_estimators, "trees; n_features:", rf.n_features_in_)
try:
    depths = [t.get_depth() for t in rf.estimators_]
    print("tree depths min/mean/max:", min(depths), round(sum(depths)/len(depths),1), max(depths))
    print("leaves mean:", round(sum(t.get_n_leaves() for t in rf.estimators_)/len(rf.estimators_)))
except Exception as e:
    print("depth introspection failed:", e)

N = 2_000_000
X = np.random.randn(N, rf.n_features_in_).astype(np.float32)
# single-thread
rf.n_jobs = 1
t0 = time.time(); rf.predict_proba(X[:200_000]); dt1 = time.time()-t0
r1 = 200_000/dt1
# all cores
import multiprocessing as mp
ncpu = mp.cpu_count()
rf.n_jobs = -1
t0 = time.time(); rf.predict_proba(X); dt = time.time()-t0
rall = N/dt
print(f"cores available: {ncpu}")
print(f"1-thread: {r1:,.0f} px/s")
print(f"{ncpu}-thread ({dt:.1f}s for {N:,}): {rall:,.0f} px/s")
for label, npx in [("Greece 1.32e9", 1.32e9), ("India ~33e9", 33e9), ("global land 1.49e12", 1.49e12)]:
    secs = npx / rall
    print(f"  {label}: {secs/3600:,.1f} core-hours-equiv on this box  ({secs/86400:.2f} days at this rate)")

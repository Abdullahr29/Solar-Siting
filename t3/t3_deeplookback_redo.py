"""Redo the deep-lookback probe (STEP6 of step56.py) with the NEW model ne30_mf32.
Score held-out inventory sites on FIXED AEF 2017 vs random land, binned by lead time
(install-2017 = 2..7 yrs). Flat ROC across lead => pre-existing-land signal (not leakage);
rising toward short lead => construction signal. Uses the CACHED 2017 embeddings (no GEE).
Recomputes deployed + ne30_mf32 (+ old rf_m2/rf_m3) on the SAME points for a clean comparison.
"""
import os, numpy as np, pandas as pd, geopandas as gpd, joblib
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
OUT = "artifacts/paper_v3/temporal_probe/big"
GPKG, INV = "global_solar_ml_pipeline.gpkg", "global_pv_facility_inventory.gpkg"
DEEP_YEAR, SEED, N_PER = 2017, 0, 1000

# --- reproduce the exact ev sample (held-out inventory, groupby year sample 1000, rs=0) ---
base = gpd.read_file(GPKG, layer="base_train"); bid = set(base["PV_ID"].tolist())
inv = gpd.read_file(INV)
held = inv[(~inv["PV_ID"].isin(bid)) & inv.year.between(2019, 2024)].copy()
parts = [sub.sample(min(N_PER, len(sub)), random_state=SEED) for _, sub in held.groupby("year")]
ev = pd.concat(parts).reset_index(drop=True)
ev_install = ev.year.to_numpy()
print(f"reproduced ev: {len(ev)} sites; per-year:\n{ev.year.value_counts().sort_index().to_string()}", flush=True)

Es = np.load(f"{OUT}/emb_deepsite_2017.npy"); Er = np.load(f"{OUT}/emb_deeprand_2017.npy")
print(f"cached emb: deepsite {Es.shape}, deeprand {Er.shape}", flush=True)
assert len(Es) == len(ev), f"ALIGN MISMATCH emb {len(Es)} vs ev {len(ev)} (inventory changed?)"
oks = np.isfinite(Es).all(1); okr = np.isfinite(Er).all(1)
lead = (ev_install - DEEP_YEAR)[oks]
print(f"valid: {oks.sum()} sites, {okr.sum()} rand", flush=True)

def roc(ps, pr):
    return roc_auc_score(np.r_[np.ones(len(ps)), np.zeros(len(pr))], np.r_[ps, pr])

MODELS = {
 "deployed_incumbent":     "artifacts/paper_v3/budget_sweep/models/px_v3_incumbent.joblib",
 "ne30_mf32":              "artifacts/paper_v3/budget_sweep/models/px_v3_ne30_mf32_ms0.34_gini.joblib",
 "ne25_mf32":              "artifacts/paper_v3/pools_t3/treecount/px_v3_ne25_mf32.joblib",
 "ne20_mf32":              "artifacts/paper_v3/pools_t3/treecount/px_v3_ne20_mf32.joblib",
 "ne15_mf32":              "artifacts/paper_v3/pools_t3/treecount/px_v3_ne15_mf32.joblib",
}
rows = {}
for name, p in MODELS.items():
    if not os.path.exists(p): print(f"  [skip] {name} missing"); continue
    rf = joblib.load(p)
    ps = rf.predict_proba(Es[oks])[:,1]; pr = rf.predict_proba(Er[okr])[:,1]
    r = {"roc_all": round(roc(ps, pr), 4)}
    for L in range(2, 8):
        m = lead == L
        if m.sum() >= 30:
            r[f"L{L}"] = round(roc(ps[m], pr), 4)
    rows[name] = r
df = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print("\n=== DEEP-LOOKBACK: ROC on fixed AEF-2017, by lead time (install-2017 yrs) ===")
print("(flat across L2->L7 = pre-existing land signal, not construction/leakage)")
print(df.to_string())
# slope L2->L7 per model (construction-signal indicator)
print("\n=== L2 vs L7 (short vs long lead): drop => some construction signal ===")
for n, r in rows.items():
    if "L2" in r and "L7" in r:
        print(f"  {n:>20}: L2={r['L2']:.4f}  L7={r['L7']:.4f}  drop={r['L2']-r['L7']:+.4f}")
df.to_csv(f"{OUT}/deeplookback_ne30_mf32.csv")
print(f"\nwrote {OUT}/deeplookback_ne30_mf32.csv", flush=True)

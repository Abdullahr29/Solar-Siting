"""Rigorous data-pool comparison for the BEST (ne30_mf32) architecture.
- per-pool n-weighted country ROC + paired bootstrap 95% CI
- paired bootstrap of (pool - cap120) to test which differences are real
Validation sites are identical across pools within a country (fixed seed), so we resample
the SAME site/rand indices across all pools each bootstrap iter -> proper paired test.
"""
import glob, os, re, numpy as np
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace"))
D = "Solar-Siting/artifacts/paper_v3/budget_sweep/scores/country"
POOLS = ["px_v3","T1000000","T2000000","T4000000","T6000000","T10000000","T200000000"]
PL = {"px_v3":"cap120","T1000000":"1M","T2000000":"2M","T4000000":"4M",
      "T6000000":"6M","T10000000":"10M","T200000000":"uncap"}
BESTG = {  # the winning BEST tag per pool
 "px_v3":"px_v3_BEST_ne30_mf32_ms0.34_gini","T1000000":"T1000000_BEST_ne30_mf32_ms0.34_gini",
 "T2000000":"T2000000_BEST_ne30_mf32_ms0.6_gini","T4000000":"T4000000_BEST_ne30_mf32_ms1.0_gini",
 "T6000000":"T6000000_BEST_ne30_mf32_ms0.6_entropy","T10000000":"T10000000_BEST_ne30_mf32_ms0.6_entropy",
 "T200000000":"T200000000_BEST_ne30_mf32_ms0.6_entropy"}

# load site_s/rand_s per (pool, country); only countries present in ALL pools (for a fair paired set)
data = {p: {} for p in POOLS}
countries = None
for p in POOLS:
    cs = set()
    for f in glob.glob(f"{D}/*__{BESTG[p]}_2019_scores.npz"):
        c = os.path.basename(f).split("__")[0]
        z = np.load(f); data[p][c] = (z["site_s"], z["rand_s"]); cs.add(c)
    countries = cs if countries is None else (countries & cs)
countries = sorted(countries, key=lambda c: -len(data["px_v3"][c][0]))
nsite = {c: len(data["px_v3"][c][0]) for c in countries}
Ntot = sum(nsite.values())
print(f"countries in common (all pools): {len(countries)}  total sites={Ntot}")
print("  " + ", ".join(f"{c}({nsite[c]})" for c in countries))

def nw_roc(pool, boot_idx=None):
    """n-weighted mean per-country ROC; boot_idx[c]=(site_ix,rand_ix) if bootstrapping."""
    num = 0.0
    for c in countries:
        s, r = data[pool][c]
        if boot_idx is not None:
            si, ri = boot_idx[c]; s, r = s[si], r[ri]
        y = np.r_[np.ones(len(s)), np.zeros(len(r))]; x = np.r_[s, r]
        num += nsite[c] * roc_auc_score(y, x)
    return num / Ntot

# point estimates
print("\n=== per-pool BEST: n-weighted country ROC (point est) ===")
pt = {p: nw_roc(p) for p in POOLS}
for p in POOLS: print(f"  {PL[p]:>7}: {pt[p]:.4f}")

# paired bootstrap
rng = np.random.default_rng(0); B = 2000
boot = {p: np.empty(B) for p in POOLS}
diff = {p: np.empty(B) for p in POOLS}   # cap120 - pool
for b in range(B):
    idx = {}
    for c in countries:
        ns = len(data["px_v3"][c][0]); nr = len(data["px_v3"][c][1])
        idx[c] = (rng.integers(0, ns, ns), rng.integers(0, nr, nr))
    vals = {p: nw_roc(p, idx) for p in POOLS}
    for p in POOLS:
        boot[p][b] = vals[p]; diff[p][b] = vals["px_v3"] - vals[p]

print("\n=== per-pool BEST ROC with 95% CI (paired bootstrap, B=2000) ===")
for p in POOLS:
    lo, hi = np.percentile(boot[p], [2.5, 97.5])
    print(f"  {PL[p]:>7}: {pt[p]:.4f}  [{lo:.4f}, {hi:.4f}]")

print("\n=== cap120 - pool  (paired diff; CI excludes 0 => cap120 significantly better) ===")
for p in POOLS:
    if p == "px_v3": continue
    d = diff[p]; lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "SIG" if (lo > 0 or hi < 0) else "tied"
    print(f"  cap120 - {PL[p]:>6}: {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")

# USA + China trend (the two dominant countries), BEST ROC across pools
print("\n=== USA & China BEST ROC across pools (they drive the n-weighted mean) ===")
for c in ["united_states", "china"]:
    if c not in countries: continue
    line = "  ".join(f"{PL[p]}:{roc_auc_score(np.r_[np.ones(len(data[p][c][0])),np.zeros(len(data[p][c][1]))], np.r_[data[p][c][0],data[p][c][1]]):.4f}" for p in POOLS)
    print(f"  {c:>13}: {line}")

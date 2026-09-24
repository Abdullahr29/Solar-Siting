"""Is cap120's data-pool win robust to country composition, or a big-country artifact?
Uses the per-country ne30_mf32 BEST ROCs (cached t-2 country scores; no GEE).
Reports: n-weighted vs macro-average (equal weight) vs mean-rank; leave-big-country-out;
and a bootstrap OVER COUNTRIES (resample the country set) -> P(pool is best) under each weighting.
This simulates 'what if the 13 countries were a different/larger sample'."""
import glob, os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace"))
SC = "Solar-Siting/artifacts/paper_v3/budget_sweep/scores/country"
PL = {"px_v3":"cap120","T1000000":"1M","T2000000":"2M","T4000000":"4M",
      "T6000000":"6M","T10000000":"10M","T200000000":"uncap"}
order = ["cap120","1M","2M","4M","6M","10M","uncap"]
rows, nsite = {}, {}
for f in glob.glob(f"{SC}/*_BEST_*_2019_scores.npz"):
    m = re.match(r"(?P<c>[a-z_]+)__(?P<pool>px_v3|T\d+)_BEST_.*_2019_scores\.npz", os.path.basename(f))
    if not m: continue
    z = np.load(f); s, r = z["site_s"], z["rand_s"]
    roc = roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(r))], np.r_[s, r])
    rows.setdefault(m["c"], {})[PL[m["pool"]]] = roc; nsite[m["c"]] = len(s)
df = pd.DataFrame(rows).T[order]
n = pd.Series(nsite).reindex(df.index).to_numpy()
M = df.to_numpy()                                        # countries x pools
countries = df.index.to_list()

def summarize(tag, weights=None, idx=None):
    sub = M if idx is None else M[idx]
    w = (np.ones(len(sub)) if weights is None else weights[idx if idx is not None else slice(None)])
    wm = np.average(sub, axis=0, weights=w)
    return dict(zip(order, np.round(wm, 4)))

print("=== pool ROC: n-weighted vs macro-average (equal per country) vs mean-rank ===")
nw = summarize("nw", n)
macro = summarize("macro")
ranks = pd.DataFrame(M, columns=order, index=countries).rank(axis=1, ascending=False)  # 1=best
meanrank = ranks.mean(0).round(2).to_dict()
cmp = pd.DataFrame({"n_weighted": nw, "macro_avg": macro, "mean_rank(1=best)": meanrank})
print(cmp.sort_values("macro_avg", ascending=False).to_string())
print(f"\nbest by n-weighted: {max(nw,key=nw.get)} | by macro-avg: {max(macro,key=macro.get)} | "
      f"by mean-rank: {min(meanrank,key=meanrank.get)}")

print("\n=== leave-one-BIG-country-out (macro-avg best pool) ===")
for drop in ["united_states","china","japan","germany"]:
    if drop in countries:
        idx = [i for i,c in enumerate(countries) if c != drop]
        mm = summarize("", None, np.array(idx))
        print(f"  drop {drop:>13}: macro-best = {max(mm,key=mm.get)}  ({mm})")

print("\n=== BOOTSTRAP over countries (resample 13 w/ replacement, B=10000) ===")
rng = np.random.default_rng(0); B = 10000
win_macro = {p:0 for p in order}; win_nw = {p:0 for p in order}
for _ in range(B):
    bi = rng.integers(0, len(countries), len(countries))
    mm = np.average(M[bi], axis=0)
    ww = np.average(M[bi], axis=0, weights=n[bi])
    win_macro[order[int(np.argmax(mm))]] += 1
    win_nw[order[int(np.argmax(ww))]] += 1
print("P(pool is BEST) under macro-avg (equal country weight):")
for p in order:
    if win_macro[p]: print(f"  {p:>7}: {100*win_macro[p]/B:5.1f}%")
print("P(pool is BEST) under n-weighted:")
for p in order:
    if win_nw[p]: print(f"  {p:>7}: {100*win_nw[p]/B:5.1f}%")

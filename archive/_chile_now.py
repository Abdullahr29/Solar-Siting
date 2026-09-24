import glob, os, re, numpy as np
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace"))
D = "Solar-Siting/artifacts/paper_v3/budget_sweep/scores/country"
PL = {"px_v3":"cap120","T1000000":"1M","T2000000":"2M","T4000000":"4M",
      "T6000000":"6M","T10000000":"10M","T200000000":"uncap"}
order = ["px_v3","T1000000","T2000000","T4000000","T6000000","T10000000","T200000000"]
d = {}
n = None
for f in sorted(glob.glob(D + "/chile__*.npz")):
    b = os.path.basename(f)
    m = re.match(r"chile__(px_v3|T\d+)_(BEST_.*|incumbent)_2019_scores\.npz", b)
    pool, kind = m.group(1), m.group(2)
    z = np.load(f); s, r = z["site_s"], z["rand_s"]; n = len(s)
    roc = roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(r))], np.r_[s, r])
    d.setdefault(pool, {})["BEST" if kind.startswith("BEST") else "inc"] = round(roc, 4)
done = sum(len(v) for v in d.values())
print(f"Chile (n={n} validation sites), {done}/14 models done")
print(f"{'pool':>8} {'incumbent':>10} {'BEST':>8} {'delta':>8}")
for p in order:
    e = d.get(p, {}); inc = e.get("inc"); best = e.get("BEST")
    dd = f"{best-inc:+.4f}" if (inc is not None and best is not None) else ""
    print(f"{PL[p]:>8} {inc if inc is not None else '—':>10} {best if best is not None else '—':>8} {dd:>8}")

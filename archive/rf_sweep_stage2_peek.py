"""Interim Stage-2 read: per-config country-forward ROC from whatever sweep npz exist so far,
next to the incumbent's published RF-R country numbers (100%-model, country_summary.csv)."""
import os, glob, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

WS = os.path.expanduser("~/Solar_Workspace")
SWS = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/sweep/scores")

# incumbent published RF-R (100%-trained) forward ROCs by iso3
inc = dict(GRC=.89526, DEU=.84994, CHN=.87584, ESP=.82065, POL=.79301, JPN=.89684,
           IND=.82733, USA=.88334, ZAF=.84686, COL=.95654, PHL=.87146, MYS=.79316, CHL=.93422)
name2iso = {"greece":"GRC","germany":"DEU","china":"CHN","spain":"ESP","poland":"POL","japan":"JPN",
            "india":"IND","united_states":"USA","south_africa":"ZAF","colombia":"COL",
            "philippines":"PHL","malaysia":"MYS","chile":"CHL"}

rows = []
for f in glob.glob(os.path.join(SWS, "cfg*_scores.npz")):
    m = re.match(r"cfg(\d+)_(.+)_(\d{4})_scores\.npz", os.path.basename(f))
    if not m: continue
    cid, country = int(m.group(1)), m.group(2)
    z = np.load(f); s, r = z["site_s"], z["rand_s"]
    y = np.r_[np.ones(len(s)), np.zeros(len(r))]
    rows.append(dict(cid=cid, iso=name2iso.get(country, country), roc=roc_auc_score(y, np.r_[s, r]), n=len(s)))
df = pd.DataFrame(rows)
for cid in sorted(df.cid.unique()):
    g = df[df.cid == cid].set_index("iso")
    common = [i for i in g.index if i in inc]
    d = pd.DataFrame({"sweep": g.roc, "incumbent_pub": pd.Series(inc), "n": g.n}).loc[common]
    d["delta"] = d.sweep - d.incumbent_pub
    wsweep = np.average(d.sweep, weights=d.n); winc = np.average(d.incumbent_pub, weights=d.n)
    tag = " [INCUMBENT cfg00, 85% model]" if cid == 0 else ""
    print(f"\n=== cfg{cid:02d}{tag}: {len(common)} countries done ===")
    print(d.round(3).to_string())
    print(f"  weighted mean over these {len(common)}: sweep {wsweep:.4f} vs incumbent-pub {winc:.4f} "
          f"(delta {wsweep-winc:+.4f})")

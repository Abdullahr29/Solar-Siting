"""FINAL head-to-head: the full validation ladder on the 6 GEE-free candidates, one consistent harness.

Models (pool x trees x lag), all trained here from scratch on the SAME code/seed:
  cap120.ne30.t2  cap120.ne15.t2  cap120.ne30.t3  cap120.ne15.t3   (px_v3 / cap120_t3 pools)
  6M.ne30.t2      6M.ne15.t2                                        (per-country 6M sample from master)
The two 6M.t3 cells are GEE-blocked (~13d of install-3 sampling) -> DERIVED in --aggregate from the
measured 6M.t2 result + the directly-measured cap120 (t3-minus-t2) delta, flagged 'inferred'.

Ladder rungs (all scored LOCALLY on cached embeddings; no GEE):
  1 pixel   ROC on px_v3_test (biased shortlist metric)
  2 country forward ROC, 13 countries, cached AEF-2019 val_emb  (n-wtd + macro + per-country)
  3 temporal honest-2024, China+USA, model retrained on install<=2023 (yr_sol<=2021), cached AEF-2022
  4 deep-lookback, fixed AEF-2017, ROC by forward lead L2..L7    (leak-resistance)
  5 LOCO, Greece/Spain/Poland, country's pos+neg removed, retrain, score cached AEF-2019 val

Run as two parallel LOTUS jobs:  --group cap120   (light)   and   --group 6M   (256G, loads master)
then  --aggregate  -> final8_master.csv + inferred 6M.t3 rows + console summary.
Resumable: one JSON per (model) under final8/ ; LOCO JSONs shared with the earlier loco/ dir.
"""
import os, sys, json, glob, re, time, argparse
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting")); sys.path.insert(0, ".")
POOLD = "artifacts/paper_v3/pools"; STAT = "artifacts/stats"
P3 = "artifacts/paper_v3/pools_t3"; VAL = f"{P3}/val_emb"; TE = f"{P3}/temporal_emb"
DB = "artifacts/paper_v3/temporal_probe/big"
OUT = f"{P3}/final8"; LOCOD = f"{P3}/loco"
os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/models", exist_ok=True)
BUDGET = 63000; NJOBS = 16
BANDS = [f"A{i:02d}" for i in range(64)]
MODELS = {  # name -> (pool_key, n_estimators, lag)
    "cap120_ne30_t2": ("px_v3", 30, "t2"), "cap120_ne15_t2": ("px_v3", 15, "t2"),
    "cap120_ne30_t3": ("cap120_t3", 30, "t3"), "cap120_ne15_t3": ("cap120_t3", 15, "t3"),
    "6M_ne30_t2": ("T6000000", 30, "t2"), "6M_ne15_t2": ("T6000000", 15, "t2"),
}
GROUPS = {"cap120": [m for m in MODELS if m.startswith("cap120")],
          "6M": [m for m in MODELS if m.startswith("6M")]}
COUNTRIES = {"greece": ("GRC", 55), "spain": ("ESP", 47), "poland": ("POL", 115)}

# ---------------- pool machinery (matches loco_debates.build_pool) ----------------
def even_alloc(counts, budget):
    counts = np.asarray(counts, np.int64); alloc = np.zeros_like(counts)
    active = counts > 0; remaining = int(min(budget, counts.sum()))
    while remaining > 0 and active.any():
        share = remaining // int(active.sum())
        if share == 0:
            take = np.argsort(-((counts-alloc)*active))[:remaining]; alloc[take] += 1; break
        take = np.minimum(counts-alloc, share)*active; alloc += take
        remaining -= int(take.sum()); active = (counts-alloc) > 0
    return alloc
_rng = np.random.default_rng(0)
def sample_country(idx, cel, cl, T):
    if len(idx) <= T: return idx
    uc, ci = np.unique(cel, return_inverse=True); cb = even_alloc(np.bincount(ci, minlength=len(uc)), T)
    out = []
    for k in range(len(uc)):
        b = int(cb[k])
        if b == 0: continue
        m = ci == k; ii, clq = idx[m], cl[m]
        if b >= len(ii): out.append(ii); continue
        us, si = np.unique(clq, return_inverse=True); sb = even_alloc(np.bincount(si, minlength=len(us)), b)
        for s in range(len(us)):
            if sb[s] == 0: continue
            pix = ii[si == s]; out.append(pix if sb[s] >= len(pix) else _rng.choice(pix, int(sb[s]), replace=False))
    return np.concatenate(out)

_master = {}
def load_master():
    if _master: return _master
    z = np.load(f"{POOLD}/px_master_train.npz", allow_pickle=False)
    _master.update(Xs=z["X_sol"], iso=z["iso_sol"], yr=z["yr_sol"], src=z["src_sol"],
                   pos_chips=z["pos_chips"], Xn=z["X_neg"], iso_n=z["iso_neg"])
    geo = pd.read_csv(f"{STAT}/chip_geo_pos.csv").set_index("chip"); g = geo.index.to_numpy()
    lat = np.floor(geo.lat.to_numpy()).astype(np.int64); lon = np.floor(geo.lon.to_numpy()).astype(np.int64)
    cby = dict(zip(g, (lat+90)*360+(lon+180))); clby = dict(zip(g, geo.cluster.to_numpy().astype(np.int64)))
    pc = _master["pos_chips"]
    _master["cell"] = np.array([cby.get(n, -1) for n in pc], np.int64)[_master["src"]]
    _master["clus"] = _master["iso"].astype(np.int64)*10_000_000 + np.array([clby.get(n, -1) for n in pc], np.int64)[_master["src"]]
    _master["fin_s"] = np.isfinite(_master["Xs"]).all(1)
    print(f"  master loaded: {len(_master['iso']):,} pos / {len(_master['iso_n']):,} neg", flush=True)
    return _master

def load_pool(pool, exclude_code=None, temporal=False):
    if pool in ("px_v3", "cap120_t3"):
        if pool == "px_v3":
            z = np.load(f"{POOLD}/px_v3_train.npz"); Xs, iso_s, yr_s = z["X_sol"], z["iso_sol"], z["yr_sol"]
            Xn, iso_n = z["X_neg"], z["iso_neg"]
        else:
            z = np.load(f"{P3}/cap120_t3_train.npz"); Xs, iso_s, yr_s = z["X_sol"], z["iso_sol"], z["yr_sol"]
            zv = np.load(f"{POOLD}/px_v3_train.npz"); Xn, iso_n = zv["X_neg"], zv["iso_neg"]
        ps = np.ones(len(iso_s), bool); ns = np.ones(len(iso_n), bool)
        if exclude_code is not None: ps &= iso_s != exclude_code; ns &= iso_n != exclude_code
        if temporal: ps &= yr_s <= 2021
        X = np.concatenate([Xs[ps], Xn[ns]]).astype(np.float32)
        y = np.r_[np.ones(int(ps.sum()), np.int8), np.zeros(int(ns.sum()), np.int8)]
    else:                                                     # 6M from master
        T = int(pool[1:]); M = load_master(); parts = []
        for u in np.unique(M["iso"]):
            if exclude_code is not None and u == exclude_code: continue
            keep = (M["iso"] == u) & M["fin_s"]
            if temporal: keep &= M["yr"] <= 2021
            idx = np.where(keep)[0]
            parts.append(idx if len(idx) <= T else sample_country(idx, M["cell"][idx], M["clus"][idx], T))
        sel = np.sort(np.concatenate(parts))
        nkeep = np.isfinite(M["Xn"]).all(1)
        if exclude_code is not None: nkeep &= M["iso_n"] != exclude_code
        nk = np.where(nkeep)[0]
        X = np.concatenate([M["Xs"][sel], M["Xn"][nk]]).astype(np.float32)
        y = np.r_[np.ones(len(sel), np.int8), np.zeros(len(nk), np.int8)]
    keep = np.isfinite(X).all(1); return X[keep], y[keep]

def fit(pool, ne, exclude=None, temporal=False):
    X, y = load_pool(pool, exclude, temporal)
    rf = RandomForestClassifier(n_estimators=ne, max_leaf_nodes=BUDGET//ne, max_features=32,
                                min_samples_leaf=20, class_weight="balanced", n_jobs=NJOBS, random_state=0).fit(X, y)
    return rf, int(len(y)), int(y.sum())

# ---------------- ladder rungs (cached-embedding scoring) ----------------
def roc(a, b): return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
def pp(rf, X): return rf.predict_proba(X)[:, 1]

def rung_pixel(rf):
    z = np.load(f"{POOLD}/px_v3_test.npz"); Xs, Xn = z["X_sol"], z["X_neg"]
    Xs = Xs[np.isfinite(Xs).all(1)]; Xn = Xn[np.isfinite(Xn).all(1)]
    return round(float(roc(pp(rf, Xs), pp(rf, Xn))), 4)

def rung_country(rf):
    rows, nsite = {}, {}
    for f in glob.glob(f"{VAL}/*_val2019.npz"):
        c = os.path.basename(f).split("_val2019")[0]; d = np.load(f)
        Xso, Xr = d["site_emb"], d["rand_emb"]; Xso = Xso[np.isfinite(Xso).all(1)]; Xr = Xr[np.isfinite(Xr).all(1)]
        rows[c] = round(float(roc(pp(rf, Xso), pp(rf, Xr))), 4); nsite[c] = len(Xso)
    s = pd.Series(rows); n = pd.Series(nsite).reindex(s.index)
    return dict(per=rows, nwtd=round(float(np.average(s, weights=n)), 4), macro=round(float(s.mean()), 4))

def rung_deeplookback(rf):
    import geopandas as gpd
    base = gpd.read_file("global_solar_ml_pipeline.gpkg", layer="base_train"); bid = set(base["PV_ID"].tolist())
    inv = gpd.read_file("global_pv_facility_inventory.gpkg")
    held = inv[(~inv["PV_ID"].isin(bid)) & inv.year.between(2019, 2024)]
    ev = pd.concat([sub.sample(min(1000, len(sub)), random_state=0) for _, sub in held.groupby("year")]).reset_index(drop=True)
    Es = np.load(f"{DB}/emb_deepsite_2017.npy"); Er = np.load(f"{DB}/emb_deeprand_2017.npy")
    oks = np.isfinite(Es).all(1); okr = np.isfinite(Er).all(1); lead = (ev.year.to_numpy() - 2017)[oks]
    ps = pp(rf, Es[oks]); pr = pp(rf, Er[okr])
    out = dict(roc_all=round(float(roc(ps, pr)), 4))
    for L in range(2, 8):
        if (lead == L).sum() >= 30: out[f"L{L}"] = round(float(roc(ps[lead == L], pr)), 4)
    return out

def rung_temporal(pool, ne):
    rf, ntr, npos = fit(pool, ne, temporal=True)
    res = {}
    for c in ("china", "usa"):
        d = np.load(f"{TE}/{c}_2022emb.npz"); Xs, Xr = d["site"], d["rand"]
        Xs = Xs[np.isfinite(Xs).all(1)]; Xr = Xr[np.isfinite(Xr).all(1)]
        res[c] = round(float(roc(pp(rf, Xs), pp(rf, Xr))), 4)
    res["n_train"] = ntr; return res

def rung_loco(name, pool, ne):
    out = {}
    for cname, (iso3, code) in COUNTRIES.items():
        ck = f"{LOCOD}/{name}__{cname}.json"
        if os.path.exists(ck):
            out[cname] = json.load(open(ck))["loco_roc"]; continue
        rf, ntr, npos = fit(pool, ne, exclude=code)
        d = np.load(f"{VAL}/{cname}_val2019.npz"); Xs, Xr = d["site_emb"], d["rand_emb"]
        Xs = Xs[np.isfinite(Xs).all(1)]; Xr = Xr[np.isfinite(Xr).all(1)]
        lr = round(float(roc(pp(rf, Xs), pp(rf, Xr))), 4)
        json.dump(dict(config=name, country=cname, loco_roc=lr, n_train=ntr), open(ck, "w"))
        out[cname] = lr; print(f"    LOCO {name} {cname}: {lr}", flush=True)
    return out

# ---------------- driver ----------------
def run_group(group):
    for name in GROUPS[group]:
        jf = f"{OUT}/{name}.json"
        if os.path.exists(jf) and not os.environ.get("FORCE"):
            print(f"[skip] {name}", flush=True); continue
        pool, ne, lag = MODELS[name]; t = time.time()
        print(f"\n=== {name} ({pool}, ne{ne}, {lag}) ===", flush=True)
        rf, ntr, npos = fit(pool, ne)                         # full model -> pixel/country/deeplookback
        joblib.dump(rf, f"{OUT}/models/{name}.joblib", compress=3)
        rec = dict(model=name, pool=pool, ne=ne, lag=lag, n_train=ntr, n_pos=npos)
        rec["pixel"] = rung_pixel(rf); print(f"  pixel={rec['pixel']} ({time.time()-t:.0f}s)", flush=True)
        rec["country"] = rung_country(rf); print(f"  country n-wtd={rec['country']['nwtd']} macro={rec['country']['macro']}", flush=True)
        rec["deep"] = rung_deeplookback(rf); print(f"  deep roc_all={rec['deep']['roc_all']} L7={rec['deep'].get('L7')}", flush=True)
        rec["temporal"] = rung_temporal(pool, ne); print(f"  temporal china={rec['temporal']['china']} usa={rec['temporal']['usa']}", flush=True)
        rec["loco"] = rung_loco(name, pool, ne); print(f"  loco={rec['loco']}", flush=True)
        json.dump(rec, open(jf, "w"), indent=1)
        print(f"  [{name} done, {time.time()-t:.0f}s]", flush=True)
    print(f"DONE group {group}", flush=True)

def aggregate():
    recs = {}
    for name in MODELS:
        jf = f"{OUT}/{name}.json"
        if os.path.exists(jf): recs[name] = json.load(open(jf))
    if not recs: print("no results yet"); return
    def row(r, tag=""):
        c, d, t, l = r["country"], r["deep"], r["temporal"], r["loco"]
        return dict(model=r["model"]+tag, pool=r["pool"], ne=r["ne"], lag=r["lag"], n_train=r.get("n_train"),
                    pixel=r["pixel"], country_nwtd=c["nwtd"], country_macro=c["macro"],
                    deep_all=d["roc_all"], deep_L7=d.get("L7"), temp_china=t["china"], temp_usa=t["usa"],
                    loco_grc=l["greece"], loco_esp=l["spain"], loco_pol=l["poland"],
                    loco_mean=round(np.mean([l["greece"], l["spain"], l["poland"]]), 4))
    table = [row(recs[m]) for m in MODELS if m in recs]
    # derive 6M.t3 = 6M.t2 + (cap120 t3 - cap120 t2) delta, per ne
    for ne in (30, 15):
        need = f"6M_ne{ne}_t3"; base6 = f"6M_ne{ne}_t2"
        c3, c2 = f"cap120_ne{ne}_t3", f"cap120_ne{ne}_t2"
        if all(k in recs for k in (base6, c3, c2)):
            b = row(recs[base6]); d3, d2 = recs[c3], recs[c2]
            def delta(path):
                import functools
                def get(r):
                    o = r
                    for p in path: o = o[p]
                    return o
                return get(d3) - get(d2)
            inf = dict(model=need+" (inferred)", pool="T6000000", ne=ne, lag="t3", n_train=b["n_train"],
                       pixel=None,
                       country_nwtd=round(b["country_nwtd"]+delta(["country","nwtd"]), 4),
                       country_macro=round(b["country_macro"]+delta(["country","macro"]), 4),
                       deep_all=round(b["deep_all"]+delta(["deep","roc_all"]), 4),
                       deep_L7=round(b["deep_L7"]+(recs[c3]["deep"].get("L7",0)-recs[c2]["deep"].get("L7",0)), 4) if b["deep_L7"] else None,
                       temp_china=round(b["temp_china"]+delta(["temporal","china"]), 4),
                       temp_usa=round(b["temp_usa"]+delta(["temporal","usa"]), 4),
                       loco_grc=round(b["loco_grc"]+(recs[c3]["loco"]["greece"]-recs[c2]["loco"]["greece"]), 4),
                       loco_esp=round(b["loco_esp"]+(recs[c3]["loco"]["spain"]-recs[c2]["loco"]["spain"]), 4),
                       loco_pol=round(b["loco_pol"]+(recs[c3]["loco"]["poland"]-recs[c2]["loco"]["poland"]), 4))
            inf["loco_mean"] = round(np.mean([inf["loco_grc"], inf["loco_esp"], inf["loco_pol"]]), 4)
            table.append(inf)
    df = pd.DataFrame(table)
    df.to_csv(f"{OUT}/final8_master.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("\n================ FINAL 8-MODEL HEAD-TO-HEAD ================")
    print(df.to_string(index=False))
    print(f"\nwrote {OUT}/final8_master.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["cap120", "6M"]); ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate: aggregate()
    elif a.group: run_group(a.group)
    else: print("need --group {cap120,6M} or --aggregate")

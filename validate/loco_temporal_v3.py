"""Stage 5: LOCO (spatial holdout) + temporal (strict 2024 forecast), for BOTH schemes.
LOCO/temporal models are DIAGNOSTICS (handicapped), never released. Ablation training is an
in-memory filter on the one extracted pool (train_rf_v3 --exclude-iso3 / --max-year), so no
re-extraction. Everything -> results_master; per-experiment figures.
"""
import os, sys, subprocess, traceback
import numpy as np, pandas as pd

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
from run_country_rf import run_country

PY = os.path.join(WS, "envs/env_solar/bin/python")
POOL = "Solar-Siting/artifacts/paper_v3/pools/px_v3_train.npz"
MDIR = "Solar-Siting/artifacts/paper_v3/models"
# LOCO: (LSIB, iso3, inv-override) across N/S x low/high
LOCO = [("Germany", "DEU", None), ("Spain", "ESP", None), ("Poland", "POL", None),
        ("China", "CHN", None), ("India", "IND", None), ("Brazil", "BRA", None),
        ("South Africa", "ZAF", None), ("Chile", "CHL", None), ("Philippines", "PHL", None)]
# temporal: needs thick 2024; validate 2024 installs on the AEF-2022 map (2-yr lead)
TEMPORAL = [("China", "CHN", None), ("United States", "USA", "United States of America"),
            ("India", "IND", None), ("Spain", "ESP", None), ("Turkey", "TUR", None),
            ("Brazil", "BRA", None)]


def train(out, scheme, exclude=None, max_year=None, scope=""):
    if os.path.exists(os.path.join(WS, out)):
        print(f"reuse {out}"); return
    cmd = [PY, "Solar-Siting/train_rf_v3.py", "--pool", POOL, "--out", out, "--test", "",
           "--scheme", scheme, "--scope", scope,
           "--experiment", "loco" if exclude else "temporal"]
    if scheme == "D":
        cmd.append("--deconf")
    if exclude:
        cmd += ["--exclude-iso3", exclude]
    if max_year is not None:
        cmd += ["--max-year", str(max_year)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def score(lsib, iso3, inv, model, prefix, year, min_install, exp, scheme):
    if os.path.exists(os.path.join(WS, prefix + "_scores.npz")):
        print(f"skip {exp} {lsib} {scheme} (scores exist — resume)"); return {"skipped": True}
    try:
        m = run_country(lsib, year=year, min_install=min_install, model_path=model,
                        out_prefix=prefix, inv_country=inv)
        for metric, val in [("roc", m["ROC"]), ("pr", m["PR"]), ("topk05", m["top5"])]:
            R.append_result(exp, "rf", scheme, iso3, metric, val, n_pos=m["n_sites"],
                            model_path=model, scores_path=prefix + "_scores.npz",
                            figure_path=prefix + "_fig.png", aef_year=year)
        print(f"### {exp} {lsib} {scheme}: ROC {m['ROC']:.3f} n={m['n_sites']}", flush=True)
        return m
    except Exception:
        traceback.print_exc(); R.log_stage(f"{exp}/{iso3}/{scheme}", prefix, "fail"); return None


def main():
    for scheme in ("R", "D"):
        # ---- LOCO ----
        for lsib, iso3, inv in LOCO:
            out = f"{MDIR}/loco/rf30_v3_{iso3}_{scheme}.joblib"
            try:
                train(out, scheme, exclude=iso3, scope=iso3)
            except subprocess.CalledProcessError:
                R.log_stage(f"loco-train/{iso3}/{scheme}", out, "fail"); continue
            prefix = f"Solar-Siting/artifacts/paper_v3/scores/loco/{iso3}_{scheme}_2019"
            score(lsib, iso3, inv, out, prefix, 2019, 2021, "loco", scheme)

        # ---- temporal (train <=2021 imagery == installs <=2023; validate 2024 on AEF-2022) ----
        tout = f"{MDIR}/temporal/rf30_v3_le2021_{scheme}.joblib"
        try:
            train(tout, scheme, max_year=2021, scope="le2021")
        except subprocess.CalledProcessError:
            R.log_stage(f"temporal-train/{scheme}", tout, "fail"); continue
        pooled_site, pooled_rand = [], []
        for lsib, iso3, inv in TEMPORAL:
            prefix = f"Solar-Siting/artifacts/paper_v3/scores/temporal/{iso3}_{scheme}_2022"
            m = score(lsib, iso3, inv, tout, prefix, 2022, 2024, "temporal", scheme)
            sp = os.path.join(WS, prefix + "_scores.npz")
            if m and os.path.exists(sp):
                z = np.load(sp); pooled_site.append(z["site_s"]); pooled_rand.append(z["rand_s"])
        # global-2024 aggregate ROC (the strongest single honest number)
        if pooled_site:
            from sklearn.metrics import roc_auc_score, average_precision_score
            ss = np.concatenate(pooled_site); rr = np.concatenate(pooled_rand)
            y = np.r_[np.ones(len(ss)), np.zeros(len(rr))]; x = np.r_[ss, rr]
            pct = np.array([(rr < s).mean() for s in ss])
            R.append_result("temporal", "rf", scheme, "global-2024", "roc", roc_auc_score(y, x),
                            n_pos=len(ss), aef_year=2022, notes="pooled across temporal countries")
            R.append_result("temporal", "rf", scheme, "global-2024", "topk05",
                            (pct >= 0.95).mean() * 100, n_pos=len(ss), aef_year=2022)
            print(f"### TEMPORAL GLOBAL-2024 {scheme}: ROC {roc_auc_score(y, x):.3f} "
                  f"top5 {(pct>=0.95).mean()*100:.1f}% n={len(ss)}", flush=True)


if __name__ == "__main__":
    main()

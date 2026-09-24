"""Stage 2 of the RF sweep — the HONEST metric. Takes the Stage-1 shortlist (top GEE-feasible
configs + the incumbent) and runs the exact country-forward validation of country_runs_v3
(same 13 countries, YEAR=2019, installs>=2021, leakage dedup) for each config's model.

Sequential on purpose (GEE concurrency=1 -> no restricted-mode 429s) and resumable: any
(config, country) whose scores npz already exists is skipped. Scores land under
artifacts/paper_v3/sweep/scores/; rank them with rf_sweep_stage2_aggregate.py.
"""
import os, sys, json, traceback
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
from run_country_rf import run_country
from country_runs_v3 import training_pv_ids, COUNTRIES, YEAR, MIN_INSTALL

SCORES = "Solar-Siting/artifacts/paper_v3/scores"  # incumbent 100%-model runs live here too
SWSCORES = "Solar-Siting/artifacts/paper_v3/sweep/scores"


def main():
    os.makedirs(os.path.join(WS, SWSCORES), exist_ok=True)
    with open(os.path.join(WS, "Solar-Siting/artifacts/paper_v3/sweep/stage2_shortlist.json")) as f:
        shortlist = json.load(f)
    pv = training_pv_ids()
    print(f"Stage 2: {len(shortlist)} configs x {len(COUNTRIES)} countries "
          f"(dedup={'yes' if pv else 'no'})", flush=True)

    for cfg in shortlist:
        cid = int(cfg["config_id"]); model = cfg["model_path"]
        if not model or not os.path.exists(os.path.join(WS, model)):
            print(f"skip cfg{cid}: model missing ({model})", flush=True); continue
        print(f"\n===== config {cid} (id{cid}) {model} =====", flush=True)
        for lsib, iso3, invname in COUNTRIES:
            prefix = f"{SWSCORES}/cfg{cid:02d}_{lsib.lower().replace(' ', '_')}_{YEAR}"
            if os.path.exists(os.path.join(WS, prefix + "_scores.npz")):
                print(f"skip cfg{cid}/{iso3} (scores exist)", flush=True); continue
            try:
                m = run_country(lsib, year=YEAR, min_install=MIN_INSTALL, model_path=model,
                                out_prefix=prefix, inv_country=invname,
                                exclude_pv_ids=(pv or None))
                print(f"### cfg{cid} {lsib}: ROC {m['ROC']:.3f} n={m['n_sites']}", flush=True)
            except SystemExit as e:
                print(f"FAIL cfg{cid}/{iso3}: {e}", flush=True)
            except Exception:
                traceback.print_exc()
    print("\nStage 2 sweep runs complete (or resumable). Aggregate with rf_sweep_stage2_aggregate.py",
          flush=True)


if __name__ == "__main__":
    main()

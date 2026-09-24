#!/bin/bash
# paper_v3 campaign orchestrator. Sequential; critical stages abort on failure, downstream
# stages continue-and-log so one bug never loses the core results. Idempotent: extractions and
# base models skip if their output already exists, so a re-launch resumes.
set -u
WS=~/Solar_Workspace
PY=$WS/envs/env_solar/bin/python
cd $WS
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
SS=Solar-Siting
P3=$SS/artifacts/paper_v3
POOLS=$P3/pools
M=$P3/models
LOGS=$P3/logs
mkdir -p $POOLS $M $M/loco $M/temporal $LOGS $P3/scores $P3/figures
MAN=$P3/run_manifest.jsonl
ts() { date +%Y-%m-%dT%H:%M:%S; }
man() { echo "{\"stage\":\"$1\",\"status\":\"$2\",\"ts\":\"$(ts)\"}" >> $MAN; }

run() {   # run <name> <critical> <cmd...>
  local name=$1 crit=$2; shift 2
  echo "===== [$(ts)] START $name =====" | tee -a $LOGS/master.log
  man "$name" start
  if "$@" > $LOGS/$name.log 2>&1; then
    echo "[$(ts)] DONE $name" | tee -a $LOGS/master.log; man "$name" ok
  else
    echo "[$(ts)] FAIL $name -> $LOGS/$name.log" | tee -a $LOGS/master.log; man "$name" fail
    tail -5 $LOGS/$name.log | tee -a $LOGS/master.log
    [ "$crit" = 1 ] && { echo "CRITICAL FAIL — aborting"; exit 1; }
  fi
}

echo "########## paper_v3 campaign start $(ts) ##########" | tee -a $LOGS/master.log

run negsplit 0 $PY $SS/negatives/regen_neg_split.py

[ -f $POOLS/px_v3_train.npz ] || run extract_train 1 $PY $SS/pipeline/extract_pixels_v3.py --split train --out $POOLS/px_v3_train.npz
[ -f $POOLS/px_v3_test.npz  ] || run extract_test  1 $PY $SS/pipeline/extract_pixels_v3.py --split test  --out $POOLS/px_v3_test.npz

[ -f $M/rf30_v3_R.joblib ] || run train_R 1 $PY $SS/train_rf_v3.py --pool $POOLS/px_v3_train.npz --test $POOLS/px_v3_test.npz --out $M/rf30_v3_R.joblib --scheme R --scope test-pixels --experiment train
[ -f $M/rf30_v3_D.joblib ] || run train_D 1 $PY $SS/train_rf_v3.py --pool $POOLS/px_v3_train.npz --test $POOLS/px_v3_test.npz --out $M/rf30_v3_D.joblib --deconf --scheme D --scope test-pixels --experiment train

run bakeoff       0 $PY $SS/pipeline/bakeoff_v3.py
run country       0 $PY $SS/country_runs_v3.py
run data_ablation 0 $PY $SS/archive/data_ablation_v3.py
run loco_temporal 0 $PY $SS/validate/loco_temporal_v3.py
run qualitative   0 $PY $SS/figures/qualitative_v3.py
run importance    0 $PY $SS/analysis/embedding_importance_v3.py
run covariate     0 $PY $SS/analysis/covariate_recompute_v3.py

# all-data released-asset candidates (both schemes; winner picked from country runs)
[ -f $POOLS/px_v3_all.npz ] || run extract_all 0 $PY $SS/pipeline/extract_pixels_v3.py --split all --out $POOLS/px_v3_all.npz
if [ -f $POOLS/px_v3_all.npz ]; then
  [ -f $M/asset_R_alldata.joblib ] || run asset_R 0 $PY $SS/train_rf_v3.py --pool $POOLS/px_v3_all.npz --test "" --out $M/asset_R_alldata.joblib --scheme R --scope all-data --experiment asset
  [ -f $M/asset_D_alldata.joblib ] || run asset_D 0 $PY $SS/train_rf_v3.py --pool $POOLS/px_v3_all.npz --test "" --deconf --out $M/asset_D_alldata.joblib --scheme D --scope all-data --experiment asset
fi

echo "########## paper_v3 campaign DONE $(ts) ##########" | tee -a $LOGS/master.log
man campaign done

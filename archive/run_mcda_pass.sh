#!/bin/bash
# Final stage: after the roads pass completes (which itself waits for the overnight run),
# run the GIS-MCDA benchmark for the install-countries and collect RF-vs-MCDA into one CSV.
set -u
cd ~/Solar_Workspace
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
PY=~/Solar_Workspace/envs/env_solar/bin/python
LOGD=Solar-Siting/artifacts/covariate/overnight
MASTER="$LOGD/overnight_master.log"
ts(){ date +%H:%M:%S; }

until grep -q "PBF ROADS PASS COMPLETE" "$MASTER" 2>/dev/null; do sleep 120; done
echo "[$(ts)] ===== MCDA BENCHMARK PASS START =====" | tee -a "$MASTER"
rm -f Solar-Siting/artifacts/results/mcda_benchmark.csv     # fresh (script appends)

for c in "Greece" "Germany" "China" "United States"; do
  slug=$(echo "$c" | tr 'A-Z ' 'a-z_')
  $PY Solar-Siting/mcda_benchmark.py "$c" --year 2021 >> "$LOGD/mcda_${slug}.log" 2>&1 \
    && echo "[$(ts)] MCDA OK $c" | tee -a "$MASTER" \
    || echo "[$(ts)] MCDA FAILED $c (see $LOGD/mcda_${slug}.log)" | tee -a "$MASTER"
done

echo "[$(ts)] ===== MCDA RESULTS =====" | tee -a "$MASTER"
cat Solar-Siting/artifacts/results/mcda_benchmark.csv 2>/dev/null | tee -a "$MASTER"
echo "[$(ts)] ===== MCDA BENCHMARK PASS COMPLETE =====" | tee -a "$MASTER"

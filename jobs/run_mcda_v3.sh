#!/bin/bash
# MCDA v3 benchmark pass: literature-sourced weights (Richards 2025 + Chen 2024), WLC + TOPSIS,
# full criteria set, v2/R model. Mirrors run_mcda_pass.sh (same PROJ_DATA framework).
set -u
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj   # established framework for GDAL/geopandas reproj
PY=~/Solar_Workspace/envs/env_solar/bin/python
LOGD=Solar-Siting/artifacts/paper_v3/logs
mkdir -p "$LOGD"
MASTER="$LOGD/mcda_v3_master.log"
ts(){ date +%H:%M:%S; }

echo "[$(ts)] ===== MCDA v3 PASS START (Richards+Chen weights x WLC+TOPSIS x land both-ways, R model) =====" | tee -a "$MASTER"
rm -f Solar-Siting/artifacts/paper_v3/mcda_v3_results.csv   # fresh (script appends; resumable per-country)

for c in "Greece" "Germany" "China" "United States"; do
  slug=$(echo "$c" | tr 'A-Z ' 'a-z_')
  echo "[$(ts)] START $c" | tee -a "$MASTER"
  $PY -u Solar-Siting/mcda_benchmark_v3.py "$c" --year 2021 >> "$LOGD/mcda_v3_${slug}.log" 2>&1 \
    && echo "[$(ts)] OK $c" | tee -a "$MASTER" \
    || echo "[$(ts)] FAILED $c (see $LOGD/mcda_v3_${slug}.log)" | tee -a "$MASTER"
done

echo "[$(ts)] ===== MCDA v3 RESULTS =====" | tee -a "$MASTER"
cat Solar-Siting/artifacts/paper_v3/mcda_v3_results.csv 2>/dev/null | tee -a "$MASTER"
echo "[$(ts)] ===== MCDA v3 PASS COMPLETE =====" | tee -a "$MASTER"

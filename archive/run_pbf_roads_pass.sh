#!/bin/bash
# After the overnight run + the ogr2ogr extractions finish, add road-distance to the 4
# .pbf countries (incremental append reads their _full table and just adds road_dist_m),
# then refresh their analysis + maps and regenerate the summary CSV.
set -u
cd ~/Solar_Workspace
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
PY=~/Solar_Workspace/envs/env_solar/bin/python
LOGD=Solar-Siting/artifacts/covariate/overnight
MASTER="$LOGD/overnight_master.log"
OSM=Solar-Siting/covariate_data/osm
ts(){ date +%H:%M:%S; }

# 1) wait for the overnight pipeline to finish (so _full tables have GSA+grid already)
until grep -q "OVERNIGHT RUN COMPLETE" "$MASTER" 2>/dev/null; do sleep 120; done
# 2) wait for all four road extractions to finish
for s in germany united_states china australia; do
  until grep -q "DONE $s" "$OSM/roads_extract_${s}.log" 2>/dev/null; do sleep 30; done
done
echo "[$(ts)] ===== PBF ROADS PASS START =====" | tee -a "$MASTER"

for c in "Germany" "United States" "China" "Australia"; do
  slug=$(echo "$c" | tr 'A-Z ' 'a-z_')
  log="$LOGD/${slug}.log"
  gp="$OSM/${slug}_roads.gpkg"
  if [ ! -s "$gp" ]; then echo "[$(ts)] $c: no roads gpkg -> skip" | tee -a "$MASTER"; continue; fi
  $PY Solar-Siting/covariate_append_manual.py "$c" --year 2021 --roads >> "$log" 2>&1 \
    && echo "[$(ts)] $c: roads append OK" | tee -a "$MASTER" \
    || echo "[$(ts)] $c: roads append FAILED (see $log)" | tee -a "$MASTER"
  $PY Solar-Siting/covariate_analysis.py "$c" --year 2021 >> "$log" 2>&1 \
    && echo "[$(ts)] $c: analysis OK" | tee -a "$MASTER" || echo "[$(ts)] $c: analysis FAILED" | tee -a "$MASTER"
  $PY Solar-Siting/covariate_maps.py "$c" --year 2021 >> "$log" 2>&1
done

$PY Solar-Siting/covariate_summary.py --year 2021 >> "$MASTER" 2>&1
echo "[$(ts)] ===== PBF ROADS PASS COMPLETE =====" | tee -a "$MASTER"

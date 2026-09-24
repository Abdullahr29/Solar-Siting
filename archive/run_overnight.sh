#!/bin/bash
# Overnight covariate pipeline for the remaining countries (Greece already done).
# Two parallel lanes (EE-concurrency-safe), each runs the full chain per country:
#   sample (GEE, n=20000) -> append GSA/grid[/roads] -> analysis -> maps
# Error-isolated: any single step/country failing is logged and the rest continue.
# Roads: only the 3 shapefile countries (Chile, S.Africa; Greece done); the 4 .pbf
# countries are GSA+grid only (pbf road extraction is too slow for unattended run).
set -u
cd ~/Solar_Workspace
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
PY=~/Solar_Workspace/envs/env_solar/bin/python
YEAR=2021
N=20000
LOGD=Solar-Siting/artifacts/covariate/overnight
mkdir -p "$LOGD"
MASTER="$LOGD/overnight_master.log"

ts(){ date +%H:%M:%S; }
mlog(){ echo "[$(ts)] $*" | tee -a "$MASTER"; }

# process_country <LSIB name> <roads: yes|no>
process_country(){
  local c="$1" roads="$2"
  local slug=$(echo "$c" | tr 'A-Z ' 'a-z_')
  local log="$LOGD/${slug}.log"
  local base="Solar-Siting/artifacts/covariate/${slug}/${slug}_${YEAR}"
  mlog "START $c ($slug)"

  # 1) sample (skip if already present)
  if [ -f "${base}_covsample.npz" ]; then
    mlog "  $c: sample exists, skip"
  else
    $PY Solar-Siting/covariate_sample.py "$c" --year $YEAR --n $N >> "$log" 2>&1 \
      && mlog "  $c: sample OK" || { mlog "  $c: sample FAILED (see $log)"; return; }
  fi

  # 2) append manual covariates (GSA + grid always; roads only for shp countries)
  local rflag=""; [ "$roads" = "yes" ] && rflag="--roads"
  $PY Solar-Siting/covariate_append_manual.py "$c" --year $YEAR --gsa --grid $rflag >> "$log" 2>&1 \
    && mlog "  $c: append OK" || mlog "  $c: append FAILED (see $log)"

  # 3) analysis (uses _full if append wrote it, else base)
  $PY Solar-Siting/covariate_analysis.py "$c" --year $YEAR >> "$log" 2>&1 \
    && mlog "  $c: analysis OK" || mlog "  $c: analysis FAILED (see $log)"

  # 4) comparison maps
  $PY Solar-Siting/covariate_maps.py "$c" --year $YEAR >> "$log" 2>&1 \
    && mlog "  $c: maps OK" || mlog "  $c: maps FAILED (see $log)"

  mlog "DONE $c"
}

lane(){ for spec in "$@"; do process_country "${spec%%|*}" "${spec##*|}"; done; }

mlog "===== OVERNIGHT RUN START (year $YEAR, n $N) ====="

# Lane A and Lane B run concurrently; countries within a lane run serially. The two
# heaviest (US, China) are kept in the SAME lane so they never hit Earth Engine at once.
lane "United States|no" "China|no" "Chile|yes" &
LA=$!
lane "Germany|no" "Australia|no" "South Africa|yes" &
LB=$!
wait $LA $LB

mlog "===== all lanes finished; aggregating summary ====="
$PY Solar-Siting/covariate_summary.py --year $YEAR >> "$MASTER" 2>&1
mlog "===== OVERNIGHT RUN COMPLETE ====="

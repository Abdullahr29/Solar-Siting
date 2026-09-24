#!/bin/bash
# Download Geofabrik -free.shp.zip road extracts for the 7 countries missing OSM roads.
# Read directly by bbox via /vsizip (no pyrosm extraction). Resumable (curl -C -).
set -u
cd ~/Solar_Workspace/Solar-Siting/covariate_data/osm
BASE=https://download.geofabrik.de
declare -A URL=(
  [spain]="$BASE/europe/spain-latest-free.shp.zip"
  [poland]="$BASE/europe/poland-latest-free.shp.zip"
  [japan]="$BASE/asia/japan-latest-free.shp.zip"
  [india]="$BASE/asia/india-latest-free.shp.zip"
  [colombia]="$BASE/south-america/colombia-latest-free.shp.zip"
  [philippines]="$BASE/asia/philippines-latest-free.shp.zip"
  [malaysia]="$BASE/asia/malaysia-singapore-brunei-latest-free.shp.zip"
)
ts(){ date +%H:%M:%S; }
echo "[$(ts)] START pull_osm_7 on $(hostname)"
for c in spain poland japan india colombia philippines malaysia; do
  out="${c}-latest-free.shp.zip"
  echo "[$(ts)] GET $c -> $out"
  curl -fsSL -C - -o "$out" "${URL[$c]}" \
    && echo "[$(ts)] OK $c $(stat -c%s "$out" 2>/dev/null || stat -f%z "$out") bytes" \
    || echo "[$(ts)] FAILED $c"
done
echo "[$(ts)] PULL COMPLETE"

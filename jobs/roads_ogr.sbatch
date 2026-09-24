#!/bin/bash
#SBATCH --job-name=roads_ogr
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/roads_ogr_%j.log
# Extract major-road lines from a .osm.pbf via GDAL's OSM driver (streams, bounded memory,
# no routable-graph build). Writes <slug>_roads.gpkg (layer 'roads') that sample_full reads.
set -e
export PATH=~/Solar_Workspace/envs/env_solar/bin:$PATH
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
cd ~/Solar_Workspace/Solar-Siting/covariate_data/osm
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
c=$1
pbf="${c}-latest.osm.pbf"
out="${c}_roads.gpkg"
if [ -f "$out" ]; then echo "$out exists -> skip"; exit 0; fi
echo "=== node $(hostname) | roads_ogr $c | start $(date) ==="
HW="'motorway','trunk','primary','secondary','tertiary','motorway_link','trunk_link','primary_link','secondary_link'"
# OSM_MAX_TMPFILE_SIZE keeps the node-resolution temp DB on disk (low RAM); interleaved reading is automatic.
ogr2ogr -f GPKG "$out" "$pbf" lines \
  -where "highway IN ($HW)" \
  -nln roads -nlt PROMOTE_TO_MULTI \
  --config OSM_MAX_TMPFILE_SIZE 1024 \
  -gt 65536
echo "=== done $(date) | $(ls -la "$out" 2>/dev/null) ==="
ogrinfo -so "$out" roads 2>/dev/null | grep -iE "Feature Count|Geometry" | head

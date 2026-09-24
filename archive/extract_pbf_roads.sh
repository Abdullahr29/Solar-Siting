#!/bin/bash
# Extract major-highway lines from the .osm.pbf countries into compact GPKGs, in parallel,
# using GDAL's OSM driver (fast C++), so the append step can read roads without pyrosm.
set -u
cd ~/Solar_Workspace/Solar-Siting/covariate_data/osm
OGR=~/Solar_Workspace/envs/env_solar/bin/ogr2ogr
export CPL_TMPDIR="$PWD/tmp"; mkdir -p "$PWD/tmp"
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
W="highway IN ('motorway','trunk','primary','secondary','tertiary','motorway_link','trunk_link','primary_link','secondary_link')"

declare -A PBF=(
  [germany]=germany-latest.osm.pbf
  [united_states]=us-latest.osm.pbf
  [china]=china-latest.osm.pbf
  [australia]=australia-latest.osm.pbf
)

for slug in "${!PBF[@]}"; do
  (
    echo "[$(date +%T)] START $slug (${PBF[$slug]})"
    rm -f "${slug}_roads.gpkg"
    $OGR --config OSM_MAX_TMPFILE_SIZE 2000 -f GPKG "${slug}_roads.gpkg" \
         "${PBF[$slug]}" lines -where "$W" -nln roads -skipfailures
    rc=$?
    sz=$(stat -c %s "${slug}_roads.gpkg" 2>/dev/null || echo 0)
    echo "[$(date +%T)] DONE $slug rc=$rc | gpkg=${sz} bytes"
  ) > "roads_extract_${slug}.log" 2>&1 &
done
wait
echo "[$(date +%T)] ALL PBF ROAD EXTRACTS DONE"

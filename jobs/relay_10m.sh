#!/usr/bin/env bash
# Drain 10m suitability tiles from Google Drive -> JASMIN, compress each to COG (DEFLATE),
# file into data_products/10m/country/<ISO3>/, delete raw + Drive copy. Runs forever (tmux).
# Robust to GEE sharding: each Drive file (whatever the shard) is COG'd independently;
# per-country mosaics are built later by mosaic_country.sh.
set -u
WS=~/Solar_Workspace
GDAL=$WS/envs/env_solar/bin
DRIVE="gdrive:solar10m_2025"
CDIR=$WS/data_products/10m/country
SLEEP=${1:-120}

while true; do
  mapfile -t files < <(rclone lsf "$DRIVE" 2>/dev/null | grep '\.tif$')
  if [ "${#files[@]}" -gt 0 ]; then
    echo "$(date +%H:%M:%S) relay: ${#files[@]} tiles on Drive"
    for f in "${files[@]}"; do
      iso=${f%%_*}
      dst=$CDIR/$iso; raw=$dst/_raw
      mkdir -p "$raw"
      rclone move "$DRIVE/$f" "$raw/" 2>/dev/null || continue
      out=$dst/${f%.tif}_cog.tif
      if "$GDAL/gdal_translate" -q -of COG -a_nodata -1 \
           -co COMPRESS=DEFLATE -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS \
           "$raw/$f" "$out" 2>/dev/null; then
        rm -f "$raw/$f"
      else
        echo "  COG FAIL $f (kept raw)"
      fi
    done
  fi
  sleep "$SLEEP"
done

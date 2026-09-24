#!/bin/bash
#SBATCH --job-name=roads_pbf
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/roads_%j.log
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
PY=~/Solar_Workspace/envs/env_solar/bin/python
echo "=== node $(hostname) | roads $1 | start $(date) ==="
$PY -u Solar-Siting/analysis/build_roads_pbf.py "$1"
echo "=== done $(date) ==="

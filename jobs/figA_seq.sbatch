#!/bin/bash
#SBATCH --job-name=figA_seq
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/figA_seq_%j.log
# Combined-set RF-vs-MCDA sampling, SEQUENTIAL (one country at a time -> GEE concurrency=1,
# no restricted-mode 429s). Resumable: mcda_bootstrap_v3 skips any country whose npz exists.
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
PY=~/Solar_Workspace/envs/env_solar/bin/python
echo "=== node $(hostname) | start $(date) | countries: $* ==="
for c in "$@"; do
  echo "----- [$(date +%H:%M:%S)] START $c -----"
  $PY -u Solar-Siting/analysis/mcda_bootstrap_v3.py "$c" && echo "----- OK $c -----" || echo "----- FAILED $c -----"
done
echo "=== done $(date) ==="

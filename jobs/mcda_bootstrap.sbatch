#!/bin/bash
#SBATCH --job-name=figA_comb
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/figA_%j.log
# Combined-set RF-vs-MCDA sampling (one country per job). env_solar direct; PROJ_DATA; EE creds.
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
PY=~/Solar_Workspace/envs/env_solar/bin/python
echo "=== node $(hostname) | $* | start $(date) ==="
$PY -u Solar-Siting/analysis/mcda_bootstrap_v3.py "$@"
echo "=== done $(date) ==="

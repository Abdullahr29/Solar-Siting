#!/bin/bash
#SBATCH --job-name=figA_post
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/figA_post_%j.log
# Local post-processing over the Fig-A per-point npz: paired bootstrap + covariate novelty +
# site profiling. No GEE. Runs after the sampling fleet (submit with --dependency=afterany:...).
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
export WS=~/Solar_Workspace
PY=~/Solar_Workspace/envs/env_solar/bin/python
echo "=== node $(hostname) | start $(date) ==="
echo "--- npz present ---"; ls Solar-Siting/artifacts/paper_v3/figA_combined/*_figA.npz 2>/dev/null | wc -l
echo "=== 1. paired bootstrap (RF vs MCDA) ==="; $PY -u Solar-Siting/analysis/bootstrap_figA.py
echo "=== 2. covariate novelty (13-country) ==="; $PY -u Solar-Siting/analysis/covariate_novelty_v3.py
echo "=== 3. site-level profiling ==="; $PY -u Solar-Siting/analysis/site_profiling_v3.py
echo "=== done $(date) ==="

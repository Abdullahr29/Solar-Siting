#!/bin/bash
# Chain the whole RF sweep with SLURM dependencies so it runs unattended:
#   Stage 1 array  ->  aggregate (afterany)  ->  Stage 2 GEE + decide (afterok)
# Run from ~/Solar_Workspace:  bash Solar-Siting/jobs/submit_sweep.sh
set -e
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
LOG=~/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs
mkdir -p "$LOG" ~/Solar_Workspace/Solar-Siting/artifacts/paper_v3/sweep/models
PY=~/Solar_Workspace/envs/env_solar/bin/python

# Stage 1: the config-sweep array
J1=$(sbatch --parsable Solar-Siting/jobs/rf_sweep_stage1.sbatch)
echo "Stage 1 (array): job $J1"

# Aggregate Stage 1 -> shortlist (runs after ALL array tasks finish, success or not)
J2=$(sbatch --parsable --dependency=afterany:$J1 --account=gbov --partition=standard \
     --qos=high --cpus-per-task=2 --mem=8G --time=00:20:00 \
     --job-name=rfsweep_agg1 --output=$LOG/rfsweep_agg1_%j.log \
     --wrap "export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj; cd ~/Solar_Workspace; $PY -u Solar-Siting/archive/rf_sweep_aggregate.py")
echo "Stage 1 aggregate: job $J2 (after $J1)"

# Stage 2: honest country validation of the shortlist + final decision (after shortlist exists)
J3=$(sbatch --parsable --dependency=afterok:$J2 Solar-Siting/jobs/rf_sweep_stage2.sbatch)
echo "Stage 2 (GEE) + decision: job $J3 (after $J2)"

echo
echo "submitted chain: $J1 -> $J2 -> $J3"
echo "watch:  squeue -u \$USER ; tail -f $LOG/rfsweep1_*_0.log"
echo "results: Solar-Siting/artifacts/paper_v3/sweep/  (sweep_stage1_results.csv, sweep_stage2_country.csv)"

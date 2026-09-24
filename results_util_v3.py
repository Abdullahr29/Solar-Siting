"""Single source of truth for the paper_v3 campaign: append one row per (experiment, model,
scope, metric) to results_master.csv, and stamp provenance into run_manifest.jsonl.

Every v3 driver imports append_result() so all experiments land in ONE index that figure
scripts can filter/join without hunting across directories. Paths are stored RELATIVE to the
workspace root so the index is portable.
"""
import os, csv, json, time, fcntl

WS = os.path.expanduser("~/Solar_Workspace")
ROOT = "Solar-Siting/artifacts/paper_v3"
MASTER = os.path.join(WS, ROOT, "results_master.csv")
MANIFEST = os.path.join(WS, ROOT, "run_manifest.jsonl")
COLS = ["experiment", "method", "scheme", "scope", "metric", "value", "n_pos", "n_neg",
        "model_path", "scores_path", "figure_path", "aef_year", "timestamp", "notes"]

for d in ["", "pools", "models", "models/bakeoff", "models/loco", "models/temporal",
          "scores", "scores/country", "scores/temporal", "scores/loco",
          "figures", "figures/bakeoff", "figures/country", "figures/temporal",
          "figures/loco", "figures/qualitative", "logs"]:
    os.makedirs(os.path.join(WS, ROOT, d), exist_ok=True)


def _rel(p):
    if not p:
        return ""
    p = str(p)
    return os.path.relpath(p, WS) if os.path.isabs(p) else p


def append_result(experiment, method, scheme, scope, metric, value, n_pos=None, n_neg=None,
                  model_path="", scores_path="", figure_path="", aef_year="", notes=""):
    """Append one metric row to results_master.csv (file-locked; header written once)."""
    new = not os.path.exists(MASTER)
    row = dict(experiment=experiment, method=method, scheme=scheme, scope=scope, metric=metric,
               value=round(float(value), 5) if value is not None else "",
               n_pos=n_pos if n_pos is not None else "", n_neg=n_neg if n_neg is not None else "",
               model_path=_rel(model_path), scores_path=_rel(scores_path),
               figure_path=_rel(figure_path), aef_year=aef_year,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), notes=notes)
    with open(MASTER, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"  [results] {experiment}/{method}/{scheme} {scope} {metric}={row['value']}", flush=True)


def log_stage(stage, cmd, status, extra=None):
    """Append a provenance line to run_manifest.jsonl."""
    rec = dict(stage=stage, cmd=cmd, status=status, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if extra:
        rec.update(extra)
    with open(MANIFEST, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

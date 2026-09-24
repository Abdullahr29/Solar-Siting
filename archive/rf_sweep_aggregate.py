"""Stage 1 aggregator. Reads all sweep/config_*.json, writes a ranked table, and emits the
Stage-2 shortlist: the top-K GEE-feasible (<=9 MB) configs by held-out pixel ROC, always
including config 0 (incumbent hyperparameters) so Stage 2 compares like-for-like."""
import os, sys, json, glob
import numpy as np, pandas as pd

WS = os.path.expanduser("~/Solar_Workspace")
SW = os.path.join(WS, "Solar-Siting/artifacts/paper_v3/sweep")
TOP_K = 5


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SW, "config_*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        sys.exit("no config_*.json found — did Stage 1 run?")
    df = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)
    df.to_csv(os.path.join(SW, "sweep_stage1_results.csv"), index=False)

    feas = df[df.feasible].sort_values("val_roc", ascending=False)
    inc = df[df.config_id == 0].iloc[0]
    cols = ["config_id", "is_incumbent", "n_estimators", "max_depth", "min_samples_leaf",
            "max_features", "max_samples", "size_mb", "feasible", "val_roc", "mean_leaves"]
    print(f"\n=== Stage 1: {len(df)} configs, {len(feas)} feasible (<=9 MB) ===")
    print(f"incumbent (id 0): size {inc.size_mb} MB, val_roc {inc.val_roc}")
    print("\nTop feasible by held-out pixel ROC:")
    print(feas[cols].head(12).to_string(index=False))
    print("\nInfeasible (>9 MB) for reference:")
    print(df[~df.feasible][cols].sort_values("val_roc", ascending=False).head(6).to_string(index=False))

    # shortlist = top-K feasible UNION incumbent
    short_ids = list(dict.fromkeys(list(feas.head(TOP_K).config_id) + [0]))
    shortlist = [df[df.config_id == c].iloc[0].to_dict() for c in short_ids]
    with open(os.path.join(SW, "stage2_shortlist.json"), "w") as f:
        json.dump(shortlist, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(f"\nStage-2 shortlist (config_ids): {short_ids}")
    print(f"wrote {SW}/stage2_shortlist.json and sweep_stage1_results.csv")


if __name__ == "__main__":
    main()

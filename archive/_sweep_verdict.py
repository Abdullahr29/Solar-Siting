import pandas as pd, json, os
S = "artifacts/paper_v3/sweep"
s1 = pd.read_csv(f"{S}/sweep_stage1_results.csv").set_index("config_id")
s2 = pd.read_csv(f"{S}/sweep_stage2_country.csv")
# site-count-weighted mean country ROC per config
def wmean(g): return (g.roc * g.n).sum() / g.n.sum()
agg = s2.groupby("config_id").apply(wmean).rename("wroc").reset_index()
agg = agg.merge(s1[["n_estimators","max_depth","min_samples_leaf","max_features","max_samples","val_roc","size_mb","is_incumbent"]], left_on="config_id", right_index=True)
agg = agg.sort_values("wroc", ascending=False)
inc = agg[agg.is_incumbent].wroc.iloc[0]
agg["d_vs_inc"] = agg.wroc - inc
print("=== stage2: site-weighted mean country ROC (feasible configs) ===")
print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nincumbent weighted ROC = {inc:.4f}")

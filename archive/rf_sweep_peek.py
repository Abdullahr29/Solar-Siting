import json, glob, os
WS = os.path.expanduser("~/Solar_Workspace")
rows = [json.load(open(f)) for f in
        glob.glob(os.path.join(WS, "Solar-Siting/artifacts/paper_v3/sweep/config_*.json"))]
rows.sort(key=lambda d: d["config_id"])
nf = sum(r["feasible"] for r in rows)
print("%d configs done | feasible (<=9MB): %d | infeasible: %d" % (len(rows), nf, len(rows) - nf))
feas = sorted([r for r in rows if r["feasible"]], key=lambda r: -r["val_roc"])
print("\ntop feasible by held-out pixel ROC:")
for r in feas[:8]:
    tag = " [INCUMBENT]" if r["config_id"] == 0 else ""
    print("  cfg%02d trees=%d d=%s leaf=%d feat=%s samp=%s  %5.2fMB  valROC=%.4f%s" % (
        r["config_id"], r["n_estimators"], r["max_depth"], r["min_samples_leaf"],
        str(r["max_features"]), r["max_samples"], r["size_mb"], r["val_roc"], tag))
inc = [r for r in rows if r["config_id"] == 0]
if inc:
    i = inc[0]
    print("\nincumbent cfg00: %.2fMB valROC=%.4f feasible=%s" % (i["size_mb"], i["val_roc"], i["feasible"]))

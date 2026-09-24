"""TZ-SAM capacity-density verification + exploration.

Per the TZ-SAM methodology (Eq.1): C_AC = A * I * eta * GCR * ILR, where A=footprint, and eta/GCR/ILR
are MODELLED per country & plant size. So density = capacity/area = I*eta*GCR*ILR should VARY
(not a single constant) if the country/size model is doing work. This quantifies that.

Computes footprint area (EPSG:6933 equal-area) per site, density MW/km2, its distribution, per-country
medians, log-log capacity~area regression, and density-vs-size — to decide whether density is
meaningful (varies) or circular (constant), and to get the number(s) usable for capacity potential.
"""
import os
import numpy as np
import geopandas as gpd

GPKG = os.path.expanduser("~/Solar_Workspace/Data/external/tz_sam_q1_2026/"
                          "TZ-SAM Q1-2026 CC-BY-NC/2026-Q1_analysis_polygons.gpkg")
EA = "EPSG:6933"


CSV = os.path.expanduser("~/Solar_Workspace/Data/external/tz_sam_q1_2026/"
                         "TZ-SAM Q1-2026 CC-BY-NC/2026-Q1_analysis_polygons.csv")


def main():
    import pandas as pd
    g = gpd.read_file(GPKG)
    meta = pd.read_csv(CSV, usecols=lambda c: c.strip().lstrip("﻿") in ("cluster_id", "country"))
    meta.columns = [c.strip().lstrip("﻿") for c in meta.columns]
    g["cluster_id"] = g["cluster_id"].astype(str)
    meta["cluster_id"] = meta["cluster_id"].astype(str)
    g = g.merge(meta, on="cluster_id", how="left")
    print("columns:", list(g.columns), "| n=", len(g))
    g = g[g["capacity_mw"] > 0].copy()
    g["area_km2"] = g.to_crs(EA).geometry.area / 1e6
    g = g[g["area_km2"] > 0].copy()
    g["density"] = g["capacity_mw"] / g["area_km2"]     # MW/km2
    # guard against degenerate geometries
    g = g[(g["density"] > 0.5) & (g["density"] < 500)].copy()

    d = g["density"].to_numpy()
    print(f"\n=== DENSITY (MW/km2) over {len(g)} sites ===")
    for q in [5, 25, 50, 75, 95]:
        print(f"  p{q:02d} = {np.percentile(d, q):.1f}")
    print(f"  mean={d.mean():.1f}  std={d.std():.1f}  CV={d.std()/d.mean():.2f}")
    print(f"  aggregate density = total_cap/total_area = "
          f"{g['capacity_mw'].sum()/g['area_km2'].sum():.1f} MW/km2 "
          f"(tot {g['capacity_mw'].sum()/1000:.0f} GW / {g['area_km2'].sum():.0f} km2)")

    # per-country variation (does the country model move density?)
    print("\n=== per-country median density (top 15 by site count) ===")
    gc = g.groupby("country")
    top = gc.size().sort_values(ascending=False).head(15).index
    for c in top:
        sub = g[g["country"] == c]
        print(f"  {c:16} n={len(sub):6d}  median_density={sub['density'].median():5.1f}  "
              f"IQR[{sub['density'].quantile(.25):.1f},{sub['density'].quantile(.75):.1f}]")
    cm = gc["density"].median()
    print(f"  -> country medians span {cm.min():.1f} to {cm.max():.1f} MW/km2 "
          f"(ratio {cm.max()/cm.min():.1f}x) across {len(cm)} countries")

    # log-log capacity ~ area: slope 1 + zero scatter => pure constant (circular);
    # slope!=1 or scatter => size/country structure
    la = np.log(g["area_km2"].to_numpy()); lc = np.log(g["capacity_mw"].to_numpy())
    A = np.vstack([la, np.ones_like(la)]).T
    slope, inter = np.linalg.lstsq(A, lc, rcond=None)[0]
    pred = A @ [slope, inter]
    r2 = 1 - ((lc - pred) ** 2).sum() / ((lc - lc.mean()) ** 2).sum()
    print(f"\n=== log(capacity) ~ log(area): slope={slope:.3f}  R2={r2:.3f} ===")
    print(f"  (slope=1 & R2=1 => capacity is exactly area*const = circular;")
    print(f"   R2={r2:.3f}<1 and/or slope!=1 => real country/size structure in density)")

    # density vs plant size
    print("\n=== density by plant-size bin ===")
    for lo, hi in [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1e6)]:
        sub = g[(g["capacity_mw"] >= lo) & (g["capacity_mw"] < hi)]
        if len(sub):
            print(f"  {lo:>4}-{hi if hi<1e6 else 'inf':>4} MW: n={len(sub):6d}  median_density={sub['density'].median():5.1f}")


if __name__ == "__main__":
    main()

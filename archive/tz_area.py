import os, numpy as np, pandas as pd, geopandas as gpd
os.chdir(os.path.expanduser("~/Solar_Workspace"))
EA = "EPSG:6933"   # global equal-area: polygon.area is true m^2, distances in metres
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
TZG = "Data/external/tz_sam_q1_2026/TZ-SAM Q1-2026 CC-BY-NC/2026-Q1_analysis_polygons.gpkg"

inv = gpd.read_file(INV); inv = inv[inv.geometry.notna() & inv.is_valid].copy()
print(f"inventory: {len(inv):,} sites | years {int(inv['year'].min())}-{int(inv['year'].max())} | "
      f"matching uses ALL of them (no year filter)")
inv_pt = inv.copy(); inv_pt["geometry"] = inv_pt.geometry.representative_point()
inv_pt = inv_pt.to_crs(EA)

tz = gpd.read_file(TZG); tz = tz[tz.geometry.notna() & tz.is_valid].copy().to_crs(EA)
tz["area_m2"] = tz.geometry.area
tz["area_ha"] = tz["area_m2"] / 1e4
tz["area_px"] = tz["area_m2"] / 100.0            # 10 m AEF pixels
j = gpd.sjoin_nearest(tz[["geometry"]], inv_pt[["geometry"]], how="left", distance_col="d")
j = j[~j.index.duplicated(keep="first")]
tz["nn_m"] = j["d"].to_numpy()
tz["new"] = tz["nn_m"] > 1000

def stats(s):
    q = np.percentile(s, [10, 25, 50, 75, 90])
    return f"p10 {q[0]:,.0f}  p25 {q[1]:,.0f}  MED {q[2]:,.0f}  p75 {q[3]:,.0f}  p90 {q[4]:,.0f}"

for lab, m in [("NEW    ", tz["new"]), ("MATCHED", ~tz["new"]), ("ALL    ", tz["new"] | ~tz["new"])]:
    sub = tz[m]
    print(f"\n=== {lab} (n={len(sub):,}) ===")
    print(f"  area m^2 : {stats(sub['area_m2'])}")
    print(f"  area ha  : {stats(sub['area_ha'])}")
    print(f"  area px  : {stats(sub['area_px'])}  (10 m pixels)")
    print(f"  capac MW : {stats(sub['capacity_mw'])}")

new = tz[tz["new"]]
print("\n=== 'missed because small?' — NEW-site size buckets ===")
bins = [0, 0.1, 0.5, 1, 2, 5, 10, 50, 1e9]
labs = ["<0.1 ha (<1000 m²)", "0.1–0.5 ha", "0.5–1 ha", "1–2 ha", "2–5 ha", "5–10 ha", "10–50 ha", ">50 ha"]
cats = pd.cut(new["area_ha"], bins=bins, labels=labs, right=False)
vc = cats.value_counts().reindex(labs)
for l in labs:
    print(f"  {l:20s} {int(vc[l]):6,}  ({100*vc[l]/len(new):4.1f}%)")
print(f"\nNEW median area {new['area_ha'].median():.2f} ha  vs  MATCHED median {tz[~tz['new']]['area_ha'].median():.2f} ha")
print(f"NEW median capacity {new['capacity_mw'].median():.2f} MW  vs  MATCHED {tz[~tz['new']]['capacity_mw'].median():.2f} MW")
print(f"NEW sites < 1 ha: {100*(new['area_ha']<1).mean():.1f}%  |  < 0.5 ha: {100*(new['area_ha']<0.5).mean():.1f}%")

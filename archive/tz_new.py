import os, numpy as np, pandas as pd, geopandas as gpd
os.chdir(os.path.expanduser("~/Solar_Workspace"))
EA = "EPSG:6933"
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
TZC = "Data/external/tz_sam_q1_2026/TZ-SAM Q1-2026 CC-BY-NC/2026-Q1_analysis_polygons.csv"

inv = gpd.read_file(INV)
inv = inv[inv.geometry.notna() & inv.is_valid].copy()
inv_yr = inv["year"] if "year" in inv.columns else gpd.read_file(INV, columns=["year"])["year"]
inv_pt = inv.copy(); inv_pt["geometry"] = inv_pt.geometry.representative_point()
inv_pt = inv_pt.to_crs(EA)

tz = pd.read_csv(TZC)
tzp = gpd.GeoDataFrame(tz, geometry=gpd.points_from_xy(tz.longitude, tz.latitude), crs="EPSG:4326").to_crs(EA)
j = gpd.sjoin_nearest(tzp[["geometry"]], inv_pt[["geometry"]], how="left", distance_col="d")
j = j[~j.index.duplicated(keep="first")]
tz["nn_m"] = j["d"].to_numpy()
tz["new"] = tz["nn_m"] > 1000
tz["yr"] = pd.to_datetime(tz["constructed_after"], errors="coerce").dt.year

N, Nnew = len(tz), int(tz["new"].sum())
new = tz[tz["new"]].copy()
print(f"OUR inventory: {len(inv):,} sites | years {int(inv_yr.min())}-{int(inv_yr.max())}")
print(f"TZ-SAM analysis sites: {N:,}")
print(f"NEW (>1 km from any inventory site): {Nnew:,}  ({100*Nnew/N:.1f}%)")
print(f"corroborate ours (<=1 km): {N-Nnew:,}  ({100*(N-Nnew)/N:.1f}%)\n")

print(f"NEW sites span {new['country'].nunique()} countries")
print("--- top 15 countries by NEW sites ---")
print(new["country"].value_counts().head(15).to_string())

print("\n--- NEW sites by construction year (constructed_after) ---")
yc = new.groupby("yr").size()
print(yc.to_string())
post = new[new["yr"] >= 2024]
print(f"\nNEW built >= 2024 (beyond our inventory max): {len(post):,}  ({100*len(post)/N:.1f}% of all TZ)")
print(f"   -> spanning {post['country'].nunique()} countries; top: "
      f"{', '.join(f'{k} {v}' for k,v in post['country'].value_counts().head(6).items())}")

print(f"\n--- capacity ---")
print(f"NEW sites total capacity: {new['capacity_mw'].sum()/1000:.1f} GW "
      f"(median {new['capacity_mw'].median():.1f} MW/site)")
print(f"all TZ-SAM capacity: {tz['capacity_mw'].sum()/1000:.1f} GW")
new.to_csv("Data/external/tz_sam_q1_2026/overlap_analysis/tz_new_sites.csv", index=False)

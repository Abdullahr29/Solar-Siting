"""Per-country install-year counts from the inventory — to choose validation / LOCO / temporal
countries on evidence. Validation uses inventory sites installed >= map_year+2 (forward),
so n(>=2021) is what makes a country's ROC meaningful; n(2024) drives temporal; total gauges
how 'low-data' (under-represented in training) a country is.
"""
import os
import geopandas as gpd
import pandas as pd

os.chdir(os.path.expanduser("~/Solar_Workspace"))
inv = gpd.read_file("Solar-Siting/global_pv_facility_inventory.gpkg")
print(f"inventory rows: {len(inv):,} | columns: {list(inv.columns)}")
yc = "year"
inv = inv[inv[yc].notna()].copy()
inv[yc] = inv[yc].astype(int)

g = inv.groupby("country")
tab = pd.DataFrame({
    "total": g.size(),
    "le2018": g.apply(lambda d: (d[yc] <= 2018).sum()),
    "ge2021": g.apply(lambda d: (d[yc] >= 2021).sum()),
    "y2022": g.apply(lambda d: (d[yc] == 2022).sum()),
    "y2023": g.apply(lambda d: (d[yc] == 2023).sum()),
    "y2024": g.apply(lambda d: (d[yc] == 2024).sum()),
}).sort_values("total", ascending=False)

pd.set_option("display.max_rows", 200); pd.set_option("display.width", 160)
print("\n===== ALL countries with >=1 site, by total (install-year buckets) =====")
print(tab.to_string())

print("\n===== HIGH-DATA (total >= 2000): temporal/LOCO candidates =====")
print(tab[tab.total >= 2000].to_string())

print("\n===== LOW-DATA but VALIDATABLE (total < 800 AND ge2021 >= 40) =====")
print(tab[(tab.total < 800) & (tab.ge2021 >= 40)].sort_values("ge2021", ascending=False).to_string())

print(f"\nyear range: {inv[yc].min()}–{inv[yc].max()}")
print("global year histogram:")
print(inv[yc].value_counts().sort_index().to_string())

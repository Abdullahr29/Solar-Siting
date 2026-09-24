"""Investigate: how many INDIVIDUAL PV facilities (PV_IDs) sit inside the sites/scenes we actually used?
16,053 = clustered scenes. We want the count of distinct source-inventory entries they aggregate.
"""
import os
os.chdir("/gws/ssde/j25b/gbov/abdullah_solar")
import geopandas as gpd
from pyogrio import list_layers

PIPE = "Solar-Siting/global_solar_ml_pipeline.gpkg"
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"

print("=== pipeline layers ===", flush=True)
print(list_layers(PIPE), flush=True)
print("=== inventory layers ===", flush=True)
print(list_layers(INV), flush=True)

for path, lyr in [(PIPE, "base_train"), (PIPE, "training_data_solar")]:
    try:
        g = gpd.read_file(path, layer=lyr, rows=3)
        print(f"\n--- {lyr} columns ({len(g.columns)}) ---", flush=True)
        print(list(g.columns), flush=True)
        # full row count without loading geometry-heavy frame
        n = len(gpd.read_file(path, layer=lyr, ignore_geometry=True))
        print(f"{lyr}: {n} rows", flush=True)
    except Exception as e:
        print(f"{lyr}: ERR {e}", flush=True)

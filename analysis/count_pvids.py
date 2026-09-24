"""Count INDIVIDUAL PV facilities (PV_IDs) vs clustered scenes actually used as positives."""
import os, ast
os.chdir("/gws/ssde/j25b/gbov/abdullah_solar")
os.environ.setdefault("PROJ_DATA", "/gws/ssde/j25b/gbov/abdullah_solar/envs/env_solar/share/proj")
import pandas as pd, geopandas as gpd
PIPE = "Solar-Siting/global_solar_ml_pipeline.gpkg"

bt = gpd.read_file(PIPE, layer="base_train", ignore_geometry=True)
ts = gpd.read_file(PIPE, layer="training_data_solar", ignore_geometry=True)

print("=== base_train (individual facilities) ===", flush=True)
print("rows:", len(bt), "| distinct PV_ID:", bt["PV_ID"].nunique(), flush=True)
print("split counts:\n", bt["dataset_split"].value_counts(dropna=False), flush=True)
print("exported flag counts:\n", bt["exported"].value_counts(dropna=False), flush=True)

print("\n=== training_data_solar (clusters/scenes) ===", flush=True)
print("clusters:", len(ts), flush=True)
print("sum(num_sites):", int(ts["num_sites"].sum()), flush=True)
print("split counts:\n", ts["dataset_split"].value_counts(dropna=False), flush=True)

# distinct PV_IDs aggregated inside clusters, via original_site_ids
def parse_ids(v):
    if v is None or (isinstance(v, float)): return []
    try: return list(ast.literal_eval(v)) if isinstance(v, str) and v.strip().startswith(("[", "(")) else [x for x in str(v).replace(";", ",").split(",") if x.strip()]
    except Exception: return [x for x in str(v).replace(";", ",").split(",") if x.strip()]

allids = set()
for v in ts["original_site_ids"]:
    for x in parse_ids(v): allids.add(str(x).strip())
print("distinct PV_IDs across original_site_ids:", len(allids), flush=True)
print("sample original_site_ids:", ts["original_site_ids"].dropna().iloc[0], flush=True)

# how many clusters/facilities were actually EXPORTED (on-disk positives)
chips_dir = "Data/aef_solar_chips/positives/chips"
if os.path.isdir(chips_dir):
    fn = [f for f in os.listdir(chips_dir) if f.startswith("POS_") and f.endswith(".tif")]
    scenes = set("_".join(f.split("_")[:4]) for f in fn)  # POS_CCC_id_year
    print(f"\non-disk positive chips: {len(fn)} | distinct scenes: {len(scenes)}", flush=True)

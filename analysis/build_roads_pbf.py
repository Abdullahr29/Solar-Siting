"""Extract major-road network from a country .osm.pbf to <slug>_roads.gpkg (pyrosm), whole-country.

Whole-file parse (one pyrosm overhead, unlike tiling which pays it per tile) — needs high RAM
(64 GB OOMs on a 2 GB pbf; run with --mem=256G). This is how china/germany/us roads were built.
sample_full picks up {slug}_roads.gpkg automatically.
"""
import os, sys, time
os.chdir(os.path.expanduser("~/Solar_Workspace"))
from covariate_append_manual import MAJOR_HW, CD

country = sys.argv[1].lower()
pbf = f"{CD}/osm/{country}-latest.osm.pbf"
gpkg = f"{CD}/osm/{country}_roads.gpkg"
if os.path.exists(gpkg):
    print(f"{gpkg} exists -> skip"); sys.exit(0)
print(f"[{time.strftime('%H:%M:%S')}] START {country} whole-country parse of {os.path.basename(pbf)}", flush=True)
from pyrosm import OSM
osm = OSM(pbf)
t0 = time.time()
r = osm.get_network(network_type="driving", custom_filter={"highway": MAJOR_HW})
print(f"[{time.strftime('%H:%M:%S')}] extracted {len(r)} road lines ({time.time()-t0:.0f}s); writing gpkg", flush=True)
r[["geometry"]].to_file(gpkg, driver="GPKG")
print(f"[{time.strftime('%H:%M:%S')}] DONE {country} -> {gpkg} ({os.path.getsize(gpkg)} bytes, {time.time()-t0:.0f}s)", flush=True)

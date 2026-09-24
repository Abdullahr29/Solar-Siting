"""Empirically estimate the GEE EECU cost of building the 6M.t-3 pool.

The build re-samples AEF(install-3) at ~43.94M positive pixels (the >120-chip, AEF-year>=2018 gap;
the 2019 cohort reuses t-2 for free). That op is reduceRegions(points, first, scale=10) over the
64-band annual embedding mosaic. We run it as a BATCH TABLE EXPORT on a probe of N points so the
completed task reports `batch_eecu_usage_seconds` (EE's own accounting), then scale linearly.
"""
import os, time, json, numpy as np
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting"))
import ee; ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
N = int(os.environ.get("N", 100000))
GAP_POINTS = 43_940_796                      # from t3_coverage_check.py
img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2019-01-01", "2020-01-01").mosaic().select(BANDS))
# land-heavy mixed region (India+Iberia+E-China boxes) so reads are representative, not all-ocean
region = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Rectangle([68, 8, 90, 30])),     # India
    ee.Feature(ee.Geometry.Rectangle([-9, 36, 3, 44])),     # Iberia
    ee.Feature(ee.Geometry.Rectangle([100, 22, 120, 40])),  # E China
]).geometry()
pts = ee.FeatureCollection.randomPoints(region, N, 7)
sampled = img.reduceRegions(pts, ee.Reducer.first(), scale=10)
task = ee.batch.Export.table.toDrive(collection=sampled, description=f"eecu_probe_{N}",
                                     folder="eecu_probe", fileFormat="CSV",
                                     selectors=BANDS[:4])   # write a few bands; reduction still computes all
t0 = time.time(); task.start()
print(f"probe: N={N:,} points, task started id={task.id}", flush=True)
while task.active():
    time.sleep(15)
st = task.status()
print("STATUS:", json.dumps(st, indent=1, default=str), flush=True)
eecu = st.get("batch_eecu_usage_seconds")
if eecu is None:
    for k, v in st.items():
        if "eecu" in k.lower(): eecu = v
print(f"\nwall={time.time()-t0:.0f}s  state={st.get('state')}", flush=True)
if eecu:
    per = float(eecu) / N
    total = per * GAP_POINTS
    print(f"\n=== EECU ESTIMATE for 6M.t-3 build ===")
    print(f"probe EECU-seconds        : {float(eecu):.1f}  ({N:,} points)")
    print(f"per-point EECU-seconds    : {per:.6f}")
    print(f"gap points to sample      : {GAP_POINTS:,}")
    print(f"TOTAL EECU-seconds        : {total:,.0f}")
    print(f"TOTAL EECU-hours          : {total/3600:,.1f}")
else:
    print("no EECU field in status; dump above has the raw keys", flush=True)

"""Global 10m solar-suitability export (2025 AEF, released R model) -> Google Drive.
Country-driven tiling: iterate geoBoundaries ADM0 (ISO3), tile each country's bbox into
TILE-degree cells, classify(2025 AEF).clip(country) per cell, export in the cell's NATIVE UTM
zone at 10m (zero resampling). Suitability = P(solar) x10000 int16, nodata=-1.

Resumable: skips a tile if its COG already exists on disk OR it's logged submitted+active.
Throttled to keep <= MAXPENDING tasks in flight so we never blow the GEE queue.

Usage:
  python export_10m.py --dry                 # print tile plan, submit nothing
  python export_10m.py --only=GRC            # one country (validation)
  python export_10m.py                       # full global run (tmux)
"""
import ee, joblib, os, sys, math, time
from geemap import ml
from shapely.geometry import shape, box

WS = os.path.expanduser("~/Solar_Workspace")
ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
DRIVE = "solar10m_2025"
TILE = 2.0
MAXPENDING = 200
TILEDIR = f"{WS}/data_products/10m/_tiles"
COUNTRYDIR = f"{WS}/data_products/10m/country"
DONE = f"{TILEDIR}/submitted.log"
os.makedirs(TILEDIR, exist_ok=True)

args = sys.argv[1:]
DRY = "--dry" in args
ONLY = None
for a in args:
    if a.startswith("--only="):
        ONLY = set(a.split("=", 1)[1].split(","))


def utm_epsg(lon, lat):
    z = int((lon + 180) // 6) + 1
    z = min(max(z, 1), 60)
    return f"EPSG:{'326' if lat >= 0 else '327'}{z:02d}"


def frange(a, b, step):
    x = math.floor(a / step) * step
    while x < b:
        yield x
        x += step


def cog_exists(iso, r, c):
    d = f"{COUNTRYDIR}/{iso}"
    if not os.path.isdir(d):
        return False
    pat = f"{iso}_{r}_{c}"
    return any(f.startswith(pat) and f.endswith("_cog.tif") for f in os.listdir(d))


# ---- model -> GEE classifier (once) ----
rf = joblib.load(f"{WS}/Solar-Siting/artifacts/paper_v3/models/asset_R_alldata.joblib")
clf = ml.strings_to_classifier(ml.rf_to_strings(rf, BANDS, processes=16, output_mode="PROBABILITY"))
img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2025-01-01", "2026-01-01").mosaic().select(BANDS))
suit = img.classify(clf).rename("suitability").multiply(10000).round().toInt16()

fc = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM0")
isos = sorted(set(fc.aggregate_array("shapeGroup").getInfo()))
if ONLY:
    isos = [i for i in isos if i in ONLY]
print(f"countries to process: {len(isos)}", flush=True)

submitted = set()
if os.path.exists(DONE):
    submitted = set(l.split()[0] for l in open(DONE) if l.strip())


def n_active():
    try:
        return sum(1 for t in ee.batch.Task.list() if t.state in ("READY", "RUNNING"))
    except Exception:
        return 0


log = open(DONE, "a")
plan, sub = 0, 0
for iso in isos:
    geom = fc.filter(ee.Filter.eq("shapeGroup", iso)).geometry()
    try:
        shp = shape(geom.getInfo())                  # client-side polygon (as run_country_rf.py does)
    except Exception as e:
        print(f"  {iso}: geom failed {str(e)[:60]}", flush=True); continue
    lon0, lat0, lon1, lat1 = shp.bounds
    img_c = suit.clip(geom).unmask(-1)
    cells = [(r, c, x, y)
             for r, y in enumerate(frange(lat0, lat1, TILE))
             for c, x in enumerate(frange(lon0, lon1, TILE))]
    # coverage filter, client-side (no per-cell GEE): keep cells that overlap country land.
    # AEF is a global land embedding, so land-intersection == "an AEF tile exists here".
    keep = [(r, c, x, y) for r, c, x, y in cells if shp.intersects(box(x, y, x + TILE, y + TILE))]
    ncov = len(keep)
    for r, c, x, y in keep:
        tid = f"{iso}_{r}_{c}"
        plan += 1
        if tid in submitted or cog_exists(iso, r, c):
            continue
        cell = ee.Geometry.Rectangle([x, y, x + TILE, y + TILE], None, False)
        crs = utm_epsg(x + TILE / 2, y + TILE / 2)
        if DRY:
            continue
        if sub % 25 == 0:                            # throttle-check periodically, not every tile
            while True:
                na = n_active()
                if na < MAXPENDING:
                    break
                print(f"  [throttle] {na} in flight >= {MAXPENDING}, sleeping 120s...", flush=True)
                time.sleep(120)
        t = ee.batch.Export.image.toDrive(
            image=img_c, description=tid, folder=DRIVE, fileNamePrefix=tid,
            region=cell, crs=crs, scale=10, maxPixels=int(1e12))
        t.start()
        log.write(f"{tid} {t.id} {crs}\n"); log.flush()
        sub += 1
        if sub % 25 == 0:
            print(f"  submitted {sub} (active~{n_active()})", flush=True)
        time.sleep(0.4)
    print(f"  {iso}: cells={len(cells)} covered={ncov} submitted_total={sub}", flush=True)

print(f"DONE. total planned tiles={plan}, submitted this run={sub}", flush=True)

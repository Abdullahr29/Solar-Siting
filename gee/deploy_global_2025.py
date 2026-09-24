"""Submit the master 100m global suitability export on 2025 AEF using the released R model.
Runs the RF classify ONCE as a GEE toAsset export -> that asset becomes the GEE deliverable AND
the source for all Zenodo continent/country sub-clips. Suitability stored int16 x10000 (0..10000)
to halve size; nodata = -1 over ocean/no-AEF (masked)."""
import ee, joblib, os
from geemap import ml
WS = os.path.expanduser("~/Solar_Workspace")
ee.Initialize(project="ee-abdullahr-solar")

MODEL = f"{WS}/Solar-Siting/artifacts/paper_v3/models/asset_R_alldata.joblib"
BANDS = [f"A{i:02d}" for i in range(64)]
ASSET = "projects/ee-abdullahr-solar/assets/solar_suitability_2025_R_100m"

rf = joblib.load(MODEL)
strs = ml.rf_to_strings(rf, BANDS, processes=32, output_mode="PROBABILITY")
size_mb = sum(len(s) for s in strs)/1e6
print(f"classifier: {len(strs)} trees, {size_mb:.1f} MB tree-strings", flush=True)
clf = ml.strings_to_classifier(strs)

img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2025-01-01","2026-01-01").mosaic().select(BANDS))
suit = (img.classify(clf).rename("suitability")
        .multiply(10000).round().toInt16().unmask(-1, sameFootprint=False))

region = ee.Geometry.Rectangle([-180,-60,180,82], None, False)
task = ee.batch.Export.image.toAsset(
    image=suit, description="solar_suit_2025_R_100m", assetId=ASSET,
    region=region, scale=100, maxPixels=int(1e13),
    pyramidingPolicy={"suitability":"mean"})
task.start()
print(f"SUBMITTED toAsset task id={task.id} -> {ASSET}", flush=True)
print("monitor: earthengine task info", task.id, flush=True)

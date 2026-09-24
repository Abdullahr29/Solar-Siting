"""Broad regional suitability heatmap (ne15 vs ne30) via GEE server-side thumbnail — to eyeball
whether speckle reads as texture at ~50km scale. One getThumbURL per model (efficient, not point sampling)."""
import os, sys, requests, numpy as np, joblib
os.chdir(os.path.expanduser("~/Solar_Workspace/Solar-Siting")); sys.path.insert(0, ".")
import gee_tree_fix; gee_tree_fix.apply()
import geemap.ml as ml, ee
ee.Initialize(project="ee-abdullahr-solar")
BANDS = [f"A{i:02d}" for i in range(64)]
TC = "artifacts/paper_v3/pools_t3/treecount"
OUT = "artifacts/paper_v3/pools_t3/speckle"; os.makedirs(OUT, exist_ok=True)
# ~0.5deg (~55km) box over a solar-rich mixed-terrain region in central Spain (Ciudad Real)
REGION = ee.Geometry.Rectangle([-4.15, 38.70, -3.65, 39.20])
DIMS = int(os.environ.get("DIMS", 1600))                       # ~34m per display pixel
ONLY = os.environ.get("ONLY")                                  # e.g. "30" to render one model
PALETTE = ["000004","1b0c41","4a0c6b","781c6d","a52c60","cf4446","ed6925","fb9b06","f7d13d","fcffa4"]  # magma
img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2019-01-01","2020-01-01").mosaic().select(BANDS))
NES = [int(ONLY)] if ONLY else [15, 30]
for ne in NES:
    try:
        rf = joblib.load(f"{TC}/px_v3_ne{ne}_mf32.joblib")
        clf = ml.strings_to_classifier(ml.rf_to_strings(rf, BANDS, processes=8, output_mode="PROBABILITY"))
        score = img.classify(clf).clip(REGION)
        vis = score.visualize(min=0.35, max=0.80, palette=PALETTE)
        url = vis.getThumbURL({"region": REGION, "dimensions": DIMS, "format": "png"})
        png = requests.get(url, timeout=600).content
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            print(f"ne{ne}: NOT a PNG (server error?): {png[:200]}", flush=True); continue
        with open(f"{OUT}/broad_spain_ne{ne}.png", "wb") as f:
            f.write(png)
        print(f"wrote broad_spain_ne{ne}.png ({len(png)/1e3:.0f} KB, ~55km @ {55000/DIMS:.0f}m/px)", flush=True)
    except Exception as e:
        print(f"ne{ne} FAILED: {repr(e)[:200]}", flush=True)
print("DONE broad render", flush=True)

"""Render a global preview of the 100m suitability master asset (2025 AEF, RF-R)."""
import ee, urllib.request, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
ee.Initialize(project="ee-abdullahr-solar")
WS = os.path.expanduser("~/Solar_Workspace")

a = ee.Image("projects/ee-abdullahr-solar/assets/solar_suitability_2025_R_100m")
img = a.updateMask(a.neq(-1)).divide(10000.0)
mag = ['000004', '3b0f70', '8c2981', 'de4968', 'fe9f6d', 'fcfdbf']
url = img.getThumbURL({'region': ee.Geometry.BBox(-180, -60, 180, 82),
                       'dimensions': 3600, 'format': 'png',
                       'min': 0, 'max': 1, 'palette': mag})
thumb = f"{WS}/data_products/100m/global/global_100m_thumb.png"
os.makedirs(os.path.dirname(thumb), exist_ok=True)
urllib.request.urlretrieve(url, thumb)
print("thumb saved", thumb, os.path.getsize(thumb), "bytes", flush=True)

im = plt.imread(thumb)
fig, ax = plt.subplots(figsize=(17, 7.2))
ax.imshow(im, extent=[-180, 180, -60, 82], aspect='auto')
ax.set_title("Global solar-suitability preview — AEF 2025, RF-R (100 m placeholder)", fontsize=14)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
cmap = LinearSegmentedColormap.from_list("mag", ['#' + c for c in mag])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, fraction=0.022, pad=0.01)
cb.set_label("P(suitable for solar)")
for out in [f"{WS}/data_products/100m/global/global_100m_preview.png",
            f"{WS}/Solar-Siting/artifacts/paper_v3/figures/global_100m_preview.png"]:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out, flush=True)

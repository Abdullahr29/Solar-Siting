"""Find big, clearly-visible utility solar farms (IND/ESP/CHN/USA/CHL) with good RF contrast;
fetch post-install Sentinel-2 -> labelled contact sheet to pick the clearest for Panel A."""
import os, glob, warnings, numpy as np, rasterio, rasterio.warp, joblib
warnings.filterwarnings("ignore")
os.chdir(os.path.expanduser("~/Solar_Workspace"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pystac_client, planetary_computer, odc.stac
CH = "Data/aef_solar_chips/positives/chips"; MK = "Data/aef_solar_chips/positives/masks_v2"
rf = joblib.load("Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib")

# fast pass: frac only, across big-farm countries
files = []
for cc in ("IND", "ESP", "CHN", "USA", "CHL"):
    files += glob.glob(f"{MK}/POS_{cc}_*")
np.random.seed(5); np.random.shuffle(files)
byfrac = []
for f in files[:1200]:
    try:
        with rasterio.open(f) as s: m = s.read(1)
        frac = (m > 0).mean()
        if 0.35 < frac < 0.75: byfrac.append((frac, os.path.basename(f)))
    except Exception: pass
byfrac.sort(reverse=True)
# contrast on the top-frac 24, keep best 9 by contrast
scored = []
for frac, b in byfrac[:24]:
    try:
        with rasterio.open(f"{CH}/{b}") as s: aef = s.read().astype(np.float32)
        X = aef.reshape(64, -1).T; fin = np.isfinite(X).all(1)
        with rasterio.open(f"{MK}/{b}") as s: m = s.read(1)
        p = np.full(len(X), np.nan); p[fin] = rf.predict_proba(X[fin])[:, 1]; p = p.reshape(m.shape)
        scored.append((float(np.nanmean(p[m > 0]) - np.nanmean(p[m == 0])), float(np.nanmean(p[m > 0])), round(float(frac), 2), b))
    except Exception: pass
scored.sort(reverse=True)
top = scored[:9]
print("top:", [(round(c[0], 3), c[1:]) for c in top], flush=True)

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for ax, (con, pm, frac, b) in zip(axs.ravel(), top):
    try:
        with rasterio.open(f"{CH}/{b}") as s: bounds, crs = s.bounds, s.crs
        with rasterio.open(f"{MK}/{b}") as s: m = s.read(1)
        lo_lon, lo_lat, hi_lon, hi_lat = rasterio.warp.transform_bounds(crs, "EPSG:4326", *bounds)
        yr = int(b.split("_")[3]) + 3
        items = cat.search(collections=["sentinel-2-l2a"], bbox=[lo_lon, lo_lat, hi_lon, hi_lat],
                           datetime=f"{yr}-04-01/{yr}-10-31", query={"eo:cloud_cover": {"lt": 15}}).item_collection()
        it = min(items, key=lambda i: i.properties["eo:cloud_cover"])
        da = odc.stac.load([it], bands=["B04", "B03", "B02"], bbox=[lo_lon, lo_lat, hi_lon, hi_lat],
                           crs="EPSG:4326", resolution=0.0001).isel(time=0)
        s2 = np.clip(np.dstack([da.B04.values, da.B03.values, da.B02.values]).astype(np.float32) / 3000, 0, 1)
        ax.imshow(s2); mh, mw = m.shape
        ax.contour(np.linspace(0, s2.shape[1]-1, mw), np.linspace(0, s2.shape[0]-1, mh), m > 0, levels=[0.5], colors="cyan", linewidths=1.0)
    except Exception as e:
        ax.text(0.5, 0.5, f"S2 fail", fontsize=7, ha="center")
    ax.set_title(f"{b}\nfrac {frac} con {con:.2f} pm {pm:.2f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig("Solar-Siting/artifacts/paper_v3/figures/chip_candidates.png", dpi=110)
print("saved", flush=True)

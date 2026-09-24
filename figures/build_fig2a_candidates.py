"""Fig 2(a) candidate generator.

Renders MANY candidate solar sites as a 1x5 tile strip so we can pick the best one
for the paper. Panels:
  1. Sentinel-2 . install-2 (before)      2. Sentinel-2 . install+1 (after)
  3. AlphaEarth . install-2 (PCA-3 RGB)   4. RF P(solar)  -- linear magma 0..1
  5. RF P(solar) -- NONLINEAR norm (supervisor ask: emphasise high-suitability area)
Footprint outlined cyan on every panel.

Selection favours: clean footprint (mask frac in a sane band), strong in-mask vs
background RF contrast, and HIGH spatial variability of the RF score (user ask).
Diversified across countries (<=CAP_PER_CTRY each).

Env knobs:
  NL_MODE = power (default) | log | logit   -- the nonlinear 5th panel
  NL_GAMMA = 2.2                            -- gamma for power mode
  N_PER_CTRY = 26   N_SELECT = 18   CAP_PER_CTRY = 2
Outputs -> artifacts/paper_v3/figures/fig2a_candidates/{<CHIP>.png, _candidates.csv}
"""
import os, io, time, warnings, glob, random
warnings.filterwarnings("ignore")
os.chdir("/gws/ssde/j25b/gbov/abdullah_solar")
import numpy as np, rasterio, rasterio.warp, joblib, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, Normalize, FuncNorm
from PIL import Image
from sklearn.decomposition import PCA

CH   = "Data/aef_solar_chips/positives/chips"
MK   = "Data/aef_solar_chips/positives/masks_v2"
RF   = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
OUT  = "Solar-Siting/artifacts/paper_v3/figures/fig2a_candidates"
os.makedirs(OUT, exist_ok=True)

N_PER_CTRY   = int(os.environ.get("N_PER_CTRY", 26))
N_SELECT     = int(os.environ.get("N_SELECT", 18))
CAP_PER_CTRY = int(os.environ.get("CAP_PER_CTRY", 2))
NL_MODE      = os.environ.get("NL_MODE", "power")
NL_GAMMA     = float(os.environ.get("NL_GAMMA", 2.2))
COUNTRIES    = os.environ.get("COUNTRIES",
    "CHN USA DEU JPN IND KOR POL ESP TUR FRA BRA GRC CHL ITA AUS GBR").split()
SEED = int(os.environ.get("SEED", 0))
random.seed(SEED); np.random.seed(SEED)

rf = joblib.load(RF)
print("RF loaded", flush=True)

def read_chip(chip):
    with rasterio.open(f"{MK}/{chip}") as s: mask = s.read(1)
    with rasterio.open(f"{CH}/{chip}") as s:
        aef = s.read().astype(np.float32); bounds, crs = s.bounds, s.crs
    return aef, mask, bounds, crs

def score_chip(aef, mask):
    C, H, W = aef.shape
    flat = aef.reshape(C, -1).T; fin = np.isfinite(flat).all(1)
    pred = np.full(H * W, np.nan, np.float32)
    pred[fin] = rf.predict_proba(flat[fin])[:, 1]
    pred = pred.reshape(H, W)
    m = mask > 0
    mfrac = m.mean()
    pin = float(np.nanmean(pred[m])) if m.any() else np.nan
    pbg = float(np.nanmean(pred[~m])) if (~m).any() else np.nan
    pstd = float(np.nanstd(pred))
    phi = float(np.nanmean(pred[fin.reshape(H, W)] > 0.6)) if fin.any() else 0.0
    return pred, dict(mfrac=mfrac, pin=pin, pbg=pbg, contrast=pin - pbg, pstd=pstd, phi=phi)

# ---------- 1) score a stratified sample ----------
cand = []
for ctry in COUNTRIES:
    files = sorted(glob.glob(f"{CH}/POS_{ctry}_*.tif"))
    if not files: continue
    pick = random.sample(files, min(N_PER_CTRY, len(files)))
    for fp in pick:
        chip = os.path.basename(fp)
        try:
            aef, mask, bounds, crs = read_chip(chip)
            _, st = score_chip(aef, mask)
        except Exception as e:
            print(f"  skip {chip}: {str(e)[:60]}", flush=True); continue
        if not (float(os.environ.get("MFRAC_LO",0.03)) <= st["mfrac"] <= float(os.environ.get("MFRAC_HI",0.35))): continue
        if st["pin"] < float(os.environ.get("PIN_MIN",0.55)) or st["contrast"] < float(os.environ.get("CONTRAST_MIN",0.18)): continue
        st.update(chip=chip, ctry=ctry)
        cand.append(st)
    print(f"{ctry}: {sum(c['ctry']==ctry for c in cand)} pass", flush=True)

# rank: reward contrast + spatial variability, mild reward for high-suitability mass
for c in cand:
    c["rank"] = 1.0 * c["contrast"] + 1.3 * c["pstd"] + 0.4 * c["phi"]
cand.sort(key=lambda c: -c["rank"])

# diversify by country
sel, per = [], {}
for c in cand:
    if per.get(c["ctry"], 0) >= CAP_PER_CTRY: continue
    sel.append(c); per[c["ctry"]] = per.get(c["ctry"], 0) + 1
    if len(sel) >= N_SELECT: break
print(f"\nselected {len(sel)} candidates from {len(cand)} passing / "
      f"{sum(min(N_PER_CTRY, len(glob.glob(f'{CH}/POS_{c}_*.tif'))) for c in COUNTRIES)} scanned", flush=True)

# write CSV
import csv
with open(f"{OUT}/_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["chip","ctry","mfrac","pin","pbg","contrast","pstd","phi","rank"])
    w.writeheader()
    for c in sel: w.writerow({k: (round(c[k],4) if isinstance(c[k],float) else c[k]) for k in w.fieldnames})
print(f"wrote {OUT}/_candidates.csv", flush=True)

# ---------- 2) S2 thumbnails (GEE, cached) ----------
import ee
ee.Initialize(project="ee-abdullahr-solar")
def s2_thumb(region, yr):
    # per-pixel SCL cloud/shadow/cirrus/snow masking + wide window -> clean composite even over
    # persistently cloudy regions (GBR). Tries L2A/SR first; falls back to L1C TOA for sparse
    # years (S2 SR coverage is thin pre-2019 in some regions, e.g. UK 2017).
    def mask_scl(img):
        scl = img.select("SCL")
        good = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
        return img.updateMask(good)
    def try_col(img):
        for att in range(3):
            try:
                url = img.getThumbURL({"region": region, "dimensions": 320, "min": 0, "max": 3000, "format": "png"})
                a = np.array(Image.open(io.BytesIO(requests.get(url, timeout=180).content))).astype(np.float32)
                a = np.clip(a[:, :, :3] / 255.0, 0, 1)
                return a if np.nanmean(a) > 0.01 else None   # all-black => masked-out => treat as fail
            except Exception:
                time.sleep(6 * (att + 1))
        return None
    win_lo = os.environ.get("S2_LO", f"{yr}-04-01"); win_hi = os.environ.get("S2_HI", f"{yr}-10-31")
    # (1) SR seasonal, SCL-masked
    sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(win_lo, win_hi).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)).map(mask_scl))
    try: sr_n = sr.size().getInfo()
    except Exception: sr_n = 0
    if sr_n == 0:  # (2) SR whole-year
        sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
              .filterDate(f"{yr}-01-01", f"{yr}-12-31").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 92)).map(mask_scl))
        try: sr_n = sr.size().getInfo()
        except Exception: sr_n = 0
    if sr_n > 0:
        a = try_col(sr.median().select(["B4", "B3", "B2"]))
        if a is not None: return a
    # (3) L1C TOA fallback (full 2017+ coverage), least-cloudy whole-year median
    toa = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(region)
           .filterDate(f"{yr}-01-01", f"{yr}-12-31").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    a = try_col(toa.median().select(["B4", "B3", "B2"]))
    if a is not None: print(f"    S2 {yr} via TOA fallback", flush=True); return a
    print(f"    S2 {yr} FAIL (SR+TOA)", flush=True); return None
    # unreachable legacy guard below
    for att in range(4):
        try:
            url = img.getThumbURL({"region": region, "dimensions": 320, "min": 0, "max": 3000, "format": "png"})
            a = np.array(Image.open(io.BytesIO(requests.get(url, timeout=180).content))).astype(np.float32)
            return np.clip(a[:, :, :3] / 255.0, 0, 1)
        except Exception as e:
            if att == 3: print(f"    S2 {yr} FAIL {str(e)[:70]}", flush=True); return None
            time.sleep(8 * (att + 1))

def nl_norm():
    if NL_MODE == "log":   return LogNorm(vmin=0.02, vmax=1.0)
    if NL_MODE == "logit":
        f = lambda p: np.log(np.clip(p,1e-3,1-1e-3)/(1-np.clip(p,1e-3,1-1e-3)))
        g = lambda z: 1/(1+np.exp(-z))
        return FuncNorm((f, g), vmin=0.02, vmax=0.98)
    return PowerNorm(gamma=NL_GAMMA, vmin=0.0, vmax=1.0)

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})

def render(c):
    chip = c["chip"]; emb = int(chip.split("_")[3])
    outpng = f"{OUT}/{chip.replace('.tif','')}.png"
    if os.path.exists(outpng) and os.environ.get("FORCE", "0") != "1":
        print(f"  skip (exists) {chip}", flush=True); return
    aef, mask, bounds, crs = read_chip(chip)
    C, H, W = aef.shape
    flat = aef.reshape(C, -1).T; fin = np.isfinite(flat).all(1)
    rgb = np.full((H*W, 3), np.nan, np.float32)
    rgb[fin] = PCA(3, random_state=0).fit_transform(flat[fin])
    rgb = rgb.reshape(H, W, 3)
    for i in range(3):
        lo, hi = np.nanpercentile(rgb[...,i], [2,98]); rgb[...,i] = np.clip((rgb[...,i]-lo)/(hi-lo+1e-9),0,1)
    aef_rgb = np.nan_to_num(rgb)
    pred = np.full(H*W, np.nan, np.float32); pred[fin] = rf.predict_proba(flat[fin])[:,1]
    pred = pred.reshape(H, W)
    lo_lon, lo_lat, hi_lon, hi_lat = rasterio.warp.transform_bounds(crs, "EPSG:4326", *bounds)
    region = ee.Geometry.Rectangle([float(lo_lon), float(lo_lat), float(hi_lon), float(hi_lat)])
    cache = f"{OUT}/_s2_{chip}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); pre, post = z["pre"], z["post"]
        pre = None if pre.dtype == object else pre; post = None if post.dtype == object else post
    else:
        pre = s2_thumb(region, emb); post = s2_thumb(region, emb+3)
        np.savez(cache, pre=pre if pre is not None else np.array(None),
                 post=post if post is not None else np.array(None))

    fig, axs = plt.subplots(1, 5, figsize=(13.2, 2.95))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.80, bottom=0.02, wspace=0.04)
    def show(ax, img, title, **kw):
        if img is None:
            ax.text(0.5,0.5,"S2 n/a",ha="center",va="center",fontsize=9); ax.set_facecolor("0.9")
        else:
            ax.imshow(img, aspect="auto", **kw)
            mh, mw = mask.shape
            ax.contour(np.linspace(0,(img.shape[1] if img.ndim>1 else W)-1, mw),
                       np.linspace(0,(img.shape[0] if img.ndim>1 else H)-1, mh),
                       mask>0, levels=[0.5], colors="cyan", linewidths=0.9)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    show(axs[0], pre,  f"Sentinel-2 . {emb} (before)")
    show(axs[1], post, f"Sentinel-2 . {emb+3} (after)")
    show(axs[2], aef_rgb, f"AlphaEarth . {emb}")
    show(axs[3], pred, "RF $P$(solar) . linear", cmap="magma", vmin=0, vmax=1)
    nlab = {"power": f"nonlinear (power $\\gamma$={NL_GAMMA:g})", "log": "nonlinear (log)",
            "logit": "nonlinear (logit)"}[NL_MODE]
    show(axs[4], pred, f"RF $P$(solar) . {nlab}", cmap="magma", norm=nl_norm())
    fig.suptitle(f"{chip}   |   {c['ctry']}   mask={c['mfrac']:.2f}   "
                 f"P_in={c['pin']:.2f}  P_bg={c['pbg']:.2f}  contrast={c['contrast']:.2f}  "
                 f"P_std={c['pstd']:.2f}",
                 fontsize=9.5, y=0.985)
    fig.savefig(f"{OUT}/{chip.replace('.tif','')}.png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered {chip}  (contrast {c['contrast']:.2f}, pstd {c['pstd']:.2f}, "
          f"S2 {'ok' if pre is not None else 'NA'})", flush=True)

for c in sel:
    try: render(c)
    except Exception as e: print(f"  render FAIL {c['chip']}: {str(e)[:90]}", flush=True)

print("\nCANDIDATES DONE ->", OUT, flush=True)

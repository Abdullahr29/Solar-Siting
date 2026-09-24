"""Build a t-3 (AEF install-3) version of the cap120 (px_v3) positive pool — minimal GEE.

Design (see memory t3-lag-bigdata-plan):
  * LOCAL: replicate extract_pixels_v3._sol selection per chip (finite masked px, cap 120 via
    rng(seed=i), i = train-split order) -> kept pixels' (row,col) -> lon/lat via geotransform,
    + their t-2 embedding (from the chip).
  * yr_sol == 2017 (install 2019): t-3 = AEF 2016 doesn't exist -> REUSE the t-2 embedding (free).
  * yr_sol >= 2018: GEE-sample AEF(year-1) at those coords (the ONLY GEE work).
  * Negatives: unchanged (reuse px_v3_train.npz X_neg) -> not touched here.
Resumable: per-chip checkpoint .npz under CKPT/. Re-running skips done chips.

    python t3_sample_pool.py [--limit N] [--batch 2000]
Assemble step (after all chips done): --assemble  -> writes cap120_t3_train.npz
"""
import argparse, os, time, glob
import numpy as np, pandas as pd, rasterio
from rasterio.transform import xy as rc_to_xy
from rasterio.warp import transform as warp_transform
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
PCH = "Data/aef_solar_chips/positives/chips"
MASKS = "Data/aef_solar_chips/positives/masks_v2"
SPLITS = "Solar-Siting/artifacts/splits"
OUTDIR = "Solar-Siting/artifacts/paper_v3/pools_t3"
CKPT = f"{OUTDIR}/ckpt_cap120"
BANDS = [f"A{i:02d}" for i in range(64)]
CAP = 120

def kept_pixels(chip, seed):
    """Return (lon, lat, x_t2) for the exact pixels extract_pixels_v3 keeps for this chip."""
    with rasterio.open(os.path.join(PCH, chip)) as s:
        a = s.read(); T = s.transform; crs = s.crs
    with rasterio.open(os.path.join(MASKS, chip)) as s:
        m = s.read(1).astype(bool)
    if not m.any(): return None
    rows, cols = np.where(m)
    x = a[:, m].T.astype(np.float32)
    fin = np.isfinite(x).all(1)
    rows, cols, x = rows[fin], cols[fin], x[fin]
    if len(x) == 0: return None
    if len(x) > CAP:
        pick = np.random.default_rng(seed).choice(len(x), CAP, replace=False)
        rows, cols, x = rows[pick], cols[pick], x[pick]
    xs, ys = rc_to_xy(T, rows, cols, offset="center")
    lon, lat = warp_transform(crs, "EPSG:4326", list(np.atleast_1d(xs)), list(np.atleast_1d(ys)))
    return np.array(lon), np.array(lat), x

def ee_sample(lon, lat, year):
    """GEE reduceRegions of AEF(year-1) at points -> (N,64) float array (NaN where missing)."""
    import ee
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{year-1}-01-01", f"{year}-01-01").mosaic().select(BANDS))
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a), float(b)]))
                               for a, b in zip(lon, lat)])
    res = img.reduceRegions(fc, __import__("ee").Reducer.first(), scale=10).getInfo()
    return np.array([[f["properties"].get(b, np.nan) for b in BANDS]
                     for f in res["features"]], np.float32)

def build(limit=None, batch=2000, procs_note=""):
    import ee
    for attempt in range(3):
        try: ee.Initialize(project="ee-abdullahr-solar"); break
        except Exception as e:
            print(f"ee.Initialize retry {attempt}: {e}", flush=True); time.sleep(10)
    os.makedirs(CKPT, exist_ok=True)
    pos = pd.read_csv(f"{SPLITS}/emb_search_split_v2.csv")
    pos = pos[pos.split == "train"].reset_index(drop=True)      # enumerate order == extractor seed i
    if limit: pos = pos.head(limit)
    codes = {c: i for i, c in enumerate(sorted(pos.iso3.dropna().unique()))}
    t0 = time.time(); done = 0; gee_pts = 0; deferred = 0; corrupt = 0
    EMPTY = dict(X=np.empty((0,64),np.float32), iso=np.empty(0,np.int32), yr=np.empty(0,np.int16))
    for i, (chip, iso, yr) in enumerate(zip(pos.chip, pos.iso3, pos.year)):
        cpath = f"{CKPT}/{i:06d}.npz"
        if os.path.exists(cpath):
            done += 1; continue
        yr = int(yr)
        try:                                                   # corrupt/unreadable chip -> skip PERMANENTLY
            r = kept_pixels(chip, i)
        except Exception as e:
            print(f"  chip{i} {chip} LOCAL-FAIL (empty ckpt): {str(e)[:80]}", flush=True)
            np.savez(cpath, **EMPTY); corrupt += 1; done += 1; continue
        if r is None:
            np.savez(cpath, **EMPTY); done += 1; continue
        lon, lat, x_t2 = r
        if yr <= 2017:                                          # floor: reuse t-2 (AEF 2017), no GEE
            x_t3 = x_t2
        else:
            x_t3 = np.full_like(x_t2, np.nan); failed = False
            for j in range(0, len(lon), batch):
                sl = slice(j, j+batch); ok = False
                for att in range(4):
                    try:
                        x_t3[sl] = ee_sample(lon[sl], lat[sl], yr); ok = True; break
                    except Exception as e:
                        print(f"  chip{i} batch{j} GEE retry {att}: {str(e)[:70]}", flush=True); time.sleep(20*(att+1))
                if not ok:
                    failed = True; break                       # GEE down -> DEFER whole chip (no ckpt)
                gee_pts += min(batch, len(lon)-j)
            if failed:
                deferred += 1
                if deferred % 25 == 1:
                    print(f"  chip{i} DEFERRED (GEE); will retry next resume link", flush=True)
                continue                                        # no checkpoint -> retried later
        np.savez(cpath, X=x_t3.astype(np.float32),
                 iso=np.full(len(x_t3), codes.get(iso, -1), np.int32),
                 yr=np.full(len(x_t3), yr, np.int16))
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(pos)} chips | gee_pts={gee_pts:,} | deferred={deferred} corrupt={corrupt} | {time.time()-t0:.0f}s", flush=True)
    print(f"DONE build: {done}/{len(pos)} chips, gee_pts~{gee_pts:,}, deferred={deferred}, corrupt={corrupt} ({time.time()-t0:.0f}s)", flush=True)

def assemble():
    files = sorted(glob.glob(f"{CKPT}/*.npz"))
    Xs, iso, yr = [], [], []
    for f in files:
        z = np.load(f)
        if len(z["X"]): Xs.append(z["X"]); iso.append(z["iso"]); yr.append(z["yr"])
    X_sol = np.concatenate(Xs); iso_sol = np.concatenate(iso); yr_sol = np.concatenate(yr)
    fin = np.isfinite(X_sol).all(1)
    print(f"assembled X_sol {X_sol.shape}, finite {fin.sum():,}/{len(X_sol):,}", flush=True)
    # reuse px_v3 negatives unchanged
    pv = np.load("Solar-Siting/artifacts/paper_v3/pools/px_v3_train.npz")
    out = f"{OUTDIR}/cap120_t3_train.npz"
    np.savez(out, X_sol=X_sol[fin], iso_sol=iso_sol[fin], yr_sol=yr_sol[fin],
             X_neg=pv["X_neg"], meta=np.array(["cap120 t-3; 2019-cohort reused at 2017 floor; neg=px_v3 unchanged"]))
    print(f"wrote {out}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    if a.assemble: assemble()
    else: build(a.limit, a.batch)

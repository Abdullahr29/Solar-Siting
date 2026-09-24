"""Sample AEF(2019) 64-d embeddings ONCE at every country's validation points (site + rand),
so t-2 and t-3 models can be scored LOCALLY (local==GEE confirmed exact). Resumable per country.
Validation imagery stays AEF 2019 -> forward installs (unchanged); only training lag differs."""
import os, glob, time, numpy as np
WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
import ee
for a in range(3):
    try: ee.Initialize(project="ee-abdullahr-solar"); break
    except Exception as e: print("init retry", e); time.sleep(10)
BANDS = [f"A{i:02d}" for i in range(64)]
SC = "Solar-Siting/artifacts/paper_v3/budget_sweep/scores/country"
OUT = "Solar-Siting/artifacts/paper_v3/pools_t3/val_emb"; os.makedirs(OUT, exist_ok=True)
img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
       .filterDate("2019-01-01", "2020-01-01").mosaic().select(BANDS))

def sample(xy, batch=1000):
    out = np.full((len(xy), 64), np.nan, np.float32)
    for j in range(0, len(xy), batch):
        sl = slice(j, j+batch)
        for att in range(4):
            try:
                fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(p[0]), float(p[1])]))
                                           for p in xy[sl]])
                res = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()
                out[sl] = [[f["properties"].get(b, np.nan) for b in BANDS] for f in res["features"]]
                break
            except Exception as e:
                print(f"    batch{j} retry{att}: {str(e)[:70]}", flush=True); time.sleep(20*(att+1))
    return out

# one npz per country holds site_xy/rand_xy (same across models) -> take px_v3_incumbent's
countries = sorted({os.path.basename(f).split("__")[0]
                    for f in glob.glob(f"{SC}/*__px_v3_incumbent_2019_scores.npz")})
print(f"countries: {len(countries)}", flush=True)
for c in countries:
    op = f"{OUT}/{c}_val2019.npz"
    if os.path.exists(op):
        print(f"[skip] {c}"); continue
    z = np.load(f"{SC}/{c}__px_v3_incumbent_2019_scores.npz")
    sx, rx = z["site_xy"], z["rand_xy"]
    t = time.time()
    Xs = sample(sx); Xr = sample(rx)
    np.savez(op, site_emb=Xs, rand_emb=Xr, site_xy=sx, rand_xy=rx)
    print(f"  {c}: sites {len(sx)} rand {len(rx)}  finite s/r "
          f"{np.isfinite(Xs).all(1).sum()}/{np.isfinite(Xr).all(1).sum()}  ({time.time()-t:.0f}s)", flush=True)
print("DONE val embeddings", flush=True)

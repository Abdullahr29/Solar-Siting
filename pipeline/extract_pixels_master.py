"""Master pixel-pool extraction for the training-data investigation (IMPROVEMENTS.md Track 1, S1).

Extract positives UNCAPPED and a large negative pool ONCE, tagged per-row with iso3 + AEF year
+ SOURCE-CHIP INDEX, so every downstream sampler — the per-country budget sweep with spatial
(site + lat/lon cell) diversity, and any ratio choice — is a pure in-memory operation on this
one pool. No re-extraction per ablation, no per-chip cap baked in (the sampler owns all balancing).

Per-pixel arrays: X (64d), iso (country code), yr (AEF year), src (index into the chip list).
The chip lists (pos_chips / neg_files) are saved alongside; join src -> chip name -> chip_geo_v3
(cluster + centroid lat/lon) to bin spatially. Buffer sizing for positives comes from the census
(sum(min(n_masked, cap))) so an uncapped extraction does not explode the len*cap pre-allocation.

Evaluate ablations on the EXISTING px_v3_test.npz (cap120) so pixel-ROC stays comparable to the
paper; only the TRAIN pool is re-extracted here.

    python Solar-Siting/extract_pixels_master.py --out <path> --cap-sol 0 --per-neg 8000 --split train
      (--cap-sol 0 = uncapped positives)
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, rasterio

WORKSPACE = os.path.expanduser("~/Solar_Workspace")
PCH = "Data/aef_solar_chips/positives/chips"
NCH = "Data/aef_solar_chips/negatives/chips"
SPLITS = "Solar-Siting/artifacts/splits"
CENSUS = "Solar-Siting/artifacts/stats/pixel_census_v3.csv"
MASKS_DIR = None


def _sol(args):
    chip, cap, idx, code, yr = args
    try:
        with rasterio.open(os.path.join(MASKS_DIR, chip)) as s:
            m = s.read(1).astype(bool)
        if not m.any():
            return None
        with rasterio.open(os.path.join(PCH, chip)) as s:
            a = s.read()
        x = a[:, m].T.astype(np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            return None
        if cap and len(x) > cap:
            x = x[np.random.default_rng(idx).choice(len(x), cap, replace=False)]
        return x, code, yr, idx
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return None


def _neg(args):
    chip, per, idx, code, yr = args
    try:
        with rasterio.open(os.path.join(NCH, chip)) as s:
            a = s.read()
        x = a.reshape(a.shape[0], -1).T.astype(np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            return None
        if per and len(x) > per:
            x = x[np.random.default_rng(idx).choice(len(x), per, replace=False)]
        return x, code, yr, idx
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return None


def collect(fn, items, cap_rows, label, procs):
    """cap_rows = exact/upper-bound total px to pre-size the buffers."""
    buf = np.empty((cap_rows, 64), np.float32)
    iso = np.empty(cap_rows, np.int32)
    yr = np.empty(cap_rows, np.int16)
    src = np.empty(cap_rows, np.int32)
    n = 0
    t = time.time()
    with Pool(procs) as p:
        for i, r in enumerate(p.imap_unordered(fn, items, chunksize=32)):
            if r is not None:
                x, c, y, s = r
                buf[n:n + len(x)] = x
                iso[n:n + len(x)] = c
                yr[n:n + len(x)] = y
                src[n:n + len(x)] = s
                n += len(x)
            if (i + 1) % 20000 == 0:
                print(f"  {label} {i+1:,}/{len(items):,} -> {n:,} px ({time.time()-t:.0f}s)", flush=True)
    print(f"  {label}: {n:,} px from {len(items):,} chips ({time.time()-t:.0f}s)", flush=True)
    return buf[:n], iso[:n], yr[:n], src[:n]


def extract(out, masks_subdir="masks_v2", neg_split="emb_neg_split_v3.csv",
            cap_sol=0, per_neg=8000, procs=32, split="train", limit=None):
    global MASKS_DIR
    os.chdir(WORKSPACE)
    MASKS_DIR = os.path.join("Data/aef_solar_chips/positives", masks_subdir)
    assert os.path.isdir(MASKS_DIR), f"no masks dir {MASKS_DIR}"

    pos = pd.read_csv(f"{SPLITS}/emb_search_split_v2.csv")
    neg = pd.read_csv(f"{SPLITS}/{neg_split}")
    if split != "all":
        pos, neg = pos[pos.split == split], neg[neg.split == split]
    if limit:
        pos, neg = pos.head(limit), neg.head(limit)
    pos = pos.reset_index(drop=True)      # src index := row position in these lists
    neg = neg.reset_index(drop=True)

    cen = pd.read_csv(CENSUS).set_index("chip")["n_masked"]
    nm = pos.chip.map(cen).fillna(0).to_numpy()
    kept = np.minimum(nm, cap_sol) if cap_sol else nm
    pos_rows = int(kept.sum()) + 64
    neg_rows = len(neg) * per_neg
    print(f"pool: {len(pos):,} pos chips (buffer {pos_rows:,} px, {pos_rows*260/1e9:.1f} GB) | "
          f"{len(neg):,} neg chips (buffer {neg_rows:,} px, {neg_rows*260/1e9:.1f} GB) | "
          f"cap_sol={cap_sol or 'UNCAPPED'} per_neg={per_neg} | split={split}", flush=True)
    if len(pos) == 0 or len(neg) == 0:
        sys.exit("ABORT: empty pool")

    names = sorted(set(pos.iso3.dropna()) | set(neg.iso3.dropna()))
    code = {c: i for i, c in enumerate(names)}
    t0 = time.time()
    sol_items = [(c, cap_sol, i, code[iso], int(y))
                 for i, (c, iso, y) in enumerate(zip(pos.chip, pos.iso3, pos.year))]
    neg_items = [(f, per_neg, i, code[iso], int(y))
                 for i, (f, iso, y) in enumerate(zip(neg.file, neg.iso3, neg.year))]
    X_sol, iso_sol, yr_sol, src_sol = collect(_sol, sol_items, pos_rows, "solar", procs)
    X_neg, iso_neg, yr_neg, src_neg = collect(_neg, neg_items, neg_rows, "negative", procs)

    print(f"\nX_sol {X_sol.shape} | X_neg {X_neg.shape} | positive share "
          f"{len(X_sol)/(len(X_sol)+len(X_neg)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X_sol=X_sol, X_neg=X_neg, iso_sol=iso_sol, iso_neg=iso_neg,
             yr_sol=yr_sol, yr_neg=yr_neg, src_sol=src_sol, src_neg=src_neg,
             pos_chips=np.asarray(pos.chip, dtype="U64"),
             neg_files=np.asarray(neg.file, dtype="U64"),
             iso_names=np.array(names),
             meta=np.array([f"masks={masks_subdir}", f"neg_split={neg_split}",
                            f"cap_sol={cap_sol}", f"per_neg={per_neg}", f"split={split}",
                            f"n_pos_chips={len(pos)}", f"n_neg_chips={len(neg)}",
                            "MASTER-UNCAPPED-src-tagged"]))
    print(f"wrote {out} ({os.path.getsize(out)/1e9:.1f} GB)", flush=True)
    return X_sol.shape, X_neg.shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks-subdir", default="masks_v2")
    ap.add_argument("--neg-split", default="emb_neg_split_v3.csv")
    ap.add_argument("--cap-sol", type=int, default=0, help="0 = uncapped (default for the master pool)")
    ap.add_argument("--per-neg", type=int, default=8000)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N chips each side")
    a = ap.parse_args()
    extract(a.out, a.masks_subdir, a.neg_split, a.cap_sol, a.per_neg, a.procs, a.split, a.limit)

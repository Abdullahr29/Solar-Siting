"""Extract the per-pixel training pool from AEF chips — v3 (masks_v2 + current negatives).

Derived from extract_pixels.py. Differences, all so the reference and every ablation differ
by EXACTLY one filter:
  * positives read from masks_v2/ (corrected binary rebuild: px=1 iff install_year>embed_year;
    +37% positive px, cap-dropped future recovered, existing solar excluded).
  * negatives from emb_neg_split_v3.csv (the current 13,440 on disk, post-remine + decontam).
  * emits per-row iso3 CODES + AEF YEAR (iso_sol/iso_neg/yr_sol/yr_neg + iso_names) so
    train_rf_v3.py can de-confound by country (Model D) and run LOCO / temporal holdout as
    in-memory filters on ONE extracted pool (no re-extraction per ablation).
The static ablation flags (--exclude-iso3, --max-year) still work here too; the in-memory
path in train_rf_v3 is the fast route used by the orchestrator.
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, rasterio

WORKSPACE = os.path.expanduser("~/Solar_Workspace")
PCH = "Data/aef_solar_chips/positives/chips"
NCH = "Data/aef_solar_chips/negatives/chips"
SPLITS = "Solar-Siting/artifacts/splits"
MASKS_DIR = None   # set in extract(); fork-inherited by workers


def _sol(args):
    chip, cap, seed, code, yr = args
    try:
        with rasterio.open(os.path.join(PCH, chip)) as s:
            a = s.read()
        with rasterio.open(os.path.join(MASKS_DIR, chip)) as s:
            m = s.read(1).astype(bool)
        if not m.any():
            return None
        x = a[:, m].T.astype(np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            return None
        if len(x) > cap:
            x = x[np.random.default_rng(seed).choice(len(x), cap, replace=False)]
        return x, code, yr
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return None


def _neg(args):
    chip, per, seed, code, yr = args
    try:
        with rasterio.open(os.path.join(NCH, chip)) as s:
            a = s.read()
        x = a.reshape(a.shape[0], -1).T.astype(np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            return None
        if len(x) > per:
            x = x[np.random.default_rng(seed).choice(len(x), per, replace=False)]
        return x, code, yr
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return None


def extract(out, masks_subdir="masks_v2", neg_split="emb_neg_split_v3.csv",
            exclude_iso3=None, max_year=None, cap_sol=120, per_neg=2800,
            procs=32, split="train", limit=None):
    global MASKS_DIR
    os.chdir(WORKSPACE)
    MASKS_DIR = os.path.join("Data/aef_solar_chips/positives", masks_subdir)
    assert os.path.isdir(MASKS_DIR), f"no masks dir {MASKS_DIR}"

    pos = pd.read_csv(f"{SPLITS}/emb_search_split_v2.csv")   # chip list unchanged; masks_v2 read at runtime
    neg = pd.read_csv(f"{SPLITS}/{neg_split}")
    if split != "all":                                        # "all" = train+test, for the released asset
        pos, neg = pos[pos.split == split], neg[neg.split == split]
    n0 = (len(pos), len(neg))

    if exclude_iso3:
        pos = pos[pos.iso3 != exclude_iso3]
        neg = neg[neg.iso3 != exclude_iso3]
    if max_year is not None:
        pos = pos[pos.year <= max_year]
        neg = neg[neg.year <= max_year]
    if limit:                                   # smoke-test only
        pos, neg = pos.head(limit), neg.head(limit)

    print(f"pool: {len(pos):,}/{n0[0]:,} positive chips | {len(neg):,}/{n0[1]:,} negative chips "
          f"| masks={MASKS_DIR} | neg_split={neg_split} | split={split}", flush=True)
    if len(pos) == 0 or len(neg) == 0:
        sys.exit("ABORT: filter left an empty pool")

    names = sorted(set(pos.iso3.dropna()) | set(neg.iso3.dropna()))
    code = {c: i for i, c in enumerate(names)}

    def collect(fn, items, per_item, label):
        buf = np.empty((len(items) * per_item, 64), np.float32)
        iso = np.empty(len(items) * per_item, np.int32)
        yr = np.empty(len(items) * per_item, np.int16)
        n = 0
        t = time.time()
        with Pool(procs) as p:
            for i, r in enumerate(p.imap_unordered(fn, items, chunksize=32)):
                if r is not None:
                    x, c, y = r
                    buf[n:n + len(x)] = x
                    iso[n:n + len(x)] = c
                    yr[n:n + len(x)] = y
                    n += len(x)
                if (i + 1) % 20000 == 0:
                    print(f"  {label} {i+1:,}/{len(items):,} -> {n:,} px ({time.time()-t:.0f}s)", flush=True)
        print(f"  {label}: {n:,} px from {len(items):,} chips ({time.time()-t:.0f}s)", flush=True)
        return buf[:n], iso[:n], yr[:n]

    t0 = time.time()
    sol_items = [(c, cap_sol, i, code[iso], int(y))
                 for i, (c, iso, y) in enumerate(zip(pos.chip, pos.iso3, pos.year))]
    neg_items = [(f, per_neg, i, code[iso], int(y))
                 for i, (f, iso, y) in enumerate(zip(neg.file, neg.iso3, neg.year))]
    X_sol, iso_sol, yr_sol = collect(_sol, sol_items, cap_sol, "solar")
    X_neg, iso_neg, yr_neg = collect(_neg, neg_items, per_neg, "negative")

    print(f"\nX_sol {X_sol.shape} | X_neg {X_neg.shape} | positive share "
          f"{len(X_sol)/(len(X_sol)+len(X_neg)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X_sol=X_sol, X_neg=X_neg, iso_sol=iso_sol, iso_neg=iso_neg,
             yr_sol=yr_sol, yr_neg=yr_neg, iso_names=np.array(names),
             meta=np.array([f"masks={masks_subdir}", f"neg_split={neg_split}",
                            f"exclude_iso3={exclude_iso3}", f"max_year={max_year}",
                            f"cap_sol={cap_sol}", f"per_neg={per_neg}", f"split={split}",
                            f"n_pos_chips={len(pos)}", f"n_neg_chips={len(neg)}"]))
    print(f"wrote {out}", flush=True)
    return X_sol.shape, X_neg.shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks-subdir", default="masks_v2")
    ap.add_argument("--neg-split", default="emb_neg_split_v3.csv")
    ap.add_argument("--exclude-iso3", default=None)
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument("--cap-sol", type=int, default=120)
    ap.add_argument("--per-neg", type=int, default=2800)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N chips each side")
    a = ap.parse_args()
    extract(a.out, a.masks_subdir, a.neg_split, a.exclude_iso3, a.max_year,
            a.cap_sol, a.per_neg, a.procs, a.split, a.limit)

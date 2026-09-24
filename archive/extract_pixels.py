"""Extract the per-pixel training pool from AEF chips, with ablation filters.

This is the single entry point for building an RF/MLP training pool, so that the
reference model and every ablation model differ by EXACTLY the filter applied here
and nothing else.

WHAT A "PIXEL" IS
-----------------
Each AEF chip is 256x256x64: a 64-dimensional AlphaEarth embedding per 10 m pixel.
  * Positive rows = pixels inside the solar mask of a positive chip (land that DID
    become solar, imaged 2 years before it was built).
  * Negative rows = pixels from negative chips (land >=5 km from any known solar site).
Unmasked pixels of positive chips are deliberately NOT used as negatives — they are
meaningful context, not confirmed non-solar.

WHY THE CAPS
------------
Solar masks vary enormously in size (a few px to thousands). Without a cap, a handful of
huge sites would dominate the pool and the model would learn those sites rather than
"solar-suitable land". `--cap-sol` takes at most N masked pixels per chip, so every site
contributes comparably — this is the "coverage" sampling that moved RF px ROC 0.891 ->
0.901 by trading depth-per-site for breadth-across-sites.

ABLATION FILTERS (the whole point)
----------------------------------
  --exclude-iso3 GRC     leave-one-country-out: drop every Greek chip from the pool, then
                         score Greece with run_country_rf.py. The gap vs the reference
                         model = that country's leakage + memorisation premium.
  --max-year 2021        temporal holdout: keep only imagery up to and including this AEF
                         year. NOTE this is the AEF YEAR, which is install_year - 2, so
                         --max-year 2021 == "trained only on sites installed <= 2023" and
                         can be validated honestly against 2024 installations.

USAGE
    python Solar-Siting/extract_pixels.py --out artifacts/stats/px_v2_reference.npz
    python Solar-Siting/extract_pixels.py --exclude-iso3 GRC --out .../px_v2_no_grc.npz
    python Solar-Siting/extract_pixels.py --max-year 2021 --out .../px_v2_to2021.npz

Writes an npz with X_sol, X_neg (float32, N x 64) plus the provenance of the run.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rasterio

WORKSPACE = os.path.expanduser("~/Solar_Workspace")
PCH = "Data/aef_solar_chips/positives/chips"
PMK = "Data/aef_solar_chips/positives/masks"
NCH = "Data/aef_solar_chips/negatives/chips"
SPLITS = "Solar-Siting/artifacts/splits"


def _sol(args):
    """Masked solar pixels from one positive chip, capped and NaN-filtered."""
    chip, cap, seed = args
    try:
        with rasterio.open(os.path.join(PCH, chip)) as s:
            a = s.read()                                  # (64, 256, 256)
        with rasterio.open(os.path.join(PMK, chip)) as s:
            m = s.read(1).astype(bool)                    # (256, 256)
        if not m.any():
            return None
        x = a[:, m].T.astype(np.float32)                  # (n_masked, 64)
        x = x[np.isfinite(x).all(axis=1)]                 # AEF has NaN nodata over water
        if len(x) == 0:
            return None
        if len(x) > cap:
            x = x[np.random.default_rng(seed).choice(len(x), cap, replace=False)]
        return x
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return None


def _neg(args):
    """A random sample of pixels from one negative chip, NaN-filtered."""
    chip, per, seed = args
    try:
        with rasterio.open(os.path.join(NCH, chip)) as s:
            a = s.read()
        x = a.reshape(a.shape[0], -1).T.astype(np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            return None                                   # a few neg chips are entirely NaN
        if len(x) > per:
            x = x[np.random.default_rng(seed).choice(len(x), per, replace=False)]
        return x
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return None


def extract(out, exclude_iso3=None, max_year=None, cap_sol=120, per_neg=2800,
            procs=32, split="train"):
    os.chdir(WORKSPACE)
    pos = pd.read_csv(f"{SPLITS}/emb_search_split_v2.csv")
    neg = pd.read_csv(f"{SPLITS}/emb_neg_split_v2.csv")
    pos, neg = pos[pos.split == split], neg[neg.split == split]
    n0 = (len(pos), len(neg))

    if exclude_iso3:
        pos = pos[pos.iso3 != exclude_iso3]
        neg = neg[neg.iso3 != exclude_iso3]
    if max_year is not None:
        pos = pos[pos.year <= max_year]
        neg = neg[neg.year <= max_year]

    print(f"pool: {len(pos):,}/{n0[0]:,} positive chips | {len(neg):,}/{n0[1]:,} negative chips")
    if exclude_iso3:
        print(f"  excluded iso3={exclude_iso3}: dropped {n0[0]-len(pos):,} pos, "
              f"{n0[1]-len(neg):,} neg chips")
    if max_year is not None:
        print(f"  max AEF year {max_year} (= installs <= {max_year+2}): dropped "
              f"{n0[0]-len(pos):,} pos, {n0[1]-len(neg):,} neg chips")
    if len(pos) == 0 or len(neg) == 0:
        sys.exit("ABORT: filter left an empty pool")

    # The kernel cgroup caps us at ~25.8 GB, so we never materialise the pool twice.
    # `np.concatenate(list_of_chunks)` would hold the list AND the result at once (~15 GB
    # peak for the negatives alone) — that is what OOM-killed earlier runs. Instead
    # preallocate the worst-case array and fill it in place, then truncate.
    def collect(fn, items, per_item, label):
        buf = np.empty((len(items) * per_item, 64), np.float32)
        n = 0
        t = time.time()
        with Pool(procs) as p:
            for i, r in enumerate(p.imap_unordered(fn, items, chunksize=32)):
                if r is not None:
                    buf[n:n + len(r)] = r
                    n += len(r)
                if (i + 1) % 20000 == 0:
                    print(f"  {label} {i+1:,}/{len(items):,} chips -> {n:,} px "
                          f"({time.time()-t:.0f}s)", flush=True)
        print(f"  {label}: {n:,} px from {len(items):,} chips ({time.time()-t:.0f}s)", flush=True)
        # Return a VIEW, not a .copy(): copying would briefly hold both the buffer and the
        # copy (~12 GB for the negatives) and blow the cgroup. n is close to the worst case
        # anyway, so the slack the view keeps alive is small. np.savez writes only the view.
        return buf[:n]

    t0 = time.time()
    X_sol = collect(_sol, [(c, cap_sol, i) for i, c in enumerate(pos.chip)], cap_sol, "solar")
    X_neg = collect(_neg, [(c, per_neg, i) for i, c in enumerate(neg.file)], per_neg, "negative")

    print(f"\nX_sol {X_sol.shape} | X_neg {X_neg.shape} | "
          f"positive share {len(X_sol)/(len(X_sol)+len(X_neg)):.3f} "
          f"({time.time()-t0:.0f}s)")
    print(f"total {(X_sol.nbytes + X_neg.nbytes)/1e9:.2f} GB")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X_sol=X_sol, X_neg=X_neg,
             meta=np.array([f"exclude_iso3={exclude_iso3}", f"max_year={max_year}",
                            f"cap_sol={cap_sol}", f"per_neg={per_neg}",
                            f"split={split}", f"n_pos_chips={len(pos)}",
                            f"n_neg_chips={len(neg)}"]))
    print(f"wrote {out}")
    return X_sol.shape, X_neg.shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-iso3", default=None, help="LOCO: drop this country (e.g. GRC)")
    ap.add_argument("--max-year", type=int, default=None,
                    help="temporal holdout: keep AEF year <= this (= installs <= year+2)")
    ap.add_argument("--cap-sol", type=int, default=120, help="max masked px per positive chip")
    ap.add_argument("--per-neg", type=int, default=2800, help="px sampled per negative chip")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    a = ap.parse_args()
    extract(a.out, a.exclude_iso3, a.max_year, a.cap_sol, a.per_neg, a.procs, a.split)

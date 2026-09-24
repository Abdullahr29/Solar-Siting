"""S0 census for the training-data investigation (IMPROVEMENTS.md, Track 1).

Counts the UNCAPPED masked-solar pixels available per positive chip (the ceiling that
`extract_pixels_v3.py`'s cap_sol=120 currently truncates), then reports the distribution
that decides the rest of the investigation:
  * total available positive px (all / train / test) and the memory a master extraction needs,
  * per-country positive mass + Lorenz/Gini (the H2 regional-skew picture),
  * cap-binding table: at each candidate cap, what fraction of chips is bound and what fraction
    of the total positive px is kept vs discarded — this picks the master-cap + the P levels,
  * a per-country cap percentile suggestion for the B (anti-skew) axis,
  * negative-px headroom (analytic; negatives are unmasked single-centre crops).

Mask-only read (1 band, 256x256) so it is a fast pass over ~82 k chips. The isfinite filter
that extraction applies is a tiny edge/nodata correction; use --finite-sample N to estimate it
on N random chips (reads the full 64-band chip for those only) and print the correction factor.

Run on LOTUS (data is local there; the mount is too slow for 82 k reads):
    sbatch wrapper, or interactively on a --qos=high node:
    python Solar-Siting/pixel_census_v3.py --procs 32
Writes artifacts/stats/pixel_census_v3.csv (per-chip) + prints the summary to stdout/log.
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, rasterio

WORKSPACE = os.path.expanduser("~/Solar_Workspace")
PCH = "Data/aef_solar_chips/positives/chips"
NCH = "Data/aef_solar_chips/negatives/chips"
SPLITS = "Solar-Siting/artifacts/splits"
MASKS_DIR = None            # set in main(); fork-inherited by workers
CHIP_SIDE = 256
CAPS = [120, 300, 600, 1200, 2000, 4000]     # candidate per-chip caps to profile


def _count(chip):
    """(n_masked_px, ok) for one chip — mask-only read."""
    try:
        with rasterio.open(os.path.join(MASKS_DIR, chip)) as s:
            m = s.read(1)
        return int((m != 0).sum())
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return -1


def _finite_count(chip):
    """(n_masked, n_masked_finite) reading the full 64-band chip — for the correction sample."""
    try:
        with rasterio.open(os.path.join(MASKS_DIR, chip)) as s:
            m = s.read(1).astype(bool)
        if not m.any():
            return (0, 0)
        with rasterio.open(os.path.join(PCH, chip)) as s:
            a = s.read()
        x = a[:, m].T.astype(np.float32)
        return (int(m.sum()), int(np.isfinite(x).all(axis=1).sum()))
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return (-1, -1)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main(a):
    global MASKS_DIR
    os.chdir(WORKSPACE)
    MASKS_DIR = os.path.join("Data/aef_solar_chips/positives", a.masks_subdir)
    assert os.path.isdir(MASKS_DIR), f"no masks dir {MASKS_DIR}"

    pos = pd.read_csv(f"{SPLITS}/emb_search_split_v2.csv")
    if a.limit:
        pos = pos.head(a.limit)
    print(f"census: {len(pos):,} positive chips | masks={MASKS_DIR}", flush=True)

    t = time.time()
    with Pool(a.procs) as p:
        counts = list(p.imap(_count, pos.chip, chunksize=64))
    pos = pos.assign(n_masked=counts)
    bad = int((pos.n_masked < 0).sum())
    pos = pos[pos.n_masked >= 0].copy()
    print(f"read {len(pos):,} chips ({bad} unreadable) in {time.time()-t:.0f}s", flush=True)

    out = f"Solar-Siting/artifacts/stats/pixel_census_v3.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pos[["chip", "iso3", "year", "split", "n_masked"]].to_csv(out, index=False)
    print(f"wrote {out}\n", flush=True)

    tot = int(pos.n_masked.sum())
    nz = pos[pos.n_masked > 0]
    print("=" * 78)
    print("POSITIVE PIXEL CENSUS")
    print("=" * 78)
    print(f"chips: {len(pos):,} total | {len(nz):,} with >=1 masked px | "
          f"{(pos.n_masked == 0).sum():,} empty")
    print(f"total masked solar px: {tot:,}  (current cap120 pool holds 6.19 M train px)")
    print(f"per-chip masked px  median {nz.n_masked.median():.0f} | mean {nz.n_masked.mean():.0f} | "
          f"p90 {nz.n_masked.quantile(.9):.0f} | p99 {nz.n_masked.quantile(.99):.0f} | "
          f"max {nz.n_masked.max():,}")
    for sp in ("train", "test"):
        s = pos[pos.split == sp]
        print(f"  {sp:5s}: {len(s):,} chips | {int(s.n_masked.sum()):,} px")

    print("\n-- cap-binding: keep = min(n, cap) summed --")
    print(f"  {'cap':>6} {'chips_bound':>12} {'px_kept':>16} {'kept%':>7} {'GB(f32x64)':>11}")
    for c in CAPS:
        kept = int(np.minimum(pos.n_masked.values, c).sum())
        bound = int((pos.n_masked > c).sum())
        gb = kept * 64 * 4 / 1e9
        print(f"  {c:>6} {bound:>12,} {kept:>16,} {100*kept/tot:>6.1f}% {gb:>10.1f}")
    print(f"  {'uncapped':>6} {0:>12} {tot:>16,} {100.0:>6.1f}% {tot*64*4/1e9:>10.1f}")

    print("\n-- per-country positive mass (top 20 by px) --")
    g = (pos.groupby("iso3").n_masked.agg(["sum", "count"])
            .sort_values("sum", ascending=False))
    g["share%"] = 100 * g["sum"] / tot
    g["cum%"] = g["share%"].cumsum()
    print(g.head(20).to_string(float_format=lambda v: f"{v:,.1f}"))
    print(f"\n  countries: {len(g)} | Gini of per-country px mass: {gini(g['sum'].values):.3f}")
    print(f"  top-5 countries hold {g['share%'].head(5).sum():.1f}% of positive px "
          f"(→ B axis needed if concentrated)")
    # suggested per-country cap for B: the mass a median country contributes
    print(f"  median per-country px: {g['sum'].median():,.0f} "
          f"(candidate per-country cap for B=cap-per-country)")

    if a.finite_sample:
        idx = np.random.default_rng(0).choice(len(nz), min(a.finite_sample, len(nz)), replace=False)
        with Pool(a.procs) as p:
            fc = list(p.imap(_finite_count, nz.chip.values[idx], chunksize=16))
        fc = np.array([r for r in fc if r[0] > 0])
        frac = fc[:, 1].sum() / fc[:, 0].sum()
        print(f"\n-- finite correction (n={len(fc)} sampled chips): "
              f"{100*frac:.2f}% of masked px are finite (extraction keeps these) --")

    nneg = len(pd.read_csv(f"{SPLITS}/emb_neg_split_v3.csv"))
    print(f"\n-- negative headroom -- {nneg:,} neg chips x ~{CHIP_SIDE**2:,} px/chip "
          f"= ~{nneg*CHIP_SIDE**2/1e6:.0f} M available; current per_neg=2800 uses "
          f"{nneg*2800/1e6:.1f} M (ratio knob has large headroom).")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks-subdir", default="masks_v2")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--finite-sample", type=int, default=2000,
                    help="estimate the isfinite-loss factor on N random chips (0 to skip)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N chips")
    main(ap.parse_args())

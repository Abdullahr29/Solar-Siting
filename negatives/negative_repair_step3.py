"""Step 3: execute the trim (delete excess pos-light negatives, densest-first to keep
the survivors spread) and simulate the re-mine to N=12,500 to report the FINAL country
proportions -- specifically flagging any country still starved after the fill.

Single coherent target: target_c = pos_share_c * N_final  (N_final = 12,500).
  * trim AUS/MEX/CHL/COL/ZAF DOWN to target (delete densest chips first)
  * leave the heavy-in-both countries (IND/CHN/USA/BRA/ESP...) above target untouched
  * waterfill the remaining budget into deficit countries (largest share-gap first)
A country is 'starved' if, after the fill, it sits well below its proportional target.

Run (JASMIN sci node, env_solar):
    python Solar-Siting/negative_repair_step3.py
"""
import os, glob
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

os.chdir(os.path.expanduser("~/Solar_Workspace"))
NEGDIR = "Data/aef_solar_chips/negatives/chips"
PROP = "Data/external/tz_sam_q1_2026/negative_contamination/negative_proportionality.csv"
OUT = "Data/external/tz_sam_q1_2026/negative_contamination"
N_FINAL = 12500
TRIM5 = ["AUS", "MEX", "CHL", "COL", "ZAF"]
EA = "EPSG:6933"


def centroids_ea(files):
    xs, ys = [], []
    cache = {}
    for p in files:
        with rasterio.open(p) as s:
            b, crs = s.bounds, s.crs
        k = crs.to_string()
        if k not in cache:
            cache[k] = Transformer.from_crs(crs, EA, always_xy=True)
        cx, cy = cache[k].transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)
        xs.append(cx); ys.append(cy)
    return np.array(xs), np.array(ys)


def trim_densest(country, keep_n):
    """Delete this country's chips down to keep_n, removing the spatially densest first
    (chip whose nearest same-country neighbour is closest) so survivors stay spread."""
    files = sorted(glob.glob(f"{NEGDIR}/NEG_{country}_*.tif"))
    n = len(files)
    if n <= keep_n:
        return 0, []
    x, y = centroids_ea(files)
    alive = np.ones(n, dtype=bool)
    removed = []
    # precompute full pairwise once (n is small, <=few hundred)
    D = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    np.fill_diagonal(D, np.inf)
    while alive.sum() > keep_n:
        Dm = D.copy()
        Dm[~alive, :] = np.inf; Dm[:, ~alive] = np.inf
        nn = Dm.min(axis=1)
        nn[~alive] = np.inf
        victim = int(np.argmin(nn))       # closest-neighbour chip = most redundant
        alive[victim] = False
        removed.append(files[victim])
    for p in removed:
        os.remove(p)
    return len(removed), removed


def main():
    df = pd.read_csv(PROP, index_col=0)
    df["target"] = (df["pos_share"] * N_FINAL).round().astype(int)

    # ---- execute trim (delete) ----
    print("=== TRIM (delete densest-first) ===")
    total_removed, log = 0, []
    for c in TRIM5:
        keep = int(df.loc[c, "target"])
        removed, files = trim_densest(c, keep)
        total_removed += removed
        log += files
        print(f"  {c}: {int(df.loc[c,'neg_now'])} -> {keep}   (deleted {removed})")
    pd.Series([os.path.basename(f) for f in log], name="chip").to_csv(
        f"{OUT}/trimmed_deleted_chips.csv", index=False)
    print(f"  deleted {total_removed} chips total\n")

    # ---- recount from disk ----
    counts = {}
    with os.scandir(NEGDIR) as it:
        for e in it:
            if e.name.endswith(".tif"):
                cc = e.name.split("_")[1]
                counts[cc] = counts.get(cc, 0) + 1
    counts = pd.Series(counts).reindex(df.index, fill_value=0).astype(float)
    ps = df["pos_share"].to_numpy()
    N_now = int(counts.sum())
    print(f"on disk after trim: {N_now:,} negatives\n")

    # ---- waterfill simulate to N_FINAL ----
    c = counts.to_numpy(dtype=float).copy()
    added = np.zeros_like(c)
    while c.sum() < N_FINAL:
        n = c.sum()
        gap = ps - c / n
        j = int(np.argmax(gap))
        if gap[j] <= 0:
            break
        c[j] += 1; added[j] += 1
    final_n = int(c.sum())
    final_share = c / final_n
    target = df["target"].to_numpy()
    fill = np.where(target > 0, c / target, np.inf)
    overlap = np.minimum(final_share, ps).sum()

    res = pd.DataFrame({
        "pos_chips": df["pos_chips"].astype(int),
        "pos_share%": (ps * 100).round(1),
        "neg_before": df["neg_now"].astype(int),   # before trim
        "on_disk": counts.astype(int),             # after trim
        "add": added.astype(int),
        "final": c.astype(int),
        "final%": (final_share * 100).round(1),
        "target": target,
        "fill%": (fill * 100).round(0),
    }, index=df.index)

    print(f"=== PROJECTED @ {final_n:,} negatives | overlap = {100*overlap:.1f}% ===\n")

    # deficit countries the fill did NOT bring to target, ranked by remaining shortfall
    deficit = res[res["final"] < res["target"]].copy()
    deficit["short"] = deficit["target"] - deficit["final"]
    deficit = deficit.sort_values("short", ascending=False)
    print("--- countries still BELOW proportional target (potential starvation) ---")
    print(deficit[["pos_chips", "on_disk", "add", "final", "target", "fill%", "short"]]
          .to_string())

    # explicit starvation flag: meaningful positives but poorly filled
    starved = deficit[(deficit["pos_chips"] >= 300) & (deficit["fill%"] < 70)]
    print(f"\n>>> STARVED (pos_chips>=300 and <70% of target): "
          f"{list(starved.index) if len(starved) else 'NONE'}")
    if len(starved):
        need = int((starved["target"] - starved["final"]).sum())
        print(f"    bringing just these to target needs ~{need} more negatives "
              f"(total ~{final_n + need:,})")

    # what would fully un-starve everyone (bring every deficit country up to target;
    # surplus countries stay where they are) -- direct, no loop.
    full = np.maximum(counts.to_numpy(dtype=float), target.astype(float))
    full_N = int(full.sum())
    full_ov = np.minimum(full / full_N, ps).sum()
    print(f"\n(for reference: full proportional fill = {full_N:,} negatives, "
          f"overlap {100*full_ov:.1f}%)")

    res.to_csv(f"{OUT}/projected_proportions_12500.csv")
    print(f"\nwrote {OUT}/projected_proportions_12500.csv & trimmed_deleted_chips.csv")


if __name__ == "__main__":
    main()

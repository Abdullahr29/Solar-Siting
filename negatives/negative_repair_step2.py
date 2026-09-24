"""Step 2: (a) permanently delete the quarantined contaminated negatives (shared GWS,
don't hoard); (b) work out the trim for the over-represented pos-light countries; and
(c) simulate where a greedy proportional re-mine lands us on the country-overlap metric,
so we can pick a footprint vs proportionality trade-off.

Proportionality metric = country-distribution overlap = sum_c min(neg_share, pos_share)
                       = 1 - total-variation distance  (1.0 = identical shape).

Run (JASMIN sci node, env_solar):
    python Solar-Siting/negative_repair_step2.py
"""
import os, shutil, glob
import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/Solar_Workspace"))
QUAR = "Data/aef_solar_chips/negatives/_quarantine_tzsam_5km"
PROP = "Data/external/tz_sam_q1_2026/negative_contamination/negative_proportionality.csv"
TRIM5 = ["AUS", "MEX", "CHL", "COL", "ZAF"]


def du_bytes(path):
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    return tot


def overlap(counts, pos_share):
    n = counts.sum()
    neg_share = counts / n if n else counts * 0
    return float(np.minimum(neg_share, pos_share).sum()), n


def main():
    # (a) delete quarantine
    if os.path.isdir(QUAR):
        freed = du_bytes(QUAR)
        nq = len(glob.glob(f"{QUAR}/*.tif"))
        shutil.rmtree(QUAR)
        print(f"deleted {nq:,} quarantined chips  ({freed/1e9:.2f} GB freed)\n")
    else:
        print("quarantine already gone\n")

    df = pd.read_csv(PROP, index_col=0)
    pos_share = df["pos_share"]
    neg_now = df["neg_now"].astype(float).copy()
    cur_ov, cur_n = overlap(neg_now, pos_share)
    print(f"CURRENT: {int(cur_n):,} negatives | overlap = {100*cur_ov:.1f}%\n")

    # (b) trim table for the pos-light surplus countries
    print("=== TRIM candidates (over-represented, positive-light) ===")
    print(f"{'ctry':>4} {'neg_now':>8} {'pos_sc':>7} {'pos_chip':>8} {'target':>7} {'trim':>6} {'->kept':>7}")
    trimmed = neg_now.copy()
    freed_slots = 0
    for c in TRIM5:
        tgt = int(round(df.loc[c, "target_holdN"]))
        keep = min(int(neg_now[c]), tgt)
        cut = int(neg_now[c]) - keep
        freed_slots += cut
        trimmed[c] = keep
        print(f"{c:>4} {int(neg_now[c]):>8} {int(df.loc[c,'pos_scenes']):>7} "
              f"{int(df.loc[c,'pos_chips']):>8} {tgt:>7} {cut:>6} {keep:>7}")
    print(f"     trimming frees {freed_slots} negatives\n")

    tr_ov, tr_n = overlap(trimmed, pos_share)
    print(f"AFTER TRIM ONLY: {int(tr_n):,} negatives | overlap = {100*tr_ov:.1f}%\n")

    # (c) greedy proportional add: repeatedly add one negative to the country most
    #     below its target share (waterfill). Only deficit countries can receive.
    counts = trimmed.to_numpy(dtype=float).copy()
    ps = pos_share.to_numpy()
    idx = df.index.to_numpy()
    # a country is "fillable" if it currently sits below its proportional share
    milestones = {}
    targets_N = [9683, 11279, 12500, 14000]  # footprint options to report
    max_add = 6000
    add = 0
    curve = []
    while add <= max_add:
        n = counts.sum()
        ov = np.minimum(counts / n, ps).sum()
        curve.append((int(n), 100 * ov))
        # pick country with largest (pos_share - current_share) among those below share
        share = counts / n
        gap = ps - share
        c = int(np.argmax(gap))
        if gap[c] <= 0:
            break                       # already fully proportional
        counts[c] += 1
        add += 1
    curve = np.array(curve)

    print("=== proportional RE-MINE (greedy waterfill into deficit countries) ===")
    print(f"{'total_neg':>10} {'net_vs_now':>11} {'net_vs_orig(11279)':>18} {'overlap':>8}")
    for TN in targets_N:
        # find nearest curve point with total >= TN
        row = curve[np.argmin(np.abs(curve[:, 0] - TN))]
        print(f"{int(row[0]):>10} {int(row[0])-int(cur_n):>+11} {int(row[0])-11279:>+18} {row[1]:>7.1f}%")
    print(f"{int(curve[-1,0]):>10} {int(curve[-1,0])-int(cur_n):>+11} "
          f"{int(curve[-1,0])-11279:>+18} {curve[-1,1]:>7.1f}%   <- full deficit fill (ceiling)")

    # per-country ADD list to reach the ~original-footprint (11279) option
    TN = 11279
    counts2 = trimmed.to_numpy(dtype=float).copy()
    add = 0
    while counts2.sum() < TN:
        n = counts2.sum(); gap = ps - counts2 / n
        c = int(np.argmax(gap))
        if gap[c] <= 0:
            break
        counts2[c] += 1; add += 1
    adds = pd.Series(counts2 - trimmed.to_numpy(), index=df.index)
    adds = adds[adds > 0].sort_values(ascending=False).astype(int)
    print(f"\n=== ADD list to reach ~{TN:,} total ({add} new negatives), top 15 ===")
    print(f"{'ctry':>4} {'neg_now':>8} {'pos_chip':>8} {'add':>5} {'->final':>8}")
    for c in adds.head(15).index:
        print(f"{c:>4} {int(neg_now[c]):>8} {int(df.loc[c,'pos_chips']):>8} "
              f"{int(adds[c]):>5} {int(counts2[df.index.get_loc(c)]):>8}")


if __name__ == "__main__":
    main()

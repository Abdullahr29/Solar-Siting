"""Step 1 of the negative-set repair:
  (a) quarantine the TZ-SAM-contaminated negatives (nearest new solar site <=5 km),
      MOVING them to a sibling folder (reversible) rather than deleting.
  (b) build the country-proportionality diagnostic: current surviving negatives per
      country vs the positive prevalence per country, and the implied deficit if we
      want negatives distributed like the positives.

Reads the per-chip distances written by negative_tzsam_contamination.py.
Run (JASMIN sci node, env_solar):
    python Solar-Siting/negative_repair_step1.py
"""
import os, glob, shutil
import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/Solar_Workspace"))
NEGDIR = "Data/aef_solar_chips/negatives/chips"
QUAR = "Data/aef_solar_chips/negatives/_quarantine_tzsam_5km"
POSDIR = "Data/aef_solar_chips/positives/chips"
NN = "Data/external/tz_sam_q1_2026/negative_contamination/negative_nn_distances.csv"
OUTDIR = "Data/external/tz_sam_q1_2026/negative_contamination"


def quarantine():
    nn = pd.read_csv(NN)
    bad = nn[nn["within5"] == True]["chip"].tolist()
    os.makedirs(QUAR, exist_ok=True)
    moved = 0
    for fn in bad:
        src = f"{NEGDIR}/{fn}"
        if os.path.exists(src):
            shutil.move(src, f"{QUAR}/{fn}")
            moved += 1
    pd.Series(bad, name="chip").to_csv(f"{OUTDIR}/quarantined_chips.csv", index=False)
    return nn, moved


def per_country_from_disk(folder, scene_key=False):
    """Count files per country from NEG_/POS_ filenames; if scene_key, count unique
    (country,id,year) scenes instead of chips (positives are gridded -> many chips/scene)."""
    chips, scenes = {}, {}
    with os.scandir(folder) as it:
        for e in it:
            if not e.name.endswith(".tif"):
                continue
            p = e.name.split("_")
            if len(p) < 4:
                continue
            c = p[1]
            chips[c] = chips.get(c, 0) + 1
            if scene_key:
                scenes.setdefault(c, set()).add((p[2], p[3]))
    if scene_key:
        return chips, {c: len(s) for c, s in scenes.items()}
    return chips, None


def main():
    nn, moved = quarantine()
    print(f"quarantined {moved:,} contaminated negatives -> {QUAR}\n", flush=True)

    # surviving negatives per country (disk truth = pre-existing minus quarantined)
    neg_all = nn.groupby("country").size()
    neg_bad = nn[nn["within5"] == True].groupby("country").size()
    neg_now = (neg_all - neg_bad.reindex(neg_all.index, fill_value=0)).rename("neg_now")

    # positive prevalence: unique scenes per country (apples-to-apples with 1-chip negs)
    pos_chips, pos_scenes = per_country_from_disk(POSDIR, scene_key=True)
    pos_sc = pd.Series(pos_scenes, name="pos_scenes")
    pos_ch = pd.Series(pos_chips, name="pos_chips")

    df = pd.concat([neg_now, neg_all.rename("neg_before"), pos_sc, pos_ch], axis=1).fillna(0)
    df = df.astype(int)
    Ntot = int(df["neg_now"].sum())
    pos_total = int(df["pos_scenes"].sum())
    df["pos_share"] = df["pos_scenes"] / pos_total
    df["neg_share"] = df["neg_now"] / max(Ntot, 1)
    # proportional target at "hold current total" budget
    df["target_holdN"] = (df["pos_share"] * Ntot).round().astype(int)
    df["deficit_holdN"] = df["target_holdN"] - df["neg_now"]

    # headline imbalance: total-variation distance between neg and pos country dists
    tv = 0.5 * (df["neg_share"] - df["pos_share"]).abs().sum()

    print(f"surviving negatives: {Ntot:,}  | positive scenes: {pos_total:,}\n")
    print(f"country-distribution TV distance (neg vs pos): {tv:.3f}  "
          f"(0 = identical shape, 1 = disjoint)\n")

    show = df.sort_values("deficit_holdN", ascending=False)
    cols = ["neg_before", "neg_now", "pos_scenes", "pos_chips",
            "pos_share", "neg_share", "target_holdN", "deficit_holdN"]
    pd.set_option("display.width", 200)
    print("=== biggest DEFICITS (want more negatives here) — top 15 ===")
    print(show[cols].head(15).round(3).to_string())
    print("\n=== biggest SURPLUS (over-represented) — bottom 10 ===")
    print(show[cols].tail(10).round(3).to_string())

    df.sort_values("deficit_holdN", ascending=False).to_csv(
        f"{OUTDIR}/negative_proportionality.csv")
    print(f"\nwrote {OUTDIR}/negative_proportionality.csv  &  quarantined_chips.csv")
    print(f"remaining on disk in {NEGDIR}: "
          f"{len(glob.glob(NEGDIR + '/*.tif')):,} negatives")


if __name__ == "__main__":
    main()

"""Contamination check: do our downloaded NEGATIVE chips overlap solar sites that
appear in the TZ-SAM Q1-2026 database (sites we didn't know about at negative-mining
time)? Any such overlap is label noise — a "presumed-unsuitable" negative that is
actually a solar site, which confounds the discriminative model.

Two checks the user asked for:
  (1) DIRECT OVERLAP  : a TZ-SAM solar polygon intersects the chip's footprint
                        (the negative chip literally contains a solar site).
  (2) WITHIN 5 km     : nearest TZ-SAM polygon is <= 5 km from the chip footprint;
                        record the distance so we get a full distribution.

Footprints are read from the actual .tif on disk (256x256 @ 10 m, per-chip UTM),
so we measure the true chip extent, not the parent scene's ml_bbox. Distances are
computed in EPSG:6933 (global equal-area, metres); a footprint-to-polygon distance
is 0 when they intersect, else the gap from the chip EDGE to the nearest site.

Run (JASMIN sci node, env_solar, PROJ_DATA set):
    python Solar-Siting/negative_tzsam_contamination.py
"""
import os, glob, re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import rasterio

os.chdir(os.path.expanduser("~/Solar_Workspace"))
EA = "EPSG:6933"
CHIPS = "Data/aef_solar_chips/negatives/chips"
TZ_GPKG = "Data/external/tz_sam_q1_2026/TZ-SAM Q1-2026 CC-BY-NC/2026-Q1_analysis_polygons.gpkg"
OUTDIR = "Data/external/tz_sam_q1_2026/negative_contamination"
os.makedirs(OUTDIR, exist_ok=True)
NAME_RE = re.compile(r"NEG_([A-Z]{3})_(\d+)_(\d{4})")


def chip_footprints():
    """Read every negative chip's bbox -> footprint polygon (in EPSG:4326)."""
    paths = sorted(glob.glob(f"{CHIPS}/*.tif"))
    rows, geoms = [], []
    tr_cache = {}
    for p in paths:
        fn = os.path.basename(p)
        m = NAME_RE.search(fn)
        with rasterio.open(p) as s:
            b, crs = s.bounds, s.crs
        key = crs.to_string()
        if key not in tr_cache:
            tr_cache[key] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        tr = tr_cache[key]
        # transform the 4 corners, take their bbox in lon/lat
        xs, ys = zip(*[(b.left, b.bottom), (b.left, b.top),
                       (b.right, b.top), (b.right, b.bottom)])
        lon, lat = tr.transform(xs, ys)
        geoms.append(box(min(lon), min(lat), max(lon), max(lat)))
        rows.append({"chip": fn,
                     "country": m.group(1) if m else None,
                     "neg_id": m.group(2) if m else None,
                     "chip_year": int(m.group(3)) if m else None})
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    return gdf.to_crs(EA)


def main():
    print("reading chip footprints from disk ...", flush=True)
    chips = chip_footprints()
    nC = len(chips)
    print(f"  {nC:,} negative chips", flush=True)

    print("loading TZ-SAM polygons ...", flush=True)
    tz = gpd.read_file(TZ_GPKG)
    tz = tz[tz.geometry.notna() & tz.is_valid].copy()
    tz["yr_after"] = pd.to_datetime(tz["constructed_after"], errors="coerce").dt.year
    tz["yr_before"] = pd.to_datetime(tz["constructed_before"], errors="coerce").dt.year
    tz = tz.to_crs(EA)
    print(f"  {len(tz):,} TZ-SAM sites", flush=True)

    # ---- (1) DIRECT OVERLAP: TZ-SAM polygon intersects chip footprint ----
    hit = gpd.sjoin(chips, tz[["geometry", "capacity_mw", "yr_after", "yr_before"]],
                    predicate="intersects", how="inner")
    # a chip may catch several sites; aggregate to per-chip
    agg = (hit.groupby(hit.index)
              .agg(n_sites=("capacity_mw", "size"),
                   cap_mw=("capacity_mw", "sum"),
                   min_yr_after=("yr_after", "min"),
                   max_yr_before=("yr_before", "max")))
    contained = chips.loc[agg.index].join(agg)
    n_contained = len(contained)

    # ---- (2) WITHIN 5 km: nearest TZ-SAM polygon to each chip footprint ----
    nn = gpd.sjoin_nearest(chips[["geometry", "chip", "country", "chip_year"]],
                           tz[["geometry", "capacity_mw", "yr_after"]],
                           how="left", distance_col="dist_m")
    nn = nn[~nn.index.duplicated(keep="first")].copy()
    d = nn["dist_m"].to_numpy()

    thr = [0, 500, 1000, 2000, 5000]  # 0 == intersects (direct overlap)
    print("\n==================== CONTAMINATION SUMMARY ====================")
    print(f"negative chips checked : {nC:,}")
    print(f"\n(1) DIRECT OVERLAP (chip footprint contains a TZ-SAM solar site):")
    print(f"    contaminated chips : {n_contained:,}  ({100*n_contained/nC:.2f}%)")
    print(f"\n(2) NEAREST TZ-SAM SITE distance-to-chip-footprint distribution:")
    for t in thr:
        n = int(np.nansum(d <= t))
        lab = "== 0 m (overlap)" if t == 0 else f"<= {t:>4} m"
        print(f"    {lab:>18} : {n:>6,}  ({100*n/nC:5.2f}%)")
    within5 = int(np.nansum(d <= 5000))
    print(f"\n    within 5 km TOTAL  : {within5:,}  ({100*within5/nC:.2f}%)  "
          f"<- these violate the >=5 km exclusion buffer vs NEW sites")
    dd = d[np.isfinite(d)]
    print(f"    distance quantiles (m): p1 {np.percentile(dd,1):,.0f} | "
          f"p10 {np.percentile(dd,10):,.0f} | median {np.percentile(dd,50):,.0f} | "
          f"p90 {np.percentile(dd,90):,.0f}")

    # temporal split of the DIRECT overlaps: was the site visible in the embedding?
    if n_contained:
        vis = contained[contained["max_yr_before"] <= contained["chip_year"]]
        fut = contained[contained["min_yr_after"] > contained["chip_year"]]
        print(f"\n(3) DIRECT overlaps by construction date vs chip embedding year:")
        print(f"    site built BEFORE chip year (visible in embedding, HARD noise): {len(vis):,}")
        print(f"    site built AFTER  chip year (pre-solar land, future site)     : {len(fut):,}")
        print(f"    (remainder: detection bracket straddles the chip year)")

    # per-country breakdown of within-5km contamination
    nn["within5"] = d <= 5000
    nn["overlap"] = d <= 0
    by = (nn.groupby("country")
             .agg(chips=("chip", "size"),
                  within5=("within5", "sum"),
                  overlap=("overlap", "sum"))
             .assign(pct5=lambda x: (100*x.within5/x.chips).round(1))
             .sort_values("within5", ascending=False))
    print("\n(4) per-country contamination (top 15 by #within-5km):")
    print(by.head(15).to_string())

    # ---- write artifacts ----
    nn[["chip", "country", "chip_year", "dist_m", "within5", "overlap"]].to_csv(
        f"{OUTDIR}/negative_nn_distances.csv", index=False)
    contained.drop(columns="geometry").to_csv(
        f"{OUTDIR}/negative_direct_overlaps.csv", index=True)
    by.to_csv(f"{OUTDIR}/contamination_by_country.csv")
    with open(f"{OUTDIR}/summary.txt", "w") as f:
        f.write(f"negatives checked: {nC}\n")
        f.write(f"direct overlap (contains site): {n_contained} ({100*n_contained/nC:.2f}%)\n")
        f.write(f"within 5km: {within5} ({100*within5/nC:.2f}%)\n")
    print(f"\nwrote {OUTDIR}/  (negative_nn_distances.csv, negative_direct_overlaps.csv, "
          f"contamination_by_country.csv, summary.txt)")


if __name__ == "__main__":
    main()

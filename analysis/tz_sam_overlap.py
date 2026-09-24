"""Quick overlap analysis: TZ-SAM Q1-2026 solar assets vs our PV facility inventory.

Side/future-reference work (NOT wired into the models). Question: how much of TZ-SAM is
already in our inventory, and how much is new? Both carry centroid lon/lat + country + a
temporal field, so we match spatially:

  * within-footprint : TZ-SAM centroid falls inside one of our inventory polygons (strict)
  * near-match       : nearest inventory centroid within 200 / 500 / 1000 m (equal-area)
both directions (TZ-SAM covered by us; us covered by TZ-SAM), plus a per-country breakdown
and the install-year / construction-date distributions (to see what TZ-SAM adds temporally).

Run (JASMIN sci node, env_solar, with PROJ_DATA set):
    python Solar-Siting/tz_sam_overlap.py
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

os.chdir(os.path.expanduser("~/Solar_Workspace"))
EA = "EPSG:6933"                          # global equal-area (metres) for distances
INV = "Solar-Siting/global_pv_facility_inventory.gpkg"
TZ_DIR = "Data/external/tz_sam_q1_2026/TZ-SAM Q1-2026 CC-BY-NC"
TZ_CSV = f"{TZ_DIR}/2026-Q1_analysis_polygons.csv"
OUTDIR = "Data/external/tz_sam_q1_2026/overlap_analysis"
os.makedirs(OUTDIR, exist_ok=True)


def load():
    inv = gpd.read_file(INV)                                  # 140k MultiPolygons, EPSG:4326
    inv = inv[inv.geometry.notna() & inv.is_valid].copy()
    tz = pd.read_csv(TZ_CSV)
    tzp = gpd.GeoDataFrame(tz, geometry=gpd.points_from_xy(tz.longitude, tz.latitude),
                           crs="EPSG:4326")
    inv_c = inv.copy(); inv_c["geometry"] = inv_c.geometry.representative_point()
    return (inv.to_crs(EA), inv_c.to_crs(EA), tzp.to_crs(EA))


def nearest_within(a_pts, b_pts, thresholds):
    """For each row of a_pts, distance to nearest b_pts centroid; counts within thresholds."""
    j = gpd.sjoin_nearest(a_pts[["geometry"]], b_pts[["geometry"]],
                          how="left", distance_col="dist_m")
    j = j[~j.index.duplicated(keep="first")]                 # ties -> keep one
    d = j["dist_m"].to_numpy()
    return d, {t: int(np.nansum(d <= t)) for t in thresholds}


def main():
    inv_poly, inv_cent, tz = load()
    nI, nT = len(inv_poly), len(tz)
    print(f"inventory: {nI:,} polygons | TZ-SAM: {nT:,} sites\n", flush=True)

    # --- strict: TZ-SAM centroid inside an inventory footprint ---
    inside = gpd.sjoin(tz[["geometry", "country", "capacity_mw"]], inv_poly[["geometry"]],
                       predicate="within", how="left")
    inside = inside[~inside.index.duplicated(keep="first")]
    n_inside = int(inside["index_right"].notna().sum())

    # --- near-match distances, both directions (centroid-to-centroid) ---
    thr = [200, 500, 1000]
    d_tz, tz_hits = nearest_within(tz, inv_cent, thr)
    d_iv, iv_hits = nearest_within(inv_cent, tz, thr)

    print("=== TZ-SAM covered by our inventory ===")
    print(f"  inside a footprint : {n_inside:,} ({100*n_inside/nT:.1f}%)")
    for t in thr:
        print(f"  centroid <= {t:>4}m   : {tz_hits[t]:,} ({100*tz_hits[t]/nT:.1f}%)")
    print(f"  median NN distance : {np.nanmedian(d_tz):,.0f} m")
    print("\n=== our inventory covered by TZ-SAM ===")
    for t in thr:
        print(f"  centroid <= {t:>4}m   : {iv_hits[t]:,} ({100*iv_hits[t]/nI:.1f}%)")
    print(f"  median NN distance : {np.nanmedian(d_iv):,.0f} m")

    # --- per-country overlap (TZ-SAM side, <=500 m) ---
    tzc = tz.copy(); tzc["matched"] = d_tz <= 500
    by = (tzc.groupby("country")
             .agg(tz_sites=("matched", "size"), matched=("matched", "sum"))
             .assign(pct=lambda x: (100 * x.matched / x.tz_sites).round(1))
             .sort_values("tz_sites", ascending=False))
    print("\n=== per-country (TZ-SAM sites, matched<=500m), top 15 by size ===")
    print(by.head(15).to_string())
    by.to_csv(f"{OUTDIR}/overlap_by_country.csv")

    # --- temporal: what does TZ-SAM add? ---
    tzc["yr_after"] = pd.to_datetime(tzc["constructed_after"], errors="coerce").dt.year
    tzc["yr_before"] = pd.to_datetime(tzc["constructed_before"], errors="coerce").dt.year
    inv_yr = gpd.read_file(INV, columns=["year"])["year"]
    print("\n=== temporal coverage ===")
    print(f"  inventory install year : min {int(inv_yr.min())} / med {int(inv_yr.median())} / max {int(inv_yr.max())}")
    print(f"  TZ-SAM constructed_after: min {int(tzc.yr_after.min())} / med {int(tzc.yr_after.median())} / max {int(tzc.yr_after.max())}")
    new = tzc[(~tzc["matched"]) & (tzc["yr_after"] >= inv_yr.max())]
    print(f"  UNMATCHED TZ-SAM built >= inventory max yr ({int(inv_yr.max())}): {len(new):,} "
          f"({100*len(new)/nT:.1f}% of TZ-SAM) -- candidate 'new' sites")

    pd.DataFrame({"tz_nn_dist_m": d_tz}).to_csv(f"{OUTDIR}/tz_nn_distances.csv", index=False)
    with open(f"{OUTDIR}/summary.txt", "w") as f:
        f.write(f"inventory {nI} polys | TZ-SAM {nT} sites\n")
        f.write(f"TZ-SAM inside footprint: {n_inside} ({100*n_inside/nT:.1f}%)\n")
        f.write(f"TZ-SAM <=500m: {tz_hits[500]} ({100*tz_hits[500]/nT:.1f}%)\n")
        f.write(f"inventory <=500m: {iv_hits[500]} ({100*iv_hits[500]/nI:.1f}%)\n")
    print(f"\nwrote {OUTDIR}/ (overlap_by_country.csv, tz_nn_distances.csv, summary.txt)")


if __name__ == "__main__":
    main()

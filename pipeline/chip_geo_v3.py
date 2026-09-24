"""Chip -> (iso3, cluster, year, centroid lat/lon) table for the spatial-diversity sampler
(IMPROVEMENTS.md Track 1). Light: reads only each chip's bounds + CRS (no pixel data) and
reprojects the centroid from its per-scene UTM to EPSG:4326.

The master pool tags each pixel with a source-chip index into pos_chips / neg_files; joining
those names to this table gives every pixel a lat/lon cell + site (cluster) for the per-country
budget sweep (spread the sample across cells -> sites -> pixels, so a few mega-farms can't
dominate). Keyed by the SAME chip lists the extractor uses.

    python Solar-Siting/chip_geo_v3.py --procs 32
Writes artifacts/stats/chip_geo_pos.csv and chip_geo_neg.csv.
"""
import argparse, os, re
from multiprocessing import Pool
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform

WS = os.path.expanduser("~/Solar_Workspace")
PCH = "Data/aef_solar_chips/positives/chips"
NCH = "Data/aef_solar_chips/negatives/chips"
SPLITS = "Solar-Siting/artifacts/splits"
DIR = None
CLUS = re.compile(r"^(?:POS|NEG)_[A-Z]{2,4}_(\d+)_")


def _geo(chip):
    try:
        with rasterio.open(os.path.join(DIR, chip)) as s:
            b, crs = s.bounds, s.crs
        cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        lon, lat = transform(crs, "EPSG:4326", [cx], [cy])
        m = CLUS.match(chip)
        return chip, (int(m.group(1)) if m else -1), round(lat[0], 5), round(lon[0], 5)
    except Exception as e:
        print(f"  skip {chip}: {e}", flush=True)
        return chip, -1, np.nan, np.nan


def run(kind, chips, iso, yr, procs, out):
    global DIR
    DIR = os.path.join(WS, PCH if kind == "pos" else NCH)
    with Pool(procs) as p:
        rows = p.map(_geo, chips, chunksize=64)
    g = pd.DataFrame(rows, columns=["chip", "cluster", "lat", "lon"])
    g["iso3"], g["year"] = iso.values, yr.values
    g = g[["chip", "iso3", "cluster", "year", "lat", "lon"]]
    g.to_csv(out, index=False)
    bad = int(g.lat.isna().sum())
    print(f"{kind}: {len(g):,} chips -> {out} ({bad} unresolved) | "
          f"lat [{g.lat.min():.1f},{g.lat.max():.1f}] lon [{g.lon.min():.1f},{g.lon.max():.1f}]", flush=True)


def main(a):
    os.chdir(WS)
    stats = "Solar-Siting/artifacts/stats"
    pos = pd.read_csv(f"{SPLITS}/emb_search_split_v2.csv")
    neg = pd.read_csv(f"{SPLITS}/emb_neg_split_v3.csv")
    if a.limit:
        pos, neg = pos.head(a.limit), neg.head(a.limit)
    run("pos", pos.chip.tolist(), pos.iso3, pos.year, a.procs, f"{stats}/chip_geo_pos.csv")
    run("neg", neg.file.tolist(), neg.iso3, neg.year, a.procs, f"{stats}/chip_geo_neg.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args())

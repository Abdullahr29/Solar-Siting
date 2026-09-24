"""Emit the CORRECTED covariate table for the re-run of the covariate analysis + MCDA.

Background: the original `_covsample_full.npz` carried a `slope` column computed by
ee.Terrain on a bare .mosaic() -> EE's ~1 deg default grid -> slope collapsed to ~0
everywhere (max 0.63 deg). The aspect probe (covariate_aspect.py) re-sampled the DEM at
native 30 m and stored the real terrain as `slope_correct` plus the folded aspect
features (`equatorwardness`, `eastness`, `slope_x_equatorward`) at the *identical*
(lon,lat) points, in `_covsample_aspect.npz`.

This script turns that into a single canonical `_covsample_corrected.npz` that the
re-run of covariate_analysis.py + mcda_benchmark.py consume:
  * `slope`  <- slope_correct  (native 30 m; REPLACES the ~0 artifact)
  * add `equatorwardness`, `eastness`, `slope_x_equatorward` as continuous covariates
  * drop raw `aspect` (circular degrees must never enter a linear/rank model) and the
    now-redundant `slope_correct` alias
  * everything else (elevation, worldcover, dw_label, in_wdpa, gsa_*, grid/road dist,
    score, lon/lat, road_n_bbox) passes through unchanged.

Usage (JASMIN sci node, env_solar):
    python Solar-Siting/covariate_corrected.py            # all 7 countries
    python Solar-Siting/covariate_corrected.py Greece     # one country
"""
import os, sys
import numpy as np

COUNTRIES = ["Greece", "Chile", "Germany", "Australia", "China",
             "United States", "South Africa"]


def build_one(country, year=2021):
    slug = country.lower().replace(" ", "_")
    base = f"Solar-Siting/artifacts/covariate/{slug}/{slug}_{year}"
    src = f"{base}_covsample_aspect.npz"
    if not os.path.exists(src):
        print(f"  SKIP {country}: no {os.path.basename(src)}", flush=True)
        return None
    d = {k: v for k, v in np.load(src).items()}
    if "slope_correct" not in d:
        print(f"  SKIP {country}: aspect table lacks slope_correct", flush=True)
        return None
    d["slope"] = d.pop("slope_correct")          # native 30 m replaces the ~0 artifact
    d.pop("aspect", None)                          # raw circular degrees: not a covariate
    out = f"{base}_covsample_corrected.npz"
    np.savez(out, **d)
    finite_slope = np.isfinite(d["slope"])
    print(f"  {country:14s} -> {os.path.basename(out)}  "
          f"cols={len(d)}  slope med={np.nanmedian(d['slope']):.1f} deg "
          f"(was ~0)  equatorwardness present={'equatorwardness' in d}", flush=True)
    return out


def main():
    os.chdir(os.path.expanduser("~/Solar_Workspace"))
    targets = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else COUNTRIES
    print(f"Building corrected covariate tables for {len(targets)} countr{'y' if len(targets)==1 else 'ies'}:")
    for c in targets:
        build_one(c)


if __name__ == "__main__":
    main()

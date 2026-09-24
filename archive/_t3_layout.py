"""Gate the t-3 build: (1) AEF-2017 floor = fraction of positives whose install-3 < 2017 (unusable),
(2) confirm we have per-pixel coords/metadata to sample t-3, (3) pool sizes. Pure local, no GEE."""
import os, numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/Solar_Workspace"))
POOLD = "Solar-Siting/artifacts/paper_v3/pools"
STAT = "Solar-Siting/artifacts/stats"

print("=== master pool keys + sizes ===")
z = np.load(f"{POOLD}/px_master_train.npz", allow_pickle=False)
for k in z.files:
    print(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")

# yr_sol distribution -> AEF-2017 floor (t-3 needs install_year-3 >= 2017 -> install_year >= 2020)
if "yr_sol" in z.files:
    yr = z["yr_sol"]
    print("\n=== positive install-year distribution (master pool) ===")
    u, c = np.unique(yr, return_counts=True)
    tot = c.sum()
    for yy, cc in zip(u, c):
        print(f"  {yy}: {cc:,} ({100*cc/tot:.1f}%)")
    # t-2 floor (current): install>=2019 (2019-2=2017 ok). t-3 floor: install>=2020 (2020-3=2017 ok)
    lost_t3 = (yr < 2020).sum()
    print(f"\n  t-2 uses install>=2019 (floor). t-3 needs install>=2020.")
    print(f"  positives LOST to t-3 AEF-2017 floor (install<2020): {lost_t3:,} / {tot:,} = {100*lost_t3/tot:.1f}%")
else:
    print("\n  NO yr_sol key -- check what year field exists:", [k for k in z.files if 'yr' in k or 'year' in k])

print("\n=== chip geo (for per-pixel coords) ===")
try:
    geo = pd.read_csv(f"{STAT}/chip_geo_pos.csv")
    print("  chip_geo_pos.csv cols:", list(geo.columns), "| rows:", len(geo))
except Exception as e:
    print("  chip_geo_pos.csv:", e)

print("\n=== px_v3 (cap120) pool sizes ===")
try:
    zz = np.load(f"{POOLD}/px_v3_train.npz")
    print("  keys:", zz.files, "| X_sol:", zz["X_sol"].shape, "| X_neg:", zz["X_neg"].shape)
except Exception as e:
    print("  px_v3_train.npz:", e)

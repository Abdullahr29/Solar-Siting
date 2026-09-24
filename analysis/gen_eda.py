"""EDA on PV facility generation data (Song et al. 2026, Nature Sust.).
Per-facility POA generation + aerosol loss, 2019-2023. Characterize, sanity-check
vs paper headline numbers, and measure overlap with our RF validation sites."""
import os, glob
import numpy as np, pandas as pd
os.environ["PROJ_DATA"] = os.path.expanduser("~/Solar_Workspace/envs/env_solar/share/proj")
WS = os.path.expanduser("~/Solar_Workspace/Solar-Siting")
GD = os.path.join(WS, "Data/external/pv_generation")

files = sorted(glob.glob(os.path.join(GD, "PV_facility_generation_year_*.csv")))
print("files:", [os.path.basename(f) for f in files])
frames = {}
for f in files:
    gy = int(f.split("_")[-1].split(".")[0])
    d = pd.read_csv(f); d["gen_year"] = gy; frames[gy] = d
    print(f"  {gy}: rows={len(d)} uniq_PV_ID={d.PV_ID.nunique()} dup={len(d)-d.PV_ID.nunique()}")
print("cols:", frames[2023].columns.tolist())

d = frames[2023].copy()
for c in ["power_POA (kWh)", "power_POA_clr (kWh)", "power_POA_cln (kWh)", "aerosol_loss (kWh)", "area_m2"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[(d.area_m2 > 0) & (d["power_POA (kWh)"] > 0)].copy()
d["yield"] = d["power_POA (kWh)"] / d.area_m2                       # kWh/m2/yr
d["aero_frac"] = d["aerosol_loss (kWh)"] / d["power_POA_cln (kWh)"]  # fraction lost

print("\n=== 2023 YIELD (kWh/m2/yr) ===")
print(d["yield"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())
print("\n=== 2023 AEROSOL LOSS FRACTION ===")
print(d["aero_frac"].describe(percentiles=[.05, .5, .95]).round(4).to_string())
glob_aero = d["aerosol_loss (kWh)"].sum() / d["power_POA_cln (kWh)"].sum()
tot_lost = d["aerosol_loss (kWh)"].sum() / 1e9
print("GLOBAL aerosol loss 2023 (paper ~5.8%%): %.2f%%  total_lost=%.1f TWh" % (glob_aero * 100, tot_lost))
chn = d[d.country == "China"]
print("CHINA aerosol loss 2023 (paper 7.7%%): %.2f%%" % (chn["aerosol_loss (kWh)"].sum() / chn["power_POA_cln (kWh)"].sum() * 100))

print("\n=== top-15 countries by facility count (2023) ===")
cc = d.groupby("country").agg(n=("PV_ID", "size"), med_yield=("yield", "median"),
                              tot_gen_TWh=("power_POA (kWh)", lambda x: x.sum() / 1e9),
                              aero_lost=("aerosol_loss (kWh)", "sum")).sort_values("n", ascending=False)
tot_cln = d.groupby("country")["power_POA_cln (kWh)"].sum()
cc["aero_pct"] = (cc["aero_lost"] / tot_cln * 100)
print(cc.drop(columns="aero_lost").head(15).round(2).to_string())

print("\n=== generation by year (all files) ===")
for gy, fr in frames.items():
    p = pd.to_numeric(fr["power_POA (kWh)"], errors="coerce")
    a = pd.to_numeric(fr["aerosol_loss (kWh)"], errors="coerce")
    print(f"  {gy}: n={len(fr)} totgen={p.sum()/1e9:.0f}TWh aero_lost={a.sum()/1e9:.1f}TWh")

# ---- overlap with our exported/validation sites ----
import geopandas as gpd
ts = gpd.read_file(os.path.join(WS, "global_solar_ml_pipeline.gpkg"), layer="training_data_solar")
inv_ids = set(pd.to_numeric(d.PV_ID, errors="coerce").dropna().astype(int))
our_ids = set(pd.to_numeric(ts.PV_ID, errors="coerce").dropna().astype(int))
print("\n=== overlap ===")
print(f"training_data_solar sites={len(ts)} uniq_PV_ID={len(our_ids)}")
print(f"  with 2023 generation: {len(our_ids & inv_ids)}")
allids = set()
for gy, fr in frames.items():
    allids |= set(pd.to_numeric(fr.PV_ID, errors="coerce").dropna().astype(int))
print(f"  with generation in ANY year: {len(our_ids & allids)}")

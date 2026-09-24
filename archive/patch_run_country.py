"""Add an optional `exclude_pv_ids` kwarg to run_country() so validation sites that were seen
in training can be dropped (leakage-free geographic breakdown). Backward-compatible: default
None reproduces the existing behaviour exactly. Idempotent; backs up once.
"""
import os, shutil, re, sys

F = os.path.expanduser("~/Solar_Workspace/Solar-Siting/run_country_rf.py")
src = open(F).read()
if "exclude_pv_ids" in src:
    print("already patched"); sys.exit(0)
shutil.copy(F, F + ".bak_predup")

# 1) add the kwarg to the signature
src = src.replace(
    "def run_country(country, year=2019, min_install=None, n_rand=6000, max_sites=3000,\n"
    "                model_path=REFERENCE_MODEL,\n"
    "                out_prefix=None, workspace=None, seed=42, inv_country=None):",
    "def run_country(country, year=2019, min_install=None, n_rand=6000, max_sites=3000,\n"
    "                model_path=REFERENCE_MODEL,\n"
    "                out_prefix=None, workspace=None, seed=42, inv_country=None,\n"
    "                exclude_pv_ids=None):")

# 2) drop training-seen sites right after the inventory is read
anchor = '    pts = inv.geometry.representative_point().to_crs("EPSG:4326")'
dedup = (
    '    if exclude_pv_ids is not None and "PV_ID" in inv.columns:\n'
    '        before = len(inv)\n'
    '        inv = inv[~inv["PV_ID"].isin(set(exclude_pv_ids))]\n'
    '        print(f"dedup: dropped {before-len(inv):,} of {before:,} validation sites seen in '
    'training", flush=True)\n'
    '        if len(inv) == 0:\n'
    '            sys.exit("all validation sites were in training after dedup")\n'
)
assert anchor in src, "anchor line not found — script structure changed"
src = src.replace(anchor, dedup + anchor)
open(F, "w").write(src)
print("patched run_country_rf.py (exclude_pv_ids); backup at run_country_rf.py.bak_predup")

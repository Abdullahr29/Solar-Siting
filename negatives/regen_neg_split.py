"""Regenerate the negative split table from the CURRENT on-disk negatives (13,440).

The old emb_neg_split_v2.csv is pre-remine (11,279 rows) and lists chips that were since
deleted (4 contaminated) while missing the 4,242 remine chips. Ground truth is disk.

Each negative chip is its own scene (single center crop), so chip-level == scene-level:
no scene-leakage concern. Split 80/20 train/test, STRATIFIED per country (iso3) with a
fixed seed so every country appears in both sides (countries with <5 chips -> all train).

Columns match what extract_pixels.py consumes: file, iso3, year, split (+ country=iso3).
"""
import os, re, csv
import numpy as np

WS = os.path.expanduser("~/Solar_Workspace")
NCH = os.path.join(WS, "Data/aef_solar_chips/negatives/chips")
OUT = os.path.join(WS, "Solar-Siting/artifacts/splits/emb_neg_split_v3.csv")
NAME_RE = re.compile(r"^NEG_([A-Z]{2,4})_(\d+)_(\d{4})\.tif$")
TEST_FRAC = 0.20
SEED = 0

files = sorted(f for f in os.listdir(NCH) if f.endswith(".tif"))
print(f"on-disk negative chips: {len(files):,}")

rows, bad = [], []
by_iso = {}
for f in files:
    m = NAME_RE.match(f)
    if not m:
        bad.append(f); continue
    iso, _id, year = m.group(1), m.group(2), int(m.group(3))
    by_iso.setdefault(iso, []).append((f, year))
if bad:
    print(f"WARNING: {len(bad)} names did not parse, e.g. {bad[:5]}")

rng = np.random.default_rng(SEED)
n_train = n_test = 0
for iso in sorted(by_iso):
    chips = sorted(by_iso[iso])
    idx = np.arange(len(chips)); rng.shuffle(idx)
    n_te = int(round(len(chips) * TEST_FRAC)) if len(chips) >= 5 else 0
    test_ids = set(idx[:n_te].tolist())
    for j, (f, year) in enumerate(chips):
        split = "test" if j in test_ids else "train"
        rows.append((f, iso, iso, year, split))
        if split == "test": n_test += 1
        else: n_train += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "country", "iso3", "year", "split"])
    w.writerows(rows)

print(f"wrote {OUT}")
print(f"  total {len(rows):,} | train {n_train:,} | test {n_test:,} "
      f"({100*n_test/len(rows):.1f}% test) | countries {len(by_iso)}")
# quick sanity: remine vs base counts
remine = sum(1 for f, *_ in rows if int(re.match(NAME_RE, f).group(2)) >= 100001)
print(f"  remine (id>=100001): {remine:,} | base: {len(rows)-remine:,}")

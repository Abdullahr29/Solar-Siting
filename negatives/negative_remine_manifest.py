"""Lock the final negative-set target at 13,450 and emit the per-country DOWNLOAD
manifest = how many NEW negatives to mine per country.

Policy (all consistent with the agreed plan):
  target_c   = round(pos_share_c * 12500)        # proportional target
  base_c     = waterfill-to-12,500 result        # balanced fill, biggest confounds first
  PRIORITY   = 8 flagged + major-EU (pos_chips>=200) deficits -> topped up to target
  add_c      = final_c - on_disk_c               # what the re-mine must sample
Surplus countries (heavy in both classes: IND/CHN/USA/BRA/ESP...) are left untouched.
"""
import pandas as pd

OUT = "Data/external/tz_sam_q1_2026/negative_contamination"
df = pd.read_csv(f"{OUT}/projected_proportions_12500.csv", index_col=0)

FLAGGED8 = ["NLD", "UKR", "EST", "ITA", "GRC", "GBR", "THA", "BGR"]
MAJOR_EU = ["HUN", "CZE", "AUT", "ROU", "LTU", "CYP", "PRT", "DNK"]
PRIORITY = FLAGGED8 + MAJOR_EU

final = df["final"].astype(float).copy()          # waterfill @12,500
for c in PRIORITY:
    final[c] = max(final[c], df.loc[c, "target"])
df["final_13450"] = final.astype(int)
df["add_new"] = (df["final_13450"] - df["on_disk"]).clip(lower=0).astype(int)

N_final = int(df["final_13450"].sum())
N_new = int(df["add_new"].sum())
print(f"LOCKED negative-set size : {N_final:,}")
print(f"currently on disk        : {int(df['on_disk'].sum()):,}")
print(f"NEW negatives to mine     : {N_new:,}\n")

man = df[df["add_new"] > 0][["on_disk", "target", "final_13450", "add_new",
                             "pos_chips"]].sort_values("add_new", ascending=False)
print(f"=== DOWNLOAD MANIFEST: {len(man)} countries, {N_new:,} new negatives ===")
print(man.to_string())

man.to_csv(f"{OUT}/remine_manifest_13450.csv")
df.to_csv(f"{OUT}/remine_final_13450_full.csv")
print(f"\nwrote {OUT}/remine_manifest_13450.csv (+ _full)")

"""After the 13,200 plan (waterfill to 12,500 + top the 8 flagged countries up to
target), is any MAJOR European solar player still starved? List every European country
by positive mass with its fill-ratio, and cost the fix if some are still short."""
import pandas as pd

OUT = "Data/external/tz_sam_q1_2026/negative_contamination"
df = pd.read_csv(f"{OUT}/projected_proportions_12500.csv", index_col=0)

FLAGGED8 = ["NLD", "UKR", "EST", "ITA", "GRC", "GBR", "THA", "BGR"]  # topped to target
EU = {"DEU","ESP","ITA","FRA","GBR","NLD","POL","GRC","PRT","HUN","CZE","ROU","AUT",
      "BEL","BGR","DNK","SWE","SVK","LTU","HRV","CYP","IRL","FIN","SVN","EST","LVA",
      "LUX","MLT","UKR","MDA","SRB","MKD","BIH","ALB","XKX","MNE","CHE","NOR","AND"}

# final counts under the 13,200 plan
final = df["final"].astype(float).copy()
for c in FLAGGED8:
    if c in final.index:
        final[c] = max(final[c], df.loc[c, "target"])
df["final_13200"] = final.astype(int)
tgt = df["target"].where(df["target"] > 0, other=1)
df["fill_13200%"] = (100 * final / tgt).round(0)

eu = df[df.index.isin(EU)].copy().sort_values("pos_chips", ascending=False)
cols = ["pos_chips", "on_disk", "final_13200", "target", "fill_13200%"]
print("=== European countries after the 13,200 plan (by positive mass) ===")
print(eu[cols].to_string())

# who is still short, and what it costs to bring them to target
short = eu[eu["final_13200"] < eu["target"]].copy()
short["need"] = short["target"] - short["final_13200"]
print("\n=== still BELOW target (European) ===")
print(short[cols + ["need"]].to_string())
print(f"\ntotal extra to fully cover ALL European deficits: {int(short['need'].sum())} "
      f"-> grand total ~{13161 + int(short['need'].sum()):,}")
# just the 'major' ones (>=200 pos_chips)
maj = short[short["pos_chips"] >= 200]
print(f"only the MAJOR ones (pos_chips>=200): {list(maj.index)} "
      f"= {int(maj['need'].sum())} extra -> ~{13161 + int(maj['need'].sum()):,}")

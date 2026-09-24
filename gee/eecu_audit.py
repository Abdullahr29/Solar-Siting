import ee, json
from collections import defaultdict
ee.Initialize(project="ee-abdullahr-solar")

tl = ee.batch.Task.list()[:400]
print("got", len(tl), "tasks")
sample = None
rows = []
for t in tl:
    s = t.status()
    st = s.get("state")
    if sample is None and st in ("COMPLETED", "RUNNING", "FAILED"):
        sample = s
    # find any EECU-like key
    eecu = None
    for k in s:
        if "eecu" in k.lower():
            eecu = s[k]
            eecu_key = k
    if eecu is not None:
        rows.append((float(eecu), st, s.get("description", ""),
                     s.get("start_timestamp_ms"), s.get("update_timestamp_ms")))

print("STATUS KEYS:", list(sample.keys()) if sample else "none")
print("rows with EECU:", len(rows))
rows.sort(reverse=True)
print("\nTOP 20 by EECU-seconds:")
for eecu, st, d, t0, t1 in rows[:20]:
    dur = (t1 - t0) / 1000 / 60 if (t0 and t1) else 0
    print(f"  {eecu:12.0f} EECU-s  {dur:7.1f} min  {st:9s} {d}")

tot = sum(r[0] for r in rows)
print(f"\nSUM reported EECU (last {len(tl)} tasks): {tot:,.0f} EECU-s")
byc = defaultdict(float); cnt = defaultdict(int)
for eecu, st, d, _, _ in rows:
    key = d.split("_")[0] if "_" in d else d
    byc[key] += eecu; cnt[key] += 1
print("\nby prefix:")
for k, v in sorted(byc.items(), key=lambda x: -x[1])[:15]:
    print(f"   {k:24s} {v:14,.0f} EECU-s   ({cnt[k]} tasks, {v/max(cnt[k],1):,.0f}/task)")

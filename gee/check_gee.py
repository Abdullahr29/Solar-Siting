"""Audit live Earth Engine tasks for project ee-abdullahr-solar.

Lists every operation still RUNNING/PENDING/READY, plus the most recent COMPLETED ones,
with description, destination (Drive folder / asset), and start/update times — so we can see
whether an export is still depositing tifs to Drive and, if so, which job spawned it.
"""
import ee

ee.Initialize(project="ee-abdullahr-solar")

try:
    ops = ee.data.listOperations()
except Exception as e:
    ops = None
    print("listOperations failed:", e)

if ops is not None:
    from collections import Counter
    states = Counter(o.get("metadata", {}).get("state", "?") for o in ops)
    print("=== operation states ===", dict(states), f"(total {len(ops)})")
    print("\n=== ACTIVE (PENDING/RUNNING) ===")
    active = [o for o in ops if o.get("metadata", {}).get("state") in
              ("PENDING", "RUNNING", "READY", "CANCELLING")]
    for o in active:
        m = o.get("metadata", {})
        dst = m.get("destinationUris", []) or m.get("description", "")
        print(f"  {m.get('state'):9} | {m.get('description','?')[:60]:60} | "
              f"start {m.get('startTime','?')} | {o.get('name','')}")
    print(f"  -> {len(active)} active operations")

    print("\n=== 15 most recent COMPLETED/FAILED ===")
    done = [o for o in ops if o.get("metadata", {}).get("state") in
            ("SUCCEEDED", "FAILED", "CANCELLED", "COMPLETED")]
    done.sort(key=lambda o: o.get("metadata", {}).get("updateTime", ""), reverse=True)
    for o in done[:15]:
        m = o.get("metadata", {})
        print(f"  {m.get('state'):9} | {m.get('description','?')[:60]:60} | "
              f"update {m.get('updateTime','?')}")

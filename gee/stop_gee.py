"""Cancel any non-terminal Earth Engine operations for ee-abdullahr-solar.

Defensive stop for the abandoned 10m export: cancels anything still PENDING/RUNNING/READY so no
further tiles can succeed and deposit tifs to Drive. Safe to run when nothing is active (no-op).
"""
import ee

ee.Initialize(project="ee-abdullahr-solar")
ops = ee.data.listOperations()
active = [o for o in ops if o.get("metadata", {}).get("state") in
         ("PENDING", "RUNNING", "READY", "CANCELLING")]
print(f"{len(ops)} operations listed; {len(active)} non-terminal")
cancelled = 0
for o in active:
    name = o.get("name", "")
    st = o.get("metadata", {}).get("state")
    if st == "CANCELLING":
        continue
    try:
        ee.data.cancelOperation(name)
        cancelled += 1
        print(f"  cancelled {st} {name}")
    except Exception as e:
        print(f"  FAILED to cancel {name}: {e}")
print(f"cancel requests sent: {cancelled}")

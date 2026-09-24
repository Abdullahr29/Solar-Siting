import ee, json
ee.Initialize(project="ee-abdullahr-solar")
aid = "projects/ee-abdullahr-solar/assets/solar_suitability_2025_R_100m"
img = ee.Image(aid)
info = img.getInfo()
b = info["bands"][0]
print("asset:", aid)
print("band:", b["id"])
print("data_type:", json.dumps(b["data_type"]))
print("dimensions:", b.get("dimensions"))
print("crs:", b.get("crs"))
print("crs_transform:", b.get("crs_transform"))
props = info.get("properties", {})
print("pyramiding/props keys:", [k for k in props][:20])
# default projection nominal scale
proj = img.projection().getInfo()
print("projection().crs:", proj.get("crs"), " transform:", proj.get("transform"))
try:
    print("nominalScale (m):", img.projection().nominalScale().getInfo())
except Exception as e:
    print("nominalScale err:", str(e)[:80])

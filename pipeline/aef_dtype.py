import ee, json
ee.Initialize(project="ee-abdullahr-solar")
ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2025-01-01", "2026-01-01")
img = ic.first()
info = img.getInfo()
bands = info["bands"]
print("n_bands:", len(bands))
b0 = bands[0]
print("band0 id:", b0["id"])
print("band0 data_type:", json.dumps(b0["data_type"]))
print("band0 dimensions:", b0.get("dimensions"))
print("band0 crs:", b0.get("crs"))
print("band0 crs_transform:", b0.get("crs_transform"))
# distinct dtypes across bands
dts = set(json.dumps(b["data_type"]) for b in bands)
print("distinct dtypes:", dts)

import county_geo

geo = county_geo.load_county_boundaries()
print("CRS:", geo.crs)
print("Sample geometry bounds (first county):", geo.geometry.iloc[0].bounds)
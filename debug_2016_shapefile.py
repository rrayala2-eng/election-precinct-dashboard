import resolver
import fetcher
import pipeline

file_set = resolver.resolve_files(2016, "General")
print("Shapefile URL:", file_set.shapefile_url)

local_path = fetcher.fetch(file_set.shapefile_url)
geo = pipeline.load_shapefile(local_path)

print("\nActual columns in the 2016 shapefile:")
print(list(geo.columns))
print("\nFirst 3 rows:")
print(geo.head(3))

print("\nFIPS column dtype:", geo['FIPS'].dtype)
print("Sample FIPS values:", geo['FIPS'].unique()[:10].tolist())
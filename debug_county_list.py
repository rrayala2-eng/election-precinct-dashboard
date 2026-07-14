import live_resolver as lr
import county_geo

geo = county_geo.load_county_boundaries()
slugs = geo['county_slug'].tolist()
found = lr.fetch_county_list('SEN', 6, slugs)
print('Counties found in SD6:', found)
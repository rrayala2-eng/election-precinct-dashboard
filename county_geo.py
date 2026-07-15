"""
county_geo.py
--------------
Fetches California's 58 county boundaries -- ONCE, ever. Unlike precinct
boundaries (which get redrawn most elections), county lines are
essentially permanent, so this file gets cached indefinitely and reused
for any year the live-results pathway needs, with no year parameter at all.

Source (confirmed real, California's own open data portal, derived from
US Census TIGER/Line):
  https://gis.data.ca.gov/api/download/v1/items/a7a5b9ebd58842e9979933cb7fe2287c/geojson?layers=0

NOT YET VERIFIED: the exact property/column names in this file (e.g.
whether the county name field is called "NAME", "COUNTY_NAME", etc).
Standard Census TIGER county layers use "NAME" -- this module assumes
that but falls back to scanning for a plausible column if "NAME" isn't
present. Run debug_county_geo.py to confirm before trusting in production.
"""

import os
import requests
import geopandas as gpd

COUNTY_GEOJSON_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/"
    "a7a5b9ebd58842e9979933cb7fe2287c/geojson?layers=0"
)
CACHE_PATH = os.environ.get("COUNTY_GEO_CACHE", "/tmp/ca_counties.geojson")
HEADERS = {"User-Agent": "election-dashboard-backend/1.0"}


def _slugify(name: str) -> str:
    """Matches the SOS site's county URL slugs, e.g. 'Los Angeles' -> 'los-angeles'."""
    return name.strip().lower().replace(" ", "-")


def _download_to_cache():
    """
    Downloads to a temp file, then atomically renames it into place.
    Same pattern as fetcher.py -- prevents a concurrent reader from ever
    seeing a partially-written (and therefore corrupted/unreadable) file.
    """
    resp = requests.get(COUNTY_GEOJSON_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    tmp_path = CACHE_PATH + ".part"
    with open(tmp_path, "wb") as f:
        f.write(resp.content)
    os.replace(tmp_path, CACHE_PATH)


def load_county_boundaries(force_refresh: bool = False) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame with one row per CA county, including a
    'county_slug' column normalized to match live_resolver.py's county
    slugs (e.g. 'sacramento', 'los-angeles') so the two can be joined
    directly.
    """
    if not os.path.exists(CACHE_PATH) or force_refresh:
        _download_to_cache()

    try:
        geo = gpd.read_file(CACHE_PATH)
    except Exception:
        # Cache file is corrupted (confirmed real cause: the old version of
        # this function wrote directly to CACHE_PATH rather than
        # atomically, so a request reading mid-write could see a partial/
        # garbage file -- and since it existed, every later request would
        # keep hitting the same corrupted file until a redeploy wiped /tmp).
        # Self-heal: delete the bad file and download fresh once, rather
        # than requiring a manual fix or a full redeploy.
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        _download_to_cache()
        geo = gpd.read_file(CACHE_PATH)

    # CONFIRMED real issue: this file comes in EPSG:3857 (Web Mercator,
    # meters) rather than plain lat/lon. Leaflet only understands lat/lon
    # (EPSG:4326) -- without this reprojection, coordinates come out as
    # huge numbers like -13,618,991 instead of -121.5, and the map
    # silently renders as an unzoomed flat world instead of California.
    if geo.crs is not None and str(geo.crs).upper() != "EPSG:4326":
        geo = geo.to_crs(epsg=4326)

    name_col = None
    for candidate in ["CountyName", "NAME", "COUNTY_NAME", "COUNTYNAME", "County", "NAMELSAD"]:
        if candidate in geo.columns:
            name_col = candidate
            break
    if name_col is None:
        raise ValueError(
            f"Could not find a county-name column in the boundary file. "
            f"Actual columns: {list(geo.columns)}. Inspect and update county_geo.py."
        )

    geo["county_slug"] = geo[name_col].apply(_slugify)

    # Real file has a 'Year' column -- likely multiple boundary-version rows
    # per county over time. Keep only the most recent per county so the
    # later join in pipeline.py can't silently fan out (one county matching
    # multiple rows would duplicate vote counts).
    if "Year" in geo.columns:
        geo = geo.sort_values("Year").drop_duplicates(subset="county_slug", keep="last")

    return geo


if __name__ == "__main__":
    # Manual test (requires outbound network access to gis.data.ca.gov)
    geo = load_county_boundaries()
    print("Columns:", list(geo.columns))
    print("Row count:", len(geo))
    print(geo[["county_slug"]].head(10))
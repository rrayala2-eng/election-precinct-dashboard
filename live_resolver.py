"""
live_resolver.py
-----------------
A SEPARATE pathway from resolver.py -- this one talks to the CA Secretary
of State's live/current-election results system, not statewidedatabase.org.

Why this exists: statewidedatabase.org (SWDB) does careful, months-long
work to produce clean precinct-level data with matching shapefiles. As of
this writing SWDB's most recent published year is 2025 -- the 2026 Primary
(held June 2, 2026, certified July 10, 2026) isn't there yet and won't be
for a while. This module is the fallback for "SWDB doesn't have this year
yet" -- it gets district/county-level (NOT precinct-level) results directly
from the SOS's real-time reporting system instead.

IMPORTANT DIFFERENCES from resolver.py's pathway:
  - No year parameter in the URLs at all -- this API only ever serves
    whatever the CURRENT election is. You can't ask it for "2024" results.
  - Data granularity is COUNTY, not precinct. There is no shapefile
    problem to solve here because county boundaries barely change --
    see county_geo.py for the static boundary file this pairs with.
  - Candidate names come bundled directly in the response -- no PDF
    parsing needed for this pathway.

Confirmed real (fetched live during development):
  https://api.sos.ca.gov/returns/state-senate/district/6
  ->
  {
    "raceTitle": "State Senate District 6 - Districtwide Results",
    "Reporting": "100% (388 of 388) precincts reporting",
    "ReportingTime": "June 3, 2026, 4:44 a.m.",
    "candidates": [
      {"Name": "Sean Frame", "Party": "Dem", "Votes": "42,780", "Percent": "26.4", "incumbent": false},
      ...
    ]
  }

NOT YET VERIFIED against the live site from this environment (network
sandbox restrictions) -- verify with debug_live_resolver.py before trusting
in production:
  - The exact URL shape for county-level sub-results
    (guessed as /returns/{slug}/district/{n}/county/{county-slug})
  - Whether every office slug below is actually correct
"""

import json
import os
import requests

API_BASE = "https://api.sos.ca.gov/returns"
HEADERS = {"User-Agent": "election-dashboard-backend/1.0"}
COUNTY_LIST_CACHE_PATH = os.environ.get("LIVE_COUNTY_LIST_CACHE", "/tmp/live_county_list_cache.json")

# Office code (matching the rest of this codebase) -> SOS live-results slug.
# Confirmed real: state-senate. Others follow the same site's own nav menu
# (seen in the fetched page) but individual district/statewide fetches for
# each haven't all been separately verified yet.
LIVE_OFFICE_SLUGS = {
    "SEN": "state-senate",
    "ASS": "state-assembly",
    "CNG": "us-rep",                     # NOTE: different from the PDF-era "us-house" guess
    "BOE": "board-of-equalization",
    "GOV": "governor",
    "LTG": "lieutenant-governor",
    "SOS": "secretary-of-state",
    "CON": "controller",
    "TRS": "treasurer",
    "ATG": "attorney-general",
    "INS": "insurance-commissioner",
    "SPI": "superintendent-of-public-instruction",
    # No confirmed slug for USS/USP on this site as of writing -- omitted
    # rather than guessed. Requesting these will raise LookupError below.
}


def _slug_for(office: str) -> str:
    slug = LIVE_OFFICE_SLUGS.get(office.upper())
    if not slug:
        raise LookupError(
            f"No live-results slug known for office='{office}'. "
            f"Known offices: {list(LIVE_OFFICE_SLUGS)}"
        )
    return slug


def fetch_district_results(office: str, district: int | None) -> dict:
    """
    Fetches district-wide (or statewide, if district is None) results
    directly as JSON. This is the ONLY call needed to get candidates +
    district-wide totals -- no separate name lookup required for this path.
    """
    slug = _slug_for(office)
    url = f"{API_BASE}/{slug}" + (f"/district/{district}" if district is not None else "")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        raise LookupError(
            f"No live results found at {url} (status {resp.status_code}). "
            f"This likely means there is no current election for this "
            f"office/district, or the office slug is wrong."
        )
    data = resp.json()
    if "candidates" not in data:
        raise LookupError(f"Unexpected response shape from {url}: {data}")
    return data


def fetch_county_list(office: str, district: int, all_county_slugs: list[str]) -> list[str]:
    """
    Returns the county slugs that ACTUALLY belong to this district, out of
    `all_county_slugs` (the full 58-county list from county_geo.py).

    CONFIRMED the original approach (scraping the HTML page for county
    links) does NOT work -- the page renders those links via JavaScript
    after load, so a plain requests.get() never sees them (confirmed: the
    raw HTML contains no per-county links or county names for this page at
    all). Brute-forcing against the already-known full county list is
    slower (up to 58 requests) but reliable.

    CACHED: this result almost never changes for a given office+district
    within one election cycle, but the brute-force scan is genuinely slow
    (confirmed: ~a minute for a 2-county district, worse for larger ones).
    Without caching this would repeat the full 58-request scan on every
    single dashboard request, which isn't acceptable. Cache is keyed by
    office+district and persists across requests/restarts via disk.
    """
    cache_key = f"{office.upper()}_{district}"
    cache = {}
    if os.path.exists(COUNTY_LIST_CACHE_PATH):
        with open(COUNTY_LIST_CACHE_PATH) as f:
            cache = json.load(f)
    if cache_key in cache:
        return cache[cache_key]

    found = []
    for slug in all_county_slugs:
        try:
            fetch_county_results(office, district, slug)
            found.append(slug)
        except LookupError:
            continue  # this county isn't part of this district -- expected for most

    cache[cache_key] = found
    with open(COUNTY_LIST_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    return found


def fetch_county_results(office: str, district: int, county_slug: str) -> dict:
    """
    Fetches results for one county within a district.

    CONFIRMED real, subtle behavior (found via testing): the API does NOT
    return an error for a county that isn't actually part of this district
    -- it silently falls back to returning the DISTRICT-WIDE totals again,
    as a single dict, with status 200. A valid county instead returns a
    LIST containing both the county-specific result and the district-wide
    total together.

    So a 200 status code alone does NOT mean "this county is really in
    this district" -- we must check that the returned raceTitle actually
    mentions "County", not just "Districtwide", before trusting it.
    """
    slug = _slug_for(office)
    url = f"{API_BASE}/{slug}/district/{district}/county/{county_slug}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        raise LookupError(f"No county-level results found at {url} (status {resp.status_code})")

    data = resp.json()
    items = data if isinstance(data, list) else [data]
    for item in items:
        if "county" in item.get("raceTitle", "").lower():
            return item

    raise LookupError(
        f"'{county_slug}' does not appear to actually be part of "
        f"{office} district {district} -- the API silently returned "
        f"district-wide totals instead of a county-specific result."
    )


def _parse_votes(v: str) -> int:
    """SOS formats vote counts with commas, e.g. '42,780' -- strip and convert."""
    return int(str(v).replace(",", "").strip() or 0)
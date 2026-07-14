"""
resolver.py
-----------
Finds the REAL download URLs for a given election year, by scraping
statewidedatabase.org rather than guessing filenames.

Why this file has to scrape instead of using a fixed URL template:
  - The site's folder prefix (d00/d10/d20...) does NOT map cleanly to year.
    e.g. 2020 General lives under /d10/, but 2022 General lives under /d20/.
  - File version suffixes (v01, v02...) can differ per county/year.
  - Exact filenames drift slightly release to release.

So the flow is:
  1. Fetch the master index page (election.html)
  2. Find the link matching (year, election_type) -> election data page
  3. Find the matching link -> geography/conversion page
  4. On each of those pages, pull out the actual .zip / .csv links we need
"""

import re
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional

BASE = "https://statewidedatabase.org"
HEADERS = {"User-Agent": "election-dashboard-backend/1.0"}


@dataclass
class ElectionFileSet:
    year: int
    election_type: str          # "General" | "Primary" | "Special"
    sov_srprec_url: Optional[str] = None      # Statement of Vote, by SR precinct (statewide)
    registration_url: Optional[str] = None    # Registration data, by SR precinct (statewide)
    shapefile_url: Optional[str] = None       # SRPREC_SHP, statewide boundary zip


def _get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _find_election_subpage(year: int, election_type: str) -> str:
    """
    Step 1: scrape election.html to find the link to this specific
    year+election_type's data page (e.g. .../d20/g22.html for 2022 General).
    """
    soup = _get_soup(f"{BASE}/election.html")  # downloads the master index page of every year

    # election_type initial: general->g, primary->p, special->s
    prefix = {"general": "g", "primary": "p", "special": "s"}[election_type.lower()]
    yy = f"{year % 100:02d}"
    target_fragment = f"{prefix}{yy}"  # e.g. "g22"

    # Look for an anchor whose href contains the target fragment, near text
    # mentioning the year and election type (site lists them as rows/links).
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if target_fragment in href.lower():
            candidates.append(href)

    if not candidates:
        raise LookupError(
            f"Could not find an election data page for {election_type} {year}. "
            f"It may not exist yet, or the site structure has changed."
        )

    # Prefer the "geo_conv" or election-data landing page over deep sub-links
    landing = [c for c in candidates if c.lower().endswith(f"{target_fragment}.html")]
    chosen = landing[0] if landing else candidates[0]
    return chosen if chosen.startswith("http") else f"{BASE}/{chosen.lstrip('/')}"


def _find_geo_conv_page(election_page_url: str, year: int, election_type: str) -> str:
    """
    Step 2: from the election data landing page, find the sibling
    '..._geo_conv.html' page that lists shapefile downloads.
    """
    prefix = {"general": "g", "primary": "p", "special": "s"}[election_type.lower()]
    yy = f"{year % 100:02d}"
    guess = election_page_url.replace(f"{prefix}{yy}.html", f"{prefix}{yy}_geo_conv.html")
    resp = requests.head(guess, headers=HEADERS, timeout=15, allow_redirects=True)
    if resp.status_code == 200: #does the file exist if exists it would return 200.
        return guess

    # fallback: search the election page itself for a "geo" link
    soup = _get_soup(election_page_url)
    for a in soup.find_all("a", href=True):
        if "geo" in a["href"].lower():
            href = a["href"]
            return href if href.startswith("http") else f"{BASE}/{href.lstrip('/')}"

    raise LookupError(f"Could not find geography/conversion page linked from {election_page_url}")


def _extract_file_links(page_url: str, keywords: list[str]) -> Optional[str]:
    """
    Generic helper: scan a page's links, return the first href whose
    text or URL matches ALL given keywords (case-insensitive) and ends in .zip/.csv.
    """
    soup = _get_soup(page_url)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = (a.get_text() or "") + " " + href
        label_low = label.lower()
        if not (href.lower().endswith(".zip") or href.lower().endswith(".csv")):
            continue
        if all(k.lower() in label_low for k in keywords):
            return href if href.startswith("http") else f"{BASE}/{href.lstrip('/')}"
    return None


def resolve_files(year: int, election_type: str = "General") -> ElectionFileSet:
    """
    Main entry point for this layer.
    Returns an ElectionFileSet with statewide URLs for:
      - SOV data by SR precinct
      - Registration data by SR precinct
      - SR precinct boundary shapefile
    Any field left None means that file wasn't found / doesn't exist for that year.
    """
    election_page = _find_election_subpage(year, election_type)
    geo_page = _find_geo_conv_page(election_page, year, election_type)

    result = ElectionFileSet(year=year, election_type=election_type)

    # Election-data page: SOV + Registration, both "by srprec" (statewide)
    result.sov_srprec_url = _extract_file_links(election_page, ["sov", "srprec"]) #vote total Files
    result.registration_url = _extract_file_links(election_page, ["registration", "srprec"]) # registration Files

    # Geography page: statewide SRPREC shapefile
    result.shapefile_url = _extract_file_links(geo_page, ["srprec", "shp"]) # boundry shape file of a precient.

    return result


if __name__ == "__main__":
    # Quick manual test (requires outbound network access to statewidedatabase.org)
    fs = resolve_files(2022, "General")
    print(fs)
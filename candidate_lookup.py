"""
candidate_lookup.py
--------------------
Automatically fetches and parses California Secretary of State's official
"Certified List of Candidates" PDFs to get real candidate names for any
office + district + year + election_type -- no manual entry required.

Source confirmed real (found via search, not guessed):
  https://elections.cdn.sos.ca.gov/statewide-elections/{year}-{primary|general}/{office-slug}.pdf
  e.g. .../2022-general/state-senate.pdf

CAVEAT (like resolver.py for the vote data): this environment can't reach
elections.cdn.sos.ca.gov to test live, so the parsing regexes below are
built from the real PDF text samples I was able to inspect via search, but
should be verified against an actual downloaded PDF before relying on it in
production. If the layout doesn't match, adjust `_parse_district_section`.

One necessary heuristic: the SOV data columns (e.g. SENDEM01, SENDEM02) don't
carry candidate names -- they're just "party + ballot slot number". Based on
our verified SD10 2022 example (SENDEM01=Mei, SENDEM02=Wahab, alphabetical by
last name), we assume slot numbers are assigned alphabetically by last name
within each party. This should be spot-checked per office, since it's an
inferred convention, not something documented by the source.
"""

import os
import re
import json
import requests
import pdfplumber
import tempfile

BASE = "https://elections.cdn.sos.ca.gov/statewide-elections"
CACHE_DIR = os.environ.get("CANDIDATE_CACHE_DIR", "/tmp/candidate_cache")
HEADERS = {"User-Agent": "election-dashboard-backend/1.0"}

# Office code -> list of filename slugs to try, in order, since naming
# has drifted slightly across years (confirmed: "state-senate" for 2022).
OFFICE_SLUGS = {
    "SEN": ["state-senate", "senate"],
    "ASS": ["state-assembly", "assembly"],
    "CNG": ["us-house", "congress", "house"],
    "GOV": ["governor"],
    "LTG": ["lieutenant-governor"],
    "ATG": ["attorney-general"],
    "SOS": ["secretary-of-state"],
    "CON": ["controller"],
    "TRS": ["treasurer"],
    "INS": ["insurance-commissioner"],
    "SPI": ["superintendent-of-public-instruction"],
    "BOE": ["board-of-equalization"],
    "USS": ["us-senate"],
}

# Fallback filenames for years/elections that only publish ONE combined PDF
# covering every office (confirmed real for 2018: cert-list-candidates.pdf;
# confirmed real for 2014: certified-list.pdf / general-certified-list.pdf --
# naming has drifted at least 3 times across the years we've tested).
COMBINED_LIST_FILENAMES = [
    "cert-list-candidates.pdf",
    "cert-list.pdf",
    "certified-list.pdf",
    "general-certified-list.pdf",
]

# Separate write-in candidate list -- votes can exist for a write-in
# candidate who never appears in the regular certified list above.
# Naming has drifted across years too (confirmed real for all of these):
#   2014: certified-write-in-list.pdf
#   2018: cert-list-write-in-candidates.pdf
#   2026: certified-list-write-in.pdf
# These are always combined multi-office documents.
WRITE_IN_LIST_FILENAMES = [
    "cert-list-write-in-candidates.pdf",
    "certified-list-write-in.pdf",
    "cert-list-write-in.pdf",
    "certified-write-in-list.pdf",
]

# Section header text (as printed in these combined PDFs) per office --
# needed to disambiguate "District 15" (which could mean Senate, Assembly,
# or Congress) when all three appear in the same document.
OFFICE_SECTION_NAMES = {
    "SEN": ["STATE SENATOR", "STATE SENATE"],
    "ASS": ["STATE ASSEMBLY MEMBER", "STATE ASSEMBLYMEMBER", "MEMBER OF THE STATE ASSEMBLY"],
    "CNG": ["UNITED STATES REPRESENTATIVE", "U.S. REPRESENTATIVE", "MEMBER OF CONGRESS"],
    "BOE": ["BOARD OF EQUALIZATION"],
}

# Party text (as printed in the PDF) -> our 3-letter column code
PARTY_CODE = {
    "democratic": "DEM", "republican": "REP", "green": "GRN",
    "libertarian": "LIB", "american independent": "AIP",
    "peace and freedom": "NLP", "no party preference": "DCL",
}


def _cache_path(year: int, election_type: str, office: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{office}_{year}_{election_type.lower()}.json")


def _download_office_pdf(year: int, election_type: str, office: str) -> tuple[str, bool]:
    """
    Returns (local_pdf_path, is_combined).
    is_combined=True means this is a whole-election, all-offices document
    (older years), so parsing needs to disambiguate by office section name.
    """
    election_slug = f"{year}-{election_type.lower()}"

    for office_slug in OFFICE_SLUGS.get(office.upper(), []):
        url = f"{BASE}/{election_slug}/{office_slug}.pdf"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.write(resp.content)
            tmp.close()
            return tmp.name, False

    # Fallback: older years publish one combined PDF for every office
    for filename in COMBINED_LIST_FILENAMES:
        url = f"{BASE}/{election_slug}/{filename}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.write(resp.content)
            tmp.close()
            return tmp.name, True

    raise LookupError(
        f"Could not find any certified candidate PDF (per-office or combined) "
        f"for office={office}, {election_type} {year}. "
        f"Tried per-office slugs: {OFFICE_SLUGS.get(office.upper())}, "
        f"and combined filenames: {COMBINED_LIST_FILENAMES}"
    )


def _download_write_in_pdf(year: int, election_type: str) -> str | None:
    """
    Best-effort fetch of the write-in candidate list. Unlike the regular
    list, this is optional -- returns None instead of raising if not found,
    since most races have no write-in candidates worth chasing.
    """
    election_slug = f"{year}-{election_type.lower()}"
    for filename in WRITE_IN_LIST_FILENAMES:
        url = f"{BASE}/{election_slug}/{filename}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.write(resp.content)
            tmp.close()
            return tmp.name
    return None


def _parse_district_section(full_text: str, district: int | None, office: str = "", combined: bool = False) -> list[dict]:
    """
    Given the full extracted PDF text, find candidates for the given district
    (or the whole document, for statewide offices with no district).

    When `combined=True` (older years with one all-offices PDF), we must
    anchor on the office's section header too -- otherwise "District 15"
    could match Senate, Assembly, or Congress ambiguously.
    """
    if district is not None:
        # End boundary must be a DIFFERENT district number than the one we're
        # matching -- otherwise a same-district page-break header (e.g. a
        # running "Page 1 State Senate District 10" footer on a page that
        # continues District 10's own candidate list) gets mistaken for the
        # start of a new section and truncates real candidates on later pages.
        if combined:
            section_names = OFFICE_SECTION_NAMES.get(office.upper())
            if not section_names:
                raise LookupError(
                    f"No section-header mapping known for office={office} in "
                    f"combined-PDF format. Add it to OFFICE_SECTION_NAMES."
                )
            start_alt = "|".join(re.escape(n) for n in section_names)
            pattern = re.compile(
                rf"(?:{start_alt})\s+DISTRICT\s+{district}\b(.*?)(?=District\s+(?!{district}\b)\d+\b|\Z)",
                re.DOTALL | re.IGNORECASE,
            )
        else:
            pattern = re.compile(
                rf"District\s+{district}\b(.*?)(?=District\s+(?!{district}\b)\d+\b|\Z)",
                re.DOTALL | re.IGNORECASE,
            )
        match = pattern.search(full_text)
        if not match:
            raise LookupError(f"District {district} not found in candidate PDF.")
        block = match.group(1)
    else:
        block = full_text  # statewide office: whole document is one race

    # Split the text on party-name occurrences; the name immediately
    # preceding each party mention is our candidate for that entry.
    party_alt = "|".join(re.escape(p.title()) for p in PARTY_CODE)
    parts = re.split(rf"\b({party_alt})\b", block)

    candidates = []
    for i in range(1, len(parts), 2):
        party_text = parts[i]
        preceding = parts[i - 1]
        name_match = re.search(
            r"([A-Z][A-Za-z.'\-]+(?: [A-Za-z.'\-]+){0,4})\*?[ \t]*$",
            preceding.strip().split("\n")[-1].strip(),
        )
        if not name_match:
            continue
        name = name_match.group(1).strip().rstrip("*").strip()
        party = PARTY_CODE[party_text.lower()]
        if len(name.split()) >= 2:  # crude filter against noise matches
            candidates.append({"name": name, "party": party})

    # de-duplicate, preserving order
    seen = set()
    deduped = []
    for c in candidates:
        key = (c["name"], c["party"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _assign_column_codes(office: str, candidates: list[dict]) -> dict:
    """
    Maps parsed (name, party) candidates to our SOV column codes
    (e.g. SENDEM01, SENDEM02, SENREP01), using the alphabetical-by-last-name
    heuristic confirmed against SD10 2022 (Mei before Wahab -> 01, 02).
    """
    by_party: dict[str, list[str]] = {}
    for c in candidates:
        by_party.setdefault(c["party"], []).append(c["name"])

    result = {}
    for party, names in by_party.items():
        names_sorted = sorted(names, key=lambda n: n.split()[-1])  # by last name
        for i, name in enumerate(names_sorted, start=1):
            column = f"{office.upper()}{party}{i:02d}"
            result[column] = name
    return result


def get_candidate_names(
    office: str,
    district: int | None,
    year: int,
    election_type: str = "General",
) -> dict:
    """
    Main entry point. Returns {column_code: candidate_name}, fetched and
    parsed automatically -- cached locally after the first successful parse.
    """
    cache_file = _cache_path(year, election_type, office.upper())
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    else:
        cache = {}

    cache_key = str(district)
    if cache_key in cache:
        return cache[cache_key]

    pdf_path, is_combined = _download_office_pdf(year, election_type, office)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    finally:
        os.remove(pdf_path)

    candidates = _parse_district_section(full_text, district, office=office, combined=is_combined)

    # Best-effort: also check the separate write-in list. A candidate can
    # receive real votes (visible in the SOV data) without ever appearing
    # in the regular certified list -- e.g. we found ASSREP01 had 6,946
    # votes in AD15 2018 with no matching name in the regular list, which
    # is very likely a write-in.
    write_in_path = _download_write_in_pdf(year, election_type)
    if write_in_path:
        try:
            with pdfplumber.open(write_in_path) as pdf:
                write_in_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            write_in_candidates = _parse_district_section(
                write_in_text, district, office=office, combined=True
            )
            candidates = candidates + write_in_candidates
        except LookupError:
            pass  # no write-in candidates for this district -- fine, ignore
        finally:
            os.remove(write_in_path)
    if not candidates:
        raise LookupError(
            f"Parsed the PDF but found no candidates for office={office}, "
            f"district={district}, {election_type} {year}. "
            f"The PDF text layout may not match the parsing regex -- inspect "
            f"manually and adjust _parse_district_section()."
        )

    name_map = _assign_column_codes(office, candidates)

    cache[cache_key] = name_map
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    return name_map


if __name__ == "__main__":
    # Manual test (requires outbound network access to elections.cdn.sos.ca.gov)
    names = get_candidate_names(office="SEN", district=10, year=2022, election_type="General")
    print(names)
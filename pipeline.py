"""
pipeline.py
-----------
Core processing: given local file paths (already downloaded by fetcher.py),
produce a ready-to-render dataset for one office+district+year.

Handles both:
  - "by srprec" SOV files (newer/preferred, joins directly to shapefile)
  - "by svprec" SOV files (older style, needs the split-precinct crosswalk
    we built for SD10 2022 -- kept here so old files still work)

Office -> district column mapping. Statewide offices (Governor, etc.) have
no district filter; every precinct is "in" that race.
"""

import os
import re
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd

DISTRICT_COLUMN = {
    "SEN": "SDDIST",   # State Senate
    "ASS": "ADDIST",   # State Assembly
    "CNG": "CDDIST",   # US Congress
    "BOE": "BEDIST",   # Board of Equalization
    # Statewide offices below have no district column -- entire state is one race
    "GOV": None, "LTG": None, "ATG": None, "SOS": None, "CON": None,
    "TRS": None, "INS": None, "SPI": None, "USS": None, "USP": None,
}

COUNTY_FIPS_MAP = {
    # partial map for reference/testing; a full 58-county table should live
    # in a shared constants file in the real deployment
    1: 6001, 43: 6085,  # Alameda, Santa Clara (used in our SD10 build)
}


def _unzip(path: str) -> str:
    """Unzips to a temp dir (or returns the same path if it's not a zip)."""
    if not path or not path.lower().endswith(".zip"):
        return path
    out_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(path) as z:
        z.extractall(out_dir)
    return out_dir


def _find_file(dir_or_path: str, ext: str) -> str:
    if os.path.isfile(dir_or_path):
        return dir_or_path
    for f in os.listdir(dir_or_path):
        if f.lower().endswith(ext):
            return os.path.join(dir_or_path, f)
    raise FileNotFoundError(f"No .{ext} file found in {dir_or_path}")


def load_sov(sov_path: str) -> pd.DataFrame:
    extracted = _unzip(sov_path)
    csv_path = _find_file(extracted, "csv")
    return pd.read_csv(csv_path, dtype={"COUNTY": int})


def load_registration(reg_path: str | None) -> pd.DataFrame | None:
    if not reg_path:
        return None
    extracted = _unzip(reg_path)
    csv_path = _find_file(extracted, "csv")
    return pd.read_csv(csv_path, dtype={"COUNTY": int})


def load_shapefile(shp_path: str) -> gpd.GeoDataFrame:
    extracted = _unzip(shp_path)
    shp_file = _find_file(extracted, "shp")
    return gpd.read_file(shp_file)


def _base_precinct_id(svprec: str) -> str:
    """Strips trailing split-precinct letters, e.g. '420010A' -> '420010'."""
    return re.sub(r"[A-H]+$", "", str(svprec))


def filter_to_district(df: pd.DataFrame, office: str, district: int | None):
    """
    Returns (filtered_df, precinct_key_column, counties_touched).
    precinct_key_column is 'SRPREC' if present, else 'SVPREC'.
    """
    key_col = "SRPREC" if "SRPREC" in df.columns else "SVPREC"

    if district is not None:
        dist_col = DISTRICT_COLUMN.get(office.upper())
        if dist_col is None:
            raise ValueError(f"Office '{office}' has no district concept (statewide race).")
        if dist_col not in df.columns:
            raise ValueError(f"Expected district column '{dist_col}' not found in this year's data.")
        filtered = df[df[dist_col] == district].copy()
    else:
        filtered = df.copy()

    counties = sorted(filtered["COUNTY"].unique().tolist())
    return filtered, key_col, counties


def extract_candidate_columns(df: pd.DataFrame, office: str) -> list[str]:
    """
    Finds all vote columns for this office, e.g. SENDEM01, SENDEM02, SENREP01...
    """
    pattern = re.compile(rf"^{office.upper()}[A-Z]{{3}}\d{{2}}$")
    return [c for c in df.columns if pattern.match(c)]


def aggregate_candidate_totals(sov_df: pd.DataFrame, office: str, district: int | None, candidate_names: dict | None = None) -> dict:
    """
    Lightweight alternative to build_precinct_dataset() for callers that
    only need vote TOTALS per candidate -- no precinct geometry at all.

    This matters for memory: build_precinct_dataset() requires loading and
    joining a full statewide shapefile, which is the single heaviest thing
    this app does. The prediction feature calls this multiple times per
    request (once for the primary, once per historical lookback year) and
    never needs geometry at all -- confirmed this was causing out-of-memory
    crashes on a 512MB deployment. This function does the same
    filter+sum+coerce work as build_precinct_dataset() but skips the
    shapefile entirely, so callers that only need totals (like
    prediction.py) never have to load it.
    """
    candidate_names = candidate_names or {}
    filtered, _key_col, counties = filter_to_district(sov_df, office, district)
    cand_cols = extract_candidate_columns(filtered, office)
    if not cand_cols:
        raise ValueError(f"No candidate columns found for office '{office}' in this dataset.")

    for col in cand_cols:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(0)

    candidates = []
    for c in cand_cols:
        total = filtered[c].sum()
        candidates.append({
            "column": c,
            "name": candidate_names.get(c, c),
            "party": c[len(office):len(office) + 3],
            "total_votes": 0 if pd.isna(total) else int(total),
        })

    return {"office": office, "district": district, "counties": counties, "candidates": candidates}


def build_precinct_dataset(
    office: str,
    district: int | None,
    sov_df: pd.DataFrame,
    reg_df: pd.DataFrame | None,
    geo: gpd.GeoDataFrame,
    candidate_names: dict | None = None,
):
    """
    The main join+compute step. Returns a dict:
      {
        "candidates": [{"column": "SENDEM01", "name": "...", "party": "DEM", "total_votes": N}, ...],
        "geojson": {...}   # FeatureCollection, one feature per precinct
      }
    """
    candidate_names = candidate_names or {}

    filtered, key_col, counties = filter_to_district(sov_df, office, district)
    cand_cols = extract_candidate_columns(filtered, office)
    if not cand_cols:
        raise ValueError(f"No candidate columns found for office '{office}' in this dataset.")

    # Guard against any of these coming in as text (a single stray/blank
    # cell in the source CSV is enough to make pandas treat the WHOLE
    # column as strings instead of numbers, which then crashes any later
    # division). Force them numeric now; anything unparseable becomes 0
    # rather than silently breaking the request.
    numeric_cols = ["TOTREG", "TOTVOTE", "ABSVOTE", "PRCVOTE"] + cand_cols
    for col in numeric_cols:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(0)

    # --- Aggregate to base precinct if this is an SVPREC (split-precinct) file ---
    if key_col == "SVPREC":
        filtered["BASE_PREC"] = filtered["SVPREC"].apply(_base_precinct_id)
        agg_cols = ["TOTREG", "TOTVOTE", "ABSVOTE", "PRCVOTE"] + cand_cols
        agg_cols = [c for c in agg_cols if c in filtered.columns]
        grouped = filtered.groupby(["COUNTY", "BASE_PREC"])[agg_cols].sum().reset_index()
        grouped = grouped.rename(columns={"BASE_PREC": "SRPREC"})
    else:
        keep = ["COUNTY", "SRPREC", "TOTREG", "TOTVOTE", "ABSVOTE", "PRCVOTE"] + cand_cols
        keep = [c for c in keep if c in filtered.columns]
        grouped = filtered[keep].copy()

    grouped["SRPREC"] = grouped["SRPREC"].astype(str)

    # --- Merge in party-registration breakdown if a registration file was provided ---
    if reg_df is not None:
        reg_cols = [c for c in reg_df.columns if c.endswith("REG") and c != "TOTREG"]
        if reg_cols:
            reg_subset = reg_df[["COUNTY", "SRPREC"] + reg_cols].copy()
            reg_subset["SRPREC"] = reg_subset["SRPREC"].astype(str)
            for col in reg_cols:
                reg_subset[col] = pd.to_numeric(reg_subset[col], errors="coerce").fillna(0)
            grouped = grouped.merge(reg_subset, on=["COUNTY", "SRPREC"], how="left")

    # --- Join to geometry ---
    # IMPORTANT: county numbering isn't consistent across sources or even
    # across years of the same source:
    #   - The SOV file's COUNTY column (1-58, alphabetical) is NEVER the
    #     right join key against a shapefile -- confirmed by testing.
    #   - Some years' shapefiles expose a 'COUNTY' field holding a short
    #     FIPS-derived file code (e.g. "001" for Alameda) -- confirmed 2022.
    #   - Other years' shapefiles expose 'FIPS' directly, already the full
    #     state+county code (e.g. "06065") -- confirmed 2016.
    # To handle both without special-casing per year, everything gets
    # normalized to the FULL FIPS number (state*1000 + county), since the
    # SOV file's own FIPS column is already in that format.
    if "FIPS" in filtered.columns:
        fips_by_county = filtered.groupby("COUNTY")["FIPS"].first()
        sov_fips = grouped["COUNTY"].map(fips_by_county)
        grouped["GEOID_FULL"] = pd.to_numeric(sov_fips, errors="coerce")
    else:
        # fallback if FIPS isn't present in this year's file -- less reliable
        grouped["GEOID_FULL"] = pd.to_numeric(grouped["COUNTY"], errors="coerce")
    grouped = grouped.dropna(subset=["GEOID_FULL"])
    grouped["GEOID_FULL"] = grouped["GEOID_FULL"].astype(int)

    geo = geo.copy()
    geo["SRPREC"] = geo["SRPREC"].astype(str)
    if "FIPS" in geo.columns:
        # Already the full code (e.g. "06065") -- just parse it directly.
        geo["GEOID_FULL"] = pd.to_numeric(geo["FIPS"], errors="coerce")
    elif "COUNTY" in geo.columns:
        # Short file-code (e.g. "001") -- California's FIPS codes are
        # state 06 * 1000 + the county's own 3-digit code, so adding 6000
        # reconstructs the full code without needing a lookup table.
        geo["GEOID_FULL"] = pd.to_numeric(geo["COUNTY"], errors="coerce") + 6000
    else:
        raise ValueError(
            "Shapefile has neither 'FIPS' nor 'COUNTY' column -- can't "
            "determine county for the join. Inspect this file's schema."
        )
    geo = geo.dropna(subset=["GEOID_FULL"])
    geo["GEOID_FULL"] = geo["GEOID_FULL"].astype(int)

    merged = geo.merge(grouped, on=["GEOID_FULL", "SRPREC"], how="inner", suffixes=("_geo", ""))

    # --- Compute per-precinct metrics ---
    total_votes_col = merged[cand_cols].sum(axis=1)
    for c in cand_cols:
        merged[f"{c}_PCT"] = (merged[c] / total_votes_col.replace(0, float("nan")) * 100).round(1)
    merged["TURNOUT_PCT"] = (merged["TOTVOTE"] / merged["TOTREG"].replace(0, float("nan")) * 100).round(1)
    if "ABSVOTE" in merged.columns and "PRCVOTE" in merged.columns:
        cast = merged["ABSVOTE"] + merged["PRCVOTE"]
        merged["MAIL_PCT"] = (merged["ABSVOTE"] / cast.replace(0, float("nan")) * 100).round(1)

    candidates = []
    for c in cand_cols:
        total = merged[c].sum()
        candidates.append({
            "column": c,
            "name": candidate_names.get(c, c),
            "party": c[len(office):len(office) + 3],
            "total_votes": 0 if pd.isna(total) else int(total),
        })

    return {
        "office": office,
        "district": district,
        "counties": counties,
        "candidates": candidates,
        "geojson": merged.to_json(),  # GeoDataFrame -> GeoJSON string
    }


def build_county_level_dataset(office, district, live_data, county_results, county_geo):
    """
    The county-level equivalent of build_precinct_dataset(), used by the
    live-results pathway (live_resolver.py) instead of statewidedatabase.org
    data. Same general shape of output, but each GeoJSON feature is a whole
    COUNTY, not a precinct -- because that's the actual granularity the
    live results API provides.

    Args:
        office, district: as elsewhere
        live_data: the dict returned by live_resolver.fetch_district_results()
                   (district-wide totals, used for the `candidates` summary)
        county_results: dict of {county_slug: county_json_dict}, one entry
                        per county in the district (from
                        live_resolver.fetch_county_results(), called once
                        per county by the caller)
        county_geo: GeoDataFrame from county_geo.load_county_boundaries()
    """
    import live_resolver  # local import to avoid a hard dependency for the SWDB-only path

    candidates = []
    for c in live_data["candidates"]:
        candidates.append({
            "name": c["Name"],
            "party": c["Party"].upper()[:3],
            "total_votes": live_resolver._parse_votes(c["Votes"]),
            "incumbent": c.get("incumbent", False),
        })

    rows = []
    for county_slug, result in county_results.items():
        row = {"county_slug": county_slug}
        for c in result.get("candidates", []):
            row[c["Name"]] = live_resolver._parse_votes(c["Votes"])
        rows.append(row)

    county_votes = pd.DataFrame(rows)
    merged = county_geo.merge(county_votes, on="county_slug", how="inner")

    cand_names = [c["name"] for c in candidates]
    total_votes_col = merged[cand_names].sum(axis=1) if cand_names else 0
    for name in cand_names:
        if name in merged.columns:
            merged[f"{name}_PCT"] = (
                merged[name] / total_votes_col.replace(0, float("nan")) * 100
            ).round(1)

    return {
        "office": office,
        "district": district,
        "granularity": "county",  # frontend should check this to render differently
        "reporting_status": live_data.get("Reporting"),
        "candidates": candidates,
        "geojson": merged.to_json(),
    }
"""
api.py
------
The actual backend API the frontend calls.

GET /api/election-data?office=SEN&district=10&year=2022&election_type=General

Two pathways, tried in order:

  1. SWDB pathway (resolver.py / fetcher.py / pipeline.py / candidate_lookup.py)
     -> precinct-level data, works for years statewidedatabase.org has
     published (2014-2025 confirmed as of writing).

  2. Live-results pathway (live_resolver.py / county_geo.py) -> ONLY tried
     if #1 fails to find the year. County-level (not precinct-level) data
     straight from the CA Secretary of State's current-election system.
     This only ever has data for whatever the CURRENT election is (no
     year parameter) -- so it's really "the fallback for the most recent
     election, before SWDB has caught up," not a general historical source.

Run locally with:
  uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import json
import os
import secrets
from concurrent.futures import ThreadPoolExecutor

import resolver
import fetcher
import pipeline
import live_resolver
import county_geo
import prediction
import result_cache
from candidate_lookup import get_candidate_names

# --- Site-wide login (native browser username/password popup) ---
# Credentials come from environment variables, NOT hardcoded, so the real
# password never ends up committed to GitHub. Set DASHBOARD_USERNAME and
# DASHBOARD_PASSWORD wherever this app runs (locally: see README; on
# Render: Settings -> Environment). If unset, falls back to admin/changeme
# for local testing only -- change this before deploying for real.
security = HTTPBasic()
DASHBOARD_USERNAME = os.environ.get("DASHBOARD_USERNAME", "admin")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme")


def require_login(credentials: HTTPBasicCredentials = Depends(security)):
    # secrets.compare_digest instead of == to avoid leaking timing info
    # about how many characters matched -- a minor but standard precaution.
    user_ok = secrets.compare_digest(credentials.username, DASHBOARD_USERNAME)
    pass_ok = secrets.compare_digest(credentials.password, DASHBOARD_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},  # this header is what triggers the browser's login popup
        )
    return credentials.username


app = FastAPI(title="Election Precinct Dashboard API", dependencies=[Depends(require_login)])

# Same-origin now that the backend serves the frontend directly (see the
# static-file route below), so this is mainly a safety net for anyone
# testing the API from a different origin (e.g. a separate local dev server).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://votersai-outreach.com",
        "https://www.votersai-outreach.com",
        "http://localhost:5500",  # local frontend dev server
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/available-years")
def available_years(office: str, district: int):
    """
    Frontend calls this first to populate the year selector.
    NOTE: a real implementation should cache this list (it rarely changes)
    rather than re-scraping the site on every page load.
    """
    # Placeholder: the site doesn't expose a single "all years for district X" index,
    # so in production this list should be maintained/cached server-side after
    # the first successful resolve for each year, OR hardcoded from known election years.
    known_general_years = [2014, 2016, 2018, 2020, 2022, 2024]
    return {"office": office, "district": district, "years": known_general_years}


def _try_swdb_pathway(office, district, year, election_type):
    """Precinct-level pathway. Raises LookupError if this year isn't on SWDB yet."""
    file_set = resolver.resolve_files(year, election_type)  # raises LookupError if year not found

    if not file_set.sov_srprec_url or not file_set.shapefile_url:
        raise LookupError(
            f"Required files not found on statewidedatabase.org for "
            f"{election_type} {year}."
        )

    local_paths = fetcher.fetch_many({
        "sov": file_set.sov_srprec_url,
        "reg": file_set.registration_url,
        "shp": file_set.shapefile_url,
    })

    sov_df = pipeline.load_sov(local_paths["sov"])
    reg_df = pipeline.load_registration(local_paths["reg"])
    geo = pipeline.load_shapefile(local_paths["shp"])

    try:
        candidate_names = get_candidate_names(
            office=office, district=district, year=year, election_type=election_type
        )
    except LookupError:
        # Don't fail the whole request if names can't be found -- fall back
        # to raw column codes so the dashboard still works, just unlabeled.
        candidate_names = {}

    result = pipeline.build_precinct_dataset(
        office=office, district=district,
        sov_df=sov_df, reg_df=reg_df, geo=geo,
        candidate_names=candidate_names,
    )
    result["granularity"] = "precinct"
    return result


def _try_live_pathway(office, district, year, election_type):
    """
    County-level pathway, for whatever election is CURRENTLY live on the
    SOS site. Raises LookupError if there's no current election matching
    this request (including if the requested year doesn't match reality --
    we can't fully verify that here since the API takes no year parameter,
    so a mismatched year request may return whatever IS current; treat
    results from this pathway as "best available," not year-exact.)
    """
    live_data = live_resolver.fetch_district_results(office, district)
    geo = county_geo.load_county_boundaries()

    if district is None:
        county_results = {}  # statewide race -- no per-county breakdown needed
    else:
        all_slugs = geo["county_slug"].tolist()
        matching_slugs = live_resolver.fetch_county_list(office, district, all_slugs)
        county_results = {}
        for slug in matching_slugs:
            try:
                county_results[slug] = live_resolver.fetch_county_results(office, district, slug)
            except LookupError:
                continue  # skip a county that fails rather than failing the whole request

    if district is None or not county_results:
        # No county breakdown available/needed -- return district totals
        # only, geojson omitted (frontend should handle granularity=='none').
        return {
            "office": office,
            "district": district,
            "granularity": "none",
            "reporting_status": live_data.get("Reporting"),
            "candidates": [
                {
                    "name": c["Name"],
                    "party": c["Party"].upper()[:3],
                    "total_votes": live_resolver._parse_votes(c["Votes"]),
                    "incumbent": c.get("incumbent", False),
                }
                for c in live_data["candidates"]
            ],
            "geojson": {"type": "FeatureCollection", "features": []},
        }

    result = pipeline.build_county_level_dataset(office, district, live_data, county_results, geo)
    return result


@app.get("/api/election-data")
def election_data(
    office: str = Query(..., description="Office code, e.g. SEN, ASS, CNG, GOV"),
    district: int | None = Query(None, description="District number; omit for statewide offices"),
    year: int = Query(...),
    election_type: str = Query("General"),
):
    swdb_error = None
    try:
        result = _try_swdb_pathway(office, district, year, election_type)
    except LookupError as e:
        swdb_error = str(e)
        result = None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result is None:
        try:
            result = _try_live_pathway(office, district, year, election_type)
        except LookupError as live_error:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No data found for {office} district={district}, "
                    f"{election_type} {year}. "
                    f"Historical source (statewidedatabase.org): {swdb_error} "
                    f"Live source (SOS): {live_error}"
                ),
            )

    result["geojson"] = (
        json.loads(result["geojson"]) if isinstance(result["geojson"], str) else result["geojson"]
    )
    result["year"] = year
    result["election_type"] = election_type
    return result


def _try_swdb_totals_only(office, district, year, election_type):
    """
    Lightweight variant of _try_swdb_pathway for callers that only need
    candidate vote TOTALS, not precinct geometry -- used by prediction,
    which never needs a map. Skips downloading/loading the shapefile
    entirely (confirmed necessary: loading the full statewide shapefile
    multiple times in one request -- once for the primary, once per
    historical lookback year -- was causing out-of-memory crashes on a
    512MB deployment).

    Also caches the final computed result (see result_cache.py) -- confirmed
    that re-parsing a full statewide CSV with pandas on every request was
    real, measurable CPU time even when the raw file itself was already
    downloaded/cached by fetcher.py. This second cache layer skips that too.
    """
    cached = result_cache.get(office, district, year, election_type)
    if cached is not None:
        return cached

    file_set = resolver.resolve_files(year, election_type)  # raises LookupError if year not found
    if not file_set.sov_srprec_url:
        raise LookupError(f"SOV data not found on statewidedatabase.org for {election_type} {year}.")

    local_paths = fetcher.fetch_many({"sov": file_set.sov_srprec_url})
    sov_df = pipeline.load_sov(local_paths["sov"])

    try:
        candidate_names = get_candidate_names(
            office=office, district=district, year=year, election_type=election_type
        )
    except LookupError:
        candidate_names = {}

    result = pipeline.aggregate_candidate_totals(sov_df, office, district, candidate_names)
    result_cache.set(office, district, year, election_type, result)
    return result


@app.get("/api/predict-general")
def predict_general(
    office: str = Query(..., description="Office code, e.g. SEN, ASS, CNG, GOV"),
    district: int | None = Query(None, description="District number; omit for statewide offices"),
    year: int = Query(..., description="The GENERAL election year being projected"),
):
    """
    Projects the likely GENERAL election winner for `year`, based on that
    year's PRIMARY results. Uses the lightweight totals-only pathway
    throughout (see _try_swdb_totals_only) since prediction never needs
    precinct geometry, only vote counts.

    The primary fetch and the historical-lean lookback are run
    CONCURRENTLY (not one after the other) -- they don't depend on each
    other, and running them in parallel removes a whole sequential stage
    from the total wait time. Confirmed real: this was the difference
    between prediction requests taking ~3x a single Results lookup vs.
    closer to ~1x.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary_future = pool.submit(_try_swdb_totals_only, office, district, year, "Primary")
        historical_future = pool.submit(
            prediction.get_historical_district_lean, office, district, year, _try_swdb_totals_only
        )

        try:
            primary_result = primary_future.result()
        except LookupError:
            try:
                primary_result = _try_live_pathway(office, district, year, "Primary")
            except LookupError as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"No primary data found for {office} district={district}, {year}. {e}",
                )

        precomputed_historical = historical_future.result()

    try:
        prediction_result = prediction.predict_general_winner(
            office=office,
            district=district,
            target_year=year,
            primary_candidates=primary_result["candidates"],
            fetch_swdb_fn=_try_swdb_totals_only,
            precomputed_historical=precomputed_historical,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return prediction_result
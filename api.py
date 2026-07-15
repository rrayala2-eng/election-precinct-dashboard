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

import threading
import datetime
from contextlib import asynccontextmanager

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


def _prewarm_cache():
    """
    Downloads (but doesn't yet need to parse) the current election cycle's
    statewide vote-data files, PLUS the historical years Prediction's
    lookback needs -- runs once in the background when the server starts,
    so the FIRST real user request doesn't pay the full 30-50s cold-
    download cost no matter which tab (Results or Prediction) or which
    specific district/office they happen to try first.

    IMPORTANT: this covers Prediction's full download needs, not just the
    target year. Prediction also fetches up to 4 PAST years' GENERAL
    results for its historical-lean comparison (see
    prediction.HISTORICAL_LOOKBACK_YEARS) -- confirmed via real testing
    that skipping those left a real gap: a user hitting Prediction cold
    (never having used Results first) still paid the full download cost
    for those historical years even after the target-year pre-warm. This
    function now pre-warms ALL of them, reusing the same lookback list
    Prediction itself uses so the two never drift out of sync.

    Why one download covers every district/office: the source files are
    STATEWIDE (one CSV covers all of California for a given year+type),
    so warming it once benefits every possible request for that year, not
    just one specific district. Only the download step is pre-warmed here
    (via fetcher.py's file cache) -- the per-district parsing still happens
    on first real request, but that part was already shown to be the
    smaller portion of the total wait.

    Silently does nothing if a given year/type isn't published yet
    (e.g. this year's General hasn't happened) -- not an error, just
    nothing to warm yet.
    """
    current_year = datetime.datetime.now().year
    # CA statewide/legislative elections only happen in even years.
    target_year = current_year if current_year % 2 == 0 else current_year - 1

    def _warm_one(year, election_type):
        try:
            file_set = resolver.resolve_files(year, election_type)
            if file_set.sov_srprec_url:
                fetcher.fetch_many({"sov": file_set.sov_srprec_url})
        except LookupError:
            pass  # not published on statewidedatabase.org yet -- nothing to warm

    # Target year: both Primary and General (whichever Results/Prediction
    # would need depending on what the user searches).
    for election_type in ["Primary", "General"]:
        _warm_one(target_year, election_type)

    # Historical lookback years: Prediction always fetches GENERAL results
    # for these, regardless of which office/district is being predicted.
    for years_back in prediction.HISTORICAL_LOOKBACK_YEARS:
        hist_year = target_year - years_back
        if hist_year >= 2000:
            _warm_one(hist_year, "General")

    # Candidate name PDFs -- one per OFFICE (not per district), since the
    # PDF-level and text-level caches in candidate_lookup.py mean fetching
    # it once for ANY district of that office now benefits every other
    # district too. District 1 is just a representative example to trigger
    # the fetch -- statewide offices (GOV, LTG, etc.) use None since they
    # have no district at all. Best-effort: a missing/unparseable PDF for
    # one office/year should never block the rest of pre-warming.
    #
    # IMPORTANT: covers BOTH the target year AND every historical lookback
    # year, not just the target year. Confirmed via real testing this was
    # a genuine gap -- prediction's historical lookback always calls
    # get_candidate_names too (since it reuses _try_swdb_totals_only), so
    # skipping those years left every historical year's candidate PDF
    # completely cold on a real user's first Prediction request, even
    # after the target year itself was fully pre-warmed.
    years_to_prewarm_candidates = [(target_year, "Primary"), (target_year, "General")]
    for years_back in prediction.HISTORICAL_LOOKBACK_YEARS:
        hist_year = target_year - years_back
        if hist_year >= 2000:
            years_to_prewarm_candidates.append((hist_year, "General"))

    for office, district_col in pipeline.DISTRICT_COLUMN.items():
        representative_district = 1 if district_col else None
        for warm_year, election_type in years_to_prewarm_candidates:
            try:
                get_candidate_names(
                    office=office, district=representative_district,
                    year=warm_year, election_type=election_type,
                )
            except LookupError:
                pass  # this office/year/type genuinely has no PDF yet -- fine

    try:
        county_geo.load_county_boundaries()  # small, static, used by the live pathway
    except Exception:
        pass  # best-effort only -- a failed pre-warm should never block startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs in a background daemon thread so it never blocks the server from
    # starting up or responding to health checks -- if it's still running
    # when the first real request comes in, that request just proceeds
    # normally (falls through to the regular download-on-demand path).
    threading.Thread(target=_prewarm_cache, daemon=True).start()
    yield


app = FastAPI(
    title="Election Precinct Dashboard API",
    dependencies=[Depends(require_login)],
    lifespan=lifespan,
)

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

    IMPORTANT: the "office wasn't on the ballot this year" case (ValueError
    from aggregate_candidate_totals -- e.g. Governor doesn't run in an
    off-cycle year) is cached too, not just successes. CONFIRMED this was a
    real gap: offices on a 4-year cycle (BOE, TRS, SPI, GOV, etc.) hit this
    ValueError on 2 of their 4 historical lookback years EVERY time, and
    without caching the failure itself, each repeat request redid the full
    download+parse just to fail the same way again.
    """
    cached = result_cache.get(office, district, year, election_type)
    if cached is not None:
        if isinstance(cached, dict) and cached.get("_cached_error") == "ValueError":
            raise ValueError(cached["detail"])
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

    try:
        result = pipeline.aggregate_candidate_totals(sov_df, office, district, candidate_names)
    except ValueError as e:
        # Cache the failure itself (a small marker dict), so a repeat
        # request for this same off-cycle combination fails FAST instead
        # of re-downloading and re-parsing the whole file again.
        result_cache.set(office, district, year, election_type, {"_cached_error": "ValueError", "detail": str(e)})
        raise

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
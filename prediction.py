"""
prediction.py
--------------
Estimates which candidate is likely to win a GENERAL election, based on
that year's PRIMARY results. This is a transparent heuristic, not a
statistical forecast -- there is no polling, fundraising, or demographic
data behind this. It outputs a labeled "lean" (Toss-up/Lean/Likely/Safe)
with the reasoning shown, not a fake precise percentage.

Reuses the SAME data pathways as the rest of the app (api.py's
_try_swdb_pathway / _try_live_pathway) -- this module adds a new kind of
OUTPUT (a prediction), not a new data source.

THREE factors, all backed by real data this app already has access to:
  1. Primary vote share (combined by party for different-party matchups;
     direct head-to-head for same-party matchups)
  2. Incumbency
  3. Historical district lean (average party vote share in this district's
     last 1-2 general elections, when available)

HONEST LIMITATION, demonstrated with a real example (SD10 2022): in a
same-party top-two matchup, the primary vote LEADER does not reliably win
the general. Mei led the 2022 SD10 primary (33.1% to Wahab's 30.0%), but
Wahab won the general (53.7% to 46.3%). Same-party primary margins are
therefore treated as a WEAKER signal than different-party primary margins
-- see WEIGHTS below.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# How much each factor contributes to the blended score, on a 0-100 scale.
# Same-party matchups deliberately weight the primary-vote signal lower
# and historical lean higher, per the SD10 2022 case above.
WEIGHTS_DIFFERENT_PARTY = {"primary": 0.55, "historical": 0.25, "incumbency": 0.20}
WEIGHTS_SAME_PARTY = {"primary": 0.30, "historical": 0.35, "incumbency": 0.35}

# Recent even years to try when looking for past GENERAL results for this
# district. Not every district was on the ballot in every one of these
# (e.g. odd-numbered Senate seats skip most of them) -- each is tried and
# skipped silently via LookupError if that year/district has no data.
HISTORICAL_LOOKBACK_YEARS = [2, 4, 6, 8]


def _party_totals(candidates: list[dict]) -> dict[str, int]:
    totals = defaultdict(int)
    for c in candidates:
        totals[c["party"]] += c.get("total_votes", 0)
    return dict(totals)


def _top_two(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda c: c.get("total_votes", 0), reverse=True)[:2]


def _lean_label(margin: float) -> str:
    """Margin is the leader's edge over the second-place score, 0-100 scale."""
    if margin < 3:
        return "Toss-up"
    elif margin < 10:
        return "Lean"
    elif margin < 20:
        return "Likely"
    return "Safe"


def get_historical_district_lean(office: str, district: int | None, before_year: int, fetch_swdb_fn) -> dict | None:
    """
    Looks back through HISTORICAL_LOOKBACK_YEARS for past GENERAL results
    in this district, and returns average party vote share across
    whichever years actually had data. Returns None if nothing found
    (e.g. a brand-new district, or all lookback attempts failed).

    `fetch_swdb_fn` is injected (rather than imported directly) so this
    module doesn't need to know which pathway function to call -- api.py
    passes in its own _try_swdb_pathway.

    Fetches all lookback years CONCURRENTLY rather than one at a time.
    These are independent network/IO-bound lookups (each a different
    year's file) with no dependency on each other, so there's no reason to
    wait for one to finish before starting the next -- confirmed this was
    a real, meaningful chunk of first-run prediction latency (up to 4
    sequential downloads+parses, one per lookback year).
    """
    candidate_years = [before_year - yb for yb in HISTORICAL_LOOKBACK_YEARS if before_year - yb >= 2000]

    def _fetch_one(year):
        try:
            result = fetch_swdb_fn(office, district, year, "General")
        except (LookupError, ValueError):
            # LookupError = this year isn't published on the source site yet.
            # ValueError = this year WAS published, but this office simply
            # wasn't on the ballot that year (e.g. Governor is only elected
            # every 4 years, so an off-cycle year like 2024 genuinely has no
            # GOV columns in that year's data at all). Both cases mean the
            # same thing for our purposes: skip this year.
            return year, None
        return year, result

    party_share_samples = []
    years_used = []

    # NOTE: cap is tied to available memory, not just "more is always better".
    # Confirmed via a real production crash on a 512MB instance: running all
    # 4 lookback fetches fully concurrently meant up to 4 large statewide
    # datasets sat in memory AT THE SAME TIME, exceeding the memory limit.
    # Set to 3 (not the full 4) after upgrading to a 2GB instance, because
    # api.py now also runs the PRIMARY fetch concurrently with this whole
    # function -- so real peak concurrent heavy fetches is primary(1) +
    # historical(3) = 4 at once, not 5. If memory pressure returns, lower
    # this further before assuming something else is wrong.
    with ThreadPoolExecutor(max_workers=min(3, len(candidate_years) or 1)) as pool:
        for year, result in pool.map(_fetch_one, candidate_years):
            if result is None:
                continue
            candidates = result.get("candidates", [])
            totals = _party_totals(candidates)
            grand_total = sum(totals.values())
            if grand_total == 0:
                continue
            party_share_samples.append({p: v / grand_total * 100 for p, v in totals.items()})
            years_used.append(year)

    # Keep deterministic ordering (most recent first) regardless of which
    # thread happened to finish first -- matters for readability of the
    # "years_used" field shown in the UI, not for the math itself.
    years_used.sort(reverse=True)

    if not party_share_samples:
        return None

    all_parties = set()
    for sample in party_share_samples:
        all_parties.update(sample.keys())

    averaged = {
        party: sum(sample.get(party, 0) for sample in party_share_samples) / len(party_share_samples)
        for party in all_parties
    }
    return {"average_party_share": averaged, "years_used": years_used}


def predict_general_winner(
    office: str,
    district: int | None,
    target_year: int,
    primary_candidates: list[dict],
    fetch_swdb_fn,
    precomputed_historical: dict | None = "unset",
) -> dict:
    """
    Main entry point. `primary_candidates` is the `candidates` list already
    returned by the app's existing primary-results fetch (same shape used
    everywhere else: {name, party, total_votes, incumbent}).

    `precomputed_historical`: optional. The primary fetch and the historical
    lookback don't actually depend on each other, so a caller can fetch both
    CONCURRENTLY (e.g. api.py running them in parallel threads) and pass the
    already-computed historical result in here, skipping a redundant
    sequential fetch. If left as the default sentinel, this function
    computes it itself via fetch_swdb_fn (original, fully self-contained
    behavior, still correct -- just slower, since it becomes one more
    sequential stage instead of running in parallel with the primary fetch).
    """
    if len(primary_candidates) < 2:
        raise ValueError("Need at least 2 primary candidates to project a general matchup.")

    finalists = _top_two(primary_candidates)
    same_party = finalists[0]["party"] == finalists[1]["party"]
    weights = WEIGHTS_SAME_PARTY if same_party else WEIGHTS_DIFFERENT_PARTY

    # --- Factor 1: primary signal ---
    if same_party:
        # Direct head-to-head share between just the two finalists.
        f1_votes, f2_votes = finalists[0]["total_votes"], finalists[1]["total_votes"]
        total = f1_votes + f2_votes
        primary_score = {
            finalists[0]["name"]: (f1_votes / total * 100) if total else 50,
            finalists[1]["name"]: (f2_votes / total * 100) if total else 50,
        }
    else:
        # Combined party share across ALL primary candidates, not just the
        # two finalists -- captures same-party voters who didn't make the
        # runoff but will likely still back their party's finalist.
        party_totals = _party_totals(primary_candidates)
        grand_total = sum(party_totals.values())
        primary_score = {}
        for f in finalists:
            party_share = (party_totals.get(f["party"], 0) / grand_total * 100) if grand_total else 50
            primary_score[f["name"]] = party_share

    # --- Factor 2: incumbency ---
    incumbency_score = {}
    for f in finalists:
        incumbency_score[f["name"]] = 65 if f.get("incumbent") else 35

    # --- Factor 3: historical district lean ---
    if precomputed_historical != "unset":
        historical = precomputed_historical
    else:
        historical = get_historical_district_lean(office, district, target_year, fetch_swdb_fn)
    historical_score = {}
    if historical:
        avg_share = historical["average_party_share"]
        for f in finalists:
            historical_score[f["name"]] = avg_share.get(f["party"], 50)
    else:
        # No historical data available -- fall back to a neutral 50/50 so
        # this factor doesn't silently distort the blend toward whichever
        # candidate happens to be listed first.
        for f in finalists:
            historical_score[f["name"]] = 50

    # --- Blend ---
    blended = {}
    for f in finalists:
        name = f["name"]
        blended[name] = (
            primary_score[name] * weights["primary"]
            + historical_score[name] * weights["historical"]
            + incumbency_score[name] * weights["incumbency"]
        )

    leader_name = max(blended, key=blended.get)
    other_name = [f["name"] for f in finalists if f["name"] != leader_name][0]
    margin = blended[leader_name] - blended[other_name]

    return {
        "office": office,
        "district": district,
        "target_year": target_year,
        "matchup_type": "same_party" if same_party else "different_party",
        "finalists": [
            {
                "name": f["name"],
                "party": f["party"],
                "incumbent": f.get("incumbent", False),
                "primary_votes": f.get("total_votes", 0),
                "blended_score": round(blended[f["name"]], 1),
            }
            for f in finalists
        ],
        "projected_leader": leader_name,
        "margin": round(margin, 1),
        "lean_label": _lean_label(margin),
        "historical_data_used": historical["years_used"] if historical else [],
        "methodology_note": (
            "Heuristic estimate only -- based on primary vote share, incumbency, "
            "and historical district lean. No polling, fundraising, or demographic "
            "data is used. Same-party matchups are historically less predictable "
            "from primary results alone (e.g. SD10 2022: the primary vote leader "
            "lost the general election)."
        ),
    }
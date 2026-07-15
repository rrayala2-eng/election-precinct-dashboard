"""
result_cache.py
-----------------
Caches the FINAL computed result of a lookup (e.g. "SD10 2022 General
candidate totals"), not just the downloaded raw file. This matters because
fetcher.py's file-level cache avoids re-downloading, but a repeat request
was still re-reading and re-parsing a full statewide CSV with pandas every
time -- real CPU work, not network time. Caching the computed answer skips
that too, on top of the existing file cache.

Confirmed useful in practice: prediction.py's historical lookback makes up
to 5 of these calls per request; caching here turns repeat requests from
~15s (parsing 5 statewide files) down to near-instant.
"""

import hashlib
import json
import os

CACHE_DIR = os.environ.get("RESULT_CACHE_DIR", "/tmp/result_cache")


def _key_for(office, district, year, election_type) -> str:
    raw = f"{office}|{district}|{year}|{election_type}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def get(office, district, year, election_type):
    """Returns the cached result dict, or None if not cached."""
    path = os.path.join(CACHE_DIR, _key_for(office, district, year, election_type) + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupted/partial cache file -- treat as a miss, not a crash


def set(office, district, year, election_type, result: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key_for(office, district, year, election_type) + ".json")
    tmp_path = path + ".part"
    with open(tmp_path, "w") as f:
        json.dump(result, f)
    os.replace(tmp_path, path)  # atomic, same reasoning as fetcher.py's download-then-rename
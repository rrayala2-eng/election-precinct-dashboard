"""
fetcher.py
----------
Downloads a URL to a local cache directory. If the file was already
downloaded before, it's reused instead of re-fetching from the source site.

This is the layer that makes repeat requests (e.g. two different candidates
both asking about "Senate District 10, 2022") cheap after the first time.
"""

import hashlib
import os
import requests

CACHE_DIR = os.environ.get("ELECTION_CACHE_DIR", "/tmp/election_cache") #this is where downloaded files store.
HEADERS = {"User-Agent": "election-dashboard-backend/1.0"}


def _cache_path_for(url: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # filename = hash of URL + original extension, so it's stable and collision-free
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = url.split(".")[-1].split("?")[0]
    return os.path.join(CACHE_DIR, f"{h}.{ext}")


def fetch(url: str, force_refresh: bool = False) -> str:
    """
    Downloads `url` if not already cached; returns the local file path.
    """
    path = _cache_path_for(url)
    if os.path.exists(path) and not force_refresh:
        return path

    resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    resp.raise_for_status()
    tmp_path = path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    os.replace(tmp_path, path)
    return path


def fetch_many(urls: dict[str, str | None]) -> dict[str, str | None]:
    """
    Takes a dict like {"sov": url_or_none, "reg": url_or_none, "shp": url_or_none}
    Returns the same keys mapped to local file paths (or None if url was None).
    """
    out = {}
    for key, url in urls.items():
        out[key] = fetch(url) if url else None
    return out
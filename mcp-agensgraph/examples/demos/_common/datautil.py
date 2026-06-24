"""OpenFlights dataset (airports + routes) loader for the demos.

Downloads the public OpenFlights data (CC-BY-SA) once into ``.data/openflights/`` and
parses it into plain dicts ready to ingest as a graph. A tiny vendored sample is used
as an offline fallback if the download is unavailable.

Source: https://github.com/jpatokal/openflights/tree/master/data
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from typing import Optional

from . import config

_BASE = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/"
_FILES = {"airports": "airports.dat", "routes": "routes.dat"}

# Offline fallback: a handful of hubs + routes so demos still run without network.
_SAMPLE_AIRPORTS = """\
3797,"John F Kennedy Intl","New York","United States","JFK","KJFK",40.639,-73.778,13,-5,"A"
507,"London Heathrow","London","United Kingdom","LHR","EGLL",51.4706,-0.461941,83,0,"E"
2188,"Tokyo Haneda","Tokyo","Japan","HND","RJTT",35.5523,139.779,35,9,"N"
3830,"Chicago O'Hare Intl","Chicago","United States","ORD","KORD",41.9786,-87.9048,672,-6,"A"
340,"Frankfurt am Main","Frankfurt","Germany","FRA","EDDF",50.0333,8.5706,364,1,"E"
"""
_SAMPLE_ROUTES = """\
AA,24,JFK,3797,LHR,507,,0,777
BA,1355,LHR,507,JFK,3797,,0,744
JL,2987,HND,2188,JFK,3797,,0,77W
UA,5209,ORD,3830,LHR,507,,0,763
LH,3320,FRA,340,ORD,3830,,0,744
"""


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _fetch(name: str) -> str:
    """Return the raw .dat text for ``name``, caching the download under .data/."""
    cache_dir = config.DATA_DIR / "openflights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / _FILES[name]
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    try:
        with urllib.request.urlopen(_BASE + _FILES[name], timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
        cached.write_text(text, encoding="utf-8")
        return text
    except Exception:
        return _SAMPLE_AIRPORTS if name == "airports" else _SAMPLE_ROUTES


def _clean(value: str) -> Optional[str]:
    value = value.strip()
    return None if value in ("", "\\N") else value


def airports(limit: Optional[int] = None) -> list[dict]:
    """Airports with a valid 3-letter IATA code (the key routes reference)."""
    rows: list[dict] = []
    reader = csv.reader(io.StringIO(_fetch("airports")))
    for r in reader:
        if len(r) < 8:
            continue
        iata = _clean(r[4])
        if not iata or len(iata) != 3:
            continue
        try:
            lat, lon = float(r[6]), float(r[7])
        except ValueError:
            continue
        rows.append(
            {
                "iata": iata,
                "name": _clean(r[1]) or iata,
                "city": _clean(r[2]),
                "country": _clean(r[3]),
                "lat": lat,
                "lon": lon,
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def routes(limit: Optional[int] = None, valid_iata: Optional[set] = None) -> list[dict]:
    """Directed routes between airports (optionally filtered to a known airport set)."""
    rows: list[dict] = []
    reader = csv.reader(io.StringIO(_fetch("routes")))
    seen = set()
    for r in reader:
        if len(r) < 9:
            continue
        src, dst, airline = _clean(r[2]), _clean(r[4]), _clean(r[0])
        if not src or not dst or not airline:
            continue
        if valid_iata is not None and (src not in valid_iata or dst not in valid_iata):
            continue
        key = (airline, src, dst)
        if key in seen:
            continue
        seen.add(key)
        try:
            stops = int(r[7])
        except (ValueError, IndexError):
            stops = 0
        rows.append(
            {
                "airline": airline,
                "src": src,
                "dst": dst,
                "stops": stops,
                "equipment": _clean(r[8]) or "",
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows

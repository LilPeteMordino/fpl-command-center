"""API endpoints, filesystem paths, and shared constants for the FPL analytics engine."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "fpl_data.db"

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_STATIC_URL = f"{FPL_BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"
ENTRY_PICKS_URL_TEMPLATE = FPL_BASE_URL + "/entry/{manager_id}/event/{event}/picks/"
MANAGER_INFO_URL_TEMPLATE = FPL_BASE_URL + "/entry/{manager_id}/"
MANAGER_HISTORY_URL_TEMPLATE = FPL_BASE_URL + "/entry/{manager_id}/history/"
LEAGUE_STANDINGS_URL_TEMPLATE = FPL_BASE_URL + "/leagues-classic/{league_id}/standings/"
EVENT_LIVE_URL_TEMPLATE = FPL_BASE_URL + "/event/{event}/live/"

# Community-maintained mirror of FPL's own official data (github.com/vaastav/Fantasy-Premier-League).
# Used only as a fallback data source when fantasy.premierleague.com is unreachable -- verified
# (2026-08) that its CSV columns match the official API's field names exactly, so rows validate
# directly against our existing Player/Team pydantic models. It has no gameweeks/events file, so
# it can restore teams + players but not fixtures/gameweeks.
VAASTAV_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
VAASTAV_TEAMS_CSV_TEMPLATE = VAASTAV_BASE_URL + "/{season}/teams.csv"
VAASTAV_PLAYERS_RAW_CSV_TEMPLATE = VAASTAV_BASE_URL + "/{season}/players_raw.csv"

# Per-gameweek merged player stats for a finished season -- verified (2026-08) live against
# 2024-25/gws/gw1.csv: columns include name (FULL name, e.g. "Mohamed Salah" -- NOT the short
# web_name FPL's own API uses), team (full club name), element, minutes, bonus, bps,
# total_points. Powers src/replay.py's Historical Gameweek Replay Mode.
VAASTAV_GW_STATS_CSV_TEMPLATE = VAASTAV_BASE_URL + "/{season}/gws/gw{gw}.csv"

REQUEST_TIMEOUT_SECONDS = 15
REQUEST_HEADERS = {"User-Agent": "fpl-analytics-engine/1.0"}
# Pause between successive FPL requests within a single sync run, to stay polite to the API.
REQUEST_DELAY_SECONDS = 0.5

# Retry/backoff for rate limiting (HTTP 429) and transient network errors.
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

# FPL stores player price as an integer in tenths of a million, e.g. 100 == GBP 10.0m.
PRICE_DIVISOR = 10

"""FPL API client and ingestion of raw payloads into the local SQLite database."""
import csv
import io
import sqlite3
import time
from datetime import date
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config, database
from src.models import Fixture, Gameweek, Player, SquadPick, Team


class FPLAPIError(RuntimeError):
    """Raised when the FPL API returns an unexpected response."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(config.REQUEST_HEADERS)
    retry = Retry(
        total=config.MAX_RETRIES,
        backoff_factor=config.BACKOFF_FACTOR,
        status_forcelist=config.RETRY_STATUS_CODES,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FPLClient:
    """Thin wrapper around the official (undocumented) FPL API endpoints."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or _build_session()

    def _get_json(self, url: str):
        response = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise FPLAPIError(
                f"GET {url} failed with status {response.status_code}", status_code=response.status_code
            )
        return response.json()

    def get_bootstrap_static(self) -> dict:
        """Raw payload containing 'elements' (players), 'teams', and 'events' (gameweeks)."""
        return self._get_json(config.BOOTSTRAP_STATIC_URL)

    def get_fixtures(self) -> list:
        return self._get_json(config.FIXTURES_URL)

    def get_manager_picks(self, manager_id: int, event: int) -> dict:
        url = config.ENTRY_PICKS_URL_TEMPLATE.format(manager_id=manager_id, event=event)
        return self._get_json(url)

    def get_manager_info(self, manager_id: int) -> dict:
        url = config.MANAGER_INFO_URL_TEMPLATE.format(manager_id=manager_id)
        return self._get_json(url)

    def get_manager_history(self, manager_id: int) -> dict:
        """Includes a 'chips' list of {name, event, time} for every chip played this season."""
        url = config.MANAGER_HISTORY_URL_TEMPLATE.format(manager_id=manager_id)
        return self._get_json(url)

    def get_league_standings(self, league_id: int, page: int = 1) -> dict:
        """page_new_entries=1 and phase=1 mirror exactly what the official FPL frontend itself
        sends on this endpoint (phase=1 pins the full-season standings phase specifically, since
        some leagues track separate mid-season phases) -- included unconditionally rather than
        only for page>1, matching real API usage instead of this client's own minimal subset."""
        url = config.LEAGUE_STANDINGS_URL_TEMPLATE.format(league_id=league_id)
        url = f"{url}?page_new_entries=1&page_standings={page}&phase=1"
        return self._get_json(url)

    def get_event_live(self, event: int) -> dict:
        """Live/final per-player stats for one gameweek: {'elements': [{'id', 'stats': {...}},
        ...]}. Powers src/live_tracker.py's Matchday Live Rank & Auto-Sub Radar -- 'stats'
        includes minutes/bonus/bps plus the full scoring breakdown for that gameweek so far."""
        url = config.EVENT_LIVE_URL_TEMPLATE.format(event=event)
        return self._get_json(url)


# --- Ingestion: raw API payloads -> validated Pydantic models -> SQLite -----

def sync_teams_and_gameweeks(conn: sqlite3.Connection, bootstrap: dict) -> None:
    teams = [Team.model_validate(t) for t in bootstrap["teams"]]
    gameweeks = [Gameweek.model_validate(e) for e in bootstrap["events"]]

    conn.executemany(
        """
        INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away,
                            strength_defence_home, strength_defence_away)
        VALUES (:id, :name, :short_name, :strength_attack_home, :strength_attack_away,
                :strength_defence_home, :strength_defence_away)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            short_name=excluded.short_name,
            strength_attack_home=excluded.strength_attack_home,
            strength_attack_away=excluded.strength_attack_away,
            strength_defence_home=excluded.strength_defence_home,
            strength_defence_away=excluded.strength_defence_away
        """,
        [t.model_dump() for t in teams],
    )

    conn.executemany(
        """
        INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished)
        VALUES (:id, :name, :deadline_time, :is_current, :is_next, :finished)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            deadline_time=excluded.deadline_time,
            is_current=excluded.is_current,
            is_next=excluded.is_next,
            finished=excluded.finished
        """,
        [g.model_dump() for g in gameweeks],
    )
    conn.commit()


def sync_players(conn: sqlite3.Connection, bootstrap: dict) -> None:
    """Populate players. now_cost is kept as the raw FPL integer (100 == GBP 10.0m)."""
    players = [Player.model_validate(p) for p in bootstrap["elements"]]

    conn.executemany(
        """
        INSERT INTO players (id, web_name, team_id, element_type, now_cost, selected_by_percent,
                              form, total_points, ep_next, xg, xa, xgi, status, news,
                              xg_per_90, xa_per_90, saves_per_90, defensive_contribution_per_90, starts_per_90,
                              starts, chance_of_playing_next_round, penalties_order, corners_order,
                              transfers_in_event, transfers_out_event)
        VALUES (:id, :web_name, :team_id, :element_type, :now_cost, :selected_by_percent,
                :form, :total_points, :ep_next, :xg, :xa, :xgi, :status, :news,
                :xg_per_90, :xa_per_90, :saves_per_90, :defensive_contribution_per_90, :starts_per_90,
                :starts, :chance_of_playing_next_round, :penalties_order, :corners_order,
                :transfers_in_event, :transfers_out_event)
        ON CONFLICT(id) DO UPDATE SET
            web_name=excluded.web_name,
            team_id=excluded.team_id,
            element_type=excluded.element_type,
            now_cost=excluded.now_cost,
            selected_by_percent=excluded.selected_by_percent,
            form=excluded.form,
            total_points=excluded.total_points,
            ep_next=excluded.ep_next,
            xg=excluded.xg,
            xa=excluded.xa,
            xgi=excluded.xgi,
            status=excluded.status,
            news=excluded.news,
            xg_per_90=excluded.xg_per_90,
            xa_per_90=excluded.xa_per_90,
            saves_per_90=excluded.saves_per_90,
            defensive_contribution_per_90=excluded.defensive_contribution_per_90,
            starts_per_90=excluded.starts_per_90,
            starts=excluded.starts,
            chance_of_playing_next_round=excluded.chance_of_playing_next_round,
            penalties_order=excluded.penalties_order,
            corners_order=excluded.corners_order,
            transfers_in_event=excluded.transfers_in_event,
            transfers_out_event=excluded.transfers_out_event
        """,
        [p.model_dump() for p in players],
    )
    conn.commit()


def sync_fixtures(conn: sqlite3.Connection, fixtures_payload: list) -> None:
    fixtures = [Fixture.model_validate(f) for f in fixtures_payload]
    conn.executemany(
        """
        INSERT INTO fixtures (id, event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished)
        VALUES (:id, :event, :team_h, :team_a, :team_h_difficulty, :team_a_difficulty, :finished)
        ON CONFLICT(id) DO UPDATE SET
            event=excluded.event,
            team_h=excluded.team_h,
            team_a=excluded.team_a,
            team_h_difficulty=excluded.team_h_difficulty,
            team_a_difficulty=excluded.team_a_difficulty,
            finished=excluded.finished
        """,
        [f.model_dump() for f in fixtures],
    )
    conn.commit()


# --- Fallback ingestion: community mirror of official data (github.com/vaastav) ------------
#
# This is a *fallback* for the same raw team/player attributes sync_teams_and_gameweeks and
# sync_players already pull from the official API -- not an independent projections model.
# It exists for when fantasy.premierleague.com is temporarily unreachable. The repo publishes
# no gameweeks/events file, so it can restore teams + players but not fixtures/gameweeks
# (fixtures.event has a foreign key into gameweeks, so fixtures can't be inserted without it
# either) -- fixture-difficulty-dependent features stay stale until the official API recovers.

def current_fpl_season() -> str:
    """The vaastav repo's season folder naming, e.g. '2026-27'. The season rolls over at the
    July transfer window, well before a new season's GW1."""
    today = date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def fetch_vaastav_csv(url: str, session: Optional[requests.Session] = None) -> list:
    """GET a vaastav CSV file and parse it into a list of {column: value} dicts (all values are
    strings, same as any CSV -- the Pydantic models already coerce string numerics/empties).
    Public (not underscore-prefixed) since src.replay's historical-gameweek fetch also needs it,
    for the per-gameweek stats CSV this module itself has no other reason to fetch."""
    sess = session or requests.Session()
    response = sess.get(url, headers=config.REQUEST_HEADERS, timeout=config.REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise FPLAPIError(f"GET {url} failed with status {response.status_code}", status_code=response.status_code)
    return list(csv.DictReader(io.StringIO(response.text)))


def fetch_vaastav_teams(season: Optional[str] = None, session: Optional[requests.Session] = None) -> list:
    url = config.VAASTAV_TEAMS_CSV_TEMPLATE.format(season=season or current_fpl_season())
    return fetch_vaastav_csv(url, session)


def fetch_vaastav_players_raw(season: Optional[str] = None, session: Optional[requests.Session] = None) -> list:
    url = config.VAASTAV_PLAYERS_RAW_CSV_TEMPLATE.format(season=season or current_fpl_season())
    return fetch_vaastav_csv(url, session)


def fetch_vaastav_fixtures(season: Optional[str] = None, session: Optional[requests.Session] = None) -> list:
    """The full season's fixture list (see config.VAASTAV_FIXTURES_CSV_TEMPLATE) -- used by
    src/backtest.py to reconstruct a historical season's gameweek-by-gameweek fixture difficulty
    without any live API access. Not used by the live-data sync fallback path (sync_all_with_fallback
    intentionally leaves fixtures/gameweeks stale when the official API is unreachable -- see the
    module docstring above sync_teams_from_vaastav_fallback -- since a *current* season's fixture
    list still needs the official API's own event/gameweek ids to be meaningful for the live app;
    a finished historical season has no such ambiguity)."""
    url = config.VAASTAV_FIXTURES_CSV_TEMPLATE.format(season=season or current_fpl_season())
    return fetch_vaastav_csv(url, session)


def sync_teams_from_vaastav_fallback(conn: sqlite3.Connection, season: Optional[str] = None) -> int:
    """Populate the teams table from the vaastav fallback. Returns the number of rows ingested;
    rows that fail validation are skipped (logged by the caller), not fatal."""
    rows = fetch_vaastav_teams(season)
    teams = []
    for row in rows:
        try:
            teams.append(Team.model_validate(row))
        except Exception:
            continue
    if not teams:
        return 0

    conn.executemany(
        """
        INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away,
                            strength_defence_home, strength_defence_away)
        VALUES (:id, :name, :short_name, :strength_attack_home, :strength_attack_away,
                :strength_defence_home, :strength_defence_away)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            short_name=excluded.short_name,
            strength_attack_home=excluded.strength_attack_home,
            strength_attack_away=excluded.strength_attack_away,
            strength_defence_home=excluded.strength_defence_home,
            strength_defence_away=excluded.strength_defence_away
        """,
        [t.model_dump() for t in teams],
    )
    conn.commit()
    return len(teams)


def sync_players_from_vaastav_fallback(conn: sqlite3.Connection, season: Optional[str] = None) -> int:
    """Populate the players table from the vaastav fallback (requires teams to already be
    populated -- team_id is a foreign key). Returns the number of rows ingested."""
    rows = fetch_vaastav_players_raw(season)
    players = []
    for row in rows:
        try:
            players.append(Player.model_validate(row))
        except Exception:
            continue
    if not players:
        return 0

    conn.executemany(
        """
        INSERT INTO players (id, web_name, team_id, element_type, now_cost, selected_by_percent,
                              form, total_points, ep_next, xg, xa, xgi, status, news,
                              xg_per_90, xa_per_90, saves_per_90, defensive_contribution_per_90, starts_per_90,
                              starts, chance_of_playing_next_round, penalties_order, corners_order,
                              transfers_in_event, transfers_out_event)
        VALUES (:id, :web_name, :team_id, :element_type, :now_cost, :selected_by_percent,
                :form, :total_points, :ep_next, :xg, :xa, :xgi, :status, :news,
                :xg_per_90, :xa_per_90, :saves_per_90, :defensive_contribution_per_90, :starts_per_90,
                :starts, :chance_of_playing_next_round, :penalties_order, :corners_order,
                :transfers_in_event, :transfers_out_event)
        ON CONFLICT(id) DO UPDATE SET
            web_name=excluded.web_name,
            team_id=excluded.team_id,
            element_type=excluded.element_type,
            now_cost=excluded.now_cost,
            selected_by_percent=excluded.selected_by_percent,
            form=excluded.form,
            total_points=excluded.total_points,
            ep_next=excluded.ep_next,
            xg=excluded.xg,
            xa=excluded.xa,
            xgi=excluded.xgi,
            status=excluded.status,
            news=excluded.news,
            xg_per_90=excluded.xg_per_90,
            xa_per_90=excluded.xa_per_90,
            saves_per_90=excluded.saves_per_90,
            defensive_contribution_per_90=excluded.defensive_contribution_per_90,
            starts_per_90=excluded.starts_per_90,
            starts=excluded.starts,
            chance_of_playing_next_round=excluded.chance_of_playing_next_round,
            penalties_order=excluded.penalties_order,
            corners_order=excluded.corners_order,
            transfers_in_event=excluded.transfers_in_event,
            transfers_out_event=excluded.transfers_out_event
        """,
        [p.model_dump() for p in players],
    )
    conn.commit()
    return len(players)


def sync_manager_squad(conn: sqlite3.Connection, manager_id: int, event: int, picks_payload: dict) -> None:
    raw_picks = picks_payload.get("picks", [])
    picks = [
        SquadPick.model_validate({**pick, "manager_id": manager_id, "event": event})
        for pick in raw_picks
    ]
    conn.executemany(
        """
        INSERT INTO user_squad (manager_id, event, player_id, position_in_squad, is_captain, is_vice)
        VALUES (:manager_id, :event, :player_id, :position_in_squad, :is_captain, :is_vice)
        ON CONFLICT(manager_id, event, player_id) DO UPDATE SET
            position_in_squad=excluded.position_in_squad,
            is_captain=excluded.is_captain,
            is_vice=excluded.is_vice
        """,
        [p.model_dump() for p in picks],
    )
    conn.commit()


def get_user_squad(conn: sqlite3.Connection, client: FPLClient, manager_id: int, event: int) -> Optional[dict]:
    """Live manager picks for the given gameweek, falling back to the locally saved pre-season
    draft (see database.save_local_draft) if the API 404s -- which is exactly what FPL's
    picks endpoint returns for a manager before the GW1 deadline has passed.

    Returns None if there is neither a live squad nor a local draft to fall back to.
    """
    try:
        payload = client.get_manager_picks(manager_id, event)
    except FPLAPIError as exc:
        if exc.status_code != 404:
            raise
        draft = database.load_local_draft(conn)
        if draft is None:
            return None
        return {
            "source": "local_draft",
            "picks": draft["player_ids"],
            "bank": draft["bank_balance"],
            "captain_id": draft["captain_id"],
            "vice_id": draft["vice_id"],
        }

    raw_picks = payload.get("picks", [])
    return {
        "source": "live",
        "picks": [p["element"] for p in raw_picks],
        "bank": payload.get("entry_history", {}).get("bank", 0),
        "captain_id": next((p["element"] for p in raw_picks if p.get("is_captain")), None),
        "vice_id": next((p["element"] for p in raw_picks if p.get("is_vice_captain")), None),
    }


# --- Automated Team ID sync: FT ledger estimation + combined hydration ------------------------

FT_STARTING_BANK = 1  # every manager starts the season with 1 free transfer
FT_CAP = 5  # matches the current FPL banked-FT limit (transfer_planner.FREE_TRANSFER_CAP)
_UNLIMITED_TRANSFER_CHIP_NAMES = {"wildcard", "freehit"}  # these gameweeks don't touch the FT bank


def estimate_free_transfers(history_payload: Optional[dict]) -> int:
    """The public FPL API doesn't expose "free transfers currently banked" directly -- this
    reconstructs it by replaying the manager's own gameweek-by-gameweek transfer history from
    /entry/{id}/history/ ('current': [{event, event_transfers, ...}], 'chips': [{name, event}]):
    starting from FT_STARTING_BANK, each played gameweek either leaves the ledger untouched
    (GW1's free/unlimited window -- same "not a real transfer decision" treatment
    optimizer.is_before_gw1_deadline/transfer_planner's is_free_gw1_step already give it
    elsewhere in this codebase -- and Wildcard/Free Hit gameweeks, whose unlimited transfers
    likewise don't draw down or get replaced by the normal +1/gameweek accrual) or advances it by
    the standard rule (bank -= transfers made that gameweek, then +1, capped at FT_CAP). The
    result is the FT bank available for the *next* (upcoming) gameweek.

    Returns FT_STARTING_BANK if there's no history yet (pre-season, or a fetch failure already
    handled upstream) -- the correct starting value regardless.
    """
    if not history_payload:
        return FT_STARTING_BANK

    chip_events = {
        c.get("event") for c in history_payload.get("chips", []) if c.get("name") in _UNLIMITED_TRANSFER_CHIP_NAMES
    }
    current = sorted(history_payload.get("current", []), key=lambda e: e.get("event", 0))

    ft = FT_STARTING_BANK
    for entry in current:
        event = entry.get("event")
        if event == 1 or event in chip_events:
            continue  # unlimited-transfer gameweeks don't touch the FT ledger at all
        transfers_made = entry.get("event_transfers", 0) or 0
        ft = min(FT_CAP, max(0, ft - transfers_made) + 1)
    return ft


def fetch_squad_state(client: FPLClient, team_id: int, current_gw: int) -> dict:
    """One combined fetch hydrating everything the Automated FPL Team ID Sync needs: the active
    15 (+ captain/vice, in bench order), exact bank (ITB), team/manager identity, overall rank,
    season transfer count, chips played this season, every classic/H2H mini-league the manager
    has joined, and the estimated banked Free Transfers for the upcoming gameweek -- from the
    picks, entry-info, and history endpoints in one call, matching the spec's 3-query hydration.

    Named fetch_squad_state (previously fetch_user_team) since Sprint 1's "UserAccountSync"
    enhancement folds mini-league discovery and identity fields into the SAME hydration call
    rather than a second, near-duplicate fetch -- this codebase's established convention is one
    enhanced function over two overlapping ones (see e.g. is_vice_eligible's promotion history).
    Every existing caller (the header sync bar, the sidebar's "Sync My Squad") already needed
    this same 3-endpoint combination, so the rename is a drop-in replacement, not a new call site.

    Picks are sorted by their FPL "position" field (1-11 Starting XI, 12-15 bench, in that exact
    order) before extraction -- squad_ids is that full sorted order, and bench_order isolates just
    the last 4 for reference. Captain/vice are still read from the explicit is_captain/
    is_vice_captain flags (not inferred from `multiplier == 2`) since that's the more direct,
    unambiguous signal already proven correct here -- multiplier is 0 for an unused bench slot,
    1 for a plain starter, and 2 for the captain, so it carries the identical information, just
    one hop removed.

    Raises FPLAPIError on any of the three requests failing (e.g. a 404 on picks pre-GW1-deadline
    -- callers should catch this the same way the existing manual "Sync My Squad" flow already
    does elsewhere in this module, since it's the identical picks-endpoint behavior).
    """
    picks_payload = client.get_manager_picks(team_id, current_gw)
    info_payload = client.get_manager_info(team_id)
    history_payload = client.get_manager_history(team_id)

    raw_picks = sorted(picks_payload.get("picks", []), key=lambda p: p.get("position", 0))
    leagues_payload = info_payload.get("leagues", {})

    return {
        "squad_ids": [p["element"] for p in raw_picks],
        "bench_order": [p["element"] for p in raw_picks if p.get("position", 0) > 11],
        "captain_id": next((p["element"] for p in raw_picks if p.get("is_captain")), None),
        "vice_id": next((p["element"] for p in raw_picks if p.get("is_vice_captain")), None),
        "bank": picks_payload.get("entry_history", {}).get("bank", 0),
        "team_value": picks_payload.get("entry_history", {}).get("value", info_payload.get("last_deadline_value")),
        "team_name": info_payload.get("name"),
        "manager_name": " ".join(
            part for part in (info_payload.get("player_first_name"), info_payload.get("player_last_name")) if part
        ),
        "overall_rank": info_payload.get("summary_overall_rank"),
        "total_transfers": info_payload.get("last_deadline_total_transfers", 0),
        "chips_played": history_payload.get("chips", []),
        "free_transfers": estimate_free_transfers(history_payload),
        "leagues_classic": [
            {
                "id": l["id"], "name": l["name"],
                "entry_rank": l.get("entry_rank"), "entry_last_rank": l.get("entry_last_rank"),
            }
            for l in leagues_payload.get("classic", [])
        ],
        "leagues_h2h": [{"id": l["id"], "name": l["name"]} for l in leagues_payload.get("h2h", [])],
    }


# --- Mini-League Rival Tracker ------------------------------------------------------------------

def fetch_minileague_standings(client: FPLClient, league_id: int, page: int = 1) -> list:
    """Just the standings page (no picks) -- top rival Team IDs, entry names, player names, and
    total points, straight from get_league_standings. A thin, explicitly-named entry point for
    callers that only need the leaderboard itself (e.g. a manual "which league is this?" lookup)
    without paying for fetch_minileague_squads' per-manager picks fetch.

    Returns [{"manager_id", "entry_name", "player_name", "total_points", "rank"}, ...] in
    standings order. Raises FPLAPIError if the standings request itself fails.
    """
    standings = client.get_league_standings(league_id, page=page)
    return [
        {
            "manager_id": r["entry"], "entry_name": r.get("entry_name", ""),
            "player_name": r.get("player_name", ""), "total_points": r.get("total"),
            "rank": r.get("rank"),
        }
        for r in standings.get("standings", {}).get("results", [])
    ]


def fetch_minileague_squads(client: FPLClient, league_id: int, current_gw: int, max_managers: int = 20) -> list:
    """Up to max_managers rival squads from a classic mini-league's standings, each with their
    picks (captain, and which of the 15 they actually started) for current_gw -- the raw data
    src.live_tracker's Local Effective Ownership (LEO) computation and Shield/Differential
    classification are built on top of.

    Capped at max_managers to keep this a bounded, sequential fetch (each manager is its own HTTP
    request, same as the existing manual "Fetch League & Compare" flow in app.py) -- classic
    leagues can run to thousands of entrants, well beyond what's useful for a single glance.

    A manager whose picks fail to fetch (e.g. a 404 for a manager who hasn't set a squad yet this
    gameweek) is silently skipped rather than failing the whole call -- matching how the existing
    Rival Radar tab already treats per-manager picks failures. Raises FPLAPIError only if the
    standings request itself fails, since without that there's nothing to iterate at all.

    Returns [{"manager_id", "entry_name", "player_name", "rank", "squad_ids": set[int],
    "starting_ids": set[int] (the 11 with FPL "position" <= 11 -- i.e. actually started, not
    benched), "captain_id": Optional[int]}, ...].
    """
    standings = client.get_league_standings(league_id)
    results = standings.get("standings", {}).get("results", [])[:max_managers]

    squads = []
    for r in results:
        try:
            picks_payload = client.get_manager_picks(r["entry"], current_gw)
        except FPLAPIError:
            continue
        raw_picks = picks_payload.get("picks", [])
        squads.append({
            "manager_id": r["entry"],
            "entry_name": r.get("entry_name", ""),
            "player_name": r.get("player_name", ""),
            "rank": r.get("rank"),
            "squad_ids": {p["element"] for p in raw_picks},
            "starting_ids": {p["element"] for p in raw_picks if p.get("position", 0) <= 11},
            "captain_id": next((p["element"] for p in raw_picks if p.get("is_captain")), None),
        })
    return squads


# --- Daily Price Change Sentinel ---------------------------------------------------------------

PRICE_DROP_NET_TRANSFER_THRESHOLD = -100_000  # net (in - out) at/below this reads as drop risk
PRICE_RISE_NET_TRANSFER_THRESHOLD = 100_000  # net (in - out) at/above this reads as rise likelihood


def compute_price_change_alerts(
    conn: sqlite3.Connection,
    squad_ids: Optional[list] = None,
    drop_threshold: int = PRICE_DROP_NET_TRANSFER_THRESHOLD,
    rise_threshold: int = PRICE_RISE_NET_TRANSFER_THRESHOLD,
) -> list:
    """Players at risk of an overnight price move, from today's transfers_in_event/
    transfers_out_event momentum (see sync_players) -- a heuristic proxy for FPL's own
    undisclosed internal "value form" algorithm, not a guaranteed predictor: real price changes
    also depend on each player's individual ownership base and how many days they've already
    trended, neither of which the public API exposes. Restricted to squad_ids when given (the
    Command Center price alert pill), otherwise scans the full player pool.

    Returns [{"player_id", "web_name", "now_cost", "net_transfers", "direction": "rise"|"drop"}],
    sorted by |net_transfers| descending (biggest movers first).
    """
    query = "SELECT id, web_name, now_cost, transfers_in_event, transfers_out_event FROM players WHERE status != 'u'"
    params: list = []
    if squad_ids:
        placeholders = ",".join(["?"] * len(squad_ids))
        query += f" AND id IN ({placeholders})"
        params.extend(squad_ids)
    rows = conn.execute(query, params).fetchall()

    alerts = []
    for row in rows:
        net = (row["transfers_in_event"] or 0) - (row["transfers_out_event"] or 0)
        if net <= drop_threshold:
            direction = "drop"
        elif net >= rise_threshold:
            direction = "rise"
        else:
            continue
        alerts.append({
            "player_id": row["id"], "web_name": row["web_name"], "now_cost": row["now_cost"],
            "net_transfers": net, "direction": direction,
        })
    alerts.sort(key=lambda a: abs(a["net_transfers"]), reverse=True)
    return alerts


def sync_all(
    conn: sqlite3.Connection,
    manager_id: Optional[int] = None,
    event: Optional[int] = None,
    client: Optional[FPLClient] = None,
) -> None:
    """Fetch bootstrap-static, fixtures, and (optionally) a manager's squad, then upsert all of it."""
    client = client or FPLClient()

    bootstrap = client.get_bootstrap_static()
    sync_teams_and_gameweeks(conn, bootstrap)
    sync_players(conn, bootstrap)

    time.sleep(config.REQUEST_DELAY_SECONDS)
    fixtures_payload = client.get_fixtures()
    sync_fixtures(conn, fixtures_payload)

    if manager_id is not None and event is not None:
        time.sleep(config.REQUEST_DELAY_SECONDS)
        picks_payload = client.get_manager_picks(manager_id, event)
        sync_manager_squad(conn, manager_id, event, picks_payload)


def sync_all_with_fallback(
    conn: sqlite3.Connection,
    manager_id: Optional[int] = None,
    event: Optional[int] = None,
    client: Optional[FPLClient] = None,
) -> dict:
    """Like sync_all, but if the official API is unreachable for teams/players, falls back to
    the vaastav community mirror so the app can keep working in a degraded mode instead of
    failing outright (see the "Fallback ingestion" section above for what it can and can't cover).

    Returns a status dict callers should use to build any user-facing message from -- never
    assume success just because this returned without raising:
        {
            "source": "official" | "fallback" | "failed",
            "players_synced": int,
            "teams_synced": int,
            "fixtures_synced": bool,
            "gameweeks_synced": bool,
            "season": str | None,   # set only when the fallback path was used
            "error": str | None,
        }
    """
    client = client or FPLClient()
    status = {
        "source": "official",
        "players_synced": 0,
        "teams_synced": 0,
        "fixtures_synced": False,
        "gameweeks_synced": False,
        "season": None,
        "error": None,
    }

    try:
        bootstrap = client.get_bootstrap_static()
        sync_teams_and_gameweeks(conn, bootstrap)
        sync_players(conn, bootstrap)
        status["teams_synced"] = len(bootstrap.get("teams", []))
        status["players_synced"] = len(bootstrap.get("elements", []))
        status["gameweeks_synced"] = True
    except (FPLAPIError, requests.RequestException) as exc:
        status["source"] = "fallback"
        status["error"] = str(exc)
        try:
            season = current_fpl_season()
            status["season"] = season
            status["teams_synced"] = sync_teams_from_vaastav_fallback(conn, season)
            status["players_synced"] = sync_players_from_vaastav_fallback(conn, season)
            # gameweeks/fixtures aren't recoverable via this fallback -- no events source exists
            # there, and fixtures.event has a foreign key into gameweeks -- left unsynced.
        except (FPLAPIError, requests.RequestException) as fallback_exc:
            status["source"] = "failed"
            status["error"] = f"{status['error']} | fallback also failed: {fallback_exc}"
            return status

    if status["gameweeks_synced"]:
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            fixtures_payload = client.get_fixtures()
            sync_fixtures(conn, fixtures_payload)
            status["fixtures_synced"] = True
        except (FPLAPIError, requests.RequestException) as exc:
            status["error"] = (status["error"] + " | " if status["error"] else "") + f"fixtures sync failed: {exc}"

    if manager_id is not None and event is not None:
        try:
            time.sleep(config.REQUEST_DELAY_SECONDS)
            picks_payload = client.get_manager_picks(manager_id, event)
            sync_manager_squad(conn, manager_id, event, picks_payload)
        except FPLAPIError:
            pass  # e.g. a pre-season 404 -- not fatal to the overall sync

    return status

"""Historical Gameweek Replay Mode.

Exercises the live-tracking pipeline (live_tracker.get_live_gameweek_status -- provisional bonus
ranking, dynamic auto-subs, captain doubling/vice promotion) against a REAL finished gameweek
from a past season, instead of waiting for the current season's first live gameweek to find out
whether it actually works. Built entirely on top of the existing vaastav fallback machinery
(fpl_api.fetch_vaastav_csv) -- the historical stats are reshaped into the exact same
{"elements": [{"id", "stats": {...}}]} payload FPLClient.get_event_live returns, so this module's
job is sourcing and identity-matching the data, never re-implementing the engine it's testing.
The one necessary exception is get_live_gameweek_status's assume_all_fixtures_finished flag (see
its own docstring): a replayed gameweek NUMBER has no fixtures-table rows of its own and can
collide with the current season's real, unrelated fixtures for that same number, so this module
passes that flag rather than letting fixture-lookup silently read the wrong season's data.

Player-identity caveat: vaastav's per-gameweek 'element' ids are scoped to THAT season's player
list, a different id space from the current season's squad (FPL reassigns element ids every
season as players join/leave the game) -- so historical rows are matched to CURRENT local players
by (team, name) instead of by raw id passthrough. This is also why the matching here is its own
thing rather than reusing src.projections.match_row_to_player: that matcher is tuned for the
name *formats* real planner-tool CSV exports use (which measurably resemble FPL's own short
web_name -- verified live against 2024-25/gws/gw1.csv that its 'name' column is a FULL name like
'Mohamed Salah', not a web_name like 'Salah'; a difflib ratio against a short web_name lands
well under match_row_to_player's own 0.72 cutoff for most players, so reusing it here would
silently match almost nobody). Team-scoped substring containment (web_name-in-fullname, falling
back to surname-only for 'F.Surname'-style web_names like 'B.Fernandes') handles this format
gap reliably instead. A past-season player no longer in the game (retired, relegated permanently,
etc.) simply won't match and is dropped from the replay -- a real, permanent, and expected gap for
those specific players, not a bug to chase.
"""
import re
from typing import Optional

from src import config
from src.fpl_api import FPLAPIError, fetch_vaastav_csv
from src.live_tracker import get_live_gameweek_status


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.strip().lower())


def _load_players_by_team(conn) -> tuple:
    """(players_by_team_id, teams_by_normalized_name) -- players_by_team_id maps team_id -> list
    of {"id", "web_name"}; teams_by_normalized_name maps a normalized club name/short-name to
    that team's id, so vaastav's 'team' column (a full club name) can be resolved locally."""
    rows = conn.execute(
        "SELECT p.id, p.web_name, p.team_id, t.name AS team_name, t.short_name AS team_short "
        "FROM players p JOIN teams t ON t.id = p.team_id"
    ).fetchall()
    players_by_team: dict = {}
    teams_by_norm: dict = {}
    for r in rows:
        players_by_team.setdefault(r["team_id"], []).append({"id": r["id"], "web_name": r["web_name"]})
        teams_by_norm[_normalize(r["team_name"])] = r["team_id"]
        teams_by_norm[_normalize(r["team_short"])] = r["team_id"]
    return players_by_team, teams_by_norm


def _match_historical_player(full_name: str, team_id: Optional[int], players_by_team: dict) -> Optional[int]:
    """Team-scoped web_name-in-full-name containment (e.g. 'salah' in 'mohamedsalah'), falling
    back to a surname-only check (the part of web_name after the last '.') for 'F.Surname'-style
    short names where the whole web_name isn't a contiguous substring of the full name (e.g.
    web_name 'B.Fernandes' against full name 'Bruno Borges Fernandes' -- 'bfernandes' isn't
    contiguous in 'brunoborgesfernandes', but 'fernandes' alone is). Returns None (dropping this
    player from the replay) if team_id can't be resolved at all, or nothing in that team matches
    either pass -- see the module docstring for why that's an accepted, permanent gap."""
    if team_id is None:
        return None
    candidates = players_by_team.get(team_id, [])
    norm_full = _normalize(full_name)

    for p in candidates:
        norm_web = _normalize(p["web_name"])
        if norm_web and norm_web in norm_full:
            return p["id"]

    for p in candidates:
        surname = p["web_name"].rsplit(".", 1)[-1]
        norm_surname = _normalize(surname)
        if norm_surname and norm_surname in norm_full:
            return p["id"]

    return None


def fetch_historical_gw_live(conn, season: str, gw: int) -> dict:
    """A past season's real per-player gameweek stats (from vaastav's data/{season}/gws/gw{N}.csv),
    reshaped into the same {"elements": [{"id", "stats": {...}}]} payload
    FPLClient.get_event_live returns for a live/current gameweek -- 'id' is the CURRENT local
    player id (see module docstring for the team+name matching that gets there), not vaastav's
    own season-scoped element id.

    Raises FPLAPIError if the CSV itself can't be fetched (a season vaastav hasn't archived, or a
    gw number beyond how far that season actually played).
    """
    url = config.VAASTAV_GW_STATS_CSV_TEMPLATE.format(season=season, gw=gw)
    rows = fetch_vaastav_csv(url)

    players_by_team, teams_by_norm = _load_players_by_team(conn)

    elements = []
    unmatched = 0
    for row in rows:
        team_id = teams_by_norm.get(_normalize(row.get("team", "")))
        pid = _match_historical_player(row.get("name", ""), team_id, players_by_team)
        if pid is None:
            unmatched += 1
            continue
        elements.append({
            "id": pid,
            "stats": {
                "minutes": int(float(row.get("minutes") or 0)),
                "total_points": int(float(row.get("total_points") or 0)),
                "bonus": int(float(row.get("bonus") or 0)),
                "bps": int(float(row.get("bps") or 0)),
            },
        })
    return {"elements": elements, "_unmatched_count": unmatched, "_total_rows": len(rows)}


class _ReplayClient:
    """A stand-in for FPLClient that satisfies get_live_gameweek_status's only requirement of it
    (a .get_event_live(event_id) method) by returning a pre-fetched historical payload instead of
    making a real request -- get_live_gameweek_status has no idea it isn't talking to the real
    live API, which is the entire point (see module docstring)."""

    def __init__(self, live_payload: dict):
        self._live_payload = live_payload

    def get_event_live(self, event_id: int) -> dict:
        return self._live_payload


def replay_gameweek(
    conn,
    season: str,
    gw: int,
    squad_ids: list,
    starting_xi_ids: list,
    bench_ids: list,
    captain_id: Optional[int] = None,
    vice_id: Optional[int] = None,
) -> dict:
    """Runs a real finished gameweek from `season` through the exact same live-tracking engine
    the Live Gameweek Radar tab uses for an actually-live gameweek. `gw` is only used to look up
    the historical stats file -- it is NOT written into local fixtures/gameweeks state, so this
    never collides with (or needs) a real local gameweek id; the returned dict's "event_id" is a
    synthetic marker for display purposes only.

    Returns the same shape as live_tracker.get_live_gameweek_status, plus "match_rate": the
    fraction of that historical gameweek's rows that matched to a current local player (see
    fetch_historical_gw_live) -- a low match_rate is a signal the replay covers this squad
    incompletely (retired/transferred-out players), not that the engine itself is wrong.
    """
    live_payload = fetch_historical_gw_live(conn, season, gw)
    unmatched, total = live_payload.pop("_unmatched_count"), live_payload.pop("_total_rows")
    match_rate = round(1 - (unmatched / total), 3) if total else 0.0

    client = _ReplayClient(live_payload)
    result = get_live_gameweek_status(
        conn, client, squad_ids, gw, starting_xi_ids, bench_ids, captain_id, vice_id,
        assume_all_fixtures_finished=True,
    )
    result["match_rate"] = match_rate
    result["season"] = season
    return result

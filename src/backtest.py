"""Full-Season Walk-Forward Backtester.

Replays an entire finished FPL season (default: 38 gameweeks) gameweek-by-gameweek through the
REAL, already-built engine -- optimizer.solve_squad_with_captaincy/solve_starting_xi,
transfer_planner.plan_transfers, live_tracker.get_live_gameweek_status -- against historical data
from the vaastav/Fantasy-Premier-League community archive, instead of re-implementing a second,
parallel simulation of any of that logic. This module's own job is narrowly: source the historical
data, reshape it into the exact same SQLite schema/payload shapes those modules already expect,
and walk the calendar forward one real gameweek at a time.

Why this needs no cross-season player-identity matching (unlike src/replay.py): a walk-forward
backtest never needs to compare against the CURRENT season's live squad at all -- it reconstructs
one fully self-contained historical season and stays inside it start to finish. vaastav's
per-gameweek "element" id is already stable and unique *within* one season, so it's used directly
as the player id throughout this module's own synthetic, in-memory SQLite database. That database
is never the same connection/object as the app's real synced database (see database.get_connection)
-- it exists only for the duration of one simulate_season() call.

--- No-lookahead-bias design -----------------------------------------------------------------
The whole point of a walk-forward backtest is that gameweek N's SQUAD/TRANSFER/LINEUP decisions
may only see information a real manager would have had before that gameweek's deadline. This is
enforced structurally, not by convention:

  - Per-90 rates (xg_per_90, xa_per_90, saves_per_90, defensive_contribution_per_90,
    starts_per_90) fed into optimizer.calculate_positional_xp are rolling cumulative averages
    computed ONLY from gameweeks 1..N-1's real results (see _accumulate/_write_players_table).
    They are updated to include gameweek N's own result only AFTER that gameweek's decision has
    already been made and scored (_accumulate is called at the end of each loop iteration).
  - Price (now_cost) is gameweek N's own real, already-public price -- see _parse_gw_row's note
    on vaastav's "value" column -- never a later gameweek's.
  - Fixture DIFFICULTY for gameweek N (and any future gameweek) is legitimate pre-match public
    information (the season's fixture list, with FPL's own pre-assigned FDR ratings, is published
    before a ball is kicked) -- this is not lookahead. Fixture RESULTS are what must stay hidden,
    and are: optimizer.team_games_played(conn) counts only fixtures this module has itself marked
    `finished=1`, which _set_walk_forward_gameweek keeps strictly to gameweeks < N (see its
    docstring) regardless of what the real historical `finished` flag says for later gameweeks.
  - Gameweek 1 has no prior gameweeks at all, so every per-90 rate is exactly 0.0 for the entire
    pool -- this is a genuine, deliberate cold start, not a bug: it's exactly the input
    optimizer.calculate_positional_xp's own pre-season DEFCON/starts-rate fallback heuristics
    (see optimizer.py's "Pre-season DEFCON fallback" section) already exist to handle, so GW1
    reuses that machinery as-is rather than inventing a second cold-start model here.

--- Documented simplifications (things the vaastav archive simply doesn't carry) ---------------
  - status/chance_of_playing_next_round: no historical injury-news archive exists to replay, so
    every player is treated as fully available (status="a", chance=None) for the whole season.
    The Press Conference/Injury Flag Gatekeeper and STATUS_MINUTES_MULTIPLIER are therefore
    effectively inert during a backtest -- a real manager's rotation/injury dodges (or failures to
    dodge one) aren't modeled.
  - selected_by_percent: vaastav's per-gameweek CSV carries a raw "selected" COUNT, not a
    percentage of all managers (that denominator isn't published historically), so this is left at
    0.0 for every player rather than built into a noisy, arbitrary-scale proxy. Effect: the
    risk_lambda EO-shielding term in RISK_PROFILE_LAMBDA has no real ownership signal to act on
    during a backtest -- --risk-profile still threads through to the Starting XI/captaincy solve,
    but its practical influence is muted for this reason.
  - penalties_order/corners_order: not present in the archive; left as None for every player, so
    the Pre-Season Scouting override pathway (apply_preseason_adjustment) is simply never invoked
    here (it depends on a saved database.preseason_adjustments row, and this module's synthetic
    conn never writes one).
  - defensive_contribution_per_90: the raw "defensive_contribution" column only exists in
    vaastav's archive from the 2025-26 season onward (verified 2026-08: present in
    2025-26/gws/gw1.csv, absent from 2024-25's). Seasons before that simply accumulate 0.0 for it
    all season, which -- exactly as in the live app pre-season -- triggers
    calculate_positional_xp's own DEFCON fallback baseline rather than a true rate.
  - "Projected Global Finish": there is no real historical rank-distribution data available
    offline to derive an actual percentile from, so this is a clearly-labeled, season-agnostic
    rule-of-thumb band (see ROUGH_PERCENTILE_BANDS) -- illustrative only, never presented as a
    precise or season-specific rank estimate.
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src import chip_planner, config, database, optimizer, transfer_planner
from src.fpl_api import FPLAPIError, fetch_vaastav_csv, fetch_vaastav_fixtures, fetch_vaastav_teams
from src.live_tracker import get_live_gameweek_status
from src.optimizer import OptimizationError

MAX_GAMEWEEKS = 38
DEFAULT_FREE_TRANSFERS_AT_GW2 = 1  # standard real-FPL rule: GW1's squad build is free/unlimited
# and outside the FT economy entirely; the first genuine "transfer" gameweek (GW2) starts with 1.

POSITION_CODE_TO_ELEMENT_TYPE = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

RISK_PROFILE_ALIASES = {
    "ev": "Pure Mathematical EV",
    "pure": "Pure Mathematical EV",
    "balanced": "Balanced Rank Protection",
    "conservative": "Conservative Shield (High EO Lock)",
    "shield": "Conservative Shield (High EO Lock)",
}


def _resolve_risk_lambda(risk_profile: str) -> float:
    """Accepts either a short CLI-friendly alias (see RISK_PROFILE_ALIASES) or one of
    optimizer.RISK_PROFILE_LAMBDA's own exact label strings."""
    label = RISK_PROFILE_ALIASES.get(risk_profile.strip().lower(), risk_profile)
    if label not in optimizer.RISK_PROFILE_LAMBDA:
        valid = sorted(set(RISK_PROFILE_ALIASES) | set(optimizer.RISK_PROFILE_LAMBDA))
        raise ValueError(f"Unknown risk_profile {risk_profile!r} -- expected one of {valid}.")
    return optimizer.RISK_PROFILE_LAMBDA[label]


# --- Historical data loading (vaastav archive) -------------------------------------------------

def _load_teams(season: str) -> list:
    """Raw teams.csv rows -- same shape fpl_api.sync_teams_from_vaastav_fallback already
    validates via the Team pydantic model, used here as plain dicts since this module writes its
    own throwaway historical-season connection rather than the live app's synced database."""
    return fetch_vaastav_teams(season)


def _seed_teams(conn: sqlite3.Connection, team_rows: list) -> None:
    rows = []
    for row in team_rows:
        try:
            rows.append({
                "id": int(row["id"]), "name": row["name"], "short_name": row["short_name"],
                "strength_attack_home": int(row["strength_attack_home"]),
                "strength_attack_away": int(row["strength_attack_away"]),
                "strength_defence_home": int(row["strength_defence_home"]),
                "strength_defence_away": int(row["strength_defence_away"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    conn.executemany(
        """
        INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away,
                            strength_defence_home, strength_defence_away)
        VALUES (:id, :name, :short_name, :strength_attack_home, :strength_attack_away,
                :strength_defence_home, :strength_defence_away)
        """,
        rows,
    )
    conn.commit()


def _seed_gameweeks(conn: sqlite3.Connection, n_gameweeks: int) -> None:
    """Seeds every gameweek 1..n_gameweeks up front -- deadline_time is left null (only
    is_before_gw1_deadline reads it, and this module never exercises that code path: GW1's squad
    is built directly via solve_squad_with_captaincy, not plan_transfers' free-GW1 special case)."""
    rows = [
        {"id": gw, "name": f"GW{gw}", "deadline_time": None, "is_current": 0, "is_next": 0, "finished": 0}
        for gw in range(1, n_gameweeks + 1)
    ]
    conn.executemany(
        "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) "
        "VALUES (:id, :name, :deadline_time, :is_current, :is_next, :finished)",
        rows,
    )
    conn.commit()


def _load_fixtures_by_event(season: str) -> dict:
    """gameweek -> list of {team_h, team_a, team_h_difficulty, team_a_difficulty} from vaastav's
    fixtures.csv. Malformed/unscheduled rows (blank event, unparsable team ids) are skipped
    individually rather than failing the whole load -- matching the established
    try/except-continue convention elsewhere in this codebase's vaastav ingestion (see
    fpl_api.sync_players_from_vaastav_fallback)."""
    rows = fetch_vaastav_fixtures(season)
    by_event: dict = defaultdict(list)
    for row in rows:
        try:
            event = int(float(row.get("event") or ""))
        except (TypeError, ValueError):
            continue
        if not (1 <= event <= MAX_GAMEWEEKS):
            continue
        try:
            by_event[event].append({
                "team_h": int(row["team_h"]),
                "team_a": int(row["team_a"]),
                "team_h_difficulty": int(float(row["team_h_difficulty"])) if row.get("team_h_difficulty") else None,
                "team_a_difficulty": int(float(row["team_a_difficulty"])) if row.get("team_a_difficulty") else None,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return dict(by_event)


def _seed_fixtures(conn: sqlite3.Connection, fixtures_by_event: dict) -> None:
    """All fixtures inserted up front with finished=0 -- _set_walk_forward_gameweek is what
    actually advances `finished` week to week (see its docstring); this just needs every fixture's
    difficulty rating on hand so a future gameweek's fixture-difficulty projection is available the
    moment the walk-forward loop reaches it (that's legitimate pre-match public info, not
    lookahead -- see the module docstring)."""
    rows = []
    fixture_id = 1
    for event, fixtures in fixtures_by_event.items():
        for fx in fixtures:
            rows.append({
                "id": fixture_id, "event": event, "team_h": fx["team_h"], "team_a": fx["team_a"],
                "team_h_difficulty": fx["team_h_difficulty"], "team_a_difficulty": fx["team_a_difficulty"],
                "finished": 0,
            })
            fixture_id += 1
    conn.executemany(
        """
        INSERT INTO fixtures (id, event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished)
        VALUES (:id, :event, :team_h, :team_a, :team_h_difficulty, :team_a_difficulty, :finished)
        """,
        rows,
    )
    conn.commit()


def _set_walk_forward_gameweek(conn: sqlite3.Connection, gw: int) -> None:
    """Advances the synthetic connection's notion of "now" to gameweek `gw`: marks it (and only
    it) is_next, and marks every EARLIER gameweek's fixtures finished=1 (gw itself and everything
    later stays finished=0) -- this is the mechanism that keeps
    optimizer.team_games_played/calculate_baseline_xmins' starts-rate strictly to gw 1..gw-1's
    real results (see the module docstring's no-lookahead-bias section)."""
    conn.execute("UPDATE gameweeks SET is_next = 0, is_current = 0")
    conn.execute("UPDATE gameweeks SET is_next = 1 WHERE id = ?", (gw,))
    conn.execute("UPDATE fixtures SET finished = CASE WHEN event < ? THEN 1 ELSE 0 END", (gw,))
    conn.commit()


def _parse_gw_row(row: dict) -> Optional[dict]:
    """One vaastav gws/gw{N}.csv row -> a plain dict of everything this module needs from it, or
    None for a row that can't even be identified by player id. "value" is that player's price
    ENTERING this gameweek (the standard, industry-common granularity for a gameweek-level
    backtest -- real FPL price changes happen daily, finer than any publicly archived historical
    dataset tracks). "defensive_contribution" is absent from the CSV for seasons before 2025-26 --
    row.get(...) returning None there is deliberate, not an error (see module docstring)."""
    try:
        element_id = int(row["element"])
    except (KeyError, ValueError, TypeError):
        return None
    return {
        "id": element_id,
        "web_name": row.get("name") or f"Player {element_id}",
        "team_name": row.get("team", ""),
        "element_type": POSITION_CODE_TO_ELEMENT_TYPE.get((row.get("position") or "").upper()),
        "now_cost": int(float(row.get("value") or 0)),
        "minutes": int(float(row.get("minutes") or 0)),
        "starts": int(float(row.get("starts") or 0)),
        "xg": float(row.get("expected_goals") or 0.0),
        "xa": float(row.get("expected_assists") or 0.0),
        "saves": int(float(row.get("saves") or 0)),
        "defcon": float(row.get("defensive_contribution") or 0.0),
        "bonus": int(float(row.get("bonus") or 0)),
        "bps": int(float(row.get("bps") or 0)),
        "total_points": int(float(row.get("total_points") or 0)),
    }


def _historical_live_payload(parsed_rows: list) -> dict:
    """Reshapes one gameweek's parsed rows into the {"elements": [{"id", "stats": {...}}]} payload
    live_tracker.get_live_gameweek_status expects from FPLClient.get_event_live -- the same trick
    src.replay.fetch_historical_gw_live uses, just with no player-identity matching needed here
    (see module docstring)."""
    return {
        "elements": [
            {
                "id": p["id"],
                "stats": {
                    "minutes": p["minutes"], "total_points": p["total_points"],
                    "bonus": p["bonus"], "bps": p["bps"],
                },
            }
            for p in parsed_rows
        ],
    }


class _StaticLiveClient:
    """Feeds a pre-fetched historical payload to get_live_gameweek_status in place of a real
    FPLClient -- reimplemented locally rather than importing src.replay's private _ReplayClient,
    since this module has none of Replay Mode's cross-season identity-matching machinery to share
    (see the module docstring); this is a 3-line stub, not worth a cross-feature dependency."""

    def __init__(self, payload: dict):
        self._payload = payload

    def get_event_live(self, _event_id: int) -> dict:
        return self._payload


# --- Rolling per-90 accumulation (no-lookahead-bias core) --------------------------------------

@dataclass
class _PlayerAccum:
    minutes: int = 0
    starts: int = 0
    xg: float = 0.0
    xa: float = 0.0
    saves: int = 0
    defcon: float = 0.0
    total_points: int = 0


def _ingest_gw_meta(meta_by_id: dict, parsed_rows: list, team_id_by_name: dict) -> None:
    """Refreshes the running player registry's STATIC info (name/team/position/price) to this
    gameweek's own latest snapshot -- exactly what a real manager would also know at this point in
    time (called only with the gameweek about to be decided/scored, never a later one)."""
    for p in parsed_rows:
        if p["element_type"] is None:
            continue
        team_id = team_id_by_name.get(p["team_name"])
        if team_id is None:
            continue
        meta_by_id[p["id"]] = {
            "web_name": p["web_name"], "team_id": team_id,
            "element_type": p["element_type"], "now_cost": p["now_cost"],
        }


def _accumulate(accum_by_id: dict, parsed_rows: list) -> None:
    """Rolls one gameweek's ACTUAL results into the running cumulative totals -- callers must only
    invoke this AFTER that gameweek's decision has already been made and scored (see the module
    docstring's no-lookahead-bias section: this is the line that draws the boundary)."""
    for p in parsed_rows:
        acc = accum_by_id.setdefault(p["id"], _PlayerAccum())
        acc.minutes += p["minutes"]
        acc.starts += p["starts"]
        acc.xg += p["xg"]
        acc.xa += p["xa"]
        acc.saves += p["saves"]
        acc.defcon += p["defcon"]
        acc.total_points += p["total_points"]


def _write_players_table(conn: sqlite3.Connection, meta_by_id: dict, accum_by_id: dict) -> None:
    """Rewrites the synthetic `players` table for the CURRENT walk-forward step: static info at
    its latest known snapshot, every per-90 rate derived strictly from `accum_by_id` (cumulative
    real results through gw N-1 only -- see _accumulate). A player with zero accumulated minutes
    so far (including the entire pool at GW1) gets all-zero per-90 rates -- a deliberate cold
    start optimizer.calculate_positional_xp's own pre-season fallbacks already handle (see module
    docstring)."""
    conn.execute("DELETE FROM players")
    rows = []
    for pid, meta in meta_by_id.items():
        acc = accum_by_id.get(pid, _PlayerAccum())
        minutes = acc.minutes

        def per90(total: float) -> float:
            return round(total / minutes * 90, 3) if minutes > 0 else 0.0

        rows.append({
            "id": pid, "web_name": meta["web_name"], "team_id": meta["team_id"],
            "element_type": meta["element_type"], "now_cost": meta["now_cost"],
            "selected_by_percent": 0.0, "form": 0.0, "total_points": acc.total_points,
            "ep_next": None, "xg": round(acc.xg, 2), "xa": round(acc.xa, 2), "xgi": round(acc.xg + acc.xa, 2),
            "status": "a", "news": "",
            "xg_per_90": per90(acc.xg), "xa_per_90": per90(acc.xa), "saves_per_90": per90(acc.saves),
            "defensive_contribution_per_90": per90(acc.defcon), "starts_per_90": per90(acc.starts),
            "starts": acc.starts, "chance_of_playing_next_round": None,
            "penalties_order": None, "corners_order": None,
            "transfers_in_event": 0, "transfers_out_event": 0,
        })
    conn.executemany(
        """
        INSERT INTO players (
            id, web_name, team_id, element_type, now_cost, selected_by_percent, form, total_points,
            ep_next, xg, xa, xgi, status, news, xg_per_90, xa_per_90, saves_per_90,
            defensive_contribution_per_90, starts_per_90, starts, chance_of_playing_next_round,
            penalties_order, corners_order, transfers_in_event, transfers_out_event
        ) VALUES (
            :id, :web_name, :team_id, :element_type, :now_cost, :selected_by_percent, :form, :total_points,
            :ep_next, :xg, :xa, :xgi, :status, :news, :xg_per_90, :xa_per_90, :saves_per_90,
            :defensive_contribution_per_90, :starts_per_90, :starts, :chance_of_playing_next_round,
            :penalties_order, :corners_order, :transfers_in_event, :transfers_out_event
        )
        """,
        rows,
    )
    conn.commit()


# --- Per-gameweek scoring -----------------------------------------------------------------------

# Starter Security floors to try, strictest first -- see _solve_starting_xi_with_floor_fallback.
# Mirrors optimizer.STARTER_SECURITY_PROFILES' own tiers (plus a final "no floor" resort) rather
# than inventing separate numbers.
_STARTER_FLOOR_FALLBACK_CHAIN = (
    optimizer.DEFAULT_STARTER_XMINS_FLOOR,
    optimizer.STARTER_SECURITY_PROFILES["balanced"],
    optimizer.STARTER_SECURITY_PROFILES["aggressive"],
    None,
)


def _solve_starting_xi_with_floor_fallback(squad_rows: list, risk_lambda: float, verbose: bool = True):
    """optimizer.solve_starting_xi, trying _STARTER_FLOOR_FALLBACK_CHAIN's floors strictest-first
    and returning the first one that's feasible. Exists because DEFAULT_STARTER_XMINS_FLOOR (75)
    can make a GW1 Pre-Season Cold-Start squad's XI selection genuinely INFEASIBLE: with zero
    real-season starts data yet, calculate_baseline_xmins' pre-season fallback (price/ownership
    banded) can legitimately read most of a 15-man squad below 75 projected minutes at once --
    unlike app.py, where a human picks a looser Starter Security profile themselves if that
    happens, this module has no one to ask, so it degrades automatically instead of crashing the
    whole 38-gameweek run over one bad week. Prints a one-line note when it had to relax below the
    strict default, so this is visible rather than silent."""
    for floor in _STARTER_FLOOR_FALLBACK_CHAIN:
        try:
            result = optimizer.solve_starting_xi(squad_rows, min_starter_xmins=floor, risk_lambda=risk_lambda)
            if verbose and floor != _STARTER_FLOOR_FALLBACK_CHAIN[0]:
                floor_desc = f"{floor} xMins" if floor is not None else "no floor"
                print(f"[backtest] Starter Security floor relaxed to {floor_desc} for this gameweek's XI (thin minutes data).")
            return result
        except OptimizationError:
            continue
    raise OptimizationError("Starting XI solver infeasible even with no minutes-security floor at all.")


def _score_gameweek(conn, client, squad_ids: list, gw: int, pool_by_id: dict, risk_lambda: float) -> dict:
    """Solves the risk-profile-aware Starting XI/Captain/Vice for a fixed 15-man squad (see
    optimizer.solve_starting_xi/transfer_planner.captain_pick_for_gw), then scores it against that
    gameweek's real historical result via live_tracker.get_live_gameweek_status -- the exact same
    auto-sub simulation, captain doubling, and vice promotion the live app itself uses. Returns
    get_live_gameweek_status's own dict plus formation/captain_id/vice_id/starting_xi_ids/bench_ids
    for the diagnostics layer above.

    Uses _solve_starting_xi_with_floor_fallback (starting at optimizer.DEFAULT_STARTER_XMINS_FLOOR)
    as the Starting XI minutes-security floor -- unlike app.py's own call sites (which always
    thread a sidebar-derived min_starter_xmins through explicitly, and which a human can always
    loosen themselves if it goes infeasible), this module has no user-facing Starter Security
    control of its own, so it can't just rely on solve_starting_xi's bare default (None -- no floor
    at all) without silently re-exposing the walk-forward backtest to the same high
    rotational-risk exposure (heavy auto-sub reliance / bench points left behind) that motivated
    raising DEFAULT_STARTER_XMINS_FLOOR in the first place -- nor can it let one infeasible
    cold-start gameweek crash the entire 38-gameweek run."""
    squad_rows = [pool_by_id[pid] for pid in squad_ids if pid in pool_by_id]
    starting_xi, bench, formation = _solve_starting_xi_with_floor_fallback(squad_rows, risk_lambda)
    captain, vice = transfer_planner.captain_pick_for_gw(starting_xi)
    result = get_live_gameweek_status(
        conn, client, squad_ids, gw, [p.id for p in starting_xi], [p.id for p in bench],
        captain.id, vice.id, assume_all_fixtures_finished=True,
    )
    result["formation"] = formation
    result["captain_id"] = captain.id
    result["vice_id"] = vice.id
    result["starting_xi_ids"] = [p.id for p in starting_xi]
    result["bench_ids"] = [p.id for p in bench]
    return result


# --- Double-Chip Strategy (2-set rules: WC/FH/BB/TC once per half-season) ----------------------
# Implements the full 2-set chip rules (one Wildcard, Free Hit, Bench Boost, and Triple Captain
# usable in each of Set 1 [GW1-19] and Set 2 [GW20-38] -- see chip_planner.CHIP_SET_1_LAST_GW,
# reused here rather than re-declaring the boundary) so a full-season backtest reflects a real
# manager's actual chip-augmented ceiling, not just transfers/captaincy/bench management.
#
# Each chip's target gameweek(s) are decided via a hybrid of two legitimate (non-lookahead)
# sources:
#   (1) The fixture SCHEDULE for the whole half (classify_gameweek_density) -- which gameweeks are
#       Single/Blank/Double -- is fair game the moment that half begins, exactly like this module's
#       existing fixture-difficulty knowledge is (see the module docstring): real fixture lists are
#       published well in advance, only RESULTS must stay hidden.
#   (2) A live, walk-forward check of the CURRENT squad/captain state at each schedule-flagged
#       candidate gameweek, evaluated strictly in chronological order with no going back -- a
#       candidate gameweek that doesn't pan out (e.g. the pre-scanned Triple Captain target no
#       longer owns that player) is simply skipped, never retroactively "saved for later".
#
# Wildcard and Free Hit change WHICH players are used (see _rebuild_squad_for_target_gw); Triple
# Captain and Bench Boost only change how that gameweek's ALREADY-DECIDED squad/lineup is SCORED
# (applied as a post-hoc adjustment to that gameweek's gross points, never touching the transfer/
# squad decision itself) -- this mirrors the real rules exactly (TC/BB don't affect who you pick).

CHIP_CODES = ("WC", "FH", "BB", "TC")
CHIP_NAMES = {"WC": "Wildcard", "FH": "Free Hit", "BB": "Bench Boost", "TC": "Triple Captain"}

CHIP_SET1_START_GW = 2  # chips need a live, locked-in squad -- GW1 is free/unlimited squad building
CHIP_SET1_END_GW = chip_planner.CHIP_SET_1_LAST_GW  # 19
CHIP_SET2_START_GW = CHIP_SET1_END_GW + 1  # 20 -- CHIP_SET2_END_GW is just MAX_GAMEWEEKS (38)

TC_TALISMAN_MIN_COST = optimizer.GW1_CAPTAIN_MIN_COST  # reuse the existing >= GBP 10.0m "premium" bar
TC1_FAVORABLE_FDR = 2  # FDR at/below this counts as a "weak defense" fixture for the Set 1 SGW fallback

WC1_TRIGGER_WINDOW = (6, 8)  # "around GW6-8" per spec
WC1_ROTATION_RISK_TRIGGER = 3  # >= this many squad players below the floor triggers an early WC1
WC1_ROTATION_FLOOR = optimizer.STARTER_SECURITY_PROFILES["balanced"]  # "regular starters lose starting status"

BB1_POST_WC_OFFSET = 1  # Bench Boost 1 fires the gameweek immediately after Wildcard 1

BB2_LATE_WINDOW = (28, 38)  # "largest late-season DGW (GW34/37)" -- scanned across this window
BB2_MIN_DOUBLE_TARGET = 12  # spec's explicit bar; logged honestly even if narrowly missed (see _bb2_trigger)
WC2_LEAD_GWS = (2, 1)  # try 2 gameweeks before the Bench Boost 2 target first, then 1


@dataclass
class ChipActivation:
    chip: str  # one of CHIP_CODES
    gw: int
    half: int  # 1 or 2
    detail: str  # human-readable trigger reason, surfaced in the chip log


def classify_gameweek_density(fixtures_by_event: dict, all_team_ids: set) -> dict:
    """Gameweek Schedule Inspector: gw -> {"team_fixture_counts": {team_id: n}, "label": "SGW"|
    "BGW"|"DGW"}, built purely from the fixture SCHEDULE (never results) -- see this section's own
    docstring for why that's legitimate pre-match public information, not lookahead.

    label reflects the single most extreme case leaguewide that gameweek (a manager's own squad
    composition isn't known at schedule-scan time): "BGW" if any team plays 0 times, else "DGW" if
    any team plays >= 2 times, else "SGW". This is the shared building block every chip trigger
    below reads from (team_fixture_counts) rather than each reimplementing its own fixture count."""
    result = {}
    for gw, fixtures in fixtures_by_event.items():
        counts = {tid: 0 for tid in all_team_ids}
        for fx in fixtures:
            if fx["team_h"] in counts:
                counts[fx["team_h"]] += 1
            if fx["team_a"] in counts:
                counts[fx["team_a"]] += 1
        if any(c == 0 for c in counts.values()):
            label = "BGW"
        elif any(c >= 2 for c in counts.values()):
            label = "DGW"
        else:
            label = "SGW"
        result[gw] = {"team_fixture_counts": counts, "label": label}
    return result


def _rotation_risk_count(squad_rows: list) -> int:
    """How many current squad members project below the 'balanced' Starter Security floor this
    gameweek -- the Wildcard 1 rotation-risk trigger's own signal (see WC1_ROTATION_RISK_TRIGGER)."""
    return sum(1 for p in squad_rows if p.xmins < WC1_ROTATION_FLOOR)


def _half_has_dgw(gw_density: dict, start: int, end: int) -> bool:
    return any(gw_density.get(gw, {}).get("label") == "DGW" for gw in range(start, end + 1))


def _scan_bb2_target_gw(gw_density: dict) -> Optional[int]:
    """Schedule-only: the gameweek in BB2_LATE_WINDOW with the most teams playing twice -- "the
    largest late-season DGW"."""
    best_gw, best_count = None, 0
    for gw in range(BB2_LATE_WINDOW[0], BB2_LATE_WINDOW[1] + 1):
        density = gw_density.get(gw)
        if not density:
            continue
        double_count = sum(1 for c in density["team_fixture_counts"].values() if c >= 2)
        if double_count > best_count:
            best_gw, best_count = gw, double_count
    return best_gw


def _scan_fh_target_gw(gw_density: dict, start: int, end: int) -> Optional[int]:
    """Schedule-only: the gameweek in [start, end] with the most teams blanking (0 fixtures) --
    reused for both Free Hit 1 ("a blank gameweek") and Free Hit 2 ("the season's largest Blank
    Gameweek"). None if no team ever blanks in the window."""
    best_gw, best_count = None, 0
    for gw in range(start, end + 1):
        density = gw_density.get(gw)
        if not density:
            continue
        blank_count = sum(1 for c in density["team_fixture_counts"].values() if c == 0)
        if blank_count > best_count:
            best_gw, best_count = gw, blank_count
    return best_gw if best_count > 0 else None


def _tc1_trigger(gw: int, gw_density: dict, half_has_dgw: bool, captain) -> Optional[str]:
    """Triple Captain 1: if ANY Double Gameweek exists anywhere in Set 1 (schedule fact, checked
    once up front -- see _half_has_dgw), fires the first gameweek the actual captain's own club has
    one; otherwise falls back to the first gameweek the captain is a premium (>= TC_TALISMAN_
    MIN_COST) talisman with a home fixture at/below TC1_FAVORABLE_FDR."""
    team_count = gw_density.get(gw, {}).get("team_fixture_counts", {}).get(captain.team_id, 1)
    if half_has_dgw:
        if team_count >= 2:
            return f"{captain.web_name}'s club has a Double Gameweek this week."
        return None
    if captain.now_cost >= TC_TALISMAN_MIN_COST and bool(captain.is_home) and captain.fixture_difficulty <= TC1_FAVORABLE_FDR:
        return (
            f"{captain.web_name} (premium, {captain.cost_millions:.1f}m) has a favorable home "
            f"fixture (FDR {captain.fixture_difficulty:.0f}) -- no Double Gameweek anywhere in Set 1."
        )
    return None


def _wc1_trigger(gw: int, squad_rows: list) -> Optional[str]:
    """Wildcard 1: fires the first gameweek in WC1_TRIGGER_WINDOW where >= WC1_ROTATION_RISK_
    TRIGGER current squad members project below the Starter Security floor."""
    if not (WC1_TRIGGER_WINDOW[0] <= gw <= WC1_TRIGGER_WINDOW[1]):
        return None
    count = _rotation_risk_count(squad_rows)
    if count >= WC1_ROTATION_RISK_TRIGGER:
        return f"{count} current squad player(s) project below the Starter Security floor this week."
    return None


def _bb1_trigger(gw: int, squad_rows: list, wc1_activation: Optional[ChipActivation]) -> Optional[str]:
    """Bench Boost 1: fires in GW1 if every one of the initial 15 projects a secure Starting XI
    minutes floor, or in the gameweek immediately after Wildcard 1."""
    if wc1_activation is not None and gw == wc1_activation.gw + BB1_POST_WC_OFFSET:
        return f"First gameweek after the GW{wc1_activation.gw} Wildcard rebuild."
    if gw == 1 and squad_rows and all(p.xmins >= optimizer.DEFAULT_STARTER_XMINS_FLOOR for p in squad_rows):
        return "Every one of the initial 15 picks projects a secure Starting XI minutes floor."
    return None


def _fh_trigger(gw: int, target_gw: Optional[int], squad_rows: list, gw_density: dict) -> Optional[str]:
    """Free Hit (both halves): fires at the pre-scanned peak Blank Gameweek for this half (see
    _scan_fh_target_gw) only if the CURRENT squad actually has a player blanking that week --
    otherwise holds (spec: "or hold for emergency fixture congestion")."""
    if target_gw is None or gw != target_gw:
        return None
    counts = gw_density.get(gw, {}).get("team_fixture_counts", {})
    blanks = [p for p in squad_rows if counts.get(p.team_id, 1) == 0]
    if not blanks:
        return None
    names = ", ".join(p.web_name for p in blanks[:3])
    suffix = "..." if len(blanks) > 3 else ""
    return f"{len(blanks)} squad player(s) blank this Blank Gameweek ({names}{suffix})."


def _scan_tc2_target(conn, squad_ids: list, set2_event_ids: list, gw_density: dict) -> Optional[tuple]:
    """Pre-scans the WHOLE Set 2 window (GW20-38) for the best premium-talisman Double Gameweek
    opportunity, projected from the squad entering Set 2 -- a legitimate forward xP PROJECTION
    (the same mechanism transfer_planner.fetch_multi_gw_projections already uses for horizon
    planning elsewhere), never a use of future RESULTS. Returns (event_id, player_id, detail) for
    the single best candidate, or None if Set 2 has no Double Gameweek at all."""
    dgw_events = [gw for gw in set2_event_ids if gw_density.get(gw, {}).get("label") == "DGW"]
    if not dgw_events:
        return None
    projections = transfer_planner.fetch_multi_gw_projections(conn, dgw_events)
    best = None
    for pid in squad_ids:
        proj = projections.get(pid)
        if not proj:
            continue
        for gw in dgw_events:
            if gw_density[gw]["team_fixture_counts"].get(proj["team_id"], 1) < 2:
                continue
            xp = proj["gw_xp"].get(gw, 0.0)
            is_premium = proj["now_cost"] >= TC_TALISMAN_MIN_COST
            score = xp + (1000.0 if is_premium else 0.0)  # premium strongly preferred, xP tie-breaks
            if best is None or score > best[0]:
                best = (score, gw, pid, proj["web_name"], xp, is_premium)
    if best is None:
        return None
    _score, gw, pid, name, xp, is_premium = best
    premium_note = " (premium)" if is_premium else ""
    detail = f"{name}{premium_note} projects {xp:.1f} xP in a Double Gameweek (GW{gw})."
    return gw, pid, detail


def _tc2_trigger(gw: int, set2_plan: dict, current_squad_ids: list) -> Optional[str]:
    if set2_plan.get("tc2_gw") != gw:
        return None
    if set2_plan.get("tc2_player_id") not in current_squad_ids:
        return None  # the originally-scanned target has since left the squad -- no retroactive re-target
    return set2_plan.get("tc2_detail")


def _bb2_trigger(gw: int, set2_plan: dict, squad_rows: list, gw_density: dict) -> Optional[str]:
    if set2_plan.get("bb2_gw") != gw:
        return None
    counts = gw_density.get(gw, {}).get("team_fixture_counts", {})
    double_count = sum(1 for p in squad_rows if counts.get(p.team_id, 1) >= 2)
    bar_note = "clears" if double_count >= BB2_MIN_DOUBLE_TARGET else "falls short of but is still this half's best chance at"
    return (
        f"Largest late-season Double Gameweek: {double_count}/15 squad players have a double "
        f"fixture this week ({bar_note} the {BB2_MIN_DOUBLE_TARGET}-player target)."
    )


def _wc2_trigger(gw: int, set2_plan: dict) -> Optional[str]:
    if set2_plan.get("wc2_gw") != gw:
        return None
    bb2_gw = set2_plan.get("bb2_gw")
    return f"Building double-fixture squad depth ahead of the planned GW{bb2_gw} Bench Boost."


def _build_set2_plan(conn, squad_ids: list, gw_density: dict) -> dict:
    """Runs ONCE, the moment the walk-forward reaches GW20 -- pre-scans Set 2's schedule (and the
    squad entering Set 2, for the Triple Captain 2 talisman projection) for each chip's target
    gameweek. See _scan_tc2_target/_scan_bb2_target_gw/_scan_fh_target_gw/_wc2_trigger."""
    set2_event_ids = list(range(CHIP_SET2_START_GW, MAX_GAMEWEEKS + 1))
    bb2_gw = _scan_bb2_target_gw(gw_density)
    wc2_gw = None
    if bb2_gw is not None:
        for lead in WC2_LEAD_GWS:
            candidate = bb2_gw - lead
            if CHIP_SET2_START_GW <= candidate < bb2_gw:
                wc2_gw = candidate
                break
    fh2_gw = _scan_fh_target_gw(gw_density, CHIP_SET2_START_GW, MAX_GAMEWEEKS)
    tc2 = _scan_tc2_target(conn, squad_ids, set2_event_ids, gw_density)
    return {
        "bb2_gw": bb2_gw,
        "wc2_gw": wc2_gw,
        "fh2_gw": fh2_gw,
        "tc2_gw": tc2[0] if tc2 else None,
        "tc2_player_id": tc2[1] if tc2 else None,
        "tc2_detail": tc2[2] if tc2 else None,
    }


def _rebuild_squad_for_target_gw(conn, budget_units: int, risk_lambda: float, target_gw: Optional[int] = None) -> list:
    """A fresh optimal 15-man squad (optimizer.solve_squad_with_captaincy) -- scored against
    `target_gw`'s own fixtures when given (used by Wildcard 2 to deliberately build for its paired
    Bench Boost 2 double-gameweek target, per the spec's "build optimal 15-man double-fixture
    depth"), or the CURRENT gameweek's fixtures otherwise (Wildcard 1, and a standalone Free Hit --
    same mechanism GW1's own squad build already uses)."""
    if target_gw is None:
        pool = optimizer.fetch_players(conn)
    else:
        projections = transfer_planner.fetch_multi_gw_projections(conn, [target_gw])
        pool = [transfer_planner.player_row_for_gw(proj, target_gw) for proj in projections.values()]
    return optimizer.solve_squad_with_captaincy(pool, budget=budget_units, risk_lambda=risk_lambda)


def _apply_chip_scoring(managed: dict, scoring_squad_ids: list, chip: Optional[str]) -> float:
    """The gross points for this gameweek, adjusted for an active TC/BB chip -- WC/FH don't need
    an adjustment here (their effect is already baked into `managed` via which squad/lineup was
    scored), so this is a no-op for those. TC: one more copy of the captain's raw live points
    (2x -> 3x). BB: the whole 15-man squad's actual points count, not just the post-auto-sub
    effective XI -- summing every squad member's own live_points already equals "Starting XI
    (with auto-subs) + bench" once the bench counts too, so no separate auto-sub bypass is needed."""
    if chip == "TC":
        return managed["provisional_total_points"] + managed["captain_doubled_points"]
    if chip == "BB":
        all_points = sum(
            status.live_points for pid, status in managed["player_status"].items() if pid in scoring_squad_ids
        )
        return all_points + managed["captain_doubled_points"]
    return managed["provisional_total_points"]


# --- Season report -------------------------------------------------------------------------------

@dataclass
class GameweekResult:
    gw: int
    managed_points: float  # gross, before any hit deduction
    managed_hit_cost: int
    managed_net_points: float
    transfers_in: list  # web_names
    transfers_out: list  # web_names
    static_points: float  # the set-and-forget benchmark squad's gross points this gameweek
    auto_sub_moves: int
    bench_points_left_behind: float
    captain_web_name: str
    captain_points: int  # the captain's own (undoubled) live points that gameweek
    captain_doubled_points: float  # the extra copy actually earned (0 if the armband was wasted)
    best_possible_captain_points: float  # what that extra copy WOULD have been, captaining hindsight's actual top scorer in the (post-auto-sub) XI
    formation: str
    chip: Optional[str] = None  # CHIP_CODES entry active this gameweek, if any -- see _apply_chip_scoring
    free_transfers_before: int = 0  # free_transfers going INTO this gameweek's decision -- lets a
    # caller verify a Wildcard/hold rolls this forward (+1, capped) rather than consuming it


# Deliberately generic/season-agnostic rule-of-thumb bands -- NOT derived from any specific
# season's real rank-distribution data (this offline tool has no access to that), so every label
# says "illustrative" and this is never presented as a precise or season-specific figure. See the
# module docstring's "Documented simplifications" section.
ROUGH_PERCENTILE_BANDS = [
    (2300, "Illustrative only -- roughly 'top 1,000' territory in a typical season"),
    (2150, "Illustrative only -- roughly 'top 10k-100k' territory in a typical season"),
    (2000, "Illustrative only -- roughly 'top 1 million' (green-arrow) territory in a typical season"),
    (1800, "Illustrative only -- roughly an average manager's season total"),
    (0, "Illustrative only -- below a typical season's average manager total"),
]


def _estimate_percentile_band(total_points: float) -> str:
    for threshold, label in ROUGH_PERCENTILE_BANDS:
        if total_points >= threshold:
            return label
    return ROUGH_PERCENTILE_BANDS[-1][1]


@dataclass
class SeasonReport:
    season: str
    gameweeks_simulated: int
    total_gross_points: float
    total_hit_cost: int
    total_points: float  # net -- gross minus hit costs; the season's real final total
    static_benchmark_points: float
    transfer_roi: float  # total_points - static_benchmark_points
    total_auto_sub_activations: int  # gameweeks with >=1 auto-sub
    total_auto_sub_moves: int
    total_bench_points_left_behind: float
    total_captain_points_earned: float
    total_captain_points_possible: float
    captaincy_points_left_on_table: float
    optimal_captaincy_weeks: int
    best_gameweeks: list = field(default_factory=list)
    worst_gameweeks: list = field(default_factory=list)
    gw_results: list = field(default_factory=list)
    estimated_percentile_band: str = ""
    chip_log: list = field(default_factory=list)  # list of ChipActivation, in gameweek order


def _build_season_report(season: str, n_gameweeks: int, gw_results: list, chip_log: Optional[list] = None) -> SeasonReport:
    total_gross = round(sum(r.managed_points for r in gw_results), 1)
    total_hit_cost = sum(r.managed_hit_cost for r in gw_results)
    total_net = round(sum(r.managed_net_points for r in gw_results), 1)
    static_total = round(sum(r.static_points for r in gw_results), 1)

    total_captain_earned = round(sum(r.captain_doubled_points for r in gw_results), 1)
    total_captain_possible = round(sum(r.best_possible_captain_points for r in gw_results), 1)
    # Compares the extra copy actually earned (captain_doubled_points -- correctly reflects a
    # vice-promotion when the nominated captain blanks, see live_tracker._live_captain_points)
    # against what that extra copy WOULD have been captaining the actual top scorer in hindsight.
    # NOT r.captain_points (the nominated captain's own raw points, ignoring any vice promotion) --
    # comparing that against a DOUBLED-bonus figure mixes units and silently undercounts optimal
    # weeks whenever the vice was promoted, even in the (only) case a promoted vice happens to have
    # legitimately been that week's actual top scorer.
    optimal_weeks = sum(1 for r in gw_results if r.captain_doubled_points >= r.best_possible_captain_points)

    ranked = sorted(gw_results, key=lambda r: r.managed_net_points, reverse=True)

    return SeasonReport(
        season=season,
        gameweeks_simulated=n_gameweeks,
        total_gross_points=total_gross,
        total_hit_cost=total_hit_cost,
        total_points=total_net,
        static_benchmark_points=static_total,
        transfer_roi=round(total_net - static_total, 1),
        total_auto_sub_activations=sum(1 for r in gw_results if r.auto_sub_moves > 0),
        total_auto_sub_moves=sum(r.auto_sub_moves for r in gw_results),
        total_bench_points_left_behind=round(sum(r.bench_points_left_behind for r in gw_results), 1),
        total_captain_points_earned=total_captain_earned,
        total_captain_points_possible=total_captain_possible,
        captaincy_points_left_on_table=round(total_captain_possible - total_captain_earned, 1),
        optimal_captaincy_weeks=optimal_weeks,
        best_gameweeks=ranked[:3],
        worst_gameweeks=list(reversed(ranked[-3:])) if len(ranked) >= 3 else list(reversed(ranked)),
        gw_results=gw_results,
        estimated_percentile_band=_estimate_percentile_band(total_net),
        chip_log=sorted(chip_log or [], key=lambda c: c.gw),
    )


# --- Walk-forward simulation loop -----------------------------------------------------------

def simulate_season(
    season: str,
    initial_budget: float = 100.0,
    risk_profile: str = "balanced",
    verbose: bool = True,
) -> SeasonReport:
    """Runs the full walk-forward backtest for one historical `season` (vaastav folder naming,
    e.g. "2024-25") and returns a SeasonReport. Stops early (rather than raising) if the archive's
    per-gameweek stats run out before gameweek 38 -- either the season genuinely had fewer weeks
    played so far (an in-progress current season) or the archive itself doesn't cover it that far;
    either way, whatever gameweeks WERE available are still reported on.

    See the module docstring for the full no-lookahead-bias design and documented simplifications.
    """
    risk_lambda = _resolve_risk_lambda(risk_profile)
    budget_units = round(initial_budget * config.PRICE_DIVISOR)

    team_rows = _load_teams(season)
    if not team_rows:
        raise FPLAPIError(f"No teams data found for season {season!r} on the vaastav archive.")
    team_id_by_name = {row["name"]: int(row["id"]) for row in team_rows if row.get("name") and row.get("id")}
    fixtures_by_event = _load_fixtures_by_event(season)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.init_db(conn)
    _seed_teams(conn, team_rows)
    _seed_gameweeks(conn, MAX_GAMEWEEKS)
    _seed_fixtures(conn, fixtures_by_event)

    meta_by_id: dict = {}
    accum_by_id: dict = {}
    gw_results: list = []
    chip_log: list = []

    managed_squad_ids: list = []
    static_squad_ids: list = []
    bank = budget_units
    free_transfers = DEFAULT_FREE_TRANSFERS_AT_GW2
    final_gw = 0

    # Double-Chip Strategy setup -- see that section's own docstring. The whole-season fixture
    # SCHEDULE (never results) is known up front, so this classification and Set 1's "does any
    # Double Gameweek exist at all" fact are computed once here rather than re-derived every week.
    gw_density = classify_gameweek_density(fixtures_by_event, set(team_id_by_name.values()))
    set1_has_dgw = _half_has_dgw(gw_density, CHIP_SET1_START_GW, CHIP_SET1_END_GW)
    set1_fh_target_gw = _scan_fh_target_gw(gw_density, CHIP_SET1_START_GW, CHIP_SET1_END_GW)
    chips_used = {1: {}, 2: {}}  # half -> {chip_code: ChipActivation}
    set2_plan: Optional[dict] = None

    for gw in range(1, MAX_GAMEWEEKS + 1):
        try:
            raw_rows = fetch_vaastav_csv(config.VAASTAV_GW_STATS_CSV_TEMPLATE.format(season=season, gw=gw))
        except FPLAPIError:
            if verbose:
                print(f"[backtest] {season}: no data for gameweek {gw} -- season data ends at GW{gw - 1}. Stopping.")
            break
        if not raw_rows:
            break

        parsed_rows = [p for p in (_parse_gw_row(r) for r in raw_rows) if p is not None]
        _ingest_gw_meta(meta_by_id, parsed_rows, team_id_by_name)
        _write_players_table(conn, meta_by_id, accum_by_id)
        _set_walk_forward_gameweek(conn, gw)

        pool = optimizer.fetch_players(conn)
        pool_by_id = {p.id: p for p in pool}

        free_transfers_before = free_transfers  # snapshot -- see GameweekResult.free_transfers_before
        wc_activation: Optional[ChipActivation] = None
        fh_activation: Optional[ChipActivation] = None
        tc_activation: Optional[ChipActivation] = None
        bb_activation: Optional[ChipActivation] = None
        fh_scoring_squad_ids: Optional[list] = None

        if gw == 1:
            squad = optimizer.solve_squad_with_captaincy(pool, budget=budget_units, risk_lambda=risk_lambda)
            managed_squad_ids = [p.id for p in squad]
            static_squad_ids = list(managed_squad_ids)
            bank = budget_units - sum(p.now_cost for p in squad)
            transfers_in_ids, transfers_out_ids, hit_cost = [], [], 0

            bb1_detail = _bb1_trigger(1, squad, None)
            if bb1_detail:
                bb_activation = ChipActivation("BB", 1, 1, bb1_detail)
                chips_used[1]["BB"] = bb_activation
                chip_log.append(bb_activation)
        else:
            half = 1 if gw <= CHIP_SET1_END_GW else 2
            if half == 2 and set2_plan is None:
                set2_plan = _build_set2_plan(conn, managed_squad_ids, gw_density)
                if verbose:
                    plan_desc = ", ".join(f"{k}={v}" for k, v in set2_plan.items() if k.endswith("_gw"))
                    print(f"[backtest] Set 2 chip schedule (pre-scanned at GW20): {plan_desc}")
            used_this_half = chips_used[half]
            squad_rows_before = [pool_by_id[pid] for pid in managed_squad_ids if pid in pool_by_id]

            if "WC" not in used_this_half:
                detail = _wc1_trigger(gw, squad_rows_before) if half == 1 else _wc2_trigger(gw, set2_plan)
                if detail:
                    wc_activation = ChipActivation("WC", gw, half, detail)

            if wc_activation is None and "FH" not in used_this_half:
                target_gw = set1_fh_target_gw if half == 1 else set2_plan["fh2_gw"]
                detail = _fh_trigger(gw, target_gw, squad_rows_before, gw_density)
                if detail:
                    fh_activation = ChipActivation("FH", gw, half, detail)

            if wc_activation is not None:
                # Wildcard 2 deliberately builds toward its paired Bench Boost 2 double-gameweek
                # target (see _rebuild_squad_for_target_gw); Wildcard 1 just optimizes for now.
                wc_target_gw = set2_plan["bb2_gw"] if (half == 2 and set2_plan.get("wc2_gw") == gw) else None
                new_squad = _rebuild_squad_for_target_gw(conn, budget_units, risk_lambda, wc_target_gw)
                new_squad_ids = {p.id for p in new_squad}
                old_squad_ids = {p.id for p in squad_rows_before}
                transfers_in_ids = list(new_squad_ids - old_squad_ids)
                transfers_out_ids = list(old_squad_ids - new_squad_ids)
                managed_squad_ids = list(new_squad_ids)
                bank = budget_units - sum(p.now_cost for p in new_squad)
                hit_cost = 0
                # Wildcard doesn't consume a free transfer, but the normal weekly FT accrual
                # continues regardless of chip usage -- same roll-forward formula plan_transfers
                # itself uses for a 0-transfer ("hold") week.
                free_transfers = min(transfer_planner.FREE_TRANSFER_CAP, free_transfers + 1)
                used_this_half["WC"] = wc_activation
                chip_log.append(wc_activation)
            elif fh_activation is not None:
                # Free Hit is a ONE-WEEK-ONLY squad -- managed_squad_ids/bank are deliberately left
                # untouched so next gameweek's decision continues from the real, pre-Free-Hit squad.
                fh_squad = _rebuild_squad_for_target_gw(conn, budget_units, risk_lambda)
                fh_scoring_squad_ids = [p.id for p in fh_squad]
                transfers_in_ids, transfers_out_ids, hit_cost = [], [], 0
                free_transfers = min(transfer_planner.FREE_TRANSFER_CAP, free_transfers + 1)
                used_this_half["FH"] = fh_activation
                chip_log.append(fh_activation)
            else:
                try:
                    roadmap = transfer_planner.plan_transfers(
                        conn, managed_squad_ids, bank, free_transfers, horizon_gws=1,
                        allow_hits=True, freeze_gkp_transfers=True,
                    )
                    step = roadmap[0]
                    transfers_in_ids, transfers_out_ids = step.transfers_in_ids, step.transfers_out_ids
                    hit_cost = step.hit_cost
                    bank = step.bank_remaining
                    free_transfers = min(
                        transfer_planner.FREE_TRANSFER_CAP,
                        max(0, free_transfers - step.transfers_made) + 1,
                    )
                    managed_squad_ids = list((set(managed_squad_ids) - set(transfers_out_ids)) | set(transfers_in_ids))
                except OptimizationError as exc:
                    if verbose:
                        print(f"[backtest] {season} GW{gw}: transfer planning failed ({exc}) -- holding squad.")
                    transfers_in_ids, transfers_out_ids, hit_cost = [], [], 0
                    free_transfers = min(transfer_planner.FREE_TRANSFER_CAP, free_transfers + 1)

        scoring_squad_ids = fh_scoring_squad_ids if fh_scoring_squad_ids is not None else managed_squad_ids

        client = _StaticLiveClient(_historical_live_payload(parsed_rows))
        managed = _score_gameweek(conn, client, scoring_squad_ids, gw, pool_by_id, risk_lambda)
        static = _score_gameweek(conn, client, static_squad_ids, gw, pool_by_id, risk_lambda)

        # Triple Captain / Bench Boost: post-scoring only (never affects the transfer/squad
        # decision itself -- see this feature's own docstring), and never on the same gameweek as
        # a just-fired Wildcard/Free Hit (real FPL: only one chip per gameweek).
        if gw != 1 and wc_activation is None and fh_activation is None:
            half = 1 if gw <= CHIP_SET1_END_GW else 2
            used_this_half = chips_used[half]
            if "TC" not in used_this_half:
                captain_row = pool_by_id.get(managed["captain_id"])
                if captain_row is not None:
                    detail = (
                        _tc1_trigger(gw, gw_density, set1_has_dgw, captain_row) if half == 1
                        else _tc2_trigger(gw, set2_plan, managed_squad_ids)
                    )
                    if detail:
                        tc_activation = ChipActivation("TC", gw, half, detail)
                        used_this_half["TC"] = tc_activation
                        chip_log.append(tc_activation)
            if tc_activation is None and "BB" not in used_this_half:
                squad_rows_now = [pool_by_id[pid] for pid in scoring_squad_ids if pid in pool_by_id]
                detail = (
                    _bb1_trigger(gw, squad_rows_now, used_this_half.get("WC")) if half == 1
                    else _bb2_trigger(gw, set2_plan, squad_rows_now, gw_density)
                )
                if detail:
                    bb_activation = ChipActivation("BB", gw, half, detail)
                    used_this_half["BB"] = bb_activation
                    chip_log.append(bb_activation)

        # gameweek_chip: whichever chip actually fired this week (at most one, see this feature's
        # own docstring), for the report's chip log -- distinct from `scoring_chip`, which is only
        # ever "TC"/"BB"/None, since WC/FH's effect is already baked into WHICH squad got scored
        # rather than needing a separate points adjustment (see _apply_chip_scoring).
        scoring_chip = "TC" if tc_activation else ("BB" if bb_activation else None)
        gameweek_chip = (
            scoring_chip or (wc_activation.chip if wc_activation else None) or (fh_activation.chip if fh_activation else None)
        )
        managed_gross = _apply_chip_scoring(managed, scoring_squad_ids, scoring_chip)
        managed_net = round(managed_gross - hit_cost, 1)
        static_gross = static["provisional_total_points"]

        if scoring_chip == "BB":
            # With Bench Boost active the whole 15 already counts, so nothing was "left behind".
            bench_left_behind = 0.0
        else:
            bench_all = sum(
                managed["player_status"][pid].live_points for pid in managed["bench_ids"] if pid in managed["player_status"]
            )
            bench_captured = sum(
                managed["player_status"][pid].live_points for pid in managed["bench_ids"]
                if pid in managed["effective_starting_xi_ids"] and pid in managed["player_status"]
            )
            bench_left_behind = round(bench_all - bench_captured, 1)

        xi_points = [
            managed["player_status"][pid].live_points
            for pid in managed["effective_starting_xi_ids"] if pid in managed["player_status"]
        ]
        best_possible_top = max(xi_points) if xi_points else 0

        captain_status = managed["player_status"].get(managed["captain_id"])

        gw_results.append(GameweekResult(
            gw=gw,
            managed_points=round(managed_gross, 1),
            managed_hit_cost=hit_cost,
            managed_net_points=managed_net,
            transfers_in=[pool_by_id[pid].web_name for pid in transfers_in_ids if pid in pool_by_id],
            transfers_out=[pool_by_id[pid].web_name for pid in transfers_out_ids if pid in pool_by_id],
            static_points=round(static_gross, 1),
            auto_sub_moves=len(managed["auto_sub_moves"]),
            bench_points_left_behind=bench_left_behind,
            captain_web_name=captain_status.web_name if captain_status else "?",
            captain_points=captain_status.live_points if captain_status else 0,
            captain_doubled_points=managed["captain_doubled_points"],
            best_possible_captain_points=best_possible_top,
            formation=managed["formation"],
            chip=gameweek_chip,
            free_transfers_before=free_transfers_before,
        ))

        if verbose:
            chip_note = f" | CHIP: {CHIP_NAMES[gw_results[-1].chip]}" if gw_results[-1].chip else ""
            print(
                f"[backtest] {season} GW{gw}: {managed_net:.1f} net pts "
                f"({managed_gross:.1f} gross - {hit_cost} hit) | static benchmark {static_gross:.1f} "
                f"| C: {gw_results[-1].captain_web_name}{chip_note}"
            )

        _accumulate(accum_by_id, parsed_rows)
        final_gw = gw

    return _build_season_report(season, final_gw, gw_results, chip_log)


# --- CLI reporting ---------------------------------------------------------------------------

def format_report(report: SeasonReport) -> str:
    lines = [
        "=" * 72,
        f"FULL-SEASON WALK-FORWARD BACKTEST -- {report.season} ({report.gameweeks_simulated} gameweeks simulated)",
        "=" * 72,
        "",
        "-- Season Totals " + "-" * 54,
        f"  Gross points:               {report.total_gross_points:.1f}",
        f"  Points lost to hits:        -{report.total_hit_cost}",
        f"  NET TOTAL POINTS:           {report.total_points:.1f}",
        f"  Estimated finish:           {report.estimated_percentile_band}",
        "",
        "-- Transfer ROI vs. Set-and-Forget Benchmark " + "-" * 25,
        f"  Set-and-forget (GW1 squad held all season, 0 transfers): {report.static_benchmark_points:.1f}",
        f"  This backtest's managed total:                            {report.total_points:.1f}",
        f"  Transfer ROI (managed - static):                          {report.transfer_roi:+.1f}",
        "",
        "-- Captaincy Efficiency " + "-" * 47,
        f"  Points earned via (C):              {report.total_captain_points_earned:.1f}",
        f"  Max possible (hindsight-optimal):   {report.total_captain_points_possible:.1f}",
        f"  Points left on the table:           {report.captaincy_points_left_on_table:.1f}",
        f"  Optimal captaincy calls:            {report.optimal_captaincy_weeks}/{report.gameweeks_simulated} gameweeks",
        "",
        "-- Bench Management " + "-" * 51,
        f"  Auto-sub activations:                {report.total_auto_sub_activations} gameweeks "
        f"({report.total_auto_sub_moves} total moves)",
        f"  Bench points left behind:            {report.total_bench_points_left_behind:.1f}",
        "",
        "-- Best 3 Gameweeks " + "-" * 51,
    ]
    for r in report.best_gameweeks:
        lines.append(f"  GW{r.gw:>2}: {r.managed_net_points:>6.1f} pts  (C: {r.captain_web_name}, {r.formation})")
    lines.append("")
    lines.append("-- Worst 3 Gameweeks " + "-" * 50)
    for r in report.worst_gameweeks:
        lines.append(f"  GW{r.gw:>2}: {r.managed_net_points:>6.1f} pts  (C: {r.captain_web_name}, {r.formation})")
    lines.append("")
    lines.append("-- Chip Log (Set 1: GW1-19, Set 2: GW20-38) " + "-" * 26)
    if report.chip_log:
        for c in report.chip_log:
            lines.append(f"  GW{c.gw:>2} [Set {c.half}] {CHIP_NAMES[c.chip]:<15} {c.detail}")
    else:
        lines.append("  No chips fired -- no trigger condition was ever met.")
    lines.extend([
        "",
        "-" * 72,
        "Note: historical injury/press-conference status and real ownership%/EO data aren't",
        "archived by the vaastav mirror this backtest reads from, so every player is treated as",
        "fully available and risk-profile EO-shielding has no ownership signal to act on here.",
        "See src/backtest.py's module docstring for the full list of documented simplifications.",
    ])
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Full-Season Walk-Forward Backtester for the FPL optimizer.")
    parser.add_argument("--season", required=True, help="vaastav archive season folder name, e.g. 2024-25")
    parser.add_argument("--budget", type=float, default=100.0, help="Starting budget in GBP millions (default 100.0)")
    parser.add_argument(
        "--risk-profile", default="balanced",
        choices=sorted(set(RISK_PROFILE_ALIASES) | set(optimizer.RISK_PROFILE_LAMBDA)),
        help="Risk/ownership profile for Starting XI & captaincy selection (default: balanced)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-gameweek progress output")
    args = parser.parse_args(argv)

    try:
        report = simulate_season(
            args.season, initial_budget=args.budget, risk_profile=args.risk_profile, verbose=not args.quiet,
        )
    except FPLAPIError as exc:
        print(f"Could not load season {args.season!r} from the vaastav archive: {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    print(format_report(report))


if __name__ == "__main__":
    main()

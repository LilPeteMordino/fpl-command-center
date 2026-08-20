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

from src import config, database, optimizer, transfer_planner
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


def _build_season_report(season: str, n_gameweeks: int, gw_results: list) -> SeasonReport:
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

    managed_squad_ids: list = []
    static_squad_ids: list = []
    bank = budget_units
    free_transfers = DEFAULT_FREE_TRANSFERS_AT_GW2
    final_gw = 0

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

        if gw == 1:
            squad = optimizer.solve_squad_with_captaincy(pool, budget=budget_units, risk_lambda=risk_lambda)
            managed_squad_ids = [p.id for p in squad]
            static_squad_ids = list(managed_squad_ids)
            bank = budget_units - sum(p.now_cost for p in squad)
            transfers_in_ids, transfers_out_ids, hit_cost = [], [], 0
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

        client = _StaticLiveClient(_historical_live_payload(parsed_rows))
        managed = _score_gameweek(conn, client, managed_squad_ids, gw, pool_by_id, risk_lambda)
        static = _score_gameweek(conn, client, static_squad_ids, gw, pool_by_id, risk_lambda)

        managed_gross = managed["provisional_total_points"]
        managed_net = round(managed_gross - hit_cost, 1)
        static_gross = static["provisional_total_points"]

        bench_all = sum(
            managed["player_status"][pid].live_points for pid in managed["bench_ids"] if pid in managed["player_status"]
        )
        bench_captured = sum(
            managed["player_status"][pid].live_points for pid in managed["bench_ids"]
            if pid in managed["effective_starting_xi_ids"] and pid in managed["player_status"]
        )

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
            bench_points_left_behind=round(bench_all - bench_captured, 1),
            captain_web_name=captain_status.web_name if captain_status else "?",
            captain_points=captain_status.live_points if captain_status else 0,
            captain_doubled_points=managed["captain_doubled_points"],
            best_possible_captain_points=best_possible_top,
            formation=managed["formation"],
        ))

        if verbose:
            print(
                f"[backtest] {season} GW{gw}: {managed_net:.1f} net pts "
                f"({managed_gross:.1f} gross - {hit_cost} hit) | static benchmark {static_gross:.1f} "
                f"| C: {gw_results[-1].captain_web_name}"
            )

        _accumulate(accum_by_id, parsed_rows)
        final_gw = gw

    return _build_season_report(season, final_gw, gw_results)


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

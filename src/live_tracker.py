"""Matchday Live Rank & Auto-Sub Radar, and the Mini-League Rival Tracker.

Real-time provisional points, in-play BPS-based bonus estimation, a simplified dynamic
auto-substitution simulation, and live Captain 2x doubling (with Vice-Captain promotion) --
built on top of the official `/event/{gw}/live/` endpoint (see fpl_api.FPLClient.get_event_live).

Distinct from the season-long projection engine (optimizer.py/transfer_planner.py, which model
*expected* points ahead of a gameweek): this module reads what has ACTUALLY happened in live or
finished matches so far this gameweek -- it has no opinion about future gameweeks at all.

Also home to the Mini-League Rival Tracker (compute_local_effective_ownership,
classify_shield_and_differential), built on top of fpl_api.fetch_minileague_squads -- grouped
here rather than in a separate module since both features are about reading real-time/near-term
state from the live API surface rather than projecting it, and both power the same "Live" side of
the dashboard.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional

PROVISIONAL_BONUS_POINTS = {1: 3, 2: 2, 3: 1}  # live-BPS rank within a fixture -> bonus points

# Same Starting XI formation floor optimizer.solve_starting_xi enforces (1 GKP, >=3 DEF, >=2 MID,
# >=1 FWD) -- duplicated as plain constants here rather than importing optimizer's ILP machinery,
# since this module only needs the position-count *rule*, not the solver itself.
MIN_STARTING_DEF = 3
MIN_STARTING_MID = 2
MIN_STARTING_FWD = 1

STATUS_ICON_PLAYING = "\U0001F7E2"  # green -- on the pitch right now
STATUS_ICON_FINISHED = "\U0001F534"  # red -- their match has concluded
STATUS_ICON_BENCH = "\U0001F7E1"  # yellow -- hasn't featured yet (bench, or kickoff still to come)


@dataclass
class LivePlayerStatus:
    player_id: int
    web_name: str
    team_id: int
    element_type: int
    minutes: int
    base_points: int  # stats.total_points as FPL's live endpoint reports it (see live_points note below)
    official_bonus: int  # stats.bonus -- 0 until FPL finalizes bonus for that match
    provisional_bonus: int  # our own live-BPS-rank estimate, only nonzero while official_bonus is still 0
    fixture_finished: bool
    fixture_started: bool
    live_points: int  # base_points, plus provisional_bonus if official bonus isn't finalized yet
    # (once official_bonus > 0, it's already folded into base_points by FPL's own live endpoint --
    # live_points must NOT add it again, see get_live_gameweek_status).

    @property
    def status_icon(self) -> str:
        if self.minutes > 0 and not self.fixture_finished:
            return STATUS_ICON_PLAYING
        if self.fixture_finished:
            return STATUS_ICON_FINISHED
        return STATUS_ICON_BENCH

    @property
    def status_label(self) -> str:
        if self.minutes > 0 and not self.fixture_finished:
            return "Playing"
        if self.fixture_finished:
            return "Finished" if self.minutes > 0 else "Finished (no minutes)"
        return "Bench / Not started"


def _team_fixture_status(conn, team_id: int, event_id: int) -> tuple:
    """(has_fixture, all_finished) for a team's fixture(s) this gameweek -- handles blank
    gameweeks (no fixture: has_fixture=False) and double gameweeks (only True once BOTH legs
    have concluded)."""
    rows = conn.execute(
        "SELECT finished FROM fixtures WHERE event = ? AND (team_h = ? OR team_a = ?)",
        (event_id, team_id, team_id),
    ).fetchall()
    if not rows:
        return False, False
    return True, all(r["finished"] for r in rows)


def _provisional_bonus_by_fixture(conn, event_id: int, live_by_player: dict, player_team: dict) -> dict:
    """player_id -> provisional bonus points (0 if not top-3 live BPS in their own fixture, or if
    that fixture's official bonus has already been finalized). Ranks players by live BPS within
    each of the gameweek's fixtures, grouping both competing teams' rostered players together --
    matching real FPL's own per-match BPS ranking -- across the FULL live payload, not just squad
    members, since a squad player's provisional bonus rank depends on all 22 players on the pitch.

    Simplification: real FPL splits bonus points across tied BPS scores (e.g. two players tied
    for 3rd both get 1pt, pushing the 4th-place player out entirely); this ranks strictly by BPS
    with no tie-splitting, which is a good live estimate but can differ from FPL's final award
    when BPS scores are exactly tied.
    """
    fixtures = conn.execute(
        "SELECT id, team_h, team_a, finished FROM fixtures WHERE event = ?", (event_id,)
    ).fetchall()
    provisional = {}
    for fx in fixtures:
        if fx["finished"]:
            continue  # official bonus should already be finalized -- no provisional estimate needed
        fixture_team_ids = {fx["team_h"], fx["team_a"]}
        candidates = [
            (pid, stats) for pid, stats in live_by_player.items()
            if player_team.get(pid) in fixture_team_ids and stats.get("bps", 0) > 0
        ]
        candidates.sort(key=lambda c: c[1].get("bps", 0), reverse=True)
        for rank, (pid, _stats) in enumerate(candidates[:3], start=1):
            provisional[pid] = PROVISIONAL_BONUS_POINTS[rank]
    return provisional


def _simulate_auto_subs(statuses: dict, starting_xi_ids: list, bench_ids: list) -> tuple:
    """Simplified single-pass simulation of FPL's auto-sub engine: for each Starting XI player
    with 0 minutes whose fixture(s) have all concluded, swap in the highest-priority bench player
    (in bench_ids order -- see optimizer.solve_starting_xi's weighted bench-slot assignment) who
    actually played (minutes > 0), keeping the formation legal (>=3 DEF, >=2 MID, >=1 FWD; a
    benched GKP only ever replaces the starting GKP).

    Real FPL's own engine is more exhaustive (it searches every ordering to maximize the final
    total, not just a single greedy priority pass) -- this is a transparent, reasonable
    approximation of that, not a guaranteed bit-for-bit match on every edge case.

    Returns (effective_starting_xi_ids, auto_sub_moves) where each move is {"out", "in"}.
    """
    xi = list(starting_xi_ids)
    bench_remaining = list(bench_ids)
    moves = []

    def position_counts(ids):
        counts = {2: 0, 3: 0, 4: 0}
        for pid in ids:
            st = statuses.get(pid)
            if st and st.element_type in counts:
                counts[st.element_type] += 1
        return counts

    for out_id in list(xi):
        out_status = statuses.get(out_id)
        if out_status is None or out_status.minutes > 0 or not out_status.fixture_finished:
            continue  # played, or still could -- no sub needed (yet)

        replacement = None
        for cand_id in bench_remaining:
            cand_status = statuses.get(cand_id)
            if cand_status is None or cand_status.minutes == 0:
                continue  # a bench player who also didn't play can't be auto-subbed in
            if out_status.element_type == 1:
                if cand_status.element_type != 1:
                    continue  # GKP can only be replaced by the bench GKP
            else:
                if cand_status.element_type == 1:
                    continue
                trial_xi = [pid for pid in xi if pid != out_id] + [cand_id]
                counts = position_counts(trial_xi)
                if counts[2] < MIN_STARTING_DEF or counts[3] < MIN_STARTING_MID or counts[4] < MIN_STARTING_FWD:
                    continue  # would break the formation floor
            replacement = cand_id
            break

        if replacement is not None:
            xi = [pid for pid in xi if pid != out_id] + [replacement]
            bench_remaining.remove(replacement)
            moves.append({"out": out_id, "in": replacement})

    return xi, moves


def _live_captain_points(statuses: dict, captain_id: Optional[int], vice_id: Optional[int]) -> tuple:
    """(active_captain_id, their_live_points) -- the captain stays active (provisionally doubled)
    as long as they've played or their match hasn't concluded yet. Only once their fixture(s) are
    ALL finished with 0 minutes does the armband promote to the vice-captain -- and only if the
    vice actually played (real FPL rule: an also-blanking vice does not get the armband; the
    captaincy is simply wasted that gameweek)."""
    captain = statuses.get(captain_id) if captain_id is not None else None
    if captain and (captain.minutes > 0 or not captain.fixture_finished):
        return captain_id, captain.live_points

    vice = statuses.get(vice_id) if vice_id is not None else None
    if vice and vice.minutes > 0:
        return vice_id, vice.live_points

    return None, 0  # captaincy wasted -- neither the captain nor vice featured


def get_live_gameweek_status(
    conn,
    client,
    squad_ids: list,
    event_id: int,
    starting_xi_ids: list,
    bench_ids: list,
    captain_id: Optional[int] = None,
    vice_id: Optional[int] = None,
    assume_all_fixtures_finished: bool = False,
) -> dict:
    """The full Matchday Live Rank & Auto-Sub Radar snapshot for one gameweek.

    assume_all_fixtures_finished: for src.replay's Historical Gameweek Replay Mode only -- a
    replayed `event_id` is a past season's gameweek NUMBER, not a real local gameweeks.id, so it
    has no rows of its own in the local `fixtures` table (which only ever holds the CURRENT
    season's schedule) and, worse, a small `event_id` like 1 can coincidentally collide with the
    current season's own real (and likely still-unplayed) fixtures for that same number. Querying
    `fixtures` at all in that situation would silently use the wrong season's data -- possibly
    reading a current, unfinished fixture as this player's match. When True, every team's fixture
    is simply treated as finished and the (fixtures-table-dependent) provisional bonus estimate is
    skipped entirely -- correct for a replay, since a finished historical gameweek's bonus in the
    source data is always already final, never provisional. Real live tracking never sets this.

    Returns {
      "event_id": int,
      "player_status": {player_id: LivePlayerStatus, ...} (every squad member),
      "effective_starting_xi_ids": [...] (Starting XI after simulated auto-subs),
      "auto_sub_moves": [{"out": pid, "in": pid}, ...],
      "active_captain_id": Optional[int] (None if the armband was wasted -- see _live_captain_points),
      "captain_doubled_points": float (the extra copy of the active captain's live points),
      "provisional_total_points": float (effective XI sum + the captain's extra copy),
    }
    """
    live_payload = client.get_event_live(event_id)
    live_by_player = {e["id"]: e.get("stats", {}) for e in live_payload.get("elements", [])}

    if not squad_ids:
        return {
            "event_id": event_id, "player_status": {}, "effective_starting_xi_ids": [],
            "auto_sub_moves": [], "active_captain_id": None, "captain_doubled_points": 0.0,
            "provisional_total_points": 0.0,
        }

    placeholders = ",".join(["?"] * len(squad_ids))
    rows = conn.execute(
        f"SELECT id, web_name, team_id, element_type FROM players WHERE id IN ({placeholders})", squad_ids,
    ).fetchall()
    player_info = {r["id"]: dict(r) for r in rows}

    # Provisional bonus ranking needs every live player's team, not just the squad's -- a squad
    # member's rank within their own match depends on all 22 players on the pitch.
    if assume_all_fixtures_finished:
        provisional_bonus_by_player: dict = {}
    else:
        player_team = {r["id"]: r["team_id"] for r in conn.execute("SELECT id, team_id FROM players").fetchall()}
        provisional_bonus_by_player = _provisional_bonus_by_fixture(conn, event_id, live_by_player, player_team)

    statuses: dict = {}
    for pid in squad_ids:
        info = player_info.get(pid)
        if info is None:
            continue
        stats = live_by_player.get(pid, {})
        minutes = stats.get("minutes", 0) or 0
        official_bonus = stats.get("bonus", 0) or 0
        base_points = stats.get("total_points", 0) or 0
        if assume_all_fixtures_finished:
            all_finished = True
        else:
            _has_fixture, all_finished = _team_fixture_status(conn, info["team_id"], event_id)
        provisional_bonus = 0 if official_bonus > 0 else provisional_bonus_by_player.get(pid, 0)
        # official_bonus, once finalized, is already folded into base_points by FPL's own live
        # endpoint -- only add our own provisional estimate, never both.
        live_points = base_points + provisional_bonus

        statuses[pid] = LivePlayerStatus(
            player_id=pid, web_name=info["web_name"], team_id=info["team_id"], element_type=info["element_type"],
            minutes=minutes, base_points=base_points, official_bonus=official_bonus,
            provisional_bonus=provisional_bonus, fixture_finished=all_finished,
            fixture_started=minutes > 0 or (stats.get("bps", 0) or 0) > 0 or all_finished,
            live_points=live_points,
        )

    effective_xi_ids, auto_sub_moves = _simulate_auto_subs(statuses, starting_xi_ids, bench_ids)
    active_captain_id, captain_live_points = _live_captain_points(statuses, captain_id, vice_id)

    provisional_total = sum(statuses[pid].live_points for pid in effective_xi_ids if pid in statuses)
    if active_captain_id is not None:
        provisional_total += captain_live_points  # the armband's extra (doubled) copy

    return {
        "event_id": event_id,
        "player_status": statuses,
        "effective_starting_xi_ids": effective_xi_ids,
        "auto_sub_moves": auto_sub_moves,
        "active_captain_id": active_captain_id,
        "captain_doubled_points": captain_live_points,
        "provisional_total_points": round(provisional_total, 1),
    }


def calculate_live_gameweek_points(conn, client, team_id: int, current_gw: int) -> dict:
    """Live Matchday Radar entry point driven by the manager's REAL FPL picks -- fetches
    /entry/{team_id}/event/{current_gw}/picks/ directly and feeds the exact starting XI, bench
    (in FPL's own auto-sub priority order), captain, and vice-captain FPL has on record into
    get_live_gameweek_status, so the scorecard tracks what actually happened to the manager's real
    team rather than this app's own optimizer-computed "best possible" XI.

    This is the one place in the app that reads a manager's *actual* live selections instead of
    recomputing an idealized one -- appropriate here specifically because a live scorecard is
    only meaningful as a report of what really happened, not a hypothetical.

    Raises FPLAPIError if the picks fetch fails (e.g. the gameweek's deadline hasn't passed yet,
    so the manager's picks for it aren't published -- callers should handle this the same way
    every other picks-endpoint call in this codebase already does).
    """
    picks_payload = client.get_manager_picks(team_id, current_gw)
    raw_picks = sorted(picks_payload.get("picks", []), key=lambda p: p.get("position", 0))

    squad_ids = [p["element"] for p in raw_picks]
    starting_xi_ids = [p["element"] for p in raw_picks if p.get("position", 0) <= 11]
    bench_ids = [p["element"] for p in raw_picks if p.get("position", 0) > 11]
    captain_id = next((p["element"] for p in raw_picks if p.get("is_captain")), None)
    vice_id = next((p["element"] for p in raw_picks if p.get("is_vice_captain")), None)

    return get_live_gameweek_status(
        conn, client, squad_ids, current_gw, starting_xi_ids, bench_ids, captain_id, vice_id,
    )


# --- Mini-League Rival Tracker --------------------------------------------------------------

# LEO% at/above this reads as a "Shield" -- a critical defensive lock the mini-league pack has
# already coalesced around (as a STARTER, not just a squad slot); going without it risks falling
# behind rivals independent of whether it's a differential play anywhere else.
SHIELD_LEO_THRESHOLD = 50.0

# LEO% at/below this (using the SAME started+captained metric, not a separate plain-ownership
# read) qualifies as a genuine "Sword" -- a high-leverage differential few rivals are exposed to.
# The caller (app.py) additionally filters this pool to high-xP picks before display; this module
# has no opinion on xP itself (see classify_shield_and_differential's docstring).
DIFFERENTIAL_LEO_THRESHOLD = 20.0


def compute_local_effective_ownership(minileague_squads: list) -> dict:
    """player_id -> Local Mini-League Effective Ownership (LEO), scoped to just the fetched
    rivals (see fpl_api.fetch_minileague_squads):

        LEO_i = (rivals who STARTED player i + rivals who CAPTAINED player i) / total rivals x 100

    Deliberately counts each rival's STARTING XI ("starting_ids", FPL "position" <= 11), not
    their full 15-man squad -- a player nailed to a rival's bench contributes nothing to their
    actual live-scoring exposure, so counting them the same as a starter would overstate the
    Shield read. A captained player is necessarily also a starter, so a rival who captains player
    i contributes 2 to that player's numerator (started once, captained once) -- exactly mirroring
    real FPL's own global EO convention (ownership% + captaincy%) at mini-league scale, just
    scoped to starters instead of full-squad ownership. Returns {} for an empty rival list."""
    n = len(minileague_squads)
    if n == 0:
        return {}

    started_count: Counter = Counter()
    captaincy_count: Counter = Counter()
    for squad in minileague_squads:
        started_count.update(squad.get("starting_ids", squad["squad_ids"]))
        captain_id = squad.get("captain_id")
        if captain_id is not None:
            captaincy_count[captain_id] += 1

    leo = {}
    for pid, started in started_count.items():
        started_pct = (started / n) * 100.0
        captaincy_pct = (captaincy_count.get(pid, 0) / n) * 100.0
        leo[pid] = round(started_pct + captaincy_pct, 1)
    return leo


def classify_shield_and_differential(
    minileague_squads: list, my_squad_ids: set, leo: Optional[dict] = None,
) -> dict:
    """Splits every player started by at least one fetched rival into the Mini-League Shield &
    Sword lens:

    - Shield assets (LEO >= SHIELD_LEO_THRESHOLD): split into "shield_owned" (already in your
      squad -- these are protecting your rank as intended) and "shield_missing" (not in your
      squad -- an exposure gap worth knowing about even if you choose not to close it).
    - Sword/differential candidates (LEO <= DIFFERENTIAL_LEO_THRESHOLD, restricted to players NOT
      already in your squad): a pool for the caller to further rank by projected_xp -- this
      module has no opinion on xP itself (that's optimizer.py's job), so it's left unsorted here
      and the caller (app.py) sorts/filters by xP before display.

    Returns {"shield_owned": [player_id, ...] (LEO desc), "shield_missing": [player_id, ...]
    (LEO desc), "differential_candidates": [player_id, ...] (unsorted)}. All empty for an empty
    rival list.
    """
    if not minileague_squads:
        return {"shield_owned": [], "shield_missing": [], "differential_candidates": []}

    leo = leo if leo is not None else compute_local_effective_ownership(minileague_squads)

    shield_ids = sorted(
        (pid for pid, score in leo.items() if score >= SHIELD_LEO_THRESHOLD),
        key=lambda pid: leo[pid], reverse=True,
    )
    shield_owned = [pid for pid in shield_ids if pid in my_squad_ids]
    shield_missing = [pid for pid in shield_ids if pid not in my_squad_ids]

    differential_candidates = [
        pid for pid, score in leo.items()
        if pid not in my_squad_ids and score <= DIFFERENTIAL_LEO_THRESHOLD
    ]

    return {
        "shield_owned": shield_owned,
        "shield_missing": shield_missing,
        "differential_candidates": differential_candidates,
    }

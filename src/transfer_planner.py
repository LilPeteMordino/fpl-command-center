"""Multi-gameweek transfer horizon planner.

Builds a per-gameweek xP projection for gameweeks N..N+horizon using the same
position-specific model as optimizer.py (calculate_positional_xp: xG/xAG for attackers,
clean-sheet probability + DEFCON for defenders, saves for keepers), with the existing
home/away venue multiplier layered on top per fixture leg, then uses a per-gameweek ILP
to decide the best transfers (including whether to hold and bank a free transfer, or take
a hit), producing a week-by-week roadmap.

Note on scope: a single joint MILP across the whole horizon (deciding every gameweek's
transfers simultaneously) would require linearizing the free-transfer carry-over with
big-M/binary tricks for what is, in the end, a personal analytics tool. Instead this
solves one gameweek at a time: for each gameweek it ILP-optimizes the squad for several
candidate transfer counts (0, 1, 2, ...) using a short lookahead-weighted score so a
transfer can be justified by fixtures a couple of weeks out, then applies whichever
count nets the highest (lookahead score - hit cost) before moving to the next gameweek.
This is the same "rolling horizon" approach most open-source FPL optimizers use once
you're past a 1-2 gameweek pure lookahead, since a fully joint model scales very poorly.
"""
from dataclasses import dataclass, field, replace
from typing import Optional

import pulp

from src import database
from src.optimizer import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    MAX_PLAYERS_PER_TEAM,
    NEUTRAL_FIXTURE_DIFFICULTY,
    PRESEASON_OOP_ATTACK_BOOST,
    PRESEASON_PENALTY_XP_BOOST,
    RATIONALE_FIXTURE_RUN_FAVORABLE,
    RATIONALE_FIXTURE_RUN_TOUGH,
    RATIONALE_HIGH_XGI_PER_90,
    SQUAD_POSITION_COUNTS,
    OptimizationError,
    PlayerRow,
    RationaleBullet,
    calculate_baseline_xmins,
    calculate_positional_xp,
    calculate_team_xp,
    captaincy_candidates,
    captaincy_score,
    ensemble_from_sources,
    ep_next_blend_weight,
    has_set_piece_duty,
    is_before_gw1_deadline,
    is_cold_start_pool,
    is_vice_eligible,
    last_season_rate_by_player_lookup,
    recent_form_by_player_lookup,
    set_piece_label,
    solve_squad,
    solve_starting_xi,
    xgi_per_90,
    team_games_played,
)

HIT_COST = 4  # points deducted per transfer beyond available free transfers
FREE_TRANSFER_CAP = 5  # matches the current FPL banked-FT limit
MAX_EXTRA_PAID_TRANSFERS = 2  # each gw, consider using all banked FTs plus up to this many hits
LOOKAHEAD_WEIGHTS = (1.0, 0.5, 0.25)  # weight given to this gw and the following two when choosing transfers

# A transfer -- even a FREE one -- carries an implicit opportunity cost: burning it forfeits the
# option value of a banked FT (flexibility for a bigger swing later, up to FREE_TRANSFER_CAP).
# Below this margin over holding, the roadmap rolls the transfer instead of taking it -- this is
# what stops noise-level xP differences between near-identical candidates from generating
# meaningless week-to-week churn (sell/buy/re-sell) that a naive "always take the highest-scoring
# t" comparison would otherwise produce. A hurdle > 0 also structurally guarantees a transfer is
# never taken for a net projected gain <= 0.0 xP, without a separate zero-check.
DEFAULT_TRANSFER_HURDLE_XP = 1.5

# A SEPARATE, stricter hurdle specifically for a candidate that actually spends a hit (t exceeds
# that gameweek's free_transfers_before) -- a -4 is a much bigger, much more irreversible cost than
# simply not banking a free transfer, so it's gated far more strictly than the plain hold-vs-
# transfer hurdle above. Compared against the same already-hit-cost-net decision_score margin
# DEFAULT_TRANSFER_HURDLE_XP uses (see plan_transfers) -- i.e. a hit-taking move needs its raw
# projected gain to clear HIT_COST + this margin, not just this margin alone.
HIT_TRANSFER_HURDLE_XP = 4.5

GKP_FREEZE_INJURY_THRESHOLD = 50  # chance_of_playing_next_round below this lifts the GKP freeze
ANTI_CHURN_MIN_GAP_GWS = 4  # a sold player can't be bought back until at least this many GWs later

HOME_FIXTURE_MULTIPLIER = 1.05
AWAY_FIXTURE_MULTIPLIER = 0.95


@dataclass
class _FixtureLeg:
    difficulty: float
    is_home: bool


@dataclass
class _RawPlayerStats:
    """Just the attributes calculate_positional_xp / calculate_baseline_xmins read, for players
    not yet built into a full PlayerRow. now_cost is included because the pre-season DEFCON (and
    xMins) fallbacks use price band as one of their signals for "is this a nailed starter" when
    live rate data isn't available yet."""
    element_type: int
    status: str
    now_cost: int
    xg_per_90: float
    xa_per_90: float
    saves_per_90: float
    defensive_contribution_per_90: float
    starts_per_90: float
    starts: int = 0
    chance_of_playing_next_round: Optional[int] = None
    selected_by_percent: float = 0.0  # xMins pre-season fallback also reads ownership
    penalties_order: Optional[int] = None
    corners_order: Optional[int] = None
    expected_goals_conceded_per_90: float = 0.0  # see optimizer._blend_player_xga


@dataclass
class GWPlan:
    event_id: int
    transfers_in: list
    transfers_out: list
    transfers_in_cost: list  # now_cost for each transfers_in entry, same order
    transfers_out_cost: list  # now_cost for each transfers_out entry, same order
    free_transfers_before: int
    transfers_made: int
    hit_cost: int
    bank_remaining: int
    formation: str
    starting_xi: list
    bench: list
    captain: PlayerRow
    vice_captain: PlayerRow
    gross_points: float
    net_points: float
    initial_selection: bool = False  # True for GW1 while still before its deadline -- see plan_transfers
    transfers_in_ids: list = field(default_factory=list)  # player ids, same order as transfers_in
    transfers_out_ids: list = field(default_factory=list)  # player ids, same order as transfers_out
    hit_justification_margin: Optional[float] = None  # the lookahead-weighted decision_score margin
    # this hit cleared over the best hit_cost == 0 alternative that same gameweek (see plan_transfers'
    # hurdle-rate comment) -- NOT the same number as net_points, and not comparable to it: net_points
    # is this single gameweek's own final score (which can legitimately be LOWER than a cheaper
    # no-hit alternative's net_points, if the hit is being justified by weighted fixture value a
    # gameweek or two further out -- see LOOKAHEAD_WEIGHTS). None when hit_cost == 0 (nothing to
    # justify) or for the free/unlimited GW1 initial-selection step (no hurdle applies there at all).


# --- Horizon gameweeks & per-gw fixture data --------------------------------

def get_horizon_event_ids(conn, horizon_gws: int) -> list:
    """The next `horizon_gws` gameweek ids, starting from the next (or current) gameweek."""
    row = conn.execute("SELECT id FROM gameweeks WHERE is_next = 1 ORDER BY id LIMIT 1").fetchone()
    if row is None:
        row = conn.execute("SELECT id FROM gameweeks WHERE is_current = 1 ORDER BY id LIMIT 1").fetchone()
    if row is None:
        return []
    rows = conn.execute(
        "SELECT id FROM gameweeks WHERE id >= ? ORDER BY id LIMIT ?",
        (row["id"], horizon_gws),
    ).fetchall()
    return [r["id"] for r in rows]


def _team_fixtures_by_event(conn, event_ids: list) -> dict:
    """(team_id, event_id) -> list of _FixtureLeg (handles blank/double gameweeks)."""
    if not event_ids:
        return {}
    placeholders = ",".join(["?"] * len(event_ids))
    rows = conn.execute(
        f"""
        SELECT event, team_h, team_a, team_h_difficulty, team_a_difficulty
        FROM fixtures WHERE event IN ({placeholders})
        """,
        event_ids,
    ).fetchall()
    fixtures: dict = {}
    for row in rows:
        fixtures.setdefault((row["team_h"], row["event"]), []).append(
            _FixtureLeg(difficulty=row["team_h_difficulty"] or NEUTRAL_FIXTURE_DIFFICULTY, is_home=True)
        )
        fixtures.setdefault((row["team_a"], row["event"]), []).append(
            _FixtureLeg(difficulty=row["team_a_difficulty"] or NEUTRAL_FIXTURE_DIFFICULTY, is_home=False)
        )
    return fixtures


def _venue_scaled_breakdown(raw_stats: _RawPlayerStats, leg: _FixtureLeg, games_played: int):
    """One fixture leg's positional xP breakdown, with the existing home/away venue multiplier
    layered on top of calculate_positional_xp's own fixture-difficulty awareness (clean-sheet
    probability for DEF/GKP, fixture-ease multiplier for attackers). games_played is the player's
    team's finished-fixture count this season -- see optimizer.calculate_positional_xp."""
    breakdown = calculate_positional_xp(raw_stats, leg.difficulty, games_played)
    venue_mult = HOME_FIXTURE_MULTIPLIER if leg.is_home else AWAY_FIXTURE_MULTIPLIER
    return replace(
        breakdown,
        total=round(breakdown.total * venue_mult, 3),
        attack_xp=round(breakdown.attack_xp * venue_mult, 3),
        defensive_xp=round(breakdown.defensive_xp * venue_mult, 3),
        saves_xp=round(breakdown.saves_xp * venue_mult, 3),
        bonus_xp=round(breakdown.bonus_xp * venue_mult, 3),
        appearance_xp=round(breakdown.appearance_xp * venue_mult, 3),
    )


def fetch_multi_gw_projections(conn, event_ids: list, ensemble_weights: Optional[dict] = None) -> dict:
    """player_id -> static info plus a per-gameweek xP/difficulty/has_fixture/breakdown/xmins.

    Per-gameweek xP comes from optimizer.calculate_positional_xp applied to that gameweek's
    specific fixture(s), not a single fixture-neutral rate reused across the whole horizon --
    a player's clean-sheet/DEFCON/attack profile is evaluated fresh against each week's opponent.

    For any player+gameweek covered by an uploaded external CSV projection (see
    src/projections.py), the internal figure is replaced by the weighted ensemble across
    whichever named sources (ensemble_weights, default optimizer.DEFAULT_ENSEMBLE_WEIGHTS) are
    available -- same rule as optimizer.fetch_players, via optimizer.ensemble_from_sources --
    but only when we also have a local fixture for that gameweek to fall back to if neither
    source covers this player (see the comment at the ensemble call below).

    The resulting per-gameweek xP is then scaled to an "Effective xP" by projected starting
    minutes for that same gameweek (xmins/90 -- see optimizer.calculate_baseline_xmins), sourced
    from an uploaded CSV's xMins column when available, the baseline formula otherwise -- mirrors
    optimizer.fetch_players exactly, so the horizon planner and the single-gameweek engine agree.
    """
    ensemble_weights = ensemble_weights or DEFAULT_ENSEMBLE_WEIGHTS
    fixtures_by_team_event = _team_fixtures_by_event(conn, event_ids)
    games_played_by_team = team_games_played(conn)
    preseason_by_player = database.get_preseason_adjustments(conn)
    recent_form_by_player = recent_form_by_player_lookup(conn)
    last_season_rate_by_player = last_season_rate_by_player_lookup(conn)
    external_rows = database.get_external_projections(conn, event_ids=event_ids, source=list(ensemble_weights.keys()))
    ensemble_sources_by_player_event: dict = {}
    xmins_sources_by_player_event: dict = {}
    for (player_id, event, source), vals in external_rows.items():
        ensemble_sources_by_player_event.setdefault((player_id, event), {})[source] = vals["xp"]
        if vals["xmins"] is not None:
            xmins_sources_by_player_event.setdefault((player_id, event), {})[source] = vals["xmins"]

    rows = conn.execute(
        """
        SELECT p.id, p.web_name, p.team_id, t.name AS team_name, p.element_type, p.now_cost,
               p.selected_by_percent, p.status, p.ep_next,
               p.xg_per_90, p.xa_per_90, p.saves_per_90, p.defensive_contribution_per_90, p.starts_per_90,
               p.starts, p.chance_of_playing_next_round, p.penalties_order, p.corners_order,
               p.expected_goals_conceded_per_90
        FROM players p
        JOIN teams t ON t.id = p.team_id
        WHERE p.status != 'u'
        """
    ).fetchall()

    projections = {}
    for row in rows:
        # Same recent-form rolling window / prior-season cold-start prior as optimizer.fetch_players
        # -- see that function's own comment for why either overrides the flat cumulative rate.
        team_games = games_played_by_team.get(row["team_id"], 0)
        row_xg_per_90, row_xa_per_90 = row["xg_per_90"] or 0.0, row["xa_per_90"] or 0.0
        if team_games == 0 and row["id"] in last_season_rate_by_player:
            row_xg_per_90, row_xa_per_90 = last_season_rate_by_player[row["id"]]
        elif row["id"] in recent_form_by_player:
            row_xg_per_90, row_xa_per_90 = recent_form_by_player[row["id"]]

        raw_stats = _RawPlayerStats(
            element_type=row["element_type"],
            status=row["status"],
            now_cost=row["now_cost"],
            xg_per_90=row_xg_per_90,
            xa_per_90=row_xa_per_90,
            saves_per_90=row["saves_per_90"] or 0.0,
            defensive_contribution_per_90=row["defensive_contribution_per_90"] or 0.0,
            starts_per_90=row["starts_per_90"] or 0.0,
            starts=row["starts"] or 0,
            chance_of_playing_next_round=row["chance_of_playing_next_round"],
            selected_by_percent=row["selected_by_percent"] or 0.0,
            penalties_order=row["penalties_order"],
            corners_order=row["corners_order"],
            expected_goals_conceded_per_90=row["expected_goals_conceded_per_90"] or 0.0,
        )
        adjustment = preseason_by_player.get(row["id"])
        if adjustment:
            # Role-only correction (see optimizer.apply_preseason_adjustment's docstring) --
            # gw-invariant, so it's applied once here rather than inside the per-event loop below.
            if adjustment.get("preseason_penalties") and raw_stats.penalties_order != 1:
                raw_stats = replace(raw_stats, penalties_order=1)
            if adjustment.get("preseason_set_pieces") and raw_stats.corners_order != 1:
                raw_stats = replace(raw_stats, corners_order=1)

        team_games_played_n = games_played_by_team.get(row["team_id"], 0)
        baseline_xmins = calculate_baseline_xmins(raw_stats, team_games_played_n)
        custom_xmins = adjustment.get("custom_xmins_override") if adjustment else None

        gw_xp, gw_difficulty, gw_has_fixture, gw_fixture_count, gw_breakdown, gw_xmins, gw_is_home = {}, {}, {}, {}, {}, {}, {}
        for event_id in event_ids:
            present_xmins = xmins_sources_by_player_event.get((row["id"], event_id), {})
            ensemble_xmins = ensemble_from_sources(present_xmins, ensemble_weights, baseline=None)
            # Pre-season manual xMins override wins over both baseline and any uploaded-CSV
            # ensemble xmins -- mirrors optimizer.fetch_players' precedence exactly.
            xmins = custom_xmins if custom_xmins is not None else (ensemble_xmins if ensemble_xmins is not None else baseline_xmins)
            gw_xmins[event_id] = xmins

            legs = fixtures_by_team_event.get((row["team_id"], event_id), [])
            gw_has_fixture[event_id] = bool(legs)
            gw_fixture_count[event_id] = len(legs)
            # Double gameweeks OR their legs together -- one home leg is enough to read as "home"
            # for the Talisman Penalty-Taker captaincy boost (see optimizer._is_talisman_boost_favorable).
            gw_is_home[event_id] = any(leg.is_home for leg in legs) if legs else None
            if legs:
                leg_breakdowns = [_venue_scaled_breakdown(raw_stats, leg, team_games_played_n) for leg in legs]
                internal_total = round(sum(b.total for b in leg_breakdowns), 3)
                representative = leg_breakdowns[0]  # for display; overwritten below if applicable

                present_xp = ensemble_sources_by_player_event.get((row["id"], event_id), {})
                ensemble_xp = ensemble_from_sources(present_xp, ensemble_weights, baseline=None)
                # Ensemble replaces (not blends with) the internal total, matching
                # optimizer.fetch_players -- the internal figure remains the fallback whenever
                # neither uploaded source covers this player. Either way, the result is then
                # scaled to Effective xP by xmins/90, and that scaled total is carried onto the
                # representative breakdown so the figure shown in the UI matches gw_xp exactly.
                if ensemble_xp is not None:
                    raw_total = ensemble_xp
                elif event_id == event_ids[0] and row["ep_next"] is not None:
                    # FPL's own ep_next describes only the immediate next round (bootstrap-static
                    # carries no further-out figure), so it's only a candidate blend input for the
                    # nearest gameweek in this horizon -- see optimizer.blend_ep_next_fallback's
                    # module-level comment for why (and when its weight fades to 0) this blend
                    # exists at all. Further-out weeks fall through to the plain internal total,
                    # same as before.
                    weight = ep_next_blend_weight(row["starts"] or 0)
                    raw_total = round(weight * row["ep_next"] + (1 - weight) * internal_total, 3)
                else:
                    raw_total = internal_total

                # Pre-season OOP/penalty-duty overrides, applied after the ensemble decision (same
                # order as optimizer.apply_preseason_adjustment) so they adjust whatever total is
                # currently in play for this gameweek, not only the internal model's own figure.
                if adjustment:
                    if adjustment.get("is_out_of_position"):
                        attack_share = sum(b.attack_xp for b in leg_breakdowns)
                        raw_total = round(raw_total + attack_share * PRESEASON_OOP_ATTACK_BOOST, 3)
                    if adjustment.get("preseason_penalties") and row["penalties_order"] != 1:
                        raw_total = round(raw_total + PRESEASON_PENALTY_XP_BOOST, 3)

                effective_total = round(raw_total * (xmins / 90.0), 3)

                gw_xp[event_id] = effective_total
                gw_breakdown[event_id] = replace(
                    representative,
                    total=effective_total,
                    external_xp=round(ensemble_xp, 3) if ensemble_xp is not None else representative.external_xp,
                    blended=(ensemble_xp is not None) or representative.blended,
                )
                gw_difficulty[event_id] = sum(leg.difficulty for leg in legs) / len(legs)
            else:
                gw_xp[event_id] = 0.0
                gw_breakdown[event_id] = None
                gw_difficulty[event_id] = NEUTRAL_FIXTURE_DIFFICULTY

        projections[row["id"]] = {
            "id": row["id"],
            "web_name": row["web_name"],
            "team_id": row["team_id"],
            "team_name": row["team_name"],
            "element_type": row["element_type"],
            "now_cost": row["now_cost"],
            "selected_by_percent": row["selected_by_percent"] or 0.0,
            "status": row["status"],
            "xg_per_90": raw_stats.xg_per_90,
            "xa_per_90": raw_stats.xa_per_90,
            "saves_per_90": raw_stats.saves_per_90,
            "defensive_contribution_per_90": raw_stats.defensive_contribution_per_90,
            "starts_per_90": raw_stats.starts_per_90,
            "starts": raw_stats.starts,
            "chance_of_playing_next_round": raw_stats.chance_of_playing_next_round,
            "penalties_order": raw_stats.penalties_order,
            "corners_order": raw_stats.corners_order,
            "gw_xp": gw_xp,
            "gw_difficulty": gw_difficulty,
            "gw_has_fixture": gw_has_fixture,
            "gw_fixture_count": gw_fixture_count,
            "gw_breakdown": gw_breakdown,
            "gw_xmins": gw_xmins,
            "gw_is_home": gw_is_home,
        }
    return projections


def _static_player_row(proj: dict) -> PlayerRow:
    """A PlayerRow carrying only the gw-invariant fields (id/cost/position/team) needed to run
    the transfer-selection ILP; projected_xp is supplied separately via the lookahead lookup."""
    return PlayerRow(
        id=proj["id"], web_name=proj["web_name"], team_id=proj["team_id"], team_name=proj["team_name"],
        element_type=proj["element_type"], now_cost=proj["now_cost"],
        selected_by_percent=proj["selected_by_percent"], form=0.0, total_points=0, ep_next=None,
        xg_per_90=proj["xg_per_90"], xa_per_90=proj["xa_per_90"], saves_per_90=proj["saves_per_90"],
        defensive_contribution_per_90=proj["defensive_contribution_per_90"], starts_per_90=proj["starts_per_90"],
        status=proj["status"], fixture_difficulty=NEUTRAL_FIXTURE_DIFFICULTY, has_fixture=True,
        projected_xp=0.0, starts=proj["starts"], chance_of_playing_next_round=proj["chance_of_playing_next_round"],
        penalties_order=proj["penalties_order"], corners_order=proj["corners_order"],
    )


def player_row_for_gw(proj: dict, event_id: int) -> PlayerRow:
    """A PlayerRow with this specific gameweek's projected (xmins-scaled) xP, for XI/captain
    selection -- xmins itself carried through too, since solve_starting_xi's minutes-security
    floor reads it directly off the PlayerRow."""
    return PlayerRow(
        id=proj["id"], web_name=proj["web_name"], team_id=proj["team_id"], team_name=proj["team_name"],
        element_type=proj["element_type"], now_cost=proj["now_cost"],
        selected_by_percent=proj["selected_by_percent"], form=0.0, total_points=0, ep_next=None,
        xg_per_90=proj["xg_per_90"], xa_per_90=proj["xa_per_90"], saves_per_90=proj["saves_per_90"],
        defensive_contribution_per_90=proj["defensive_contribution_per_90"], starts_per_90=proj["starts_per_90"],
        status=proj["status"], fixture_difficulty=proj["gw_difficulty"][event_id],
        has_fixture=proj["gw_has_fixture"][event_id], projected_xp=proj["gw_xp"][event_id],
        xp_breakdown=proj["gw_breakdown"].get(event_id),
        starts=proj["starts"], chance_of_playing_next_round=proj["chance_of_playing_next_round"],
        penalties_order=proj["penalties_order"], corners_order=proj["corners_order"],
        xmins=proj["gw_xmins"][event_id],
        is_home=proj["gw_is_home"][event_id],
    )


def _lookahead_scores(projections: dict, lookahead_events: list) -> dict:
    """player_id -> weighted sum of projected xP over the lookahead window, used only to decide
    which players to transfer (so a transfer can be justified by a fixture swing a week or two out)."""
    scores = {}
    for pid, proj in projections.items():
        score = sum(
            weight * proj["gw_xp"].get(event_id, 0.0)
            for weight, event_id in zip(LOOKAHEAD_WEIGHTS, lookahead_events)
        )
        scores[pid] = round(score, 3)
    return scores


# --- Per-gameweek transfer ILP ------------------------------------------------

def _solve_transfer_squad(
    pool: list,
    old_squad_ids: set,
    num_transfers: int,
    budget: int,
    xp_lookup: dict,
    locked_ids: Optional[set] = None,
    excluded_ids: Optional[set] = None,
) -> set:
    """The 15-man squad, reachable from old_squad_ids via exactly num_transfers swaps, that
    maximizes sum(xp_lookup) subject to budget/position/club constraints.

    locked_ids: current-squad players who must stay in the squad regardless of how the solver
    would otherwise spend this gameweek's swaps -- see plan_transfers' GKP freeze rule.
    excluded_ids: players (typically not currently owned) who can't be bought this gameweek --
    see plan_transfers' anti-churn rebuy-prevention rule. A player who is BOTH locked and excluded
    (shouldn't normally happen -- locked implies "currently owned", excluded implies "not
    currently owned") would be a contradictory/infeasible constraint pair; callers are expected to
    keep the two sets disjoint.
    """
    if num_transfers == 0:
        return set(old_squad_ids)

    prob = pulp.LpProblem("fpl_transfer_horizon", pulp.LpMaximize)
    in_squad = {p.id: pulp.LpVariable(f"in_{p.id}", cat="Binary") for p in pool}

    prob += pulp.lpSum(in_squad[p.id] * xp_lookup.get(p.id, 0.0) for p in pool)

    prob += pulp.lpSum(in_squad[p.id] * p.now_cost for p in pool) <= budget

    for element_type, count in SQUAD_POSITION_COUNTS.items():
        prob += pulp.lpSum(in_squad[p.id] for p in pool if p.element_type == element_type) == count

    team_ids = {p.team_id for p in pool}
    for team_id in team_ids:
        prob += pulp.lpSum(in_squad[p.id] for p in pool if p.team_id == team_id) <= MAX_PLAYERS_PER_TEAM

    old_pool = [p for p in pool if p.id in old_squad_ids]
    new_pool = [p for p in pool if p.id not in old_squad_ids]

    prob += pulp.lpSum(in_squad[p.id] for p in old_pool) == len(old_squad_ids) - num_transfers
    prob += pulp.lpSum(in_squad[p.id] for p in new_pool) == num_transfers

    if locked_ids:
        for pid in locked_ids & old_squad_ids:
            if pid in in_squad:
                prob += in_squad[pid] == 1
    if excluded_ids:
        for pid in excluded_ids - old_squad_ids:
            if pid in in_squad:
                prob += in_squad[pid] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError(f"Transfer solver (t={num_transfers}) returned status: {pulp.LpStatus[status]}")

    return {p.id for p in pool if pulp.value(in_squad[p.id]) > 0.5}


def captain_pick_for_gw(starting_xi: list):
    """Captain = the highest optimizer.captaincy_score among captaincy-eligible starters (see
    optimizer.is_captaincy_eligible -- MID/FWD always, DEF only with confirmed penalty/corner
    duty, GKP never -- and, during a GW1 Pre-Season Cold-Start window,
    optimizer.captaincy_candidates further restricts the field to premium-priced or
    standout-projection candidates only). Vice = the highest-scoring eligible starter (other than
    the captain) that ALSO clears the Vice-Captain Lock (is_vice_eligible), falling back to the
    plain runner-up if nobody clears it. Mirrors optimizer.get_captain_recommendations exactly --
    see its docstring for the full reasoning. A legal Starting XI always has >= 3 base-eligible
    MID/FWD (the formation floor guarantees it), so the eligible pool here can never come back
    empty."""
    cold_start = is_cold_start_pool(starting_xi)
    eligible_ids = captaincy_candidates(starting_xi)
    ranked = sorted(
        (p for p in starting_xi if p.id in eligible_ids),
        key=lambda p: captaincy_score(p, cold_start), reverse=True,
    )
    captain = ranked[0]
    eligible_vice = [p for p in ranked[1:] if is_vice_eligible(p)]
    vice = eligible_vice[0] if eligible_vice else ranked[1]
    return captain, vice


def team_xp_by_gameweek(
    conn,
    squad_ids: list,
    horizon_gws: Optional[int] = None,
    event_ids: Optional[list] = None,
    projections: Optional[dict] = None,
    ensemble_weights: Optional[dict] = None,
    min_starter_xmins: Optional[float] = None,
) -> dict:
    """event_id -> that gameweek's Team Starting XI xP (see optimizer.calculate_team_xp) for the
    given squad. Gameweeks where fewer than 11 of the squad's players resolve locally -- or where
    min_starter_xmins makes the formation infeasible for this squad -- are omitted. Pass
    event_ids/projections through if the caller already has them (e.g. to get a 3-GW and 5-GW
    cumulative total, and the same numbers for a second squad, from one fetch) instead of
    refetching for every horizon/squad combination -- ensemble_weights is ignored once a
    `projections` dict is passed in, since it was already baked into that fetch.
    """
    if event_ids is None:
        event_ids = get_horizon_event_ids(conn, horizon_gws)
    if not event_ids:
        return {}
    if projections is None:
        projections = fetch_multi_gw_projections(conn, event_ids, ensemble_weights=ensemble_weights)

    per_gw = {}
    for event_id in event_ids:
        rows = [player_row_for_gw(projections[pid], event_id) for pid in squad_ids if pid in projections]
        if len(rows) < 11:
            continue
        try:
            starting_xi, _bench, _formation = solve_starting_xi(rows, min_starter_xmins=min_starter_xmins)
        except OptimizationError:
            continue
        captain, _vice = captain_pick_for_gw(starting_xi)
        per_gw[event_id] = calculate_team_xp(starting_xi, captain)
    return per_gw


# --- Public entry point -------------------------------------------------------

def _locked_gkp_ids(squad_ids: set, projections: dict, freeze_gkp: bool, chip_active: bool) -> set:
    """Current-squad goalkeeper(s) the Set-and-Forget rule locks out of this gameweek's transfer
    ILP -- i.e. the solver can't spend a swap on them. Both squad GKPs are eligible for the lock
    (there's no "starting vs bench" distinction at squad-selection time -- see plan_transfers),
    individually lifted if that specific player is a doubt (chance_of_playing_next_round below
    GKP_FREEZE_INJURY_THRESHOLD -- None means "no doubt" and stays locked, same convention as
    optimizer.CHANCE_OF_PLAYING_DEFAULT elsewhere), or entirely lifted for the gameweek when a
    Wildcard/Free Hit is active (chip_active)."""
    if not freeze_gkp or chip_active:
        return set()
    locked = set()
    for pid in squad_ids:
        proj = projections.get(pid)
        if proj is None or proj["element_type"] != 1:
            continue
        chance = proj.get("chance_of_playing_next_round")
        if chance is not None and chance < GKP_FREEZE_INJURY_THRESHOLD:
            continue  # injury/suspension exception -- not locked
        locked.add(pid)
    return locked


def plan_transfers(
    conn,
    current_squad_ids: list,
    bank: int,
    free_transfers: int,
    horizon_gws: int = 5,
    allow_hits: bool = True,
    ensemble_weights: Optional[dict] = None,
    min_starter_xmins: Optional[float] = None,
    freeze_gkp_transfers: bool = True,
    transfer_hurdle_xp: float = DEFAULT_TRANSFER_HURDLE_XP,
    hit_transfer_hurdle_xp: float = HIT_TRANSFER_HURDLE_XP,
    chip_active_event_ids: Optional[set] = None,
) -> list:
    """ILP-driven multi-gameweek transfer roadmap.

    bank is in integer cost units (e.g. 5 == GBP 0.5m), matching now_cost. When allow_hits is
    False, transfers are capped at whatever free transfers are available that gameweek (no -4s).
    min_starter_xmins (see optimizer.STARTER_SECURITY_PROFILES) is applied to each gameweek's
    Starting XI selection, same as team_xp_by_gameweek -- it does not affect which 15 players the
    transfer ILP picks, only who among them can start.

    Three anti-churn rules gate every gameweek's transfer decision after the free/unlimited GW1
    window (see is_free_gw1_step below -- none of these three apply there, since squad-building
    before GW1's deadline isn't a "transfer" in the FPL sense at all):
      - Hurdle rate: a transfer (free or paid) only executes when its net projected gain over
        simply holding clears `transfer_hurdle_xp` -- otherwise the free transfer rolls. This also
        structurally guarantees a transfer is never taken for a net gain <= 0.0 xP. A candidate
        that actually SPENDS a hit (its transfer count exceeds that gameweek's free transfers, so
        hit_cost > 0) must clear the separate, stricter `hit_transfer_hurdle_xp` on that SAME
        already-hit-cost-net margin instead -- a -4 is a much bigger, more irreversible cost than
        simply not banking a free transfer, so unnecessary hit churn is gated far more strictly.
      - GKP freeze ("Set-and-Forget"): the squad's goalkeeper(s) are locked out of routine
        transfers unless injured/suspended or a Wildcard/Free Hit is active -- see
        `freeze_gkp_transfers`/`chip_active_event_ids`/_locked_gkp_ids.
      - Rebuy prevention: a player sold at gameweek t can't be bought back before t + 4 within
        this same roadmap, unless a Wildcard/Free Hit is active that gameweek.

    `chip_active_event_ids` (event ids where a Wildcard or Free Hit is being played) defaults to
    none active -- there's no chip-selection UI wired to this yet, only the engine-level hook.

    Returns a list of GWPlan, one per gameweek covered by the horizon.
    """
    if len(set(current_squad_ids)) != 15:
        raise OptimizationError(f"current_squad_ids must contain exactly 15 unique players, got {len(set(current_squad_ids))}.")

    event_ids = get_horizon_event_ids(conn, horizon_gws)
    if not event_ids:
        raise OptimizationError("No upcoming gameweeks found; run sync_data.py to refresh fixtures/gameweeks.")

    projections = fetch_multi_gw_projections(conn, event_ids, ensemble_weights=ensemble_weights)
    missing = [pid for pid in current_squad_ids if pid not in projections]
    if missing:
        raise OptimizationError(f"Player id(s) not found or unavailable: {missing}")

    decision_pool = [_static_player_row(proj) for proj in projections.values()]

    # Real FPL rule: squad changes before the GW1 deadline are free and unlimited -- that's an
    # initial-squad-selection window, not a "transfer" decision, so it gets no hit cost and
    # doesn't touch the free-transfer count at all. Only the *first* horizon gameweek can ever
    # be this case, since every gameweek after GW1 is necessarily past that deadline.
    gw1_still_free = is_before_gw1_deadline(conn)

    squad_ids = set(current_squad_ids)
    bank_remaining = bank
    ft_available = free_transfers
    chip_active_event_ids = chip_active_event_ids or set()
    sold_at: dict = {}  # player_id -> event_id they were sold at, for the rebuy-prevention rule
    roadmap = []

    for i, event_id in enumerate(event_ids):
        ft_before = ft_available
        lookahead_events = event_ids[i:i + len(LOOKAHEAD_WEIGHTS)]
        xp_lookup = _lookahead_scores(projections, lookahead_events)

        squad_value = sum(projections[pid]["now_cost"] for pid in squad_ids)
        budget = bank_remaining + squad_value

        is_free_gw1_step = i == 0 and event_id == 1 and gw1_still_free

        if is_free_gw1_step:
            # No "reachable via exactly t swaps" constraint -- with changes free and unlimited,
            # the only thing that matters is the single best 15 within budget/position/club
            # rules, so solve directly instead of the ILP-per-candidate-t loop below. None of the
            # hurdle/GKP-freeze/anti-churn rules apply here either -- see plan_transfers' docstring.
            scored_pool = [replace(p, projected_xp=xp_lookup.get(p.id, 0.0)) for p in decision_pool]
            new_squad = solve_squad(scored_pool, budget=budget)
            new_squad_ids = {p.id for p in new_squad}
            best = {
                "t": len(new_squad_ids - squad_ids),
                "squad_ids": new_squad_ids,
                "hit_cost": 0,
                "decision_score": sum(xp_lookup.get(pid, 0.0) for pid in new_squad_ids),
            }
        else:
            chip_active = event_id in chip_active_event_ids
            locked_ids = _locked_gkp_ids(squad_ids, projections, freeze_gkp_transfers, chip_active)
            excluded_ids = (
                set()
                if chip_active
                else {pid for pid, sold_event in sold_at.items() if event_id < sold_event + ANTI_CHURN_MIN_GAP_GWS}
            )

            candidates: dict = {}
            extra_allowed = MAX_EXTRA_PAID_TRANSFERS if allow_hits else 0
            max_t = ft_before + extra_allowed
            for t in range(0, max_t + 1):
                try:
                    candidate_ids = _solve_transfer_squad(
                        decision_pool, squad_ids, t, budget, xp_lookup,
                        locked_ids=locked_ids, excluded_ids=excluded_ids,
                    )
                except OptimizationError:
                    continue
                hit_cost = HIT_COST * max(0, t - ft_before)
                decision_score = sum(xp_lookup.get(pid, 0.0) for pid in candidate_ids) - hit_cost
                candidates[t] = {"t": t, "squad_ids": candidate_ids, "hit_cost": hit_cost, "decision_score": decision_score}

            if 0 not in candidates:
                # t=0 (hold) should always be feasible -- it's a no-op on budget/position/club
                # constraints -- but compute it defensively in case _solve_transfer_squad's
                # early-return path was somehow bypassed.
                candidates[0] = {
                    "t": 0, "squad_ids": set(squad_ids), "hit_cost": 0,
                    "decision_score": sum(xp_lookup.get(pid, 0.0) for pid in squad_ids),
                }

            # Hurdle rate: only a candidate whose margin over holding clears transfer_hurdle_xp is
            # eligible to be picked over t=0 -- see plan_transfers' docstring. A candidate that
            # actually spends a hit (hit_cost > 0, i.e. t exceeds that gameweek's free transfers)
            # must clear the separate, stricter hit_transfer_hurdle_xp instead -- but critically,
            # measured against the best FREE alternative (any t with hit_cost == 0, which is
            # sometimes t=1+ rather than t=0 itself, e.g. this gameweek already has a banked free
            # transfer), NOT against t=0 (hold) directly. Comparing a hit candidate's margin
            # against pure hold lets a genuinely good free transfer's own gain "carry" one or more
            # much weaker EXTRA hit-transfers bundled on top of it -- the bundle as a whole clears
            # the hold-vs-hurdle bar easily even when the incremental hit-transfer(s) alone barely
            # beat the free option and would never clear hit_transfer_hurdle_xp on their own merit.
            hold_score = candidates[0]["decision_score"]
            free_score = max((c["decision_score"] for c in candidates.values() if c["hit_cost"] == 0), default=hold_score)
            eligible = [
                c for t, c in candidates.items()
                if t == 0 or (
                    (c["decision_score"] - hold_score) >= transfer_hurdle_xp if c["hit_cost"] == 0
                    else (c["decision_score"] - free_score) >= hit_transfer_hurdle_xp
                )
            ]
            best = max(eligible, key=lambda c: c["decision_score"])

        if best is None:
            raise OptimizationError(f"No feasible squad found for gameweek {event_id}.")

        hit_justification_margin = (
            round(best["decision_score"] - free_score, 3)
            if not is_free_gw1_step and best["hit_cost"] > 0
            else None
        )

        transfers_in = sorted(best["squad_ids"] - squad_ids)
        transfers_out = sorted(squad_ids - best["squad_ids"])
        cost_delta = (
            sum(projections[pid]["now_cost"] for pid in transfers_in)
            - sum(projections[pid]["now_cost"] for pid in transfers_out)
        )
        bank_remaining -= cost_delta
        squad_ids = best["squad_ids"]
        if not is_free_gw1_step:
            for pid in transfers_out:
                sold_at[pid] = event_id
        # Free/unlimited GW1 changes sit entirely outside the FT economy -- ft_available carries
        # forward unchanged into GW2, rather than rolling/consuming based on a swap count that
        # was never really a "transfer" in the FPL sense.
        ft_available = ft_before if is_free_gw1_step else min(FREE_TRANSFER_CAP, max(0, ft_before - best["t"]) + 1)

        gw_squad_rows = [player_row_for_gw(projections[pid], event_id) for pid in squad_ids]
        starting_xi, bench, formation = solve_starting_xi(gw_squad_rows, min_starter_xmins=min_starter_xmins)
        captain, vice = captain_pick_for_gw(starting_xi)

        gross_points = round(sum(p.projected_xp for p in starting_xi) + captain.projected_xp, 3)
        net_points = round(gross_points - best["hit_cost"], 3)

        roadmap.append(
            GWPlan(
                event_id=event_id,
                transfers_in=[projections[pid]["web_name"] for pid in transfers_in],
                transfers_out=[projections[pid]["web_name"] for pid in transfers_out],
                transfers_in_cost=[projections[pid]["now_cost"] for pid in transfers_in],
                transfers_out_cost=[projections[pid]["now_cost"] for pid in transfers_out],
                transfers_in_ids=list(transfers_in),
                transfers_out_ids=list(transfers_out),
                free_transfers_before=ft_before,
                transfers_made=best["t"],
                hit_cost=best["hit_cost"],
                bank_remaining=bank_remaining,
                formation=formation,
                starting_xi=starting_xi,
                bench=bench,
                captain=captain,
                vice_captain=vice,
                gross_points=gross_points,
                net_points=net_points,
                initial_selection=is_free_gw1_step,
                hit_justification_margin=hit_justification_margin,
            )
        )

    return roadmap


# --- Transfer roadmap rationale generation -----------------------------------

def _rationale_player_lookup(conn, roadmap: list, current_squad: list, horizon_gws: int):
    """Multi-GW projections plus a resolver that can produce a display PlayerRow for any player id
    referenced anywhere in the roadmap -- including an OUT player who no longer appears in any
    later roadmap step once they've been transferred away, so their fixture/underlying-stats data
    would otherwise be lost. Falls back to the current_squad snapshot for anyone the fresh
    projections fetch doesn't cover (e.g. a player who leaves the pool entirely, such as an
    injury/unavailability status change)."""
    event_ids = get_horizon_event_ids(conn, horizon_gws)
    projections = fetch_multi_gw_projections(conn, event_ids) if event_ids else {}

    def avg_difficulty(pid) -> Optional[float]:
        proj = projections.get(pid)
        if proj is None:
            return None
        difficulties = [proj["gw_difficulty"][eid] for eid in event_ids if proj["gw_has_fixture"].get(eid)]
        return round(sum(difficulties) / len(difficulties), 2) if difficulties else None

    def player_for_rationale(pid) -> Optional[PlayerRow]:
        proj = projections.get(pid)
        if proj is not None:
            # Prefer a specific gameweek's row (via player_row_for_gw) over the gw-invariant
            # _static_player_row when a fixture exists in the horizon -- only that form carries
            # an xp_breakdown (cs_prob/defcon_prob), which the position-aware Underlying Stats
            # bullet below needs for GKP/DEF comparisons. Falls back to the static row for a
            # player with no fixture anywhere in the horizon (e.g. mid-horizon blank gameweek).
            representative_event = next((eid for eid in event_ids if proj["gw_has_fixture"].get(eid)), None)
            if representative_event is not None:
                return player_row_for_gw(proj, representative_event)
            return _static_player_row(proj)
        return next((p for p in current_squad if p.id == pid), None)

    return avg_difficulty, player_for_rationale


def _underlying_stats_bullet(out_player, in_player) -> RationaleBullet:
    """Position-aware 'Underlying Stats' bullet -- ranked on whichever metrics actually decide a
    transfer at that position, not one generic figure for every position:
      - GKP transfer (either side): save rate + clean sheet odds + budget value.
      - DEF transfer (either side, no GKP involved): clean sheet horizon + DEFCON floor +
        attacking threat.
      - MID/FWD transfer: xGI + open-play threat (existing xGI-based comparison; set pieces are
        already covered by the separate Set Pieces & Role bullet above).
    Falls back to the plain xGI comparison whenever a position-specific breakdown (cs_prob/
    defcon_prob) isn't available for both sides -- e.g. a player with no fixture anywhere in the
    horizon, so only the gw-invariant static row (no xp_breakdown) could be resolved for them.
    """
    positions = {out_player.element_type, in_player.element_type}
    out_b, in_b = out_player.xp_breakdown, in_player.xp_breakdown
    xp_gain = in_player.projected_xp - out_player.projected_xp
    gain_note = f" Net projected gain of {xp_gain:+.1f} xP this gameweek."

    if 1 in positions and out_b is not None and in_b is not None:  # GKP transfer
        return RationaleBullet(
            text=(
                f"{in_player.web_name} ({in_b.cs_prob * 100:.0f}% CS odds, {in_player.saves_per_90:.1f} saves/90, "
                f"£{in_player.cost_millions:.1f}m) vs {out_player.web_name} ({out_b.cs_prob * 100:.0f}% CS odds, "
                f"{out_player.saves_per_90:.1f} saves/90) -- ranked on save rate, clean sheet odds, and budget value."
                + gain_note
            ),
            tags=[],
        )
    if 2 in positions and out_b is not None and in_b is not None:  # DEF transfer (no GKP)
        return RationaleBullet(
            text=(
                f"{in_player.web_name} ({in_b.cs_prob * 100:.0f}% CS odds, {in_b.defcon_prob * 100:.0f}% DEFCON floor, "
                f"{in_player.xg_per_90:.2f} xG90/{in_player.xa_per_90:.2f} xA90) vs {out_player.web_name} "
                f"({out_b.cs_prob * 100:.0f}% CS odds, {out_b.defcon_prob * 100:.0f}% DEFCON floor) -- ranked on "
                f"clean sheet horizon, DEFCON floor, and attacking threat." + gain_note
            ),
            tags=[],
        )
    out_xgi, in_xgi = xgi_per_90(out_player), xgi_per_90(in_player)  # MID/FWD, or no breakdown available
    return RationaleBullet(
        text=(
            f"{in_player.web_name} ({in_xgi:.2f} xGI/90) vs {out_player.web_name} ({out_xgi:.2f} xGI/90) "
            f"-- ranked on xGI and open-play threat." + gain_note
        ),
        tags=["⚡ High xGI"] if in_xgi >= RATIONALE_HIGH_XGI_PER_90 else [],
    )


def _transfer_pair_rationale(out_player, in_player, out_fdr, in_fdr, horizon_gws: int) -> list:
    """Fixtures & Swings / Set Pieces & Role / (position-aware) Underlying Stats bullets for one
    OUT->IN pair."""
    bullets = []

    if out_fdr is not None and in_fdr is not None:
        tags = []
        text = f"{out_player.web_name} faces an avg FDR of {out_fdr:.1f} over the next {horizon_gws} GWs"
        if out_fdr >= RATIONALE_FIXTURE_RUN_TOUGH:
            text += " (a tough run)"
        text += f" -> {in_player.web_name} averages FDR {in_fdr:.1f}"
        if in_fdr <= RATIONALE_FIXTURE_RUN_FAVORABLE:
            text += " (a favorable run)"
            tags.append("📅 Fixture Swing")
        bullets.append(RationaleBullet(text=text + ".", tags=tags))

    if has_set_piece_duty(in_player):
        if has_set_piece_duty(out_player):
            text = f"{in_player.web_name} continues to hold first-choice {set_piece_label(in_player)} duty."
        else:
            text = (
                f"{in_player.web_name} adds first-choice {set_piece_label(in_player)} duty that "
                f"{out_player.web_name} didn't carry."
            )
        bullets.append(RationaleBullet(text=text, tags=["🎯 Set Pieces"]))

    bullets.append(_underlying_stats_bullet(out_player, in_player))
    return bullets


def _hit_or_roll_rationale(plan: GWPlan) -> "RationaleBullet":
    """Justifies a -4 (or larger) hit by its expected point delta, or explains banking a free
    transfer when the roadmap chooses to hold instead.

    Bug found live: this used to quote plan.net_points itself as "the net gain" that justified a
    hit -- but net_points is this single gameweek's own absolute score, not a margin over
    anything. A hit-taking squad is chosen because it wins on plan_transfers' lookahead-weighted
    decision_score across several gameweeks (see LOOKAHEAD_WEIGHTS), which can legitimately mean
    THIS gameweek's own net_points ends up lower than the best free-only alternative would have
    scored -- the hit pays for itself over the next gameweek or two, not necessarily this one. The
    old wording read as "you're net_points points better off," which is simply false when that's
    the case, and is exactly the kind of thing that looks like a self-contradictory recommendation
    from the outside. Now quotes plan.hit_justification_margin -- the actual lookahead-weighted
    margin over the best hit_cost == 0 alternative that the hurdle rate checked -- and says
    explicitly that it's a multi-gameweek margin, not this week's own points."""
    if plan.transfers_made and plan.hit_cost:
        if plan.hit_justification_margin is not None:
            margin_text = (
                f"a projected +{plan.hit_justification_margin:.1f} xP margin over the best FREE "
                f"alternative that gameweek, weighted across the next few gameweeks' fixtures -- "
                f"NOT necessarily GW{plan.event_id}'s own net points alone, which can legitimately "
                f"come out lower than the free alternative would have scored this single week; the "
                f"hit is expected to pay for itself over the following gameweek(s) instead"
            )
        else:
            margin_text = f"a net gain of {plan.net_points:+.1f} xP after the hit"
        return RationaleBullet(
            text=(
                f"GW{plan.event_id}: taking a -{plan.hit_cost}pt hit for {plan.transfers_made} transfer(s) "
                f"is justified by {margin_text} -- the roadmap only ever takes a hit when the ILP finds "
                f"it clears the stricter hit-only hurdle."
            ),
            tags=[],
        )
    if plan.transfers_made:
        return RationaleBullet(
            text=(
                f"GW{plan.event_id}: {plan.transfers_made} transfer(s) made using free transfers only "
                f"(no hit), from {plan.free_transfers_before} FT available before this move."
            ),
            tags=[],
        )
    banked = min(FREE_TRANSFER_CAP, plan.free_transfers_before + 1)
    return RationaleBullet(
        text=(
            f"GW{plan.event_id}: Hold / Roll Transfer -- banking the free transfer rather than forcing a "
            f"move, carrying {banked} FT (capped at {FREE_TRANSFER_CAP}) into the next gameweek."
        ),
        tags=[],
    )


def generate_transfer_rationale(conn, roadmap: list, current_squad: list, horizon_gws: int = 4) -> list:
    """Metric-driven, plain-English explanation for each step of a plan_transfers roadmap:
    Fixtures & Swings, Set Pieces & Role, Underlying Stats (net xP gain) for every suggested
    transfer, plus Hit/Roll Logic justifying -4 hits or explaining a banked free transfer.

    Deviates from the spec's literal `(transfer_plan, current_squad_df, horizon_gws=4)` signature
    the same way optimizer.generate_squad_rationale deviates from its own spec: `current_squad_df`
    becomes `current_squad: list[PlayerRow]` to match this module's list[PlayerRow] convention
    everywhere else, and `conn` is added (not in the literal spec signature) so OUT players --
    who no longer appear in any later roadmap step once transferred away -- can still be resolved
    to real multi-GW fixture/projection data via a fresh fetch_multi_gw_projections call, rather
    than only whatever fixture figure happened to be attached to their very last appearance.

    Returns one dict per non-initial-selection roadmap step:
      {"event_id", "transfers": [{"out": PlayerRow, "in": PlayerRow, "bullets": [RationaleBullet]}],
       "hit_roll_bullet": RationaleBullet}
    (the free/unlimited pre-GW1 "initial_selection" step, if present, is skipped -- there's no
    hit/roll or fixture-swing decision to explain for a squad's very first pick.)
    """
    steps = [plan for plan in roadmap if not plan.initial_selection]
    if not steps:
        return []

    avg_difficulty, player_for_rationale = _rationale_player_lookup(conn, roadmap, current_squad, horizon_gws)

    results = []
    for plan in steps:
        transfers = []
        for out_id, in_id in zip(plan.transfers_out_ids, plan.transfers_in_ids):
            out_player = player_for_rationale(out_id)
            in_player = player_for_rationale(in_id)
            if out_player is None or in_player is None:
                continue
            bullets = _transfer_pair_rationale(
                out_player, in_player, avg_difficulty(out_id), avg_difficulty(in_id), horizon_gws
            )
            transfers.append({"out": out_player, "in": in_player, "bullets": bullets})

        results.append({
            "event_id": plan.event_id,
            "transfers": transfers,
            "hit_roll_bullet": _hit_or_roll_rationale(plan),
        })

    return results

"""Real assertion-based regression tests for the MILP constraints in src/optimizer.py -- formation
locking, the Press Conference/Injury Flag hard-exclusion gate, the Vice-Captain Lock, the Sub 1
Security Constraint, and the basic squad-building constraints (budget/position/club limits) that
test_optimizer.py (the repo-root manual smoke script) only ever prints for a human to eyeball.
"""
from collections import Counter

import pytest

from src.optimizer import (
    EP_NEXT_BLEND_FADE_OUT_STARTS,
    EP_NEXT_BLEND_MAX_WEIGHT,
    FORMATION_CHOICES,
    GW1_CAPTAIN_MIN_COST,
    MAX_PLAYERS_PER_TEAM,
    RECENT_FORM_MIN_GAMES,
    RECENT_FORM_WINDOW_GAMES,
    SQUAD_POSITION_COUNTS,
    VICE_CAPTAIN_XMINS_FLOOR,
    OptimizationError,
    PlayerRow,
    _blend_player_xga,
    _formation_label,
    _is_hard_excluded,
    _projected_minutes_fraction,
    blend_ep_next_fallback,
    calculate_positional_xp,
    captaincy_candidates,
    captaincy_score,
    ep_next_blend_weight,
    is_captaincy_eligible,
    is_cold_start_pool,
    is_vice_eligible,
    order_bench,
    parse_formation_lock,
    recent_form_rate,
    solve_starting_xi,
    solve_starting_xi_with_fallback,
)
from tests.conftest import make_player


# --- Formation Lock -----------------------------------------------------------------------------

@pytest.mark.parametrize("formation", [f for f in FORMATION_CHOICES if f != "Auto (Best xP)"])
def test_formation_lock_produces_exact_shape(synthetic_squad, formation):
    starting_xi, bench, label = solve_starting_xi(synthetic_squad, formation_lock=formation)
    assert label == formation
    assert _formation_label(starting_xi) == formation
    assert len(starting_xi) == 11
    assert len(bench) == 4


def test_formation_lock_auto_uses_flexible_bounds(synthetic_squad):
    _starting_xi, _bench, label = solve_starting_xi(synthetic_squad, formation_lock="Auto (Best xP)")
    def_n, mid_n, fwd_n = (int(x) for x in label.split("-"))
    assert 3 <= def_n <= 5
    assert 2 <= mid_n <= 5
    assert 1 <= fwd_n <= 3


def test_formation_lock_invalid_raises():
    with pytest.raises(OptimizationError):
        parse_formation_lock("not-a-formation")


def test_formation_lock_none_and_auto_label_both_parse_to_none():
    assert parse_formation_lock(None) is None
    assert parse_formation_lock("Auto (Best xP)") is None


def test_formation_lock_valid_shapes_parse_to_exact_tuple():
    assert parse_formation_lock("3-5-2") == (3, 5, 2)
    assert parse_formation_lock("5-2-3") == (5, 2, 3)


# --- Press Conference / Injury Flag hard-exclusion -----------------------------------------------

def test_hard_excluded_player_never_starts(synthetic_pool):
    # Take the single highest-xP forward (the solver would pick them by default) and mark them
    # unavailable -- if the hard-exclusion constraint works, they must never appear in the XI even
    # though they're still in the 15-man squad (see _solve_lineup_milp's docstring: hard-exclusion
    # only ever blocks s[p.id], never squad membership x[p.id]).
    forwards = sorted((p for p in synthetic_pool if p.element_type == 4), key=lambda p: p.projected_xp)
    best_fwd = forwards[-1]
    best_fwd.status = "i"
    assert _is_hard_excluded(best_fwd)

    from src.optimizer import solve_squad

    squad = solve_squad(synthetic_pool, locked_ids={best_fwd.id})
    assert best_fwd.id in {p.id for p in squad}  # locked into the SQUAD...

    starting_xi, bench, _formation = solve_starting_xi(squad)
    assert best_fwd.id not in {p.id for p in starting_xi}  # ...but never starts
    assert best_fwd.id in {p.id for p in bench}

    ordered_bench = order_bench(bench)
    # and never lands in Sub 1 (the front of the ordered outfield bench), even though their raw
    # xP would otherwise put them there -- order_bench pushes hard-excluded players to the back.
    outfield_bench = [p for p in ordered_bench if p.element_type != 1]
    if outfield_bench:
        assert outfield_bench[0].id != best_fwd.id


def test_chance_zero_also_hard_excludes(synthetic_pool):
    mids = [p for p in synthetic_pool if p.element_type == 3]
    target = mids[-1]
    target.chance_of_playing_next_round = 0
    assert _is_hard_excluded(target)


# --- Vice-Captain Lock ---------------------------------------------------------------------------

def test_vice_eligible_requires_xmins_floor():
    eligible = make_player(9001, 3, 1, cost=80, xp=6.0, xmins=VICE_CAPTAIN_XMINS_FLOOR, chance_of_playing_next_round=100)
    ineligible = make_player(9002, 3, 1, cost=80, xp=6.0, xmins=VICE_CAPTAIN_XMINS_FLOOR - 1, chance_of_playing_next_round=100)
    assert is_vice_eligible(eligible)
    assert not is_vice_eligible(ineligible)


def test_vice_eligible_treats_none_chance_as_eligible():
    # Deliberately relaxed vs _is_minutes_secure's strict reading -- see optimizer.py's own
    # comments on why (a strict-only reading leaves almost nobody eligible pool-wide).
    p = make_player(9003, 3, 1, cost=80, xp=6.0, xmins=VICE_CAPTAIN_XMINS_FLOOR, chance_of_playing_next_round=None)
    assert is_vice_eligible(p)


def test_vice_eligible_rejects_chance_75():
    p = make_player(9004, 3, 1, cost=80, xp=6.0, xmins=VICE_CAPTAIN_XMINS_FLOOR, chance_of_playing_next_round=75)
    assert not is_vice_eligible(p)


def test_get_captain_recommendations_falls_back_when_nobody_is_vice_eligible(synthetic_squad):
    # Every starter in synthetic_pool has xmins=80 (< VICE_CAPTAIN_XMINS_FLOOR=85) by default, so
    # NONE are strictly vice-eligible here -- this deliberately exercises the documented fallback
    # (plain 2nd-highest-xP starter) rather than the eligible branch, proving a vice is still
    # always returned instead of None when the pool can't clear the eligibility bar at all.
    from src.optimizer import get_captain_recommendations

    class _FakeConn:
        """get_captain_recommendations calls fetch_players(conn), which queries the DB -- stub
        it out (see the monkeypatch below) so this stays a pure in-memory test."""

    import src.optimizer as optimizer_module

    original_fetch_players = optimizer_module.fetch_players
    try:
        optimizer_module.fetch_players = lambda conn, ensemble_weights=None: synthetic_squad
        rec = get_captain_recommendations(_FakeConn(), [p.id for p in synthetic_squad])
    finally:
        optimizer_module.fetch_players = original_fetch_players

    assert rec.get("vice_captain") is not None


def test_vice_lock_prefers_eligible_starter_when_one_exists():
    # A squad where the top-2-by-xP starter is xmins-thin but a 3rd option clears the floor --
    # the vice should skip the plain 2nd place and pick the eligible one instead.
    from src.optimizer import get_captain_recommendations

    squad = [
        make_player(1, 1, 1, cost=45, xp=3.0, xmins=90),
        make_player(2, 1, 2, cost=40, xp=2.5, xmins=90),
        make_player(3, 2, 1, cost=45, xp=5.0, xmins=90),
        make_player(4, 2, 2, cost=45, xp=4.0, xmins=90),
        make_player(5, 2, 3, cost=45, xp=3.5, xmins=90),
        make_player(6, 2, 4, cost=45, xp=3.0, xmins=90),
        make_player(7, 2, 5, cost=40, xp=2.5, xmins=90),
        make_player(8, 3, 1, cost=90, xp=8.0, xmins=90),  # captain: clear top scorer
        make_player(9, 3, 2, cost=60, xp=6.0, xmins=50),  # plain 2nd-highest, but xmins-thin
        make_player(10, 3, 3, cost=55, xp=5.5, xmins=90, chance_of_playing_next_round=100),  # vice-eligible
        make_player(11, 3, 4, cost=50, xp=4.0, xmins=90),
        make_player(12, 3, 5, cost=45, xp=3.5, xmins=90),
        make_player(13, 4, 1, cost=70, xp=4.5, xmins=90),
        make_player(14, 4, 2, cost=55, xp=3.5, xmins=90),
        make_player(15, 4, 3, cost=45, xp=3.0, xmins=90),
    ]

    class _FakeConn:
        pass

    import src.optimizer as optimizer_module

    original_fetch_players = optimizer_module.fetch_players
    try:
        optimizer_module.fetch_players = lambda conn, ensemble_weights=None: squad
        rec = get_captain_recommendations(_FakeConn(), [p.id for p in squad])
    finally:
        optimizer_module.fetch_players = original_fetch_players

    assert rec["captain"]["player"].id == 8
    assert rec["vice_captain"]["player"].id == 10  # not 9 -- 9 fails the xmins floor


# --- Sub 1 Security Constraint & bench ordering --------------------------------------------------

def test_order_bench_puts_highest_xp_outfield_player_first(synthetic_pool):
    from src.optimizer import solve_squad

    squad = solve_squad(synthetic_pool)
    _starting_xi, bench, _formation = solve_starting_xi(squad)
    ordered = order_bench(list(bench))

    outfield = [p for p in ordered if p.element_type != 1]
    if len(outfield) >= 2:
        assert outfield[0].projected_xp >= outfield[1].projected_xp
    if outfield:
        assert ordered[0].element_type == 1 or outfield[0] is ordered[0]  # GKP (if present) leads


# --- Basic squad-building constraints -------------------------------------------------------------

def test_solve_squad_respects_budget_position_and_club_limits(synthetic_pool):
    from src.optimizer import BUDGET_LIMIT, solve_squad

    squad = solve_squad(synthetic_pool)
    assert len(squad) == 15
    assert sum(p.now_cost for p in squad) <= BUDGET_LIMIT

    counts = Counter(p.element_type for p in squad)
    for element_type, required in SQUAD_POSITION_COUNTS.items():
        assert counts[element_type] == required

    team_counts = Counter(p.team_id for p in squad)
    assert max(team_counts.values()) <= MAX_PLAYERS_PER_TEAM


def test_solve_squad_honors_locked_and_excluded_ids(synthetic_pool):
    from src.optimizer import solve_squad

    gkp_ids = [p.id for p in synthetic_pool if p.element_type == 1]
    lock_id, exclude_id = gkp_ids[0], gkp_ids[1]

    squad = solve_squad(synthetic_pool, locked_ids={lock_id}, excluded_ids={exclude_id})
    squad_ids = {p.id for p in squad}
    assert lock_id in squad_ids
    assert exclude_id not in squad_ids


# --- Captaincy Position & Talisman Filtering -------------------------------------------------------

def test_is_captaincy_eligible_positions():
    gkp = make_player(1, 1, 1, cost=50, xp=5.0)
    def_no_duty = make_player(2, 2, 1, cost=55, xp=5.0)
    def_penalty_taker = make_player(3, 2, 1, cost=55, xp=5.0, penalties_order=1)
    def_corner_taker = make_player(4, 2, 1, cost=55, xp=5.0, corners_order=1)
    mid = make_player(5, 3, 1, cost=55, xp=5.0)
    fwd = make_player(6, 4, 1, cost=55, xp=5.0)

    assert not is_captaincy_eligible(gkp)
    assert not is_captaincy_eligible(def_no_duty)
    assert is_captaincy_eligible(def_penalty_taker)
    assert is_captaincy_eligible(def_corner_taker)
    assert is_captaincy_eligible(mid)
    assert is_captaincy_eligible(fwd)


def test_gkp_with_penalty_duty_is_still_never_captaincy_eligible():
    # penalties_order == 1 clears the gate for a DEF, but GKP has no such carve-out at all.
    gkp_pens = make_player(1, 1, 1, cost=50, xp=5.0, penalties_order=1)
    assert not is_captaincy_eligible(gkp_pens)


def test_is_vice_eligible_requires_captaincy_position_too():
    def_high_minutes = make_player(
        1, 2, 1, cost=55, xp=5.0, xmins=VICE_CAPTAIN_XMINS_FLOOR, chance_of_playing_next_round=100,
    )
    assert not is_vice_eligible(def_high_minutes)  # DEF, no set-piece duty -- fails the position gate first

    def_with_duty = make_player(
        2, 2, 1, cost=55, xp=5.0, xmins=VICE_CAPTAIN_XMINS_FLOOR, chance_of_playing_next_round=100,
        penalties_order=1,
    )
    assert is_vice_eligible(def_with_duty)


# --- Talisman Penalty-Taker boost -------------------------------------------------------------------

def test_captaincy_score_boosts_favorable_penalty_taker_only():
    baseline = make_player(1, 4, 1, cost=90, xp=8.0, fixture_difficulty=3.0)
    favorable_easy_fdr = make_player(2, 4, 1, cost=90, xp=8.0, penalties_order=1, fixture_difficulty=2.0)
    favorable_home = make_player(3, 4, 1, cost=90, xp=8.0, penalties_order=1, fixture_difficulty=4.0, is_home=True)
    unfavorable = make_player(4, 4, 1, cost=90, xp=8.0, penalties_order=1, fixture_difficulty=4.0, is_home=False)

    assert captaincy_score(baseline) == pytest.approx(8.0)
    assert captaincy_score(favorable_easy_fdr) == pytest.approx(8.0 * 1.15)
    assert captaincy_score(favorable_home) == pytest.approx(8.0 * 1.15)
    assert captaincy_score(unfavorable) == pytest.approx(8.0)  # penalty taker, but tough away fixture


# --- GW1 Pre-Season Cold-Start Anchor ----------------------------------------------------------------

def test_is_cold_start_pool():
    cold = [make_player(1, 3, 1, cost=90, xp=8.0, starts=0), make_player(2, 4, 1, cost=90, xp=8.0, starts=0)]
    warm = [make_player(1, 3, 1, cost=90, xp=8.0, starts=0), make_player(2, 4, 1, cost=90, xp=8.0, starts=5)]
    assert is_cold_start_pool(cold)
    assert not is_cold_start_pool(warm)
    assert not is_cold_start_pool([])


def test_captaincy_candidates_no_restriction_outside_cold_start():
    premium = make_player(1, 4, 1, cost=GW1_CAPTAIN_MIN_COST, xp=6.0, starts=5)
    cheap = make_player(2, 3, 2, cost=45, xp=1.0, starts=5)
    ids = captaincy_candidates([premium, cheap])
    assert ids == {premium.id, cheap.id}


def test_captaincy_candidates_restricts_to_premium_or_top_n_during_cold_start():
    premium = make_player(1, 4, 1, cost=GW1_CAPTAIN_MIN_COST, xp=6.0, starts=0)
    mid_a = make_player(2, 3, 2, cost=45, xp=5.5, starts=0)
    mid_b = make_player(3, 3, 3, cost=45, xp=5.0, starts=0)
    mid_c = make_player(4, 3, 4, cost=45, xp=4.5, starts=0)  # 4th-best, not premium -- excluded
    mid_d = make_player(5, 3, 5, cost=45, xp=4.0, starts=0)  # 5th-best, not premium -- excluded
    ids = captaincy_candidates([premium, mid_a, mid_b, mid_c, mid_d])
    assert premium.id in ids
    assert mid_a.id in ids
    assert mid_b.id in ids
    assert mid_c.id not in ids
    assert mid_d.id not in ids


# --- GW1 Pre-Season Cold-Start Minutes-Fraction Fallback ----------------------------------------------
# Bug found via a real GW1 2026-27 squad: calculate_positional_xp scales appearance_xp/attack_xp/
# defensive_xp/bonus_xp by _projected_minutes_fraction, which (before this fix) read starts_per_90
# ONLY -- necessarily 0.0 for every player before a single match of the season has been played, so
# EVERY player's projected_xp came out at exactly 0.0 pre-season, including nailed, obviously-
# starting superstars. That made captaincy_score/captaincy_candidates' ranking pure tie-break noise
# (whoever happened to sort first) at exactly the real GW1 deadline, independent of and on top of
# GW1_COLD_START_PRICE_BONUS's own multiplicative-on-a-zero-base issue. calculate_baseline_xmins
# already had a correct pre-season fallback (_preseason_starts_rate_fallback); this reuses it.

def test_projected_minutes_fraction_falls_back_to_price_ownership_proxy_pre_season():
    nailed_premium = make_player(1, 4, 1, cost=150, xp=0.0, starts=0, starts_per_90=0.0)
    clear_backup = make_player(2, 1, 2, cost=40, xp=0.0, starts=0, starts_per_90=0.0, selected_by_percent=1.0)

    assert _projected_minutes_fraction(nailed_premium, games_played=0) > 0.9
    assert _projected_minutes_fraction(clear_backup, games_played=0) < 0.2
    # And decisively higher for the nailed player -- not a coin flip between the two.
    assert (
        _projected_minutes_fraction(nailed_premium, games_played=0)
        > _projected_minutes_fraction(clear_backup, games_played=0)
    )


def test_projected_minutes_fraction_uses_real_starts_rate_once_the_season_is_underway():
    # Once games have actually been played, price/ownership no longer matters -- a rotation-risk
    # player's own real starts_per_90 rate governs directly, same as before this fix.
    rotation_risk = make_player(1, 4, 1, cost=150, xp=0.0, starts=2, starts_per_90=0.3)
    assert _projected_minutes_fraction(rotation_risk, games_played=5) == pytest.approx(0.3)


def test_calculate_positional_xp_is_not_flat_zero_pre_season_for_an_established_player():
    # True cold start: zero accumulated rate data of every kind, matching a genuine pre-GW1 squad.
    haaland_like = make_player(
        1, 4, 1, cost=155, xp=0.0, starts=0, starts_per_90=0.0,
        xg_per_90=0.0, xa_per_90=0.0, defensive_contribution_per_90=0.0,
    )
    breakdown = calculate_positional_xp(haaland_like, fixture_difficulty=3.0, games_played=0)
    assert breakdown.total > 0.0
    assert breakdown.appearance_xp > 0.0


def test_calculate_positional_xp_still_reads_a_clear_cheap_backup_as_low_minutes_pre_season():
    nailed = make_player(
        1, 4, 1, cost=155, xp=0.0, starts=0, starts_per_90=0.0,
        xg_per_90=0.0, xa_per_90=0.0, defensive_contribution_per_90=0.0,
    )
    backup = make_player(
        2, 1, 2, cost=40, xp=0.0, starts=0, starts_per_90=0.0, selected_by_percent=1.0,
        xg_per_90=0.0, xa_per_90=0.0, saves_per_90=0.0, defensive_contribution_per_90=0.0,
    )
    nailed_breakdown = calculate_positional_xp(nailed, fixture_difficulty=3.0, games_played=0)
    backup_breakdown = calculate_positional_xp(backup, fixture_difficulty=3.0, games_played=0)
    assert nailed_breakdown.appearance_xp > backup_breakdown.appearance_xp


def test_calculate_positional_xp_uses_real_data_once_the_season_is_underway_ignoring_price():
    # games_played > 0 -- back to reading the player's own real (zero) starts_per_90 directly,
    # regardless of price, exactly like before this fix (a genuinely benched premium stays at 0).
    benched_premium = make_player(
        1, 4, 1, cost=155, xp=0.0, starts=0, starts_per_90=0.0,
        xg_per_90=0.0, xa_per_90=0.0, defensive_contribution_per_90=0.0,
    )
    breakdown = calculate_positional_xp(benched_premium, fixture_difficulty=3.0, games_played=5)
    assert breakdown.total == 0.0
    assert breakdown.appearance_xp == 0.0


# --- Built-in ep_next fallback blend --------------------------------------------------------------
# Even with the minutes-fraction fallback above, the internal model's own xG/xA-rate terms are
# still exactly 0.0 (or built from a tiny, noisy sample) early in a player's own season -- these
# tests cover blending in FPL's own ep_next to compensate, fading out as real starts accumulate.

def test_ep_next_blend_weight_is_maximal_at_zero_starts_and_fades_to_zero():
    assert ep_next_blend_weight(0) == pytest.approx(EP_NEXT_BLEND_MAX_WEIGHT)
    assert ep_next_blend_weight(EP_NEXT_BLEND_FADE_OUT_STARTS) == 0.0
    assert ep_next_blend_weight(EP_NEXT_BLEND_FADE_OUT_STARTS + 5) == 0.0
    mid = ep_next_blend_weight(EP_NEXT_BLEND_FADE_OUT_STARTS // 2)
    assert 0.0 < mid < EP_NEXT_BLEND_MAX_WEIGHT


def test_blend_ep_next_fallback_pulls_a_true_cold_start_total_toward_ep_next():
    cold = make_player(
        1, 4, 1, cost=155, xp=0.0, starts=0, starts_per_90=0.0,
        xg_per_90=0.0, xa_per_90=0.0, defensive_contribution_per_90=0.0,
    )
    breakdown = calculate_positional_xp(cold, fixture_difficulty=3.0, games_played=0)
    internal_only = breakdown.total

    blended = blend_ep_next_fallback(breakdown, ep_next=8.0, starts=0)
    assert blended.blended is True
    assert blended.external_xp == pytest.approx(8.0)
    # Pulled meaningfully toward ep_next=8.0, not left at the (much lower) internal-only figure.
    assert blended.total > internal_only
    assert blended.total == pytest.approx(EP_NEXT_BLEND_MAX_WEIGHT * 8.0 + (1 - EP_NEXT_BLEND_MAX_WEIGHT) * internal_only)


def test_blend_ep_next_fallback_has_no_effect_once_starts_have_faded_it_out():
    warm = make_player(1, 4, 1, cost=155, xp=0.0, starts=EP_NEXT_BLEND_FADE_OUT_STARTS, starts_per_90=0.5)
    breakdown = calculate_positional_xp(warm, fixture_difficulty=3.0, games_played=10)
    blended = blend_ep_next_fallback(breakdown, ep_next=8.0, starts=EP_NEXT_BLEND_FADE_OUT_STARTS)
    assert blended.total == breakdown.total
    assert blended.blended is False


def test_blend_ep_next_fallback_no_op_when_ep_next_is_none():
    cold = make_player(1, 4, 1, cost=155, xp=0.0, starts=0, starts_per_90=0.0, xg_per_90=0.0, xa_per_90=0.0)
    breakdown = calculate_positional_xp(cold, fixture_difficulty=3.0, games_played=0)
    blended = blend_ep_next_fallback(breakdown, ep_next=None, starts=0)
    assert blended is breakdown


# --- get_captain_recommendations integration ----------------------------------------------------------

def _fake_conn_with_squad(squad):
    class _FakeConn:
        pass

    import src.optimizer as optimizer_module

    original_fetch_players = optimizer_module.fetch_players
    optimizer_module.fetch_players = lambda conn, ensemble_weights=None: squad
    return _FakeConn(), optimizer_module, original_fetch_players


def test_get_captain_recommendations_never_picks_gkp_even_with_highest_raw_xp():
    from src.optimizer import get_captain_recommendations

    squad = [
        make_player(1, 1, 1, cost=55, xp=15.0, xmins=90),  # GKP with an absurd raw xP -- never captain
        make_player(2, 1, 2, cost=40, xp=2.0, xmins=90),
        make_player(3, 2, 1, cost=45, xp=4.0, xmins=90),
        make_player(4, 2, 2, cost=45, xp=3.5, xmins=90),
        make_player(5, 2, 3, cost=45, xp=3.0, xmins=90),
        make_player(6, 2, 4, cost=45, xp=2.5, xmins=90),
        make_player(7, 2, 5, cost=40, xp=2.0, xmins=90),
        make_player(8, 3, 1, cost=90, xp=8.0, xmins=90),  # expected captain: top eligible scorer
        make_player(9, 3, 2, cost=60, xp=6.0, xmins=90, chance_of_playing_next_round=100),
        make_player(10, 3, 3, cost=55, xp=5.5, xmins=90, chance_of_playing_next_round=100),
        make_player(11, 3, 4, cost=50, xp=4.0, xmins=90),
        make_player(12, 3, 5, cost=45, xp=3.5, xmins=90),
        make_player(13, 4, 1, cost=70, xp=4.5, xmins=90),
        make_player(14, 4, 2, cost=55, xp=3.5, xmins=90),
        make_player(15, 4, 3, cost=45, xp=3.0, xmins=90),
    ]
    fake_conn, optimizer_module, original = _fake_conn_with_squad(squad)
    try:
        rec = get_captain_recommendations(fake_conn, [p.id for p in squad])
    finally:
        optimizer_module.fetch_players = original

    assert rec["captain"]["player"].id == 8
    assert rec["captain"]["player"].element_type in (3, 4)


def test_get_captain_recommendations_gw1_cold_start_prefers_premium_over_marginal_cheap_edge():
    from src.optimizer import get_captain_recommendations

    # Every player has starts=0 -- a genuine GW1 cold start. The cheap MID (id=8) has a very
    # slightly higher raw xP than the premium MID (id=9) purely from pre-season noise -- without
    # the cold-start anchor's price boost, this cheap punt would win the armband outright.
    squad = [
        make_player(1, 1, 1, cost=45, xp=3.0, xmins=90, starts=0),
        make_player(2, 1, 2, cost=40, xp=2.5, xmins=90, starts=0),
        make_player(3, 2, 1, cost=45, xp=4.0, xmins=90, starts=0),
        make_player(4, 2, 2, cost=45, xp=3.5, xmins=90, starts=0),
        make_player(5, 2, 3, cost=45, xp=3.0, xmins=90, starts=0),
        make_player(6, 2, 4, cost=45, xp=2.5, xmins=90, starts=0),
        make_player(7, 2, 5, cost=40, xp=2.0, xmins=90, starts=0),
        make_player(8, 3, 1, cost=45, xp=6.05, xmins=90, starts=0),  # cheap punt, marginally highest raw xP
        make_player(9, 3, 2, cost=100, xp=6.0, xmins=90, starts=0),  # premium -- should win via cold-start anchor
        make_player(10, 3, 3, cost=55, xp=5.5, xmins=90, starts=0),
        make_player(11, 3, 4, cost=50, xp=4.0, xmins=90, starts=0),
        make_player(12, 3, 5, cost=45, xp=3.5, xmins=90, starts=0),
        make_player(13, 4, 1, cost=70, xp=4.5, xmins=90, starts=0),
        make_player(14, 4, 2, cost=55, xp=3.5, xmins=90, starts=0),
        make_player(15, 4, 3, cost=45, xp=3.0, xmins=90, starts=0),
    ]
    fake_conn, optimizer_module, original = _fake_conn_with_squad(squad)
    try:
        rec = get_captain_recommendations(fake_conn, [p.id for p in squad])
    finally:
        optimizer_module.fetch_players = original

    assert rec["captain"]["player"].id == 9


# --- solve_starting_xi_with_fallback -----------------------------------------------------------
# Real bug found live: several squad members genuinely below the sidebar's Starter Security floor
# at once (thin/no recent minutes) made solve_starting_xi raise OptimizationError outright, which
# used to just dead-end the whole page -- no graceful way forward short of a user manually finding
# the sidebar control themselves. This mirrors src/backtest.py's own fallback chain, but anchored
# at whatever floor the caller actually requested (there's a human on the other end here, unlike
# backtest.py's unattended 38-gameweek run).

def _squad_with_thin_defense() -> list:
    """15 players where DEF is the bottleneck: only 2 of 5 clear a 60 (or 75) xMins floor -- below
    every formation's 3-DEF minimum -- but all 5 clear 45 (Aggressive). Every other position is
    comfortably staffed at every floor, so DEF is the only thing this fallback chain has to fix."""
    players = [
        make_player(1, 1, 1, cost=45, xp=2.0, xmins=90),
        make_player(2, 1, 2, cost=40, xp=1.0, xmins=0),
        make_player(3, 2, 1, cost=45, xp=3.0, xmins=90),
        make_player(4, 2, 2, cost=45, xp=3.0, xmins=90),
        make_player(5, 2, 3, cost=45, xp=2.5, xmins=50),
        make_player(6, 2, 4, cost=45, xp=2.5, xmins=50),
        make_player(7, 2, 5, cost=45, xp=2.5, xmins=50),
        make_player(8, 3, 1, cost=55, xp=4.0, xmins=90),
        make_player(9, 3, 2, cost=55, xp=4.0, xmins=90),
        make_player(10, 3, 3, cost=55, xp=4.0, xmins=90),
        make_player(11, 3, 4, cost=55, xp=4.0, xmins=90),
        make_player(12, 3, 5, cost=55, xp=4.0, xmins=90),
        make_player(13, 4, 1, cost=60, xp=5.0, xmins=90),
        make_player(14, 4, 2, cost=60, xp=1.0, xmins=0),
        make_player(15, 4, 3, cost=60, xp=1.0, xmins=0),
    ]
    assert len(players) == 15
    return players


def test_fallback_returns_immediately_when_the_requested_floor_is_already_feasible():
    squad = _squad_with_thin_defense()
    xi, bench, formation, floor_used, was_relaxed = solve_starting_xi_with_fallback(squad, min_starter_xmins=45.0)
    assert floor_used == 45.0
    assert was_relaxed is False
    assert len(xi) == 11


def test_fallback_relaxes_from_balanced_down_to_aggressive_when_balanced_is_infeasible():
    squad = _squad_with_thin_defense()
    # 60.0 (Balanced) only has 2 DEF clearing it -- infeasible (every formation needs >= 3 DEF).
    xi, bench, formation, floor_used, was_relaxed = solve_starting_xi_with_fallback(squad, min_starter_xmins=60.0)
    assert was_relaxed is True
    assert floor_used == 45.0
    assert len(xi) == 11
    assert sum(1 for p in xi if p.element_type == 2) >= 3


def test_fallback_relaxes_from_conservative_all_the_way_down_to_aggressive():
    squad = _squad_with_thin_defense()
    # 75.0 (Conservative) and 60.0 (Balanced) both only have 2 DEF clearing them.
    xi, bench, formation, floor_used, was_relaxed = solve_starting_xi_with_fallback(squad, min_starter_xmins=75.0)
    assert was_relaxed is True
    assert floor_used == 45.0


def test_fallback_with_no_requested_floor_does_not_retry_anything():
    squad = _squad_with_thin_defense()
    xi, bench, formation, floor_used, was_relaxed = solve_starting_xi_with_fallback(squad, min_starter_xmins=None)
    assert floor_used is None
    assert was_relaxed is False


def test_fallback_still_raises_when_even_no_floor_at_all_is_infeasible():
    # Only 2 DEF in the whole squad -- no formation can ever field the required 3, regardless of
    # any minutes floor. A genuine squad-construction problem, not a minutes-security one.
    squad = [
        make_player(1, 1, 7, cost=45, xp=2.0, xmins=90),
        make_player(2, 1, 8, cost=40, xp=1.0, xmins=90),
        make_player(3, 2, 7, cost=45, xp=3.0, xmins=90),
        make_player(4, 2, 8, cost=45, xp=3.0, xmins=90),
    ] + [make_player(5 + i, 3, i + 1, cost=55, xp=4.0, xmins=90) for i in range(8)] + [
        make_player(13, 4, 1, cost=60, xp=5.0, xmins=90),
        make_player(14, 4, 2, cost=60, xp=5.0, xmins=90),
        make_player(15, 4, 3, cost=60, xp=5.0, xmins=90),
    ]
    assert len(squad) == 15
    club_counts = Counter(p.team_id for p in squad)
    assert max(club_counts.values()) <= MAX_PLAYERS_PER_TEAM  # sanity: infeasible for the DEF count, not the club limit
    with pytest.raises(OptimizationError):
        solve_starting_xi_with_fallback(squad, min_starter_xmins=60.0)


# --- Per-player xGC blend -----------------------------------------------------------------------

def test_blend_player_xga_falls_back_to_team_proxy_when_player_has_no_real_data():
    assert _blend_player_xga(0.0, 1.4) == 1.4


def test_blend_player_xga_blends_toward_the_players_own_real_rate():
    blended = _blend_player_xga(2.0, 1.0)
    assert 1.0 < blended < 2.0
    assert blended == pytest.approx(1.5)  # 50/50 at PLAYER_XGA_BLEND_WEIGHT's default


# --- Recent-form rolling window ------------------------------------------------------------------

def _gw_row(minutes, xg, xa):
    return {"minutes": minutes, "expected_goals": xg, "expected_assists": xa}


def test_recent_form_rate_returns_none_below_the_minimum_game_count():
    gw_rows = [_gw_row(90, 0.5, 0.2), _gw_row(90, 0.3, 0.1)]  # only 2 real games, need >= 3
    assert RECENT_FORM_MIN_GAMES > len(gw_rows)
    assert recent_form_rate(gw_rows) is None


def test_recent_form_rate_computes_a_per_90_rate_over_the_window():
    gw_rows = [_gw_row(90, 1.0, 0.5) for _ in range(RECENT_FORM_MIN_GAMES)]
    rate = recent_form_rate(gw_rows)
    assert rate is not None
    xg_per_90, xa_per_90 = rate
    assert xg_per_90 == pytest.approx(1.0)
    assert xa_per_90 == pytest.approx(0.5)


def test_recent_form_rate_only_uses_the_last_window_games_not_the_whole_history():
    # A long-past cold/blank spell shouldn't dilute a currently-hot rolling window.
    old_cold_games = [_gw_row(90, 0.0, 0.0) for _ in range(10)]
    recent_hot_games = [_gw_row(90, 2.0, 0.0) for _ in range(RECENT_FORM_WINDOW_GAMES)]
    rate = recent_form_rate(old_cold_games + recent_hot_games)
    assert rate is not None
    assert rate[0] == pytest.approx(2.0)


def test_recent_form_rate_ignores_games_with_zero_minutes():
    gw_rows = [_gw_row(0, 0.0, 0.0)] * 5 + [_gw_row(90, 1.0, 0.0) for _ in range(RECENT_FORM_MIN_GAMES)]
    rate = recent_form_rate(gw_rows)
    assert rate is not None
    assert rate[0] == pytest.approx(1.0)  # the 0-minute blanks don't drag the rate down


# --- fetch_players integration: history-based overrides actually apply ---------------------------

def test_fetch_players_uses_last_season_rate_at_a_true_cold_start(monkeypatch):
    import sqlite3
    from src import database, optimizer as optimizer_module

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.init_db(conn)
    conn.execute(
        "INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away, "
        "strength_defence_home, strength_defence_away) VALUES (1, 'Team1', 'T1', 1100, 1100, 1100, 1100)"
    )
    conn.execute(
        "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) "
        "VALUES (1, 'GW1', '2099-01-01T00:00:00Z', 0, 1, 0)"
    )  # no fixtures marked finished -- team_games_played == 0, a true cold start
    conn.execute(
        """
        INSERT INTO players (
            id, web_name, team_id, element_type, now_cost, selected_by_percent, form, total_points,
            ep_next, xg, xa, xgi, status, news, xg_per_90, xa_per_90, saves_per_90,
            defensive_contribution_per_90, starts_per_90, starts, chance_of_playing_next_round,
            penalties_order, corners_order, transfers_in_event, transfers_out_event
        ) VALUES (1, 'ColdStartStar', 1, 4, 120, 30.0, 0.0, 0, 0.0, 0, 0, 0, 'a', '', 0.0, 0.0, 0.0, 0, 0.0, 0, NULL, NULL, NULL, 0, 0)
        """
    )  # xg_per_90/starts all genuinely 0 -- nothing this season yet
    conn.execute(
        "INSERT INTO player_season_history (player_id, season_name, minutes, starts, total_points, "
        "expected_goals, expected_assists, expected_goals_conceded) VALUES (1, '2025/26', 3000, 33, 220, 25.0, 5.0, 0.0)"
    )  # a real, strong prior season: 25.0 xG over 3000 mins = 0.75 xG/90
    conn.commit()

    players = optimizer_module.fetch_players(conn)
    player = next(p for p in players if p.id == 1)
    assert player.xg_per_90 == pytest.approx(0.75)
    assert player.xa_per_90 == pytest.approx(0.15)  # 5.0 / 3000 * 90

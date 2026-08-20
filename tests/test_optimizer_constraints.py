"""Real assertion-based regression tests for the MILP constraints in src/optimizer.py -- formation
locking, the Press Conference/Injury Flag hard-exclusion gate, the Vice-Captain Lock, the Sub 1
Security Constraint, and the basic squad-building constraints (budget/position/club limits) that
test_optimizer.py (the repo-root manual smoke script) only ever prints for a human to eyeball.
"""
from collections import Counter

import pytest

from src.optimizer import (
    FORMATION_CHOICES,
    GW1_CAPTAIN_MIN_COST,
    MAX_PLAYERS_PER_TEAM,
    SQUAD_POSITION_COUNTS,
    VICE_CAPTAIN_XMINS_FLOOR,
    OptimizationError,
    _formation_label,
    _is_hard_excluded,
    captaincy_candidates,
    captaincy_score,
    is_captaincy_eligible,
    is_cold_start_pool,
    is_vice_eligible,
    order_bench,
    parse_formation_lock,
    solve_starting_xi,
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

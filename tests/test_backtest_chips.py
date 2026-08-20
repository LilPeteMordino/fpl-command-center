"""Real assertion-based regression tests for the Double-Chip Strategy in src/backtest.py:
Gameweek Schedule Inspector (SGW/BGW/DGW classification) and each chip's individual trigger rule.
Deliberately fabricated, deterministic inputs throughout -- no network access -- since which real
historical gameweeks actually trigger a given chip is season-dependent (see the module's own
verification notes); these tests instead pin down each trigger's PURE decision logic in isolation.
"""
import pytest

from src import backtest
from tests.conftest import make_player


# --- Gameweek Schedule Inspector ------------------------------------------------------------------

def test_classify_gameweek_density_sgw_bgw_dgw():
    fixtures_by_event = {
        1: [{"team_h": 1, "team_a": 2}, {"team_h": 3, "team_a": 4}],  # every team plays exactly once
        2: [{"team_h": 1, "team_a": 2}],  # teams 3, 4 blank
        3: [{"team_h": 1, "team_a": 2}, {"team_h": 1, "team_a": 3}, {"team_h": 4, "team_a": 2}],  # team 1 plays twice
    }
    density = backtest.classify_gameweek_density(fixtures_by_event, {1, 2, 3, 4})
    assert density[1]["label"] == "SGW"
    assert density[2]["label"] == "BGW"
    assert density[2]["team_fixture_counts"][3] == 0
    assert density[2]["team_fixture_counts"][4] == 0
    assert density[3]["label"] == "DGW"
    assert density[3]["team_fixture_counts"][1] == 2


def test_classify_gameweek_density_bgw_wins_over_dgw_same_week():
    # A team playing twice AND another team blanking the same gw -- BGW is the more extreme label.
    fixtures_by_event = {1: [{"team_h": 1, "team_a": 2}, {"team_h": 1, "team_a": 3}]}
    density = backtest.classify_gameweek_density(fixtures_by_event, {1, 2, 3, 4})
    assert density[1]["label"] == "BGW"  # team 4 never appears -> 0 fixtures


# --- Schedule-only scans -------------------------------------------------------------------------

def test_half_has_dgw():
    density = {5: {"label": "SGW"}, 6: {"label": "DGW"}, 7: {"label": "SGW"}}
    assert backtest._half_has_dgw(density, 5, 7)
    assert not backtest._half_has_dgw(density, 5, 5)


def test_scan_bb2_target_gw_picks_the_most_doubles_in_window():
    density = {
        28: {"team_fixture_counts": {1: 2, 2: 1}},
        30: {"team_fixture_counts": {1: 2, 2: 2, 3: 2}},
        35: {"team_fixture_counts": {1: 1, 2: 1}},
    }
    assert backtest._scan_bb2_target_gw(density) == 30


def test_scan_bb2_target_gw_none_when_no_doubles_at_all():
    density = {28: {"team_fixture_counts": {1: 1, 2: 1}}}
    assert backtest._scan_bb2_target_gw(density) is None


def test_scan_fh_target_gw_picks_the_most_blanks_in_window():
    density = {
        10: {"team_fixture_counts": {1: 0, 2: 1}},
        11: {"team_fixture_counts": {1: 0, 2: 0, 3: 0}},
        12: {"team_fixture_counts": {1: 1, 2: 1}},
    }
    assert backtest._scan_fh_target_gw(density, 10, 12) == 11


def test_scan_fh_target_gw_none_when_no_blanks():
    density = {10: {"team_fixture_counts": {1: 1, 2: 1}}}
    assert backtest._scan_fh_target_gw(density, 10, 10) is None


# --- Set 1 live triggers ---------------------------------------------------------------------------

def test_wc1_trigger_only_within_window():
    squad = [make_player(i, 3, 1, cost=50, xp=4.0, xmins=30.0) for i in range(5)]  # all rotation-risk
    assert backtest._wc1_trigger(5, squad) is None  # before the window
    assert backtest._wc1_trigger(9, squad) is None  # after the window
    assert backtest._wc1_trigger(7, squad) is not None


def test_wc1_trigger_requires_the_risk_count_bar():
    secure = [make_player(i, 3, 1, cost=50, xp=4.0, xmins=90.0) for i in range(15)]
    assert backtest._wc1_trigger(7, secure) is None
    risky = [make_player(i, 3, 1, cost=50, xp=4.0, xmins=30.0) for i in range(3)] + secure[3:]
    assert backtest._wc1_trigger(7, risky) is not None


def test_tc1_trigger_dgw_branch_fires_only_for_captains_own_club_double():
    density = {7: {"team_fixture_counts": {1: 2, 2: 1}}}
    captain_with_double = make_player(1, 4, 1, cost=140, xp=10.0)
    captain_without_double = make_player(2, 4, 2, cost=140, xp=10.0)
    assert backtest._tc1_trigger(7, density, half_has_dgw=True, captain=captain_with_double) is not None
    assert backtest._tc1_trigger(7, density, half_has_dgw=True, captain=captain_without_double) is None


def test_tc1_trigger_sgw_fallback_requires_premium_home_easy_fixture():
    density = {5: {"team_fixture_counts": {1: 1}}}
    premium_home_easy = make_player(1, 4, 1, cost=140, xp=8.0, is_home=True, fixture_difficulty=2)
    cheap_home_easy = make_player(2, 4, 1, cost=45, xp=8.0, is_home=True, fixture_difficulty=2)
    premium_away = make_player(3, 4, 1, cost=140, xp=8.0, is_home=False, fixture_difficulty=2)
    premium_home_hard = make_player(4, 4, 1, cost=140, xp=8.0, is_home=True, fixture_difficulty=4)
    assert backtest._tc1_trigger(5, density, half_has_dgw=False, captain=premium_home_easy) is not None
    assert backtest._tc1_trigger(5, density, half_has_dgw=False, captain=cheap_home_easy) is None
    assert backtest._tc1_trigger(5, density, half_has_dgw=False, captain=premium_away) is None
    assert backtest._tc1_trigger(5, density, half_has_dgw=False, captain=premium_home_hard) is None


def test_bb1_trigger_gw1_all_secure():
    secure = [make_player(i, 3, 1, cost=50, xp=4.0, xmins=90.0) for i in range(15)]
    assert backtest._bb1_trigger(1, secure, None) is not None
    thin = secure[:-1] + [make_player(99, 3, 1, cost=50, xp=4.0, xmins=10.0)]
    assert backtest._bb1_trigger(1, thin, None) is None


def test_bb1_trigger_gameweek_after_wildcard():
    wc1 = backtest.ChipActivation("WC", 7, 1, "test")
    assert backtest._bb1_trigger(8, [], wc1) is not None
    assert backtest._bb1_trigger(9, [], wc1) is None


def test_fh_trigger_fires_only_at_target_gw_with_a_real_squad_blank():
    density = {11: {"team_fixture_counts": {1: 0, 2: 1}}}
    blanking_player = make_player(1, 3, 1, cost=50, xp=4.0)  # team 1 -- blanks at gw 11
    playing_player = make_player(2, 3, 2, cost=50, xp=4.0)  # team 2 -- plays at gw 11
    assert backtest._fh_trigger(11, 11, [blanking_player], density) is not None
    assert backtest._fh_trigger(11, 11, [playing_player], density) is None
    assert backtest._fh_trigger(10, 11, [blanking_player], density) is None  # not the target gw
    assert backtest._fh_trigger(11, None, [blanking_player], density) is None  # no target at all


# --- Set 2 live triggers (pre-scanned set2_plan) ---------------------------------------------------

def test_tc2_trigger_requires_target_gw_and_still_owning_the_player():
    plan = {"tc2_gw": 25, "tc2_player_id": 99, "tc2_detail": "test detail"}
    assert backtest._tc2_trigger(25, plan, current_squad_ids=[99, 1, 2]) == "test detail"
    assert backtest._tc2_trigger(25, plan, current_squad_ids=[1, 2]) is None  # player since sold
    assert backtest._tc2_trigger(24, plan, current_squad_ids=[99, 1, 2]) is None  # wrong gw


def test_bb2_trigger_fires_at_target_gw_and_reports_double_count():
    plan = {"bb2_gw": 34}
    # A real 15-man squad -- 3 players on team 1 (which has a double GW34), 12 elsewhere (single).
    density = {34: {"team_fixture_counts": {1: 2, 2: 1}}}
    squad = [make_player(i, 3, 1, cost=50, xp=4.0) for i in range(3)] + [
        make_player(i, 3, 2, cost=50, xp=4.0) for i in range(3, 15)
    ]
    detail = backtest._bb2_trigger(34, plan, squad, density)
    assert detail is not None
    assert "3/15" in detail
    assert backtest._bb2_trigger(33, plan, squad, density) is None


def test_wc2_trigger_fires_only_at_planned_gw():
    plan = {"wc2_gw": 32, "bb2_gw": 34}
    assert backtest._wc2_trigger(32, plan) is not None
    assert backtest._wc2_trigger(31, plan) is None


# --- Chip scoring adjustment -------------------------------------------------------------------------

def test_apply_chip_scoring_no_chip_returns_plain_total():
    managed = {"provisional_total_points": 55.0, "captain_doubled_points": 10.0, "player_status": {}}
    assert backtest._apply_chip_scoring(managed, [], None) == 55.0


def test_apply_chip_scoring_triple_captain_adds_one_more_captain_copy():
    managed = {"provisional_total_points": 55.0, "captain_doubled_points": 10.0, "player_status": {}}
    assert backtest._apply_chip_scoring(managed, [], "TC") == pytest.approx(65.0)


def test_apply_chip_scoring_bench_boost_sums_the_whole_squad():
    class _Status:
        def __init__(self, pts):
            self.live_points = pts

    managed = {
        "provisional_total_points": 999.0,  # deliberately wrong/unused for BB -- must be ignored
        "captain_doubled_points": 8.0,
        "player_status": {1: _Status(10), 2: _Status(5), 3: _Status(0)},
    }
    # Only ids 1 and 2 are in scoring_squad_ids -- id 3's points must not be counted.
    assert backtest._apply_chip_scoring(managed, [1, 2], "BB") == pytest.approx(10 + 5 + 8.0)

"""Real assertion-based regression tests for the deterministic, network-free pieces of
src/backtest.py: historical CSV row parsing, the walk-forward per-90 accumulator (the module's
own no-lookahead-bias guarantee), the synthetic players-table writer, and the diagnostics math in
_build_season_report. Deliberately does NOT exercise simulate_season() itself -- that needs real
network access to the vaastav archive and runs 38 gameweeks of MILP solves, so it's a manual/CLI
smoke-test concern (`python -m src.backtest --season <season>`), not part of this offline suite --
see conftest.py's own module docstring for why that split matters.
"""
import sqlite3

import pytest

from src import backtest, database


# --- _parse_gw_row -------------------------------------------------------------------------------

def test_parse_gw_row_reads_core_fields():
    row = {
        "element": "355", "name": "Erling Haaland", "team": "Man City", "position": "FWD",
        "value": "141", "minutes": "90", "starts": "1", "expected_goals": "1.23",
        "expected_assists": "0.10", "saves": "0", "bonus": "3", "bps": "45", "total_points": "12",
    }
    parsed = backtest._parse_gw_row(row)
    assert parsed["id"] == 355
    assert parsed["web_name"] == "Erling Haaland"
    assert parsed["team_name"] == "Man City"
    assert parsed["element_type"] == 4
    assert parsed["now_cost"] == 141
    assert parsed["minutes"] == 90
    assert parsed["starts"] == 1
    assert parsed["xg"] == pytest.approx(1.23)
    assert parsed["xa"] == pytest.approx(0.10)
    assert parsed["total_points"] == 12


def test_parse_gw_row_defcon_absent_defaults_to_zero():
    """Pre-2025-26 seasons have no defensive_contribution column at all -- .get() returning None
    must default to 0.0, not crash (see module docstring)."""
    row = {"element": "1", "name": "P", "team": "T", "position": "DEF", "value": "45"}
    parsed = backtest._parse_gw_row(row)
    assert parsed["defcon"] == 0.0
    assert parsed["minutes"] == 0


def test_parse_gw_row_defcon_present_is_read():
    row = {"element": "1", "name": "P", "team": "T", "position": "DEF", "value": "45", "defensive_contribution": "8"}
    parsed = backtest._parse_gw_row(row)
    assert parsed["defcon"] == pytest.approx(8.0)


def test_parse_gw_row_missing_element_returns_none():
    assert backtest._parse_gw_row({"name": "No element id"}) is None


def test_parse_gw_row_unknown_position_maps_to_none_element_type():
    row = {"element": "1", "name": "P", "team": "T", "position": "??", "value": "45"}
    parsed = backtest._parse_gw_row(row)
    assert parsed["element_type"] is None


def test_parse_gw_row_gk_and_gkp_both_map_to_goalkeeper():
    for code in ("GK", "GKP", "gk"):
        row = {"element": "1", "name": "P", "team": "T", "position": code, "value": "45"}
        assert backtest._parse_gw_row(row)["element_type"] == 1


# --- Walk-forward accumulator (no-lookahead-bias core) --------------------------------------------

def test_accumulate_then_per90_reflects_only_ingested_gameweeks():
    """The crux of the module's no-lookahead guarantee: a player's per-90 rate written to the
    synthetic players table must be computed strictly from whatever has been _accumulate()'d so
    far -- calling _write_players_table BEFORE a gameweek's _accumulate() call must not see that
    gameweek's stats at all."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.init_db(conn)

    meta_by_id = {1: {"web_name": "Test Player", "team_id": 1, "element_type": 4, "now_cost": 80}}
    accum_by_id: dict = {}

    # GW1: nothing accumulated yet -- cold start, must be all-zero rates (triggers optimizer's own
    # pre-season fallback rather than a divide-by-zero or a leaked future rate).
    backtest._write_players_table(conn, meta_by_id, accum_by_id)
    row = conn.execute("SELECT xg_per_90, starts_per_90 FROM players WHERE id = 1").fetchone()
    assert row["xg_per_90"] == 0.0
    assert row["starts_per_90"] == 0.0

    # Ingest GW1's real result (90 mins, 1.0 xG, 1 start) -- only NOW should it show up.
    gw1_rows = [{"id": 1, "minutes": 90, "starts": 1, "xg": 1.0, "xa": 0.0, "saves": 0, "defcon": 0.0, "total_points": 8}]
    backtest._accumulate(accum_by_id, gw1_rows)
    backtest._write_players_table(conn, meta_by_id, accum_by_id)
    row = conn.execute("SELECT xg_per_90, starts_per_90, total_points FROM players WHERE id = 1").fetchone()
    assert row["xg_per_90"] == pytest.approx(1.0)  # 1.0 xG over exactly 90 mins => 1.0 per 90
    assert row["starts_per_90"] == pytest.approx(1.0)
    assert row["total_points"] == 8

    # A second gameweek's worth of accumulation should blend into the rate, not replace it.
    gw2_rows = [{"id": 1, "minutes": 90, "starts": 1, "xg": 0.0, "xa": 0.0, "saves": 0, "defcon": 0.0, "total_points": 2}]
    backtest._accumulate(accum_by_id, gw2_rows)
    backtest._write_players_table(conn, meta_by_id, accum_by_id)
    row = conn.execute("SELECT xg_per_90, total_points FROM players WHERE id = 1").fetchone()
    assert row["xg_per_90"] == pytest.approx(0.5)  # 1.0 total xG over 180 mins => 0.5 per 90
    assert row["total_points"] == 10  # cumulative, not just the latest gw


def test_ingest_gw_meta_skips_unresolvable_team_and_position():
    meta_by_id: dict = {}
    parsed_rows = [
        {"id": 1, "web_name": "A", "team_name": "Nowhere FC", "element_type": 2, "now_cost": 45},  # unknown team
        {"id": 2, "web_name": "B", "team_name": "Team1", "element_type": None, "now_cost": 45},  # unknown position
        {"id": 3, "web_name": "C", "team_name": "Team1", "element_type": 3, "now_cost": 55},  # valid
    ]
    backtest._ingest_gw_meta(meta_by_id, parsed_rows, team_id_by_name={"Team1": 1})
    assert set(meta_by_id) == {3}
    assert meta_by_id[3]["team_id"] == 1


# --- Historical live payload / stub client --------------------------------------------------------

def test_historical_live_payload_shape():
    parsed_rows = [{"id": 7, "minutes": 90, "total_points": 6, "bonus": 1, "bps": 30}]
    payload = backtest._historical_live_payload(parsed_rows)
    assert payload == {"elements": [{"id": 7, "stats": {"minutes": 90, "total_points": 6, "bonus": 1, "bps": 30}}]}


def test_static_live_client_ignores_event_id_argument():
    payload = {"elements": []}
    client = backtest._StaticLiveClient(payload)
    assert client.get_event_live(1) is payload
    assert client.get_event_live(38) is payload


# --- Risk profile resolution ----------------------------------------------------------------------

def test_resolve_risk_lambda_accepts_aliases_and_real_labels():
    assert backtest._resolve_risk_lambda("balanced") == 0.4
    assert backtest._resolve_risk_lambda("ev") == 0.0
    assert backtest._resolve_risk_lambda("conservative") == 1.0
    assert backtest._resolve_risk_lambda("Pure Mathematical EV") == 0.0


def test_resolve_risk_lambda_rejects_unknown_profile():
    with pytest.raises(ValueError):
        backtest._resolve_risk_lambda("not-a-real-profile")


# --- Season report diagnostics ----------------------------------------------------------------------

def _make_gw(gw, net, hit=0, gross=None, auto_subs=0, bench_left=0.0, cap_earned=0.0, cap_possible=0.0, cap_pts=0, static=0.0):
    return backtest.GameweekResult(
        gw=gw, managed_points=gross if gross is not None else net + hit, managed_hit_cost=hit,
        managed_net_points=net, transfers_in=[], transfers_out=[], static_points=static,
        auto_sub_moves=auto_subs, bench_points_left_behind=bench_left, captain_web_name="X",
        captain_points=cap_pts, captain_doubled_points=cap_earned, best_possible_captain_points=cap_possible,
        formation="4-4-2",
    )


def test_build_season_report_totals_and_transfer_roi():
    gw_results = [
        _make_gw(1, net=50, hit=0, static=50, cap_earned=10, cap_possible=10, cap_pts=5),
        _make_gw(2, net=40, hit=4, gross=44, static=45, cap_earned=6, cap_possible=12, cap_pts=6),
    ]
    report = backtest._build_season_report("test-season", 2, gw_results)
    assert report.total_gross_points == pytest.approx(94.0)
    assert report.total_hit_cost == 4
    assert report.total_points == pytest.approx(90.0)
    assert report.static_benchmark_points == pytest.approx(95.0)
    assert report.transfer_roi == pytest.approx(90.0 - 95.0)
    assert report.total_captain_points_earned == pytest.approx(16.0)
    assert report.total_captain_points_possible == pytest.approx(22.0)
    assert report.captaincy_points_left_on_table == pytest.approx(6.0)
    # GW1's captain earned (10) >= best possible (10) -- an optimal call; GW2's (6 < 12) is not.
    assert report.optimal_captaincy_weeks == 1


def test_build_season_report_best_and_worst_gameweeks_are_ranked_by_net_points():
    gw_results = [_make_gw(1, net=20), _make_gw(2, net=80), _make_gw(3, net=50), _make_gw(4, net=10)]
    report = backtest._build_season_report("test-season", 4, gw_results)
    assert [r.gw for r in report.best_gameweeks] == [2, 3, 1]
    assert [r.gw for r in report.worst_gameweeks] == [4, 1, 3]


def test_estimate_percentile_band_is_monotonic_and_always_returns_a_label():
    assert "top 1,000" in backtest._estimate_percentile_band(2400)
    assert "average manager total" in backtest._estimate_percentile_band(500)
    for pts in (-100, 0, 1799, 1800, 1999, 2000, 2149, 2150, 2299, 2300, 3000):
        assert isinstance(backtest._estimate_percentile_band(pts), str) and backtest._estimate_percentile_band(pts)

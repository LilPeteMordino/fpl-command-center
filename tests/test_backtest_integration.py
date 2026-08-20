"""Deterministic, offline integration tests for src/backtest.py's Double-Chip Strategy WIRING --
distinct from tests/test_backtest_chips.py, which pins down each chip's individual trigger
DECISION in isolation (pure functions, fabricated inputs). This file instead runs the REAL
simulate_season() walk-forward loop end-to-end against a small synthetic season and forces a
chosen chip to fire on a chosen gameweek (by monkeypatching that one trigger function, with every
other trigger patched to never fire, for clean isolation), then asserts the SURROUNDING state
machine behaved correctly: a Wildcard changes the squad without consuming a free transfer, a Free
Hit's squad substitution never touches the persistent squad, Triple Captain adds exactly one more
copy of the captain's points, Bench Boost counts the whole 15 and zeroes "left behind".

No network access: _load_teams/_load_fixtures_by_event/fetch_vaastav_csv are all monkeypatched to
serve a small, fully fabricated player/fixture universe (module-level constants below), so this
runs in well under a second and never depends on today's real vaastav archive contents.
"""
import pytest

from src import backtest

N_TEAMS = 6
# 3 GKP / 8 DEF / 8 MID / 5 FWD = 24 players -- comfortably more than SQUAD_POSITION_COUNTS'
# minimums (2/5/5/3) so solve_squad has a real choice, spread round-robin across N_TEAMS so no
# team is ever forced over MAX_PLAYERS_PER_TEAM (3) in the chosen 15.
_POSITION_LAYOUT = [("GK", 3), ("DEF", 8), ("MID", 8), ("FWD", 5)]


def _synthetic_teams() -> list:
    return [
        {
            "id": str(tid), "name": f"Team{tid}", "short_name": f"T{tid}",
            "strength_attack_home": "1100", "strength_attack_away": "1100",
            "strength_defence_home": "1100", "strength_defence_away": "1100",
        }
        for tid in range(1, N_TEAMS + 1)
    ]


def _player_universe() -> list:
    """[(element_id, position_code, team_id, cost), ...] -- a fixed, deterministic 24-player pool."""
    players = []
    element_id = 1
    for position, count in _POSITION_LAYOUT:
        for i in range(count):
            team_id = (element_id % N_TEAMS) + 1
            cost = 40 + (element_id % 6) * 5  # 40..65, deterministic spread
            players.append((element_id, position, team_id, cost))
            element_id += 1
    return players


def _synthetic_fixtures_by_event(n_gw: int) -> dict:
    """One round of fixtures per gameweek (teams paired 1v2, 3v4, 5v6), neutral difficulty
    throughout -- deliberately plain (this file tests chip WIRING via forced triggers, not the
    schedule-driven trigger DECISIONS themselves, which tests/test_backtest_chips.py already
    covers)."""
    fixtures_by_event = {}
    for gw in range(1, n_gw + 1):
        fixtures_by_event[gw] = [
            {"team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
            {"team_h": 3, "team_a": 4, "team_h_difficulty": 3, "team_a_difficulty": 3},
            {"team_h": 5, "team_a": 6, "team_h_difficulty": 3, "team_a_difficulty": 3},
        ]
    return fixtures_by_event


def _synthetic_gw_rows(gw: int, universe: list) -> list:
    """One vaastav-shaped CSV row per player for gameweek `gw` -- everyone plays a full 90
    minutes/1 start every week (keeps the Starter Security floor a non-issue here; that mechanism
    already has its own coverage elsewhere) with a small, deterministic per-player total_points
    spread so scoring differences between players/gameweeks are real and stable, not incidental."""
    rows = []
    for element_id, position, team_id, cost in universe:
        total_points = 3 + ((element_id + gw) % 8)  # deterministic, varies by player and gw
        rows.append({
            "element": str(element_id),
            "name": f"Player{element_id}",
            "team": f"Team{team_id}",
            "position": position,
            "value": str(cost),
            "minutes": "90",
            "starts": "1",
            "expected_goals": "0.3" if position in ("MID", "FWD") else "0.0",
            "expected_assists": "0.2" if position in ("MID", "FWD") else "0.0",
            "saves": "3" if position == "GK" else "0",
            "bonus": "1" if total_points >= 8 else "0",
            "bps": str(20 + total_points),
            "total_points": str(total_points),
        })
    return rows


def _patch_data_sources(monkeypatch, n_gw: int) -> None:
    universe = _player_universe()
    teams = _synthetic_teams()
    fixtures_by_event = _synthetic_fixtures_by_event(n_gw)
    gw_rows_by_gw = {gw: _synthetic_gw_rows(gw, universe) for gw in range(1, n_gw + 1)}

    def fake_load_teams(season):
        return teams

    def fake_load_fixtures_by_event(season):
        return fixtures_by_event

    def fake_fetch_vaastav_csv(url, session=None):
        # url is .../gws/gw{N}.csv -- pull N back out rather than threading gw through separately,
        # matching exactly how simulate_season itself calls this (see config.VAASTAV_GW_STATS_CSV_TEMPLATE).
        gw = int(url.rsplit("gw", 1)[1].split(".")[0])
        if gw not in gw_rows_by_gw:
            raise backtest.FPLAPIError(f"no synthetic data beyond gw {n_gw}")
        return gw_rows_by_gw[gw]

    monkeypatch.setattr(backtest, "_load_teams", fake_load_teams)
    monkeypatch.setattr(backtest, "_load_fixtures_by_event", fake_load_fixtures_by_event)
    monkeypatch.setattr(backtest, "fetch_vaastav_csv", fake_fetch_vaastav_csv)


def _patch_all_triggers_off(monkeypatch) -> None:
    """Every chip trigger forced to never fire -- the baseline a test then selectively overrides
    for the ONE chip under test, so no other chip can accidentally fire from the synthetic data
    (e.g. every player having identical 90-minute weeks could otherwise trip Bench Boost 1's
    "all 15 secure" GW1 condition) and contaminate the result."""
    for name in ("_wc1_trigger", "_fh_trigger", "_tc1_trigger", "_bb1_trigger"):
        monkeypatch.setattr(backtest, name, lambda *a, **k: None)


def _result_at(report, gw: int):
    return next(r for r in report.gw_results if r.gw == gw)


# --- Wildcard: changes the squad, never consumes a free transfer ---------------------------------

def test_wildcard_changes_squad_and_does_not_consume_free_transfer(monkeypatch):
    _patch_data_sources(monkeypatch, n_gw=4)
    _patch_all_triggers_off(monkeypatch)
    monkeypatch.setattr(backtest, "_wc1_trigger", lambda gw, squad_rows: "forced WC for test" if gw == 3 else None)

    report = backtest.simulate_season("synthetic", risk_profile="balanced", verbose=False)

    gw3 = _result_at(report, 3)
    assert gw3.chip == "WC"
    assert gw3.managed_hit_cost == 0
    gw4 = _result_at(report, 4)
    # Free transfers roll forward by 1 (capped) exactly like a normal hold week -- a Wildcard
    # spends none of the free_transfers_before it had going in.
    assert gw4.free_transfers_before == min(backtest.transfer_planner.FREE_TRANSFER_CAP, gw3.free_transfers_before + 1)


def test_wildcard_never_fires_twice_in_the_same_half(monkeypatch):
    _patch_data_sources(monkeypatch, n_gw=4)
    _patch_all_triggers_off(monkeypatch)
    # Deliberately "fires" every gameweek from gw2 onward -- only the FIRST should actually count.
    monkeypatch.setattr(backtest, "_wc1_trigger", lambda gw, squad_rows: "forced WC every week" if gw >= 2 else None)

    report = backtest.simulate_season("synthetic", risk_profile="balanced", verbose=False)
    wc_gameweeks = [r.gw for r in report.gw_results if r.chip == "WC"]
    assert wc_gameweeks == [2]


# --- Free Hit: one-week-only squad substitution, never touches the persistent squad --------------

def test_free_hit_never_touches_the_persistent_squad(monkeypatch):
    _patch_data_sources(monkeypatch, n_gw=4)
    _patch_all_triggers_off(monkeypatch)
    monkeypatch.setattr(
        backtest, "_fh_trigger",
        lambda gw, target_gw, squad_rows, gw_density: "forced FH for test" if gw == 3 else None,
    )

    report = backtest.simulate_season("synthetic", risk_profile="balanced", verbose=False)

    gw3 = _result_at(report, 3)
    assert gw3.chip == "FH"
    assert gw3.managed_hit_cost == 0
    # The persistent squad is never touched by a Free Hit -- transfers_in/out for that gameweek
    # stay empty even though the SQUAD ACTUALLY SCORED that week was a temporary rebuild.
    assert gw3.transfers_in == []
    assert gw3.transfers_out == []


# --- Triple Captain: exactly one more copy of the captain's points, nothing else changes ---------

def test_triple_captain_adds_exactly_one_more_captain_copy(monkeypatch):
    _patch_data_sources(monkeypatch, n_gw=3)
    _patch_all_triggers_off(monkeypatch)
    report_without_tc = backtest.simulate_season("synthetic", risk_profile="balanced", verbose=False)

    _patch_data_sources(monkeypatch, n_gw=3)
    _patch_all_triggers_off(monkeypatch)
    monkeypatch.setattr(
        backtest, "_tc1_trigger",
        lambda gw, gw_density, half_has_dgw, captain: "forced TC for test" if gw == 3 else None,
    )
    report_with_tc = backtest.simulate_season("synthetic", risk_profile="balanced", verbose=False)

    gw3_without = _result_at(report_without_tc, 3)
    gw3_with = _result_at(report_with_tc, 3)
    assert gw3_with.chip == "TC"
    assert gw3_without.chip is None
    # Same underlying gameweek/squad/captain either way (deterministic synthetic data) -- Triple
    # Captain's entire effect is exactly one more copy of the captain's own live points on top.
    assert gw3_with.captain_web_name == gw3_without.captain_web_name
    assert gw3_with.managed_points == pytest.approx(gw3_without.managed_points + gw3_without.captain_doubled_points)


# --- Bench Boost: the whole 15 counts, nothing is "left behind" ----------------------------------

def test_bench_boost_counts_the_whole_squad_and_zeroes_points_left_behind(monkeypatch):
    _patch_data_sources(monkeypatch, n_gw=3)
    _patch_all_triggers_off(monkeypatch)
    report_without_bb = backtest.simulate_season("synthetic", risk_profile="balanced", verbose=False)

    _patch_data_sources(monkeypatch, n_gw=3)
    _patch_all_triggers_off(monkeypatch)
    monkeypatch.setattr(
        backtest, "_bb1_trigger",
        lambda gw, squad_rows, wc1_activation: "forced BB for test" if gw == 3 else None,
    )
    report_with_bb = backtest.simulate_season("synthetic", risk_profile="balanced", verbose=False)

    gw3_without = _result_at(report_without_bb, 3)
    gw3_with = _result_at(report_with_bb, 3)
    assert gw3_with.chip == "BB"
    assert gw3_with.bench_points_left_behind == 0.0
    # Bench Boost is purely additive (the whole squad already counted for the non-BB run too, via
    # auto-subs, minus whatever was left on the bench) -- it can never score less than the plain run.
    assert gw3_with.managed_points >= gw3_without.managed_points


# --- Wildcard 2: builds toward a specific future target gameweek's fixtures, not just "now" ------

def test_rebuild_squad_for_target_gw_optimizes_for_the_target_gameweeks_own_fixtures():
    """Standalone unit test (no full simulate_season walk needed) for the specific mechanic
    Wildcard 2 relies on: _rebuild_squad_for_target_gw(..., target_gw=X) must score candidates
    against gameweek X's fixtures, not the gameweek the rebuild itself happens on.

    Design: team 1 and team 2 have IDENTICAL strong underlying rates -- the only thing that ever
    differs between them is which gameweek's fixture is favorable, so any squad-composition shift
    can only be explained by target_gw actually changing which gameweek got scored (not a rate
    mismatch). Both always have a real fixture (never a genuine blank) specifically to sidestep
    optimizer.fetch_players' blank-fallback vs transfer_planner.fetch_multi_gw_projections'
    blank-zeroes-out difference entirely, which isn't what this test is about. Four extra "filler"
    teams (3-6) with weak rates but stable neutral fixtures both weeks supply the remaining squad
    slots needed for MAX_PLAYERS_PER_TEAM(3) feasibility -- 2 teams alone can field at most 6 of
    the 15 required players."""
    import sqlite3

    from src import database

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.init_db(conn)

    for tid in range(1, 9):  # teams 1/2 (the two under test), 3-6 (filler), 7/8 (pure opponents)
        conn.execute(
            "INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away, "
            "strength_defence_home, strength_defence_away) VALUES (?, ?, ?, 1100, 1100, 1100, 1100)",
            (tid, f"Team{tid}", f"T{tid}"),
        )
    conn.execute(
        "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) "
        "VALUES (1, 'GW1', '2020-01-01T00:00:00Z', 0, 1, 0)"
    )
    conn.execute(
        "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) "
        "VALUES (2, 'GW2', '2099-01-01T00:00:00Z', 0, 0, 0)"
    )

    def _fixture(event, team_h, team_a, fdr_h, fdr_a):
        conn.execute(
            "INSERT INTO fixtures (event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (event, team_h, team_a, fdr_h, fdr_a),
        )

    # GW1 ("now"): team 1 gets an easy fixture (FDR 1), team 2 a hard one (FDR 5).
    _fixture(1, 1, 7, 1, 5)
    _fixture(1, 2, 8, 5, 1)
    _fixture(1, 3, 4, 3, 3)
    _fixture(1, 5, 6, 3, 3)
    # GW2 (the WC2 target): team 1's fixture flips hard (FDR 5), team 2 gets a favorable (FDR 2)
    # DOUBLE gameweek (two legs) -- exactly the double-fixture-depth scenario Wildcard 2 targets.
    _fixture(2, 1, 7, 5, 1)
    _fixture(2, 2, 8, 2, 4)
    _fixture(2, 8, 2, 4, 2)
    _fixture(2, 3, 4, 3, 3)
    _fixture(2, 5, 6, 3, 3)

    def _insert_player(pid, team_id, element_type, cost, web_name, xg, xa):
        conn.execute(
            """
            INSERT INTO players (
                id, web_name, team_id, element_type, now_cost, selected_by_percent, form, total_points,
                ep_next, xg, xa, xgi, status, news, xg_per_90, xa_per_90, saves_per_90,
                defensive_contribution_per_90, starts_per_90, starts, chance_of_playing_next_round,
                penalties_order, corners_order, transfers_in_event, transfers_out_event
            ) VALUES (?, ?, ?, ?, ?, 5.0, 3.0, 20, 3.0, ?, ?, ?, 'a', '', ?, ?, 0.0, 0.0, 1.0, 10, NULL, NULL, NULL, 0, 0)
            """,
            (pid, web_name, team_id, element_type, cost, xg, xa, xg + xa, xg, xa),
        )

    pid = 1
    team1_ids, team2_ids = [], []
    # Team 1 & 2: 1 GKP + 1 DEF + 1 MID each, strong rates (0.6/0.4) -- the fixture swing above is
    # the ONLY thing that ever differs between them.
    for team_id, id_bucket in ((1, team1_ids), (2, team2_ids)):
        for element_type in (1, 2, 3):
            xg, xa = (0.0, 0.0) if element_type == 1 else (0.6, 0.4)
            _insert_player(pid, team_id, element_type, 50, f"Team{team_id}P{pid}", xg, xa)
            id_bucket.append(pid)
            pid += 1
    # Filler teams 3-6: 1 DEF + 1 MID + 1 FWD each, deliberately weak rates -- always feasible
    # fallback content, never competitive with whichever of team 1/2 is currently favored.
    for team_id in (3, 4, 5, 6):
        for element_type in (2, 3, 4):
            _insert_player(pid, team_id, element_type, 45, f"Team{team_id}P{pid}", 0.1, 0.05)
            pid += 1
    conn.commit()

    budget_units = 1000
    squad_for_now = backtest._rebuild_squad_for_target_gw(conn, budget_units, risk_lambda=0.0, target_gw=None)
    squad_for_gw2 = backtest._rebuild_squad_for_target_gw(conn, budget_units, risk_lambda=0.0, target_gw=2)
    now_by_id = {p.id: p for p in squad_for_now}
    gw2_by_id = {p.id: p for p in squad_for_gw2}

    # Both team 1 and team 2's strong-rate players comfortably outscore the weak filler teams
    # regardless of fixture (DEF's attacking term isn't even fixture-scaled in this model, and
    # MID's fixture-difficulty multiplier is a modest +-20% either side of a much bigger rate
    # gap) -- so squad MEMBERSHIP alone doesn't flip here, and isn't the right signal. What must
    # flip is each player's own projected_xp between the two builds, which is exactly what
    # target_gw is supposed to control.
    team1_mid_id, team2_mid_id = team1_ids[2], team2_ids[2]
    assert team1_mid_id in now_by_id and team1_mid_id in gw2_by_id
    assert team2_mid_id in now_by_id and team2_mid_id in gw2_by_id

    # Team 1 is favorable "now" (FDR 1) but hard at the target gameweek (FDR 5) -- its
    # projected_xp must drop between the two builds.
    assert now_by_id[team1_mid_id].projected_xp > gw2_by_id[team1_mid_id].projected_xp
    # Team 2 is the reverse: hard "now" (FDR 5) but a favorable DOUBLE at the target gameweek
    # (FDR 2, twice) -- its projected_xp must rise, and end up ahead of team 1's now-discounted
    # figure -- proof the rebuild is genuinely scored against target_gw's own fixtures, not
    # whichever gameweek the rebuild itself runs on.
    assert gw2_by_id[team2_mid_id].projected_xp > now_by_id[team2_mid_id].projected_xp
    assert gw2_by_id[team2_mid_id].projected_xp > gw2_by_id[team1_mid_id].projected_xp

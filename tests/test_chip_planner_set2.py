"""Real assertion-based regression tests for chip_planner.py's Set 2 (GW20-38) macro chip
planner: solve_second_half_chip_strategy and its per-chip candidate functions
(_tc2_macro_candidates/_bb2_macro_candidates/_wc2_macro_candidates -- _fh_macro_candidates is
reused as-is from Set 1 and already has no dedicated coverage of its own, out of scope here).
Built on a small, fully self-contained in-memory SQLite DB, matching the established pattern in
tests/test_transfer_planner_rules.py -- deterministic, no network access.
"""
import sqlite3

import pytest

from src import chip_planner, database, transfer_planner

N_TEAMS = 6


def _insert_player(
    conn, pid, team_id, element_type, cost, *,
    web_name=None, xg=0.3, xa=0.1, saves=0.0, starts_per_90=0.9, starts=10,
):
    conn.execute(
        """
        INSERT INTO players (
            id, web_name, team_id, element_type, now_cost, selected_by_percent, form, total_points,
            ep_next, xg, xa, xgi, status, news, xg_per_90, xa_per_90, saves_per_90,
            defensive_contribution_per_90, starts_per_90, starts, chance_of_playing_next_round,
            penalties_order, corners_order, transfers_in_event, transfers_out_event
        ) VALUES (?, ?, ?, ?, ?, 5.0, 3.0, 20, 3.0, 0, 0, 0, 'a', '', ?, ?, ?, 0, ?, ?, NULL, NULL, NULL, 0, 0)
        """,
        (pid, web_name or f"P{pid}", team_id, element_type, cost, xg, xa, saves, starts_per_90, starts),
    )


@pytest.fixture
def set2_db():
    """6 teams, gameweeks 20-38 (gw20 is_next), one neutral round of fixtures per gameweek, plus
    a real 15-man squad -- including a premium (>= GW1_CAPTAIN_MIN_COST) MID on team 1. Individual
    tests layer their own deliberate double/blank fixtures on top of this base."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.init_db(conn)

    for tid in range(1, N_TEAMS + 1):
        conn.execute(
            "INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away, "
            "strength_defence_home, strength_defence_away) VALUES (?, ?, ?, 1100, 1100, 1100, 1100)",
            (tid, f"Team{tid}", f"T{tid}"),
        )
    for gw in range(20, 39):
        conn.execute(
            "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) VALUES (?, ?, ?, 0, ?, 0)",
            (gw, f"GW{gw}", "2099-01-01T00:00:00Z", 1 if gw == 20 else 0),
        )
        for team_a, team_b in [(1, 2), (3, 4), (5, 6)]:
            conn.execute(
                "INSERT INTO fixtures (event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished) "
                "VALUES (?, ?, ?, 3, 3, 0)",
                (gw, team_a, team_b),
            )

    pid = 1
    squad_ids = []
    _insert_player(conn, pid, 1, 1, 45, saves=3.0, web_name="GK1"); squad_ids.append(pid); pid += 1
    _insert_player(conn, pid, 2, 1, 40, saves=1.0, web_name="GK2"); squad_ids.append(pid); pid += 1
    for i in range(5):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 2, 45 + i, xg=0.05, xa=0.05, web_name=f"DEF{pid}")
        squad_ids.append(pid); pid += 1
    for i in range(4):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 3, 55 + i, xg=0.25, xa=0.2, web_name=f"MID{pid}")
        squad_ids.append(pid); pid += 1
    premium_id = pid
    _insert_player(conn, pid, 1, 3, 120, xg=0.6, xa=0.4, web_name="PremiumTalisman")
    squad_ids.append(pid); pid += 1
    for i in range(3):
        team = ((i + 3) % N_TEAMS) + 1
        _insert_player(conn, pid, team, 4, 60 + i, xg=0.4, xa=0.1, web_name=f"FWD{pid}")
        squad_ids.append(pid); pid += 1

    assert len(squad_ids) == 15
    conn.commit()
    return conn, squad_ids, {"premium_id": premium_id}


def _add_double(conn, event, team_h, team_a, fdr_h=2, fdr_a=4):
    conn.execute(
        "INSERT INTO fixtures (event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (event, team_h, team_a, fdr_h, fdr_a),
    )


def _blank_team(conn, event, team_id):
    conn.execute("DELETE FROM fixtures WHERE event = ? AND (team_h = ? OR team_a = ?)", (event, team_id, team_id))


# --- Triple Captain 2 ------------------------------------------------------------------------------

def test_tc2_prefers_premium_double_gameweek_and_labels_it(set2_db):
    conn, squad_ids, ids = set2_db
    _add_double(conn, 30, 1, 2)  # team 1 (premium talisman's club) doubles at gw30
    conn.commit()
    event_ids = list(range(20, 39))
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    candidates = chip_planner._tc2_macro_candidates(squad_ids, projections, event_ids)
    assert candidates, "expected at least one double-gameweek candidate"
    best = candidates[0]
    assert best["event_id"] == 30
    assert "premium" in best["reason"].lower()
    assert "PremiumTalisman" in best["reason"]


def test_tc2_still_surfaces_a_non_premium_double_if_thats_all_there_is(set2_db):
    conn, squad_ids, ids = set2_db
    _add_double(conn, 28, 3, 4)  # a non-premium filler team doubles -- no premium double anywhere
    conn.commit()
    event_ids = list(range(20, 39))
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    candidates = chip_planner._tc2_macro_candidates(squad_ids, projections, event_ids)
    assert candidates
    assert candidates[0]["event_id"] == 28
    assert "premium" not in candidates[0]["reason"].lower()


def test_tc2_empty_when_no_double_gameweek_exists_at_all(set2_db):
    conn, squad_ids, ids = set2_db
    event_ids = list(range(20, 39))
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)
    assert chip_planner._tc2_macro_candidates(squad_ids, projections, event_ids) == []


# --- Bench Boost 2 ----------------------------------------------------------------------------------

def test_bb2_picks_the_gameweek_with_the_most_squad_doubles(set2_db):
    conn, squad_ids, ids = set2_db
    _add_double(conn, 30, 1, 2)  # 2 squad players double (GK1's club + PremiumTalisman's club, team 1)
    _add_double(conn, 34, 3, 4)  # 1 squad player doubles (a single DEF/MID/FWD on team 3 or 4)
    conn.commit()
    event_ids = list(range(20, 39))
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    candidates = chip_planner._bb2_macro_candidates(squad_ids, projections, event_ids)
    assert candidates
    assert candidates[0]["event_id"] == 30
    assert candidates[0]["data_driven"] is True
    assert "falls short of" in candidates[0]["reason"]  # well under the 12-player bar in this small fixture


def test_bb2_falls_back_to_a_placeholder_when_nothing_doubles(set2_db):
    conn, squad_ids, ids = set2_db
    event_ids = list(range(20, 39))
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    candidates = chip_planner._bb2_macro_candidates(squad_ids, projections, event_ids)
    assert len(candidates) == 1
    assert candidates[0]["event_id"] == event_ids[-1]
    assert candidates[0]["data_driven"] is False


# --- Wildcard 2 --------------------------------------------------------------------------------------

def test_wc2_targets_two_gameweeks_before_bench_boost_2_when_available():
    event_ids = list(range(20, 39))
    candidates = chip_planner._wc2_macro_candidates(event_ids, bb2_event_id=35)
    assert candidates[0]["event_id"] == 33  # 35 - 2, the preferred lead time


def test_wc2_falls_back_to_one_gameweek_before_if_two_is_out_of_window():
    event_ids = list(range(33, 39))  # 33-38 -- gw32 (34-2) isn't in this window, gw33 (34-1) is
    candidates = chip_planner._wc2_macro_candidates(event_ids, bb2_event_id=34)
    assert candidates and candidates[0]["event_id"] == 33  # 34 - 1


def test_wc2_empty_when_bench_boost_2_never_found_a_target():
    event_ids = list(range(20, 39))
    assert chip_planner._wc2_macro_candidates(event_ids, bb2_event_id=None) == []


# --- Full second-half roadmap: collision handling & dependency ordering --------------------------

def test_solve_second_half_chip_strategy_wildcard_precedes_bench_boost_by_lead_time(set2_db):
    conn, squad_ids, ids = set2_db
    _add_double(conn, 35, 1, 2)  # the squad's biggest double -- premium talisman's club included
    conn.commit()

    roadmap = chip_planner.solve_second_half_chip_strategy(conn, squad_ids)
    by_chip = {r.chip: r for r in roadmap}

    assert "BB" in by_chip and by_chip["BB"].event_id == 35
    assert "WC" in by_chip and by_chip["WC"].event_id == 33  # 2 gws ahead of the Bench Boost target
    # Triple Captain's only real double-gameweek opportunity (gw35) collides with the already-
    # claimed Bench Boost gameweek, so it correctly finds nothing rather than double-booking GW35.
    assert "TC" not in by_chip


def test_solve_second_half_chip_strategy_respects_available_chips_filter(set2_db):
    conn, squad_ids, ids = set2_db
    _add_double(conn, 35, 1, 2)
    conn.commit()

    roadmap = chip_planner.solve_second_half_chip_strategy(conn, squad_ids, available_chips=["FH"])
    assert {r.chip for r in roadmap} <= {"FH"}


def test_solve_second_half_chip_strategy_no_gameweek_gets_two_chips(set2_db):
    conn, squad_ids, ids = set2_db
    _add_double(conn, 35, 1, 2)
    _blank_team(conn, 36, 2)
    conn.commit()

    roadmap = chip_planner.solve_second_half_chip_strategy(conn, squad_ids)
    event_ids_used = [r.event_id for r in roadmap]
    assert len(event_ids_used) == len(set(event_ids_used))


def test_solve_second_half_chip_strategy_out_of_window_returns_empty(set2_db):
    conn, squad_ids, ids = set2_db
    # A gw_start/gw_end pair entirely inside Set 1 -- Set 2's own window clamps it away to nothing.
    assert chip_planner.solve_second_half_chip_strategy(conn, squad_ids, gw_start=1, gw_end=5) == []

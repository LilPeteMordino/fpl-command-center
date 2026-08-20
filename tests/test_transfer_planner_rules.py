"""Real assertion-based regression tests for the anti-churn rules in src/transfer_planner.py's
plan_transfers: the hurdle rate, the FT banking cap, the GKP freeze ("Set-and-Forget"), and the
allow_hits bound. Built on a small, fully self-contained in-memory SQLite DB (not the real synced
one) so these run deterministically in CI with no network access and no dependency on today's
real player pool -- see conftest.py's own module docstring for why that matters.
"""
import sqlite3

import pytest

from src import database
from src.transfer_planner import FREE_TRANSFER_CAP, plan_transfers

N_TEAMS = 6


def _insert_player(
    conn, pid, team_id, element_type, cost, *,
    web_name=None, xg=0.3, xa=0.1, saves=0.0, defcon=0.0, starts_per_90=0.9, starts=10,
    status="a", chance=None,
):
    conn.execute(
        """
        INSERT INTO players (
            id, web_name, team_id, element_type, now_cost, selected_by_percent, form, total_points,
            ep_next, xg, xa, xgi, status, news, xg_per_90, xa_per_90, saves_per_90,
            defensive_contribution_per_90, starts_per_90, starts, chance_of_playing_next_round,
            penalties_order, corners_order, transfers_in_event, transfers_out_event
        ) VALUES (?, ?, ?, ?, ?, 5.0, 3.0, 20, 3.0, 0, 0, 0, ?, '', ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, 0)
        """,
        (pid, web_name or f"P{pid}", team_id, element_type, cost, status, xg, xa, saves, defcon, starts_per_90, starts, chance),
    )


@pytest.fixture
def transfer_db():
    """A minimal, deterministic in-memory DB: 6 teams, 7 gameweeks (gw2 is_next, so
    get_horizon_event_ids starts there -- gw1's deadline is set safely in the past so
    is_before_gw1_deadline() is False and every plan_transfers step below exercises the NORMAL
    transfer rules, not the free/unlimited GW1 window), one round of fixtures per gameweek
    (neutral difficulty throughout), and a real 15-man starting squad plus a few deliberately
    stronger/weaker alternatives in the wider pool for the tests to transfer toward/away from.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.init_db(conn)

    for tid in range(1, N_TEAMS + 1):
        conn.execute(
            "INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away, "
            "strength_defence_home, strength_defence_away) VALUES (?, ?, ?, 1100, 1100, 1100, 1100)",
            (tid, f"Team{tid}", f"T{tid}"),
        )

    conn.execute(
        "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) "
        "VALUES (1, 'GW1', '2020-01-01T00:00:00Z', 0, 0, 1)"
    )
    for gw in range(2, 8):
        conn.execute(
            "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) VALUES (?, ?, ?, 0, ?, 0)",
            (gw, f"GW{gw}", f"2099-01-0{min(gw, 9)}T00:00:00Z", 1 if gw == 2 else 0),
        )
        for team_a, team_b in [(1, 2), (3, 4), (5, 6)]:
            conn.execute(
                "INSERT INTO fixtures (event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished) "
                "VALUES (?, ?, ?, 3, 3, 0)",
                (gw, team_a, team_b),
            )

    pid = 1
    squad_ids = []
    # 2 GKP -- a weak one worth upgrading on pure merit, but should NEVER be touched while
    # freeze_gkp_transfers=True.
    _insert_player(conn, pid, 1, 1, 40, xg=0.0, xa=0.0, saves=2.0, starts_per_90=1.0); squad_ids.append(pid); pid += 1
    _insert_player(conn, pid, 2, 1, 40, xg=0.0, xa=0.0, saves=1.0, starts_per_90=0.3); squad_ids.append(pid); pid += 1
    # 5 DEF, modest/solid underlying rates, one per team where possible.
    for i in range(5):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 2, 45 + i, xg=0.05, xa=0.08, defcon=6.0, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    # 5 MID -- including one deliberately weak "P{id}" the tests transfer OUT.
    for i in range(4):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 3, 55 + i, xg=0.25, xa=0.2, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    weak_mid_id = pid
    _insert_player(conn, pid, 6, 3, 50, xg=0.0, xa=0.0, starts_per_90=0.9, web_name="WeakMid")
    squad_ids.append(pid); pid += 1
    # 3 FWD -- offset by +3 teams (not the plain (i % N_TEAMS) + 1 the GKP/DEF/MID loops above use)
    # so this loop doesn't stack a 4th player onto team 1: GKP/DEF/MID above already each put one
    # player on team 1 at i=0, so a naive FWD loop would push team 1 to 4 -- over
    # MAX_PLAYERS_PER_TEAM (3) -- making the initial squad itself invalid and silently blocking any
    # transfer ILP call that includes the club-limit constraint (t=0 "hold" never validates it, so
    # this went unnoticed until a t=1/t=2 candidate needed solving).
    for i in range(3):
        _insert_player(conn, pid, ((i + 3) % N_TEAMS) + 1, 4, 60 + i, xg=0.4, xa=0.1, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1

    assert len(squad_ids) == 15

    # Pool-only alternatives (not in the initial squad):
    strong_gkp_id = pid
    _insert_player(conn, pid, 3, 1, 45, xg=0.0, xa=0.0, saves=4.0, starts_per_90=1.0, web_name="StrongGKP")
    pid += 1
    strong_mid_id = pid
    _insert_player(conn, pid, 6, 3, 50, xg=0.6, xa=0.4, starts_per_90=0.95, web_name="StrongMid")
    pid += 1
    marginal_mid_id = pid
    _insert_player(conn, pid, 6, 3, 50, xg=0.02, xa=0.02, starts_per_90=0.9, web_name="MarginalMid")
    pid += 1

    conn.commit()
    return conn, squad_ids, {
        "weak_mid_id": weak_mid_id, "strong_gkp_id": strong_gkp_id,
        "strong_mid_id": strong_mid_id, "marginal_mid_id": marginal_mid_id,
    }


def test_gkp_freeze_never_transfers_a_goalkeeper(transfer_db):
    conn, squad_ids, ids = transfer_db
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=2, horizon_gws=3,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    for step in roadmap:
        assert ids["strong_gkp_id"] not in step.transfers_in_ids
        for out_id in step.transfers_out_ids:
            # None of the GKPs originally in the squad (ids 1 or 2) should ever be sold.
            assert out_id not in (1, 2)


def test_hurdle_rate_blocks_marginal_transfer(transfer_db):
    conn, squad_ids, ids = transfer_db
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=1, horizon_gws=2,
        allow_hits=True, freeze_gkp_transfers=True, transfer_hurdle_xp=1.5,
    )
    all_transfers_in = {pid for step in roadmap for pid in step.transfers_in_ids}
    # MarginalMid is barely better than WeakMid -- shouldn't clear a 1.5 xP hurdle over holding.
    assert ids["marginal_mid_id"] not in all_transfers_in


def test_clear_upgrade_executes_and_clears_hurdle(transfer_db):
    conn, squad_ids, ids = transfer_db
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=1, horizon_gws=2,
        allow_hits=True, freeze_gkp_transfers=True, transfer_hurdle_xp=1.5,
    )
    # StrongMid clearly outperforms WeakMid (0.0 vs 0.6+0.4 underlying rate) -- some step in this
    # short horizon should pick it up.
    all_transfers_in = {pid for step in roadmap for pid in step.transfers_in_ids}
    assert ids["strong_mid_id"] in all_transfers_in


def test_ft_available_never_exceeds_cap(transfer_db):
    conn, squad_ids, _ids = transfer_db
    # Roll every gameweek (no attractive transfer) by setting an impossibly high hurdle --
    # free_transfers_before should climb but never past FREE_TRANSFER_CAP.
    roadmap = plan_transfers(
        conn, squad_ids, bank=0, free_transfers=1, horizon_gws=6,
        allow_hits=True, freeze_gkp_transfers=True, transfer_hurdle_xp=999.0,
    )
    for step in roadmap:
        assert step.free_transfers_before <= FREE_TRANSFER_CAP
        assert step.transfers_made == 0


def test_allow_hits_false_never_exceeds_free_transfers(transfer_db):
    conn, squad_ids, _ids = transfer_db
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=1, horizon_gws=3,
        allow_hits=False, freeze_gkp_transfers=True, transfer_hurdle_xp=0.0,
    )
    for step in roadmap:
        assert step.transfers_made <= step.free_transfers_before
        assert step.hit_cost == 0

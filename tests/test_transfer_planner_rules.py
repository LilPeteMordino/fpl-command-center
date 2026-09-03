"""Real assertion-based regression tests for the anti-churn rules in src/transfer_planner.py's
plan_transfers: the hurdle rate, the FT banking cap, the GKP freeze ("Set-and-Forget"), and the
allow_hits bound. Built on a small, fully self-contained in-memory SQLite DB (not the real synced
one) so these run deterministically in CI with no network access and no dependency on today's
real player pool -- see conftest.py's own module docstring for why that matters.
"""
import sqlite3

import pytest

from src import database
from src.transfer_planner import FREE_TRANSFER_CAP, _hit_or_roll_rationale, plan_transfers

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


# --- Stricter hit-only transfer hurdle (HIT_TRANSFER_HURDLE_XP) -----------------------------------
# A separate, deliberately self-contained fixture (not transfer_db) so each test can control
# exactly which single upgrade candidate is available -- with both a modest and a big upgrade in
# the same pool simultaneously, the ILP always just picks the objectively-better one, which would
# make "the modest one specifically gets blocked" unobservable.

def _build_hit_hurdle_db(upgrade_xg: float, upgrade_xa: float, upgrade_web_name: str):
    """Same team/gameweek/fixture/squad shape as transfer_db (see its own docstring for why the
    FWD loop is offset), but with exactly ONE pool-only alternative to WeakMid -- verified live
    (see the commit introducing this fixture) that (xg=0.6, xa=0.4) nets a 2-gw lookahead delta of
    5.85 over WeakMid (clears the OLD 1.5 hurdle but not a forced hit's -4, i.e. net margin 1.85 <
    HIT_TRANSFER_HURDLE_XP) while (xg=1.6, xa=0.9) nets +14.2 (net margin ~10.2, clears it easily)."""
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
    for gw in range(2, 4):
        conn.execute(
            "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) VALUES (?, ?, ?, 0, ?, 0)",
            (gw, f"GW{gw}", "2099-01-01T00:00:00Z", 1 if gw == 2 else 0),
        )
        for team_a, team_b in [(1, 2), (3, 4), (5, 6)]:
            conn.execute(
                "INSERT INTO fixtures (event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished) "
                "VALUES (?, ?, ?, 3, 3, 0)",
                (gw, team_a, team_b),
            )

    pid = 1
    squad_ids = []
    _insert_player(conn, pid, 1, 1, 40, xg=0.0, xa=0.0, saves=2.0, starts_per_90=1.0); squad_ids.append(pid); pid += 1
    _insert_player(conn, pid, 2, 1, 40, xg=0.0, xa=0.0, saves=1.0, starts_per_90=0.3); squad_ids.append(pid); pid += 1
    for i in range(5):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 2, 45 + i, xg=0.05, xa=0.08, defcon=6.0, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    for i in range(4):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 3, 55 + i, xg=0.25, xa=0.2, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    weak_mid_id = pid
    _insert_player(conn, pid, 6, 3, 50, xg=0.0, xa=0.0, starts_per_90=0.9, web_name="WeakMid")
    squad_ids.append(pid); pid += 1
    for i in range(3):
        _insert_player(conn, pid, ((i + 3) % N_TEAMS) + 1, 4, 60 + i, xg=0.4, xa=0.1, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    assert len(squad_ids) == 15

    upgrade_id = pid
    _insert_player(conn, pid, 6, 3, 50, xg=upgrade_xg, xa=upgrade_xa, starts_per_90=0.95, web_name=upgrade_web_name)

    conn.commit()
    return conn, squad_ids, weak_mid_id, upgrade_id


def test_hit_transfer_hurdle_blocks_a_modest_upgrade_that_clears_only_the_old_free_hurdle():
    conn, squad_ids, weak_mid_id, modest_id = _build_hit_hurdle_db(0.6, 0.4, "ModestUpgrade")
    # free_transfers=0 forces ANY transfer this single-gw horizon to spend a hit (t=1 > ft_before=0).
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=0, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    # Net margin over holding is ~1.85 xP -- comfortably clears the plain 1.5 hurdle, but not
    # HIT_TRANSFER_HURDLE_XP (4.5) -- must roll (hold) instead of taking the hit.
    assert step.transfers_made == 0
    assert step.hit_cost == 0
    assert modest_id not in step.transfers_in_ids
    assert weak_mid_id not in step.transfers_out_ids


def test_hit_transfer_hurdle_allows_a_big_enough_upgrade_to_take_the_hit():
    conn, squad_ids, weak_mid_id, big_id = _build_hit_hurdle_db(1.6, 0.9, "BigUpgrade")
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=0, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    # Net margin over holding is ~10.2 xP -- clears HIT_TRANSFER_HURDLE_XP (4.5) easily, so the
    # hit is worth taking.
    assert step.transfers_made == 1
    assert step.hit_cost == 4  # sanity: a single hit costs HIT_COST (4)
    assert big_id in step.transfers_in_ids
    assert weak_mid_id in step.transfers_out_ids


def test_hit_transfer_hurdle_not_applied_when_a_free_transfer_covers_it():
    """The SAME modest upgrade that gets blocked as a hit (previous test) should still go through
    when a free transfer is actually available -- hit_transfer_hurdle_xp only gates candidates
    that spend a hit, never a plain free swap."""
    conn, squad_ids, weak_mid_id, modest_id = _build_hit_hurdle_db(0.6, 0.4, "ModestUpgrade")
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=1, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    assert step.hit_cost == 0
    assert modest_id in step.transfers_in_ids
    assert weak_mid_id in step.transfers_out_ids


# --- Hit hurdle must be measured against the best FREE alternative, not against pure hold ---------
# Bug found live (a real GW1 2026-27 squad): with a genuinely good free transfer ALSO available
# that gameweek, the hit hurdle was being checked against t=0 (hold) for every candidate t, so a
# 3-transfer/-8 bundle cleared HIT_TRANSFER_HURDLE_XP purely by riding on the one good FREE
# transfer's own margin -- even though the two EXTRA hit-transfers, evaluated on their own merit,
# barely beat holding and would never have cleared the hurdle by themselves. Fixed by comparing a
# hit-spending candidate's margin against the best hit_cost == 0 candidate (which may itself be
# t=1, not t=0), not against t=0 directly.

def _build_bundled_hit_hurdle_db():
    """Same shape as _build_hit_hurdle_db, but with TWO weak MIDs in the squad and two pool-only
    upgrades: StrongMid is a big, clear upgrade over WeakMid1 (the same (xg=1.6, xa=0.9) profile
    _build_hit_hurdle_db's own "BigUpgrade" uses, worth ~10.2 xP over holding on its own -- clears
    even the strict hit hurdle by itself) and ModestMid is only a marginal upgrade over WeakMid2
    (the same (xg=0.6, xa=0.4) "ModestUpgrade" profile, worth ~1.85 xP over holding -- clears the
    plain 1.5 hurdle but NOT HIT_TRANSFER_HURDLE_XP's 4.5 on its own)."""
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
    for gw in range(2, 4):
        conn.execute(
            "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) VALUES (?, ?, ?, 0, ?, 0)",
            (gw, f"GW{gw}", "2099-01-01T00:00:00Z", 1 if gw == 2 else 0),
        )
        for team_a, team_b in [(1, 2), (3, 4), (5, 6)]:
            conn.execute(
                "INSERT INTO fixtures (event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished) "
                "VALUES (?, ?, ?, 3, 3, 0)",
                (gw, team_a, team_b),
            )

    pid = 1
    squad_ids = []
    _insert_player(conn, pid, 1, 1, 40, xg=0.0, xa=0.0, saves=2.0, starts_per_90=1.0); squad_ids.append(pid); pid += 1
    _insert_player(conn, pid, 2, 1, 40, xg=0.0, xa=0.0, saves=1.0, starts_per_90=0.3); squad_ids.append(pid); pid += 1
    for i in range(5):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 2, 45 + i, xg=0.05, xa=0.08, defcon=6.0, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    for i in range(3):
        _insert_player(conn, pid, (i % N_TEAMS) + 1, 3, 55 + i, xg=0.25, xa=0.2, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    weak_mid1_id = pid
    _insert_player(conn, pid, 6, 3, 50, xg=0.0, xa=0.0, starts_per_90=0.9, web_name="WeakMid1")
    squad_ids.append(pid); pid += 1
    weak_mid2_id = pid
    _insert_player(conn, pid, 5, 3, 50, xg=0.0, xa=0.0, starts_per_90=0.9, web_name="WeakMid2")
    squad_ids.append(pid); pid += 1
    for i in range(3):
        _insert_player(conn, pid, ((i + 3) % N_TEAMS) + 1, 4, 60 + i, xg=0.4, xa=0.1, starts_per_90=0.9)
        squad_ids.append(pid); pid += 1
    assert len(squad_ids) == 15

    strong_id = pid
    _insert_player(conn, pid, 6, 3, 50, xg=1.6, xa=0.9, starts_per_90=0.95, web_name="StrongMid")
    pid += 1
    modest_id = pid
    _insert_player(conn, pid, 5, 3, 50, xg=0.6, xa=0.4, starts_per_90=0.95, web_name="ModestMid")

    conn.commit()
    return conn, squad_ids, weak_mid1_id, weak_mid2_id, strong_id, modest_id


def test_hit_transfer_hurdle_rejects_a_marginal_extra_transfer_riding_on_a_good_free_one():
    conn, squad_ids, weak_mid1_id, weak_mid2_id, strong_id, modest_id = _build_bundled_hit_hurdle_db()
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=1, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    # The free transfer (StrongMid in, WeakMid1 out) is a clear, easy win -- takes it.
    assert step.transfers_made == 1
    assert step.hit_cost == 0
    assert strong_id in step.transfers_in_ids
    assert weak_mid1_id in step.transfers_out_ids
    # But the SECOND, hit-costing transfer (ModestMid for WeakMid2) does NOT get bundled in --
    # its own margin over the free-transfer baseline (~1.85 xP) doesn't clear
    # HIT_TRANSFER_HURDLE_XP (4.5), even though the bundle's margin over pure hold (~12 xP) would
    # have wrongly cleared it under the old (buggy) hold-only comparison.
    assert modest_id not in step.transfers_in_ids
    assert weak_mid2_id not in step.transfers_out_ids


def test_hit_transfer_hurdle_still_allows_two_transfers_when_both_individually_clear_it():
    conn, squad_ids, weak_mid1_id, weak_mid2_id, strong_id, modest_id = _build_bundled_hit_hurdle_db()
    # Same fixture, but demand a much bigger free transfer isn't enough on its own by raising the
    # base hurdle sky-high -- forces free_transfers=2 instead, so BOTH swaps are free (hit_cost=0)
    # and should both go through regardless of the hit-only hurdle (which never applies here).
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=2, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    assert step.hit_cost == 0
    assert strong_id in step.transfers_in_ids
    assert modest_id in step.transfers_in_ids


# --- hit_justification_margin & its rationale text -------------------------------------------------
# Bug found live: the "Hit/Roll Logic" rationale used to quote plan.net_points itself as "the net
# gain" justifying a hit -- but net_points is that single gameweek's own absolute score, not a
# margin over anything, and can legitimately be LOWER than the best free-only alternative would
# have scored that same week (the hit pays off over a couple of gameweeks' weighted fixtures, not
# necessarily this one -- see LOOKAHEAD_WEIGHTS). That false "you're net_points points better off"
# framing is exactly what made a real, defensible multi-gameweek trade-off look self-contradictory.

def test_hit_justification_margin_is_the_true_hurdle_margin_not_net_points():
    conn, squad_ids, weak_mid_id, big_id = _build_hit_hurdle_db(1.6, 0.9, "BigUpgrade")
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=0, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    assert step.hit_cost == 4
    # The margin that actually justified the hit (~10.2 xP over holding, per _build_hit_hurdle_db's
    # own docstring) is NOT the same number as net_points (this gameweek's absolute score).
    assert step.hit_justification_margin is not None
    assert step.hit_justification_margin != pytest.approx(step.net_points)
    assert step.hit_justification_margin > 4.5  # cleared HIT_TRANSFER_HURDLE_XP


def test_hit_justification_margin_is_none_when_no_hit_was_taken():
    conn, squad_ids, weak_mid_id, modest_id = _build_hit_hurdle_db(0.6, 0.4, "ModestUpgrade")
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=1, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    assert step.hit_cost == 0
    assert step.hit_justification_margin is None


def test_hit_or_roll_rationale_does_not_misquote_net_points_as_the_justifying_margin():
    conn, squad_ids, weak_mid_id, big_id = _build_hit_hurdle_db(1.6, 0.9, "BigUpgrade")
    roadmap = plan_transfers(
        conn, squad_ids, bank=50, free_transfers=0, horizon_gws=1,
        allow_hits=True, freeze_gkp_transfers=True,
    )
    step = roadmap[0]
    bullet = _hit_or_roll_rationale(step)
    # Must quote the real margin, not net_points -- and must NOT claim it's "this gameweek's" own
    # net gain, since it's a multi-gameweek-weighted figure.
    assert f"{step.hit_justification_margin:.1f}" in bullet.text
    assert f"{step.net_points:+.1f} xP after the hit" not in bullet.text
    assert "not necessarily" in bullet.text.lower() or "own net points" in bullet.text.lower()

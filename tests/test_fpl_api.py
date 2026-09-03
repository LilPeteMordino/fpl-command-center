"""Real assertion-based regression tests for src/fpl_api.py's deadline_has_passed -- the helper
behind the GW1 squad-sync fix (bootstrap-static's is_current flag can stay False for hours after
a gameweek's real deadline has passed, most visibly right after GW1, which used to make squad
sync misreport a live season as "pre-season still"). Deterministic, no network access.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src import fpl_api


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_deadline_in_the_past_has_passed():
    past = _iso(datetime.now(timezone.utc) - timedelta(hours=3))
    assert fpl_api.deadline_has_passed(past) is True


def test_deadline_in_the_future_has_not_passed():
    future = _iso(datetime.now(timezone.utc) + timedelta(days=2))
    assert fpl_api.deadline_has_passed(future) is False


def test_missing_deadline_has_not_passed():
    assert fpl_api.deadline_has_passed(None) is False
    assert fpl_api.deadline_has_passed("") is False


def test_unparseable_deadline_has_not_passed():
    assert fpl_api.deadline_has_passed("not-a-real-timestamp") is False


def test_accepts_the_exact_bootstrap_static_z_suffixed_format():
    # FPL's own API returns e.g. '2026-08-15T17:30:00Z' -- confirm the 'Z' suffix (not a
    # '+00:00' offset) parses correctly rather than raising.
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert past.endswith("Z")
    assert fpl_api.deadline_has_passed(past) is True


def test_accepts_an_explicit_non_utc_offset():
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    assert fpl_api.deadline_has_passed(past) is True


def test_naive_offset_less_timestamp_is_treated_as_utc_rather_than_raising():
    # Not FPL's real format (bootstrap-static always sends 'Z'), but a defensive edge case: a
    # bare timestamp with no timezone info must not raise TypeError when compared to an
    # offset-aware "now" -- it should be treated as UTC.
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    assert fpl_api.deadline_has_passed(past) is True
    assert fpl_api.deadline_has_passed(future) is False


# --- compute_price_change_alerts -----------------------------------------------------------------
# Real bug/gap found live: the net-transfers-only heuristic missed real risk on lower-ownership
# players, and had no way to know a price had already moved today (would keep showing a stale
# "act now" pill for something already resolved). Both fixed by also reading FPL's own
# cost_change_event (today's already-realized move) and price_change_percent (their own
# undocumented "progress toward next change" figure) -- see compute_price_change_alerts.

import sqlite3

from src import database


def _price_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.init_db(conn)
    return conn


def _insert_price_player(conn, pid, web_name, transfers_in=0, transfers_out=0, cost_change_event=0, price_change_percent=0.0):
    conn.execute(
        """
        INSERT INTO players (
            id, web_name, team_id, element_type, now_cost, selected_by_percent, form, total_points,
            ep_next, xg, xa, xgi, status, news, transfers_in_event, transfers_out_event,
            cost_change_event, price_change_percent
        ) VALUES (?, ?, 1, 3, 50, 5.0, 3.0, 20, 3.0, 0, 0, 0, 'a', '', ?, ?, ?, ?)
        """,
        (pid, web_name, transfers_in, transfers_out, cost_change_event, price_change_percent),
    )
    conn.commit()


def _seed_price_team(conn):
    conn.execute(
        "INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away, "
        "strength_defence_home, strength_defence_away) VALUES (1, 'Team1', 'T1', 1100, 1100, 1100, 1100)"
    )


def test_price_change_percent_alone_can_trigger_a_drop_alert_the_net_transfers_heuristic_misses():
    conn = _price_db()
    _seed_price_team(conn)
    # Modest absolute transfer counts (below the net-transfers threshold) but FPL's own percent
    # figure already deep into drop territory -- a low-ownership player crossing FPL's real
    # threshold on small absolute numbers, exactly what the old heuristic-only check would miss.
    _insert_price_player(conn, 1, "LowOwnershipDropper", transfers_in=100, transfers_out=200, price_change_percent=-60.0)
    alerts = fpl_api.compute_price_change_alerts(conn)
    assert len(alerts) == 1
    assert alerts[0]["direction"] == "drop"


def test_already_moved_today_is_excluded_even_if_still_trending():
    conn = _price_db()
    _seed_price_team(conn)
    # Huge net-transfer momentum AND a big price_change_percent, but the price has ALREADY
    # dropped today (cost_change_event != 0) -- nothing left to act on, shouldn't be flagged.
    _insert_price_player(
        conn, 1, "AlreadyMoved", transfers_in=0, transfers_out=500_000,
        cost_change_event=-1, price_change_percent=-95.0,
    )
    alerts = fpl_api.compute_price_change_alerts(conn)
    assert alerts == []


def test_net_transfers_heuristic_still_works_on_its_own():
    conn = _price_db()
    _seed_price_team(conn)
    _insert_price_player(conn, 1, "BigRiser", transfers_in=500_000, transfers_out=0, price_change_percent=0.0)
    alerts = fpl_api.compute_price_change_alerts(conn)
    assert len(alerts) == 1
    assert alerts[0]["direction"] == "rise"


def test_neither_signal_crossing_threshold_is_not_flagged():
    conn = _price_db()
    _seed_price_team(conn)
    _insert_price_player(conn, 1, "Quiet", transfers_in=1000, transfers_out=900, price_change_percent=5.0)
    assert fpl_api.compute_price_change_alerts(conn) == []


# --- sync_player_history --------------------------------------------------------------------------

class _FakeHistoryClient:
    """A minimal stand-in for FPLClient.get_element_summary -- no real network access."""

    def __init__(self, responses: dict):
        self.responses = responses  # player_id -> dict (or an exception instance to raise)
        self.requested_ids: list = []

    def get_element_summary(self, player_id: int) -> dict:
        self.requested_ids.append(player_id)
        response = self.responses.get(player_id)
        if isinstance(response, Exception):
            raise response
        return response or {"history": [], "history_past": []}


def test_sync_player_history_populates_both_tables():
    conn = _price_db()
    _seed_price_team(conn)
    _insert_price_player(conn, 1, "Player1")
    client = _FakeHistoryClient({
        1: {
            "history": [
                {"round": 1, "minutes": 90, "starts": 1, "expected_goals": "0.5", "expected_assists": "0.2",
                 "expected_goals_conceded": "1.0", "total_points": 6},
            ],
            "history_past": [
                {"season_name": "2025/26", "minutes": 3000, "starts": 33, "total_points": 200,
                 "expected_goals": "20.0", "expected_assists": "8.0", "expected_goals_conceded": "0.0"},
            ],
        },
    })
    result = fpl_api.sync_player_history(conn, client, [1])
    assert result == {"synced": 1, "failed": []}

    gw_row = conn.execute("SELECT * FROM player_gw_history WHERE player_id=1").fetchone()
    assert gw_row["round"] == 1
    assert gw_row["expected_goals"] == pytest.approx(0.5)

    season_row = conn.execute("SELECT * FROM player_season_history WHERE player_id=1").fetchone()
    assert season_row["season_name"] == "2025/26"
    assert season_row["expected_goals"] == pytest.approx(20.0)


def test_sync_player_history_skips_a_failed_player_without_aborting_the_batch():
    conn = _price_db()
    _seed_price_team(conn)
    _insert_price_player(conn, 1, "Player1")
    _insert_price_player(conn, 2, "Player2")
    client = _FakeHistoryClient({
        1: fpl_api.FPLAPIError("not found", status_code=404),
        2: {"history": [{"round": 1, "minutes": 90, "starts": 1, "expected_goals": "0.1",
                          "expected_assists": "0.0", "expected_goals_conceded": "1.0", "total_points": 2}],
            "history_past": []},
    })
    result = fpl_api.sync_player_history(conn, client, [1, 2])
    assert result == {"synced": 1, "failed": [1]}
    assert conn.execute("SELECT COUNT(*) c FROM player_gw_history WHERE player_id=2").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM player_gw_history WHERE player_id=1").fetchone()["c"] == 0


def test_sync_player_history_calls_progress_callback_for_every_player():
    conn = _price_db()
    _seed_price_team(conn)
    _insert_price_player(conn, 1, "Player1")
    _insert_price_player(conn, 2, "Player2")
    client = _FakeHistoryClient({})
    calls = []
    fpl_api.sync_player_history(conn, client, [1, 2], progress_callback=lambda done, total, name: calls.append((done, total)))
    assert calls == [(1, 2), (2, 2)]

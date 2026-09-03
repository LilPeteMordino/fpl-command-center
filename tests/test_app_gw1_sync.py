"""Real assertion-based regression tests for the GW1 squad-sync bug fix in app.py:
_picks_event_id (which gameweek to fetch live PICKS for) is deliberately distinct from
_current_or_next_event_id (which gameweek to PLAN/project for) because FPL's bootstrap-static can
leave a just-passed gameweek's is_current flag stuck False -- with is_next already advanced to the
following gameweek -- for hours after the real deadline. Using the flag-based lookup for picks
fetching in that window asks for a future, still-blank gameweek's picks and 404s, which used to
render as a misleading "Pre-season active" banner even once the season/GW1 was genuinely live.

Deterministic, no network access -- builds a minimal in-memory gameweeks table directly (matching
the pattern in tests/test_chip_planner_set2.py), no need for the full database.init_db schema
since _picks_event_id only touches the gameweeks table.
"""
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

import app


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def gw_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE gameweeks (
            id INTEGER PRIMARY KEY, name TEXT, deadline_time TEXT,
            is_current INTEGER, is_next INTEGER, finished INTEGER
        )
        """
    )
    return conn


def _insert_gw(conn, gw_id, deadline, is_current=0, is_next=0, finished=0):
    conn.execute(
        "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (gw_id, f"GW{gw_id}", deadline, is_current, is_next, finished),
    )


def test_gw1_deadline_just_passed_but_is_current_still_stale_resolves_to_gw1(gw_conn):
    # The exact reported bug: GW1's deadline passed minutes ago, but bootstrap-static hasn't
    # flipped is_current to True yet, and is_next has already moved on to GW2.
    now = datetime.now(timezone.utc)
    _insert_gw(gw_conn, 1, _iso(now - timedelta(minutes=10)), is_current=0, is_next=0, finished=0)
    _insert_gw(gw_conn, 2, _iso(now + timedelta(days=7)), is_current=0, is_next=1, finished=0)
    gw_conn.commit()

    assert app._picks_event_id(gw_conn) == 1
    # The flag-based lookup (used for projections, not picks) legitimately still says GW2 is next
    # -- that distinction is exactly why _picks_event_id exists as a separate function.
    assert app._current_or_next_event_id(gw_conn) == 2


def test_genuine_pre_season_before_any_deadline_falls_back_to_flag_based_lookup(gw_conn):
    now = datetime.now(timezone.utc)
    _insert_gw(gw_conn, 1, _iso(now + timedelta(days=3)), is_current=0, is_next=1, finished=0)
    gw_conn.commit()

    # No deadline has passed yet -- correctly defers to the is_next flag, same as before the fix.
    assert app._picks_event_id(gw_conn) == 1


def test_mid_season_uses_the_gameweek_whose_deadline_actually_passed_not_the_upcoming_one(gw_conn):
    now = datetime.now(timezone.utc)
    _insert_gw(gw_conn, 10, _iso(now - timedelta(days=1)), is_current=1, is_next=0, finished=0)
    _insert_gw(gw_conn, 11, _iso(now + timedelta(days=6)), is_current=0, is_next=1, finished=0)
    gw_conn.commit()

    assert app._picks_event_id(gw_conn) == 10


def test_no_gameweeks_at_all_returns_none(gw_conn):
    assert app._picks_event_id(gw_conn) is None


def test_picks_event_id_ignores_finished_gameweeks(gw_conn):
    now = datetime.now(timezone.utc)
    _insert_gw(gw_conn, 1, _iso(now - timedelta(days=200)), is_current=0, is_next=0, finished=1)
    _insert_gw(gw_conn, 2, _iso(now - timedelta(minutes=5)), is_current=0, is_next=0, finished=0)
    gw_conn.commit()

    assert app._picks_event_id(gw_conn) == 2


# --- _sync_squad_with_gw1_fallback: the live-probe safety net -----------------------------------

class _FakeClientAlwaysGW1Works:
    """Simulates the exact failure mode: whatever event_id was resolved 404s (stale/misleading
    data), but event=1's picks are genuinely live."""

    def __init__(self):
        self.requested_events = []

    def fetch_squad_state_stub(self, team_id, event_id):
        self.requested_events.append(event_id)
        if event_id == 1:
            return {"squad_ids": list(range(1, 16)), "bench_order": [], "captain_id": 1, "vice_id": 2,
                     "bank": 0, "team_value": 1000, "team_name": "T", "manager_name": "M",
                     "overall_rank": 1, "total_transfers": 0, "chips_played": [],
                     "free_transfers": 1, "leagues_classic": [], "leagues_h2h": []}
        raise app.FPLAPIError("not found", status_code=404)


def test_gw1_fallback_probe_recovers_when_the_resolved_event_404s(monkeypatch):
    fake = _FakeClientAlwaysGW1Works()
    monkeypatch.setattr(app.fpl_api, "fetch_squad_state", lambda client, team_id, event_id: fake.fetch_squad_state_stub(team_id, event_id))

    team = app._sync_squad_with_gw1_fallback(client=object(), team_id=123, event_id=2)
    assert team["squad_ids"] == list(range(1, 16))
    assert fake.requested_events == [2, 1]  # tried the resolved event first, then the GW1 probe


def test_gw1_fallback_probe_not_attempted_twice_when_event_id_is_already_1(monkeypatch):
    calls = []

    def stub(client, team_id, event_id):
        calls.append(event_id)
        raise app.FPLAPIError("not found", status_code=404)

    monkeypatch.setattr(app.fpl_api, "fetch_squad_state", stub)

    with pytest.raises(app.FPLAPIError):
        app._sync_squad_with_gw1_fallback(client=object(), team_id=123, event_id=1)
    assert calls == [1]  # no redundant second attempt at the same event


def test_gw1_fallback_probe_reraises_when_both_attempts_fail(monkeypatch):
    def stub(client, team_id, event_id):
        raise app.FPLAPIError("not found", status_code=404)

    monkeypatch.setattr(app.fpl_api, "fetch_squad_state", stub)

    with pytest.raises(app.FPLAPIError):
        app._sync_squad_with_gw1_fallback(client=object(), team_id=123, event_id=5)


def test_gw1_fallback_probe_does_not_swallow_non_404_errors(monkeypatch):
    def stub(client, team_id, event_id):
        raise app.FPLAPIError("server error", status_code=500)

    monkeypatch.setattr(app.fpl_api, "fetch_squad_state", stub)

    with pytest.raises(app.FPLAPIError) as exc_info:
        app._sync_squad_with_gw1_fallback(client=object(), team_id=123, event_id=2)
    assert exc_info.value.status_code == 500

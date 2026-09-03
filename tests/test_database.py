"""Real assertion-based regression tests for src/database.py.

Real bug found live: clear_all_data's DELETE order didn't include the newer
player_gw_history/player_season_history tables (added for the recent-form rolling window and
prior-season cold-start fallback) -- with PRAGMA foreign_keys = ON (see get_connection), deleting
`players` while those child tables still held rows referencing player_id raised
sqlite3.IntegrityError: FOREIGN KEY constraint failed, breaking the main "Sync Live FPL Data"
button entirely for anyone who'd run "Sync Player History" first.
"""
import sqlite3

from src import database


def _seeded_conn() -> sqlite3.Connection:
    """A DB with at least one row in every table clear_all_data touches, foreign keys ON --
    matches get_connection()'s real pragma, since that's exactly what let this bug bite live."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    database.init_db(conn)

    conn.execute(
        "INSERT INTO teams (id, name, short_name, strength_attack_home, strength_attack_away, "
        "strength_defence_home, strength_defence_away) VALUES (1, 'Team1', 'T1', 1100, 1100, 1100, 1100)"
    )
    conn.execute(
        "INSERT INTO gameweeks (id, name, deadline_time, is_current, is_next, finished) "
        "VALUES (1, 'GW1', '2099-01-01T00:00:00Z', 1, 0, 1)"
    )
    conn.execute(
        """
        INSERT INTO players (
            id, web_name, team_id, element_type, now_cost, selected_by_percent, form, total_points,
            ep_next, xg, xa, xgi, status, news
        ) VALUES (1, 'Player1', 1, 3, 50, 5.0, 3.0, 20, 3.0, 0, 0, 0, 'a', '')
        """
    )
    conn.execute(
        "INSERT INTO fixtures (event, team_h, team_a, team_h_difficulty, team_a_difficulty, finished) "
        "VALUES (1, 1, 1, 3, 3, 0)"
    )
    conn.execute(
        "INSERT INTO user_squad (manager_id, event, player_id, position_in_squad, is_captain, is_vice) "
        "VALUES (1, 1, 1, 1, 0, 0)"
    )
    conn.execute(
        "INSERT INTO external_projections (player_id, event, xp, source) VALUES (1, 1, 5.0, 'custom')"
    )
    conn.execute("INSERT INTO preseason_adjustments (player_id) VALUES (1)")
    # The two tables that caused the real bug -- both FK-reference players(id).
    conn.execute(
        "INSERT INTO player_gw_history (player_id, round, minutes, starts, expected_goals, "
        "expected_assists, expected_goals_conceded, total_points) VALUES (1, 1, 90, 1, 0.5, 0.2, 1.0, 6)"
    )
    conn.execute(
        "INSERT INTO player_season_history (player_id, season_name, minutes, starts, total_points, "
        "expected_goals, expected_assists, expected_goals_conceded) VALUES (1, '2025/26', 3000, 33, 200, 20.0, 8.0, 0.0)"
    )
    conn.commit()
    return conn


def test_clear_all_data_does_not_raise_with_history_tables_populated():
    conn = _seeded_conn()
    database.clear_all_data(conn)  # must not raise sqlite3.IntegrityError


def test_clear_all_data_actually_empties_every_table_including_history():
    conn = _seeded_conn()
    database.clear_all_data(conn)
    tables = [
        "user_squad", "external_projections", "preseason_adjustments", "player_gw_history",
        "player_season_history", "fixtures", "players", "gameweeks", "teams",
    ]
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        assert count == 0, f"{table} still has rows after clear_all_data"

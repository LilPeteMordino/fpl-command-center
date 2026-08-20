"""SQLite schema definition and connection handling for the FPL analytics engine."""
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Optional, Union

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER
);

CREATE TABLE IF NOT EXISTS gameweeks (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    deadline_time TEXT,
    is_current INTEGER NOT NULL DEFAULT 0,
    is_next INTEGER NOT NULL DEFAULT 0,
    finished INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    web_name TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    element_type INTEGER NOT NULL,
    now_cost INTEGER NOT NULL,
    selected_by_percent REAL,
    form REAL,
    total_points INTEGER,
    ep_next REAL,
    xg REAL,
    xa REAL,
    xgi REAL,
    status TEXT,
    news TEXT,
    xg_per_90 REAL,
    xa_per_90 REAL,
    saves_per_90 REAL,
    defensive_contribution_per_90 REAL,
    starts_per_90 REAL,
    starts INTEGER,
    chance_of_playing_next_round INTEGER,
    penalties_order INTEGER,
    corners_order INTEGER,
    transfers_in_event INTEGER,
    transfers_out_event INTEGER
);

CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    event INTEGER REFERENCES gameweeks(id),
    team_h INTEGER NOT NULL REFERENCES teams(id),
    team_a INTEGER NOT NULL REFERENCES teams(id),
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    finished INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_squad (
    manager_id INTEGER NOT NULL,
    event INTEGER NOT NULL REFERENCES gameweeks(id),
    player_id INTEGER NOT NULL REFERENCES players(id),
    position_in_squad INTEGER NOT NULL,
    is_captain INTEGER NOT NULL DEFAULT 0,
    is_vice INTEGER NOT NULL DEFAULT 0,
    bank_balance INTEGER,
    PRIMARY KEY (manager_id, event, player_id)
);

-- Per-gameweek xP figures parsed from user-uploaded external projections CSVs (e.g. manually
-- exported from the user's own FPL Review / FPL Form accounts -- see src/projections.py).
-- Multiple named sources can coexist per player+gameweek; see optimizer.ensemble_from_sources /
-- get_ensemble_xp for how they're weighted-averaged into the engine's own positional xP model.
CREATE TABLE IF NOT EXISTS external_projections (
    player_id INTEGER NOT NULL REFERENCES players(id),
    event INTEGER NOT NULL REFERENCES gameweeks(id),
    xp REAL NOT NULL,
    xmins REAL,
    source TEXT NOT NULL DEFAULT 'custom',
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, event, source)
);

-- Manual pre-season scouting overrides (see the "Pre-Season Scouting & Overrides" sidebar in
-- app.py): out-of-position deployments, penalty/set-piece duty observed in friendlies before the
-- live FPL API's own penalties_order/corners_order catch up, and a per-player xMins gatekeeper
-- override for a nailed-vs-rotation read the season's real minutes data doesn't exist yet to
-- support. One row per player who has ANY override set; a player with no row uses the model's
-- own pre-season fallbacks untouched -- see optimizer.apply_preseason_adjustment.
CREATE TABLE IF NOT EXISTS preseason_adjustments (
    player_id INTEGER PRIMARY KEY REFERENCES players(id),
    preseason_mins_status TEXT,
    is_out_of_position INTEGER NOT NULL DEFAULT 0,
    preseason_penalties INTEGER NOT NULL DEFAULT 0,
    preseason_set_pieces INTEGER NOT NULL DEFAULT 0,
    custom_xmins_override REAL
);

-- Sidebar UI preferences (Formation Lock, Starter Security profile, Risk Profile, Squad
-- Locks/Blacklist, ...) that previously reset every session since only squad/draft state
-- persisted to SQLite -- see save_preference/get_preference. value is JSON-encoded so one table
-- can hold a string, number, bool, or list under a single schema; not FK-linked to
-- players/gameweeks/teams, so clear_all_data() (a live-data resync) deliberately leaves it alone
-- -- these are user settings, not synced FPL data.
CREATE TABLE IF NOT EXISTS user_preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_players_team_id ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event);
CREATE INDEX IF NOT EXISTS idx_user_squad_manager_event ON user_squad(manager_id, event);
CREATE INDEX IF NOT EXISTS idx_external_projections_event ON external_projections(event);
"""


def get_connection() -> sqlite3.Connection:
    """Open a connection to the local FPL SQLite database, creating the data dir if needed."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after the initial players table shipped. CREATE TABLE IF NOT EXISTS is a no-op
# on an existing table, so these are backfilled via ALTER TABLE for databases created before
# the positional xP engine (xG/xAG/DEFCON) was added.
PLAYERS_MIGRATION_COLUMNS = {
    "xg_per_90": "REAL",
    "xa_per_90": "REAL",
    "saves_per_90": "REAL",
    "defensive_contribution_per_90": "REAL",
    "starts_per_90": "REAL",
    "starts": "INTEGER",
    "chance_of_playing_next_round": "INTEGER",
    "penalties_order": "INTEGER",
    "corners_order": "INTEGER",
    "transfers_in_event": "INTEGER",
    "transfers_out_event": "INTEGER",
}


def _migrate_players_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(players)").fetchall()}
    for column, sql_type in PLAYERS_MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE players ADD COLUMN {column} {sql_type}")
    conn.commit()


USER_SQUAD_MIGRATION_COLUMNS = {"bank_balance": "INTEGER"}


def _migrate_user_squad_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(user_squad)").fetchall()}
    for column, sql_type in USER_SQUAD_MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE user_squad ADD COLUMN {column} {sql_type}")
    conn.commit()


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create all tables/indexes if they don't already exist, and backfill any columns added
    to an existing table since it was first created."""
    owns_conn = conn is None
    conn = conn or get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        _migrate_players_columns(conn)
        _migrate_user_squad_columns(conn)
    finally:
        if owns_conn:
            conn.close()


def clear_all_data(conn: sqlite3.Connection) -> None:
    """Delete all rows from every table (schema stays intact). Children before parents for FKs.

    user_squad (including any locally saved pre-season draft) is wiped along with everything
    else, since its rows reference players/gameweeks by foreign key and can't outlive them.
    Callers that want a draft to survive a live-data refresh should save it with
    load_local_draft() beforehand and restore it with save_local_draft() afterwards -- see
    app.py's "Sync Live Data" handler.
    """
    for table in (
        "user_squad", "external_projections", "preseason_adjustments", "fixtures", "players", "gameweeks", "teams",
    ):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


# --- Local pre-season draft ("My Team" before the FPL API unlocks real GW1 picks) -----------

LOCAL_DRAFT_MANAGER_ID = 0  # sentinel: no real FPL manager has id 0


def _any_gameweek_id(conn: sqlite3.Connection) -> int:
    """A valid gameweeks.id to satisfy user_squad's FK when storing a local draft. The actual
    event value doesn't mean anything for a local draft -- rows are identified by manager_id."""
    row = conn.execute("SELECT id FROM gameweeks ORDER BY id LIMIT 1").fetchone()
    if row is None:
        raise ValueError("No gameweeks found locally -- sync live data before saving a draft.")
    return row["id"]


def save_local_draft(
    conn: sqlite3.Connection,
    player_ids: list,
    bank_balance: int,
    captain_id: int | None = None,
    vice_id: int | None = None,
) -> None:
    """Persist a 15-player pre-season draft squad locally."""
    if len(set(player_ids)) != 15:
        raise ValueError(f"A draft must contain exactly 15 unique players, got {len(set(player_ids))}.")
    event_id = _any_gameweek_id(conn)
    conn.execute("DELETE FROM user_squad WHERE manager_id = ?", (LOCAL_DRAFT_MANAGER_ID,))
    rows = [
        (LOCAL_DRAFT_MANAGER_ID, event_id, pid, i + 1, int(pid == captain_id), int(pid == vice_id), bank_balance)
        for i, pid in enumerate(player_ids)
    ]
    conn.executemany(
        """
        INSERT INTO user_squad (manager_id, event, player_id, position_in_squad, is_captain, is_vice, bank_balance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def load_local_draft(conn: sqlite3.Connection) -> dict | None:
    """{"player_ids": [...], "bank_balance": int, "captain_id": ..., "vice_id": ...}, or None
    if no draft has been saved yet."""
    rows = conn.execute(
        "SELECT player_id, is_captain, is_vice, bank_balance FROM user_squad "
        "WHERE manager_id = ? ORDER BY position_in_squad",
        (LOCAL_DRAFT_MANAGER_ID,),
    ).fetchall()
    if not rows:
        return None
    return {
        "player_ids": [r["player_id"] for r in rows],
        "bank_balance": rows[0]["bank_balance"] if rows[0]["bank_balance"] is not None else 0,
        "captain_id": next((r["player_id"] for r in rows if r["is_captain"]), None),
        "vice_id": next((r["player_id"] for r in rows if r["is_vice"]), None),
    }


def has_local_draft(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM user_squad WHERE manager_id = ? LIMIT 1", (LOCAL_DRAFT_MANAGER_ID,)
    ).fetchone()
    return row is not None


# --- External xP projections (user-uploaded CSVs, e.g. FPL Review / FPL Form exports) ----------
# Multiple named sources can coexist for the same player+gameweek (PK is (player_id, event,
# source)) -- see optimizer.ensemble_from_sources/get_ensemble_xp for how they're combined.
# "source" is a free-form label; the app currently uses "fpl_review" and "fpl_form" for its two
# upload slots, plus "custom" for anything else ingested directly (see src/projections.py).

EXTERNAL_PROJECTION_SOURCES = ("fpl_review", "fpl_form", "custom")


def save_external_projections(conn: sqlite3.Connection, rows: list, source: str = "custom") -> int:
    """rows: list of {"player_id": int, "event": int, "xp": float, "xmins": float | None}.
    Upserts (player_id, event, source) so re-uploading the same export overwrites cleanly.
    Returns the number of rows written."""
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO external_projections (player_id, event, xp, xmins, source, uploaded_at)
        VALUES (:player_id, :event, :xp, :xmins, :source, datetime('now'))
        ON CONFLICT(player_id, event, source) DO UPDATE SET
            xp = excluded.xp, xmins = excluded.xmins, uploaded_at = excluded.uploaded_at
        """,
        [{**row, "source": source} for row in rows],
    )
    conn.commit()
    return len(rows)


def get_external_projections(
    conn: sqlite3.Connection,
    event_ids: Optional[list] = None,
    source: Optional[Union[str, list]] = None,
) -> dict:
    """{(player_id, event, source): {"xp": float, "xmins": float | None}}, optionally restricted
    to a set of gameweek ids and/or a source name (or list of source names) -- source=None
    (the default) returns every source. The key always carries `source` (even for a single-source
    query) so a caller combining multiple sources for the same player+gameweek doesn't lose one
    to a dict-key collision."""
    query = "SELECT player_id, event, xp, xmins, source FROM external_projections WHERE 1=1"
    params: list = []
    if event_ids:
        placeholders = ",".join(["?"] * len(event_ids))
        query += f" AND event IN ({placeholders})"
        params.extend(event_ids)
    if source is not None:
        sources = [source] if isinstance(source, str) else list(source)
        placeholders = ",".join(["?"] * len(sources))
        query += f" AND source IN ({placeholders})"
        params.extend(sources)
    rows = conn.execute(query, params).fetchall()
    return {
        (r["player_id"], r["event"], r["source"]): {"xp": r["xp"], "xmins": r["xmins"]} for r in rows
    }


def has_external_projections(conn: sqlite3.Connection, source: Optional[Union[str, list]] = None) -> bool:
    """True if any external projection row exists -- for a specific source (or list of sources)
    when given, for any source at all when not."""
    query = "SELECT 1 FROM external_projections WHERE 1=1"
    params: list = []
    if source is not None:
        sources = [source] if isinstance(source, str) else list(source)
        placeholders = ",".join(["?"] * len(sources))
        query += f" AND source IN ({placeholders})"
        params.extend(sources)
    query += " LIMIT 1"
    return conn.execute(query, params).fetchone() is not None


def clear_external_projections(conn: sqlite3.Connection, source: Optional[str] = None) -> None:
    if source is None:
        conn.execute("DELETE FROM external_projections")
    else:
        conn.execute("DELETE FROM external_projections WHERE source = ?", (source,))
    conn.commit()


# --- Pre-season scouting & tactical overrides -----------------------------------------------

def save_preseason_adjustment(
    conn: sqlite3.Connection,
    player_id: int,
    preseason_mins_status: Optional[str] = None,
    is_out_of_position: bool = False,
    preseason_penalties: bool = False,
    preseason_set_pieces: bool = False,
    custom_xmins_override: Optional[float] = None,
) -> None:
    """Upserts one player's pre-season override row -- see the "Pre-Season Scouting & Overrides"
    sidebar in app.py. Called with the full current state of that player's form each time (not a
    partial patch), so an unchecked toggle correctly clears back to False/None rather than being
    silently skipped."""
    conn.execute(
        """
        INSERT INTO preseason_adjustments
            (player_id, preseason_mins_status, is_out_of_position, preseason_penalties,
             preseason_set_pieces, custom_xmins_override)
        VALUES (:player_id, :preseason_mins_status, :is_out_of_position, :preseason_penalties,
                :preseason_set_pieces, :custom_xmins_override)
        ON CONFLICT(player_id) DO UPDATE SET
            preseason_mins_status = excluded.preseason_mins_status,
            is_out_of_position = excluded.is_out_of_position,
            preseason_penalties = excluded.preseason_penalties,
            preseason_set_pieces = excluded.preseason_set_pieces,
            custom_xmins_override = excluded.custom_xmins_override
        """,
        {
            "player_id": player_id,
            "preseason_mins_status": preseason_mins_status,
            "is_out_of_position": int(is_out_of_position),
            "preseason_penalties": int(preseason_penalties),
            "preseason_set_pieces": int(preseason_set_pieces),
            "custom_xmins_override": custom_xmins_override,
        },
    )
    conn.commit()


def get_preseason_adjustments(conn: sqlite3.Connection) -> dict:
    """{player_id: {"preseason_mins_status", "is_out_of_position", "preseason_penalties",
    "preseason_set_pieces", "custom_xmins_override"}} for every player with a saved override --
    consumed by optimizer.fetch_players / transfer_planner.fetch_multi_gw_projections."""
    rows = conn.execute(
        "SELECT player_id, preseason_mins_status, is_out_of_position, preseason_penalties, "
        "preseason_set_pieces, custom_xmins_override FROM preseason_adjustments"
    ).fetchall()
    return {
        r["player_id"]: {
            "preseason_mins_status": r["preseason_mins_status"],
            "is_out_of_position": bool(r["is_out_of_position"]),
            "preseason_penalties": bool(r["preseason_penalties"]),
            "preseason_set_pieces": bool(r["preseason_set_pieces"]),
            "custom_xmins_override": r["custom_xmins_override"],
        }
        for r in rows
    }


def get_preseason_adjustment(conn: sqlite3.Connection, player_id: int) -> Optional[dict]:
    """A single player's saved override, or None if they don't have one -- used by the sidebar
    form to pre-fill the widgets for whichever player is currently selected."""
    return get_preseason_adjustments(conn).get(player_id)


def clear_preseason_adjustment(conn: sqlite3.Connection, player_id: int) -> None:
    conn.execute("DELETE FROM preseason_adjustments WHERE player_id = ?", (player_id,))
    conn.commit()


# --- UI preference persistence (sidebar widgets: Formation Lock, Starter Security, Risk Profile,
# Squad Locks/Blacklist) -----------------------------------------------------------------------
# A generic key/value store (JSON-encoded value) rather than one column per widget -- new sidebar
# controls can persist themselves by picking a new key, with no schema migration required.

def save_preference(conn: sqlite3.Connection, key: str, value: Any) -> None:
    """Upserts one UI preference. value can be any JSON-serializable type (str/int/float/bool/
    list/dict/None) -- e.g. save_preference(conn, "formation_lock", "4-3-3") or
    save_preference(conn, "squad_blacklist_ids", [12, 57])."""
    conn.execute(
        """
        INSERT INTO user_preferences (key, value, updated_at)
        VALUES (:key, :value, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        {"key": key, "value": json.dumps(value)},
    )
    conn.commit()


def get_preference(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    """The saved value for `key`, JSON-decoded back to its original type, or `default` if this
    preference has never been saved."""
    row = conn.execute("SELECT value FROM user_preferences WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def get_all_preferences(conn: sqlite3.Connection) -> dict:
    """{key: decoded value} for every saved preference -- lets the sidebar prefill its widgets
    with one query at startup instead of one get_preference() call per control."""
    rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def clear_preference(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM user_preferences WHERE key = ?", (key,))
    conn.commit()


@contextmanager
def db_connection():
    """Context manager yielding a connection that is always closed on exit."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {config.DB_PATH}")

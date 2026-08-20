"""Ingestion of user-uploaded external xP projections CSVs (e.g. manually exported from the
user's own FPL Review or FPL Form account, or any similarly-shaped "wide" per-gameweek
projection export), and comparison of two such sources against each other.

This is a deliberately *local file* loader, not a web fetcher. FPL Review has no public/free API
(an unauthenticated request to fplreview.com/free-planner/ returns HTTP 403 -- it's gated behind
a personal login). FPL Form (fplform.com) advertises its own "Export FPL Form Data" CSV feature,
but reaching it also requires loading your own squad through the site first -- there's no
anonymous/public endpoint either. So for both, the only way to use their numbers here is for the
user to export their own CSV from their own account and upload it. Nothing in this module makes
an HTTP request to either site, or handles any credential for either.

Column layout varies between planner tools and even between export options on the same tool, so
headers are matched with flexible, case/spacing-insensitive detection rather than a fixed schema:
  - a player-name column (Name / Player / web_name / ...)
  - an optional team column (Team / Club / ...), used to disambiguate same-named players
  - one "points" column per gameweek (GW1, 1_Pts, GW1_xPts, Pts_1, ...), optionally paired with
    an expected-minutes column (GW1_xMins, 1_xMins, xMins_1, ...)

If no name column or no gameweek columns can be detected, CSVFormatError is raised with the exact
headers seen, so the format can be diagnosed and this parser adjusted -- it never silently no-ops.

Gameweek numbering: most planner exports number columns relative to the next unplayed gameweek
(GW1 = the upcoming gameweek) rather than the fixed season gameweek id, since that's how the
underlying tools are actually used. gw_numbering="relative" (the default) maps column N to the
Nth gameweek in transfer_planner.get_horizon_event_ids; pass gw_numbering="absolute" if your
export instead labels columns with the real season gameweek number.

Parsed rows are matched against the local `players` table (exact web_name+team match, falling
back to a fuzzy name match within the resolved team, or the whole pool if team is unknown/
ambiguous) and stored via database.save_external_projections, tagged with a `source` label
("fpl_review", "fpl_form", or "custom" -- see database.EXTERNAL_PROJECTION_SOURCES). They are
never used standalone: optimizer.fetch_players and transfer_planner.fetch_multi_gw_projections
combine whichever named sources are available into a weighted ensemble (see
optimizer.ensemble_from_sources / get_ensemble_xp), falling back to this module's own positional
xP model only for a player neither uploaded source covers -- so a bad or partial upload can't
silently replace the whole projection engine, and compute_divergence_table (below) surfaces
players where two sources disagree, for a sanity check before trusting either.
"""
import csv
import difflib
import io
import re
from dataclasses import dataclass
from typing import Optional

from src import database

DEFAULT_SOURCE = "custom"  # the app's two named upload slots pass source="fpl_review"/"fpl_form" explicitly

NAME_HEADERS = {"name", "player", "webname", "playername", "fullname", "player_name"}
TEAM_HEADERS = {"team", "club", "teamname", "teamshort", "team_name"}
POS_HEADERS = {"pos", "position", "elementtype"}

_GW = r"(\d{1,2})"
POINTS_PATTERNS = [
    re.compile(rf"^gw{_GW}$"),
    re.compile(rf"^gw{_GW}pts$"),
    re.compile(rf"^gw{_GW}xpts$"),
    re.compile(rf"^gw{_GW}points$"),
    re.compile(rf"^{_GW}pts$"),
    re.compile(rf"^{_GW}xpts$"),
    re.compile(rf"^pts{_GW}$"),
    re.compile(rf"^xpts{_GW}$"),
    re.compile(rf"^ep{_GW}$"),
]
MINS_PATTERNS = [
    re.compile(rf"^gw{_GW}xmins$"),
    re.compile(rf"^gw{_GW}mins$"),
    re.compile(rf"^{_GW}xmins$"),
    re.compile(rf"^{_GW}mins$"),
    re.compile(rf"^xmins{_GW}$"),
    re.compile(rf"^mins{_GW}$"),
]

FUZZY_NAME_CUTOFF = 0.72


class CSVFormatError(ValueError):
    """The uploaded file doesn't look like a projections export we can parse. The message always
    includes the headers actually seen, so the mismatch can be diagnosed from the error alone."""


@dataclass
class ParsedCSV:
    rows: list  # [{"name": str, "team": Optional[str], "pos": Optional[str], "gws": {n: {"xp": float, "xmins": Optional[float]}}}]
    name_column: str
    team_column: Optional[str]
    gw_columns: dict  # gw number -> {"pts": header, "mins": header | None}
    headers: list


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.strip().lower())


def normalize_name(name: str) -> str:
    """Public (not underscore-prefixed) since src.replay's historical-gameweek name matching also
    needs it -- same promotion convention as xgi_per_90/has_set_piece_duty/is_vice_eligible."""
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"-", "nan", "none", "na", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_projections_csv(file_obj) -> ParsedCSV:
    """Parses an uploaded CSV (file-like object, e.g. Streamlit's UploadedFile) into rows keyed
    by detected gameweek columns. Raises CSVFormatError if it can't find a name column or any
    per-gameweek points columns."""
    raw = file_obj.read()
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames
    if not headers:
        raise CSVFormatError("The uploaded file has no header row, or is empty.")

    normalized = {h: _normalize_header(h) for h in headers}
    name_col = next((h for h in headers if normalized[h] in NAME_HEADERS), None)
    team_col = next((h for h in headers if normalized[h] in TEAM_HEADERS), None)
    pos_col = next((h for h in headers if normalized[h] in POS_HEADERS), None)

    if name_col is None:
        raise CSVFormatError(
            "Couldn't find a player-name column (looked for one of: "
            f"{sorted(NAME_HEADERS)}). Headers found in the file: {headers}"
        )

    gw_columns: dict = {}
    for h in headers:
        n = normalized[h]
        matched = False
        for pat in POINTS_PATTERNS:
            m = pat.match(n)
            if m:
                gw_columns.setdefault(int(m.group(1)), {})["pts"] = h
                matched = True
                break
        if matched:
            continue
        for pat in MINS_PATTERNS:
            m = pat.match(n)
            if m:
                gw_columns.setdefault(int(m.group(1)), {})["mins"] = h
                break

    gw_columns = {gw: cols for gw, cols in gw_columns.items() if "pts" in cols}
    if not gw_columns:
        raise CSVFormatError(
            "Couldn't find any per-gameweek points columns (looked for headers like 'GW1', "
            f"'1_Pts', 'GW1_xPts', ...). Headers found in the file: {headers}"
        )

    rows = []
    for raw_row in reader:
        name = (raw_row.get(name_col) or "").strip()
        if not name:
            continue
        team = (raw_row.get(team_col) or "").strip() or None if team_col else None
        pos = (raw_row.get(pos_col) or "").strip() or None if pos_col else None
        gws = {}
        for gw, cols in gw_columns.items():
            xp = _to_float(raw_row.get(cols["pts"]))
            if xp is None:
                continue
            xmins = _to_float(raw_row.get(cols["mins"])) if cols.get("mins") else None
            gws[gw] = {"xp": xp, "xmins": xmins}
        if gws:
            rows.append({"name": name, "team": team, "pos": pos, "gws": gws})

    return ParsedCSV(rows=rows, name_column=name_col, team_column=team_col, gw_columns=gw_columns, headers=headers)


# --- Matching parsed rows to local players -----------------------------------------------------

def _load_player_index(conn):
    rows = conn.execute(
        "SELECT p.id, p.web_name, p.team_id, t.name AS team_name, t.short_name AS team_short "
        "FROM players p JOIN teams t ON t.id = p.team_id"
    ).fetchall()
    by_name: dict = {}
    teams_by_norm: dict = {}
    for r in rows:
        by_name.setdefault(normalize_name(r["web_name"]), []).append(r)
        teams_by_norm[normalize_name(r["team_name"])] = r["team_id"]
        teams_by_norm[normalize_name(r["team_short"])] = r["team_id"]
    return rows, by_name, teams_by_norm


def match_row_to_player(row: dict, rows, by_name: dict, teams_by_norm: dict) -> Optional[int]:
    """Exact web_name match, disambiguated by team when the name isn't unique; falls back to a
    fuzzy name match (within the resolved team when known, the whole pool otherwise)."""
    team_id = teams_by_norm.get(normalize_name(row["team"])) if row.get("team") else None
    candidates = by_name.get(normalize_name(row["name"]), [])

    if team_id is not None:
        exact = [c for c in candidates if c["team_id"] == team_id]
        if exact:
            return exact[0]["id"]
    elif len(candidates) == 1:
        return candidates[0]["id"]
    elif candidates:
        return candidates[0]["id"]  # ambiguous team, but the name alone is unique enough to guess

    pool = [r for r in rows if team_id is None or r["team_id"] == team_id]
    pool_names = {r["web_name"]: r for r in pool}
    close = difflib.get_close_matches(row["name"], list(pool_names.keys()), n=1, cutoff=FUZZY_NAME_CUTOFF)
    if close:
        return pool_names[close[0]]["id"]
    return None


# --- Gameweek-number resolution ------------------------------------------------------------------

def _resolve_event_ids(conn, gw_numbers, gw_numbering: str) -> dict:
    """CSV gw number -> local gameweeks.id. 'relative' (default) treats column N as the Nth
    upcoming gameweek (matches how rolling-planner exports are normally labelled); 'absolute'
    treats the column number as the real season gameweek id directly."""
    if not gw_numbers:
        return {}
    if gw_numbering == "absolute":
        valid = {r["id"] for r in conn.execute("SELECT id FROM gameweeks").fetchall()}
        return {gw: gw for gw in gw_numbers if gw in valid}

    from src.transfer_planner import get_horizon_event_ids  # deferred: avoids a hard import cycle risk

    ordered = get_horizon_event_ids(conn, max(gw_numbers))
    return {gw: ordered[gw - 1] for gw in gw_numbers if 0 < gw <= len(ordered)}


# --- Public entry point ---------------------------------------------------------------------------

def ingest_projections_csv(
    conn, file_obj, source: str = DEFAULT_SOURCE, gw_numbering: str = "relative"
) -> dict:
    """Parses, matches, and saves an uploaded projections CSV. Returns a stats dict:
        {
            "rows_total": int, "matched_players": int, "rows_saved": int,
            "unmatched_names": [str, ...],
            "gameweeks_detected": [int, ...],       # CSV column numbers
            "gameweeks_unresolved": [int, ...],     # detected but no matching local gameweek
            "name_column": str, "team_column": str | None,
        }
    Raises CSVFormatError if the file itself can't be parsed at all (see parse_projections_csv).
    """
    parsed = parse_projections_csv(file_obj)
    rows, by_name, teams_by_norm = _load_player_index(conn)
    gw_to_event = _resolve_event_ids(conn, set(parsed.gw_columns.keys()), gw_numbering)

    to_save = []
    matched_names = set()
    unmatched_names = []
    unresolved_gws = set(parsed.gw_columns.keys()) - set(gw_to_event.keys())

    for row in parsed.rows:
        player_id = match_row_to_player(row, rows, by_name, teams_by_norm)
        if player_id is None:
            unmatched_names.append(row["name"])
            continue
        matched_names.add(row["name"])
        for gw, vals in row["gws"].items():
            event_id = gw_to_event.get(gw)
            if event_id is None:
                continue
            to_save.append({"player_id": player_id, "event": event_id, "xp": vals["xp"], "xmins": vals["xmins"]})

    saved = database.save_external_projections(conn, to_save, source=source)

    return {
        "rows_total": len(parsed.rows),
        "matched_players": len(matched_names),
        "rows_saved": saved,
        "unmatched_names": sorted(unmatched_names),
        "gameweeks_detected": sorted(parsed.gw_columns.keys()),
        "gameweeks_unresolved": sorted(unresolved_gws),
        "name_column": parsed.name_column,
        "team_column": parsed.team_column,
    }


# --- Model divergence (comparing two uploaded sources against each other) ----------------------

DIVERGENCE_THRESHOLD_DEFAULT = 1.5


def compute_divergence_table(
    conn, event_id: int, source_a: str = "fpl_review", source_b: str = "fpl_form",
    threshold: float = DIVERGENCE_THRESHOLD_DEFAULT,
) -> list:
    """Players where both `source_a` and `source_b` have an uploaded projection for `event_id`
    and disagree by more than `threshold` xP. Sorted by absolute disagreement, descending. Each
    row: {"player_id", "web_name", "team_name", source_a: xp, source_b: xp, "diff": source_a - source_b}.
    Players missing either source are excluded -- there's nothing to compare them against."""
    rows = database.get_external_projections(conn, event_ids=[event_id], source=[source_a, source_b])
    by_player: dict = {}
    for (player_id, _event, source), vals in rows.items():
        by_player.setdefault(player_id, {})[source] = vals["xp"]

    player_ids = list(by_player.keys())
    if not player_ids:
        return []
    placeholders = ",".join(["?"] * len(player_ids))
    info_rows = conn.execute(
        f"""
        SELECT p.id, p.web_name, t.name AS team_name FROM players p
        JOIN teams t ON t.id = p.team_id WHERE p.id IN ({placeholders})
        """,
        player_ids,
    ).fetchall()
    player_info = {r["id"]: {"web_name": r["web_name"], "team_name": r["team_name"]} for r in info_rows}

    table = []
    for player_id, vals in by_player.items():
        if source_a not in vals or source_b not in vals:
            continue
        diff = vals[source_a] - vals[source_b]
        if abs(diff) <= threshold:
            continue
        info = player_info.get(player_id, {"web_name": f"player #{player_id}", "team_name": "?"})
        table.append(
            {
                "player_id": player_id,
                "web_name": info["web_name"],
                "team_name": info["team_name"],
                source_a: round(vals[source_a], 2),
                source_b: round(vals[source_b], 2),
                "diff": round(diff, 2),
            }
        )
    table.sort(key=lambda r: abs(r["diff"]), reverse=True)
    return table

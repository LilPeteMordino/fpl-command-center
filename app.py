"""Streamlit dashboard tying together the FPL analytics engine: live data sync,
squad/pitch view, multi-gameweek transfer planning, squad generators, a rolling
fixture-difficulty matrix, and mini-league rival comparison.

Run with: streamlit run app.py
"""
import html
import time
from collections import Counter
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import chip_planner, config, database, fpl_api, live_tracker, optimizer, projections, replay, transfer_planner
from src.fpl_api import FPLAPIError, FPLClient, sync_all_with_fallback
from src.optimizer import OptimizationError

st.set_page_config(page_title="FPL Analytics Engine", page_icon="⚽", layout="wide")

STYLE_CSS_PATH = Path(__file__).resolve().parent / "assets" / "style.css"


def _inject_global_css() -> None:
    """Dark-theme overrides for Streamlit's own chrome (metrics/buttons/tabs/expanders/inputs)
    plus the custom classes the HTML components below render into (pitch/player cards, glass
    cards, fixture pills/ticker) -- see assets/style.css. Missing the file degrades to
    Streamlit's default theme rather than crashing the app."""
    try:
        css = STYLE_CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning(f"Stylesheet not found at {STYLE_CSS_PATH} -- using default Streamlit theme.")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _inject_pwa_meta() -> None:
    """Mobile PWA setup: installable home-screen bookmark / standalone display mode, no offline
    caching (no service worker registered here, by design -- see the manifest's own comment).
    Requires [server] enableStaticServing = true in .streamlit/config.toml so /app/static/... is
    actually served; static/manifest.json holds the manifest itself.

    Deliberately routed through a <script> (unsafe_allow_javascript=True) that manipulates
    document.head directly, NOT a literal st.html(<link>/<meta> string) -- verified live that
    st.html sanitizes its body with DOMPurify first, which strips head-only tags like <link> and
    <meta> outright (DOMPurify's default profile treats them as non-content elements), so they
    never reach the DOM at all with the straightforward approach. A <script> tag survives that
    same sanitization specifically because unsafe_allow_javascript=True opts into it, and since
    st.html's content is NOT iframed, document.head here is the real page head -- this is the
    one reliable way to land these tags where the browser will actually honor them (a stray
    <meta viewport>/<link rel=manifest> sitting in the body's DOM is simply ignored by browsers,
    manifest-installability checks included)."""
    st.html(
        """
        <script>
        (function() {
            const tags = [
                ['link', {rel: 'manifest', href: '/app/static/manifest.json'}],
                ['meta', {name: 'apple-mobile-web-app-capable', content: 'yes'}],
                ['meta', {name: 'apple-mobile-web-app-status-bar-style', content: 'black-translucent'}],
                ['meta', {name: 'apple-mobile-web-app-title', content: 'FPL Solver'}],
            ];
            for (const [tag, attrs] of tags) {
                const key = attrs.name || attrs.rel;
                if (document.head.querySelector(`${tag}[${attrs.name ? 'name' : 'rel'}="${key}"]`)) continue;
                const el = document.createElement(tag);
                for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
                document.head.appendChild(el);
            }
            const viewport = document.head.querySelector('meta[name="viewport"]');
            if (viewport) {
                viewport.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no');
            }
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


NAV_PAGES = [
    "Manager Command Center",
    "My Squad & Pitch View",
    "Horizon Transfer Planner",
    "Squad Optimizer & Generators",
    "Live Gameweek Radar",
    "Fixture Difficulty Matrix",
    "Rival Radar & Mini-League",
    "Chip Strategy & Tactics",
]


# --- Shared helpers -----------------------------------------------------------

def _current_or_next_event_id(conn):
    row = conn.execute("SELECT id FROM gameweeks WHERE is_next = 1 ORDER BY id LIMIT 1").fetchone()
    if row is None:
        row = conn.execute("SELECT id FROM gameweeks WHERE is_current = 1 ORDER BY id LIMIT 1").fetchone()
    return row["id"] if row else None


def _picks_event_id(conn):
    """The gameweek to fetch a manager's live PICKS for -- distinct from
    _current_or_next_event_id, which prefers the *upcoming* gameweek (is_next) for projection
    purposes. Picks only exist once a deadline has actually passed, and FPL's own bootstrap-static
    can leave is_current stuck False (with is_next already advanced to the following gameweek) for
    HOURS after a deadline genuinely passes -- most visibly right after the GW1 deadline, when
    asking for is_next's event would fetch an always-404 future gameweek's picks and misreport a
    live, already-picked-for season as "pre-season still".

    Resolves instead from the raw deadline_time column: the highest-id unfinished gameweek whose
    deadline has already passed (see fpl_api.deadline_has_passed) is the one whose picks should be
    live. Falls back to _current_or_next_event_id's flag-based logic when no gameweek's deadline
    has passed yet (genuine pre-season, before GW1) or deadline_time data isn't available."""
    rows = conn.execute(
        "SELECT id, deadline_time FROM gameweeks WHERE finished = 0 ORDER BY id"
    ).fetchall()
    passed_ids = [r["id"] for r in rows if fpl_api.deadline_has_passed(r["deadline_time"])]
    if passed_ids:
        return max(passed_ids)
    return _current_or_next_event_id(conn)


def _current_live_event_id(conn):
    """The gameweek currently IN PROGRESS (is_current=1), or None if no gameweek is live right
    now (pre-deadline, or between gameweeks) -- distinct from _current_or_next_event_id, which
    prefers the *upcoming* gameweek for projection purposes. The Live Gameweek Radar only makes
    sense once a gameweek's matches have actually kicked off."""
    row = conn.execute("SELECT id FROM gameweeks WHERE is_current = 1 ORDER BY id LIMIT 1").fetchone()
    return row["id"] if row else None


def _team_opponent_labels(conn, event_id) -> dict:
    """team_id -> 'OPP (H)' / 'OPP (A)' / 'OPP1 (H) + OPP2 (A)' for the given event."""
    if event_id is None:
        return {}
    rows = conn.execute(
        """
        SELECT f.team_h, f.team_a, th.short_name AS home_short, ta.short_name AS away_short
        FROM fixtures f
        JOIN teams th ON th.id = f.team_h
        JOIN teams ta ON ta.id = f.team_a
        WHERE f.event = ?
        """,
        (event_id,),
    ).fetchall()
    legs: dict = {}
    for row in rows:
        legs.setdefault(row["team_h"], []).append(f'{row["away_short"]} (H)')
        legs.setdefault(row["team_a"], []).append(f'{row["home_short"]} (A)')
    return {team_id: " + ".join(team_legs) for team_id, team_legs in legs.items()}


def _squad_to_dataframe(players) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Position": p.position,
                "Name": p.web_name,
                "Team": p.team_name,
                "Cost (£m)": p.cost_millions,
                "Ownership %": p.selected_by_percent,
                "Projected xP": p.projected_xp,
                "Attack xP": p.xp_breakdown.attack_xp if p.xp_breakdown else None,
                "Defensive xP": p.xp_breakdown.defensive_xp if p.xp_breakdown else None,
                "DEFCON Prob %": round(p.xp_breakdown.defcon_prob * 100, 1) if p.xp_breakdown else None,
                "CS Prob %": round(p.xp_breakdown.cs_prob * 100, 1) if p.xp_breakdown else None,
            }
            for p in players
        ]
    )


def _db_status(conn) -> str:
    try:
        n_players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        n_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    except Exception:
        return "Database not initialized yet."
    if n_players == 0:
        return "No data yet -- use 'Sync Live FPL Data' below."
    return f"{n_players} players / {n_teams} teams cached locally."


def _ensemble_weights() -> dict:
    """The current FPL Review / FPL Form ensemble split, from the sidebar slider (see
    render_sidebar) -- {"fpl_review": pct/100, "fpl_form": (100-pct)/100}. Falls back to
    optimizer.DEFAULT_ENSEMBLE_WEIGHTS (50/50) before the slider has been rendered/touched."""
    pct = st.session_state.get("fpl_review_weight_pct")
    if pct is None:
        return dict(optimizer.DEFAULT_ENSEMBLE_WEIGHTS)
    return {"fpl_review": pct / 100, "fpl_form": (100 - pct) / 100}


STARTER_SECURITY_OPTIONS = {
    "Conservative (xMins ≥ 75)": optimizer.STARTER_SECURITY_PROFILES["conservative"],
    "Balanced (xMins ≥ 60)": optimizer.STARTER_SECURITY_PROFILES["balanced"],
    "Punt / Aggressive (xMins ≥ 45)": optimizer.STARTER_SECURITY_PROFILES["aggressive"],
}
STARTER_SECURITY_DEFAULT_LABEL = "Balanced (xMins ≥ 60)"


def _min_starter_xmins() -> float:
    """The active Starting XI minutes-security floor, from the sidebar's Starter Security radio
    (see render_sidebar) -- falls back to Balanced before the radio has been rendered/touched."""
    label = st.session_state.get("starter_security_label", STARTER_SECURITY_DEFAULT_LABEL)
    return STARTER_SECURITY_OPTIONS.get(label, optimizer.DEFAULT_STARTER_XMINS_FLOOR)


def _starter_floor_relaxed_message(floor_used: Optional[float]) -> str:
    """Shown whenever optimizer.solve_starting_xi_with_fallback had to relax below the sidebar's
    own Starter Security setting to find a legal XI at all -- real case this covers: several
    squad members genuinely below the requested floor at once (thin/no recent minutes), which
    used to just dead-end the whole page with a bare 'Infeasible' error and no way forward short
    of a user manually finding the sidebar control themselves."""
    floor_desc = f"{floor_used:.0f} xMins" if floor_used is not None else "no minutes floor at all"
    return (
        f"Your Starter Security setting couldn't build a legal Starting XI from this squad -- too "
        f"many players are currently below that floor at once. Automatically relaxed to **{floor_desc}** "
        f"for this view. Consider a looser Starter Security profile in the sidebar, or address the "
        f"underlying minutes risk with a transfer -- the players below your floor are exactly who a "
        f"transfer suggestion would flag."
    )


def _risk_lambda() -> float:
    """The active Risk & Ownership Strategy's lambda weight (see optimizer.RISK_PROFILE_LAMBDA),
    from the sidebar's Optimization Strategy select_slider -- falls back to the default profile
    before the slider has been rendered/touched."""
    label = st.session_state.get("risk_profile_label", optimizer.DEFAULT_RISK_PROFILE)
    return optimizer.RISK_PROFILE_LAMBDA.get(label, optimizer.RISK_PROFILE_LAMBDA[optimizer.DEFAULT_RISK_PROFILE])


def _formation_lock() -> Optional[str]:
    """The active Formation Lock (see optimizer.FORMATION_CHOICES), from the sidebar's Formation
    Lock selectbox -- None for 'Auto (Best xP)' (or before the selectbox has been
    rendered/touched), which optimizer.py's solvers already treat as 'no lock, use the default
    flexible bounds'."""
    label = st.session_state.get("formation_lock_label", optimizer.AUTO_FORMATION_LABEL)
    return None if label == optimizer.AUTO_FORMATION_LABEL else label


def _squad_blacklist() -> set:
    """Player ids the sidebar's Blacklist multiselect always excludes from generated squads."""
    return set(st.session_state.get("squad_blacklist_ids", set()))


def _squad_locks() -> set:
    """Player ids the sidebar's Force Lock multiselect always includes in generated squads.
    Blacklist wins on overlap (see the warning render_sidebar shows for that case)."""
    return set(st.session_state.get("squad_lock_ids", set())) - _squad_blacklist()


# --- Sidebar preference persistence ---------------------------------------------------------
# Formation Lock, Starter Security, Risk Profile, and Squad Locks/Blacklist previously reset to
# their defaults every session -- only squad/draft state made it to SQLite. These are the only
# sidebar controls with a real "correct" per-manager default that's worth remembering (data
# source uploads and pre-season overrides already persist via their own tables/keys).
PERSISTED_LABEL_PREF_KEYS = ("starter_security_label", "formation_lock_label", "risk_profile_label")


def _init_persisted_prefs(conn) -> None:
    """Seed session_state from previously saved preferences, once per session -- the key-bound
    widgets below then behave exactly as before (read/write session_state via their own `key=`);
    on_change/post-widget calls persist any change back to SQLite via database.save_preference."""
    if st.session_state.get("_prefs_loaded"):
        return
    for key in PERSISTED_LABEL_PREF_KEYS:
        saved = database.get_preference(conn, key)
        if saved is not None:
            st.session_state.setdefault(key, saved)
    # Squad Locks/Blacklist persist by player id (stable across a resync), not by the multiselect's
    # own label strings (which embed the player's web_name/team and could shift) -- stashed here so
    # the multiselect defaults can be reconstructed once player_options is available below.
    st.session_state.setdefault("_persisted_lock_ids", set(database.get_preference(conn, "squad_lock_ids", []) or []))
    st.session_state.setdefault("_persisted_blacklist_ids", set(database.get_preference(conn, "squad_blacklist_ids", []) or []))
    st.session_state._prefs_loaded = True


def _persist_pref(conn, key: str) -> None:
    """on_change callback: writes a key-bound widget's current session_state value back to SQLite."""
    database.save_preference(conn, key, st.session_state[key])


def _render_projection_upload_slot(conn, source: str, label: str, key_prefix: str, gw_numbering: str) -> None:
    """One upload slot (FPL Review or FPL Form) within the ensemble expander: file uploader +
    ingest button + match-rate feedback + a clear button once something's loaded for `source`."""
    loaded = database.has_external_projections(conn, source=source)
    st.caption(f"✅ {label} loaded" if loaded else "Nothing loaded yet")

    uploaded = st.file_uploader(f"{label} CSV", type=["csv"], key=f"{key_prefix}_csv")
    if uploaded is not None and st.button(f"Ingest {label}", use_container_width=True, key=f"{key_prefix}_ingest"):
        try:
            stats = projections.ingest_projections_csv(conn, uploaded, source=source, gw_numbering=gw_numbering)
            st.cache_data.clear()
            gws_desc = ", ".join(str(g) for g in stats["gameweeks_detected"])
            st.success(
                f"Matched {stats['matched_players']}/{stats['rows_total']} players, saved "
                f"{stats['rows_saved']} player-gameweek rows (gw columns: {gws_desc})."
            )
            if stats["unmatched_names"]:
                shown = ", ".join(stats["unmatched_names"][:10])
                more = " ..." if len(stats["unmatched_names"]) > 10 else ""
                st.warning(f"Couldn't match: {shown}{more}")
            if stats["gameweeks_unresolved"]:
                st.warning(
                    f"Couldn't resolve column(s) {stats['gameweeks_unresolved']} to a local "
                    f"gameweek -- try the other numbering option above, or sync live data first."
                )
        except projections.CSVFormatError as exc:
            st.error(str(exc))

    if loaded and st.button(f"Clear {label}", use_container_width=True, key=f"{key_prefix}_clear"):
        database.clear_external_projections(conn, source=source)
        st.cache_data.clear()
        st.success(f"Cleared {label} data.")


def _ensure_sample_squad(conn) -> None:
    sample = optimizer.build_optimal_squad(conn, mode="balanced", ensemble_weights=_ensemble_weights(), locked_ids=_squad_locks(), excluded_ids=_squad_blacklist(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock())
    st.session_state.squad_ids = [p.id for p in sample]
    st.session_state.bank = 0
    st.session_state.manager_synced = False


PRESEASON_MINS_STATUS_OPTIONS = ["-- none --", "Nailed Starter", "Rotation/Shared", "Injured/Late Return"]


def _render_preseason_scouting_expander(conn) -> None:
    """The "Pre-Season Scouting & Overrides" sidebar drawer: quick per-player toggles for
    out-of-position deployments, penalty/set-piece duty observed in friendlies before the live
    FPL API's own penalties_order/corners_order catch up, and a manual xMins gatekeeper override
    -- see database.preseason_adjustments / optimizer.apply_preseason_adjustment for how these
    feed into the engine."""
    with st.expander("\U0001F3CB️ Pre-Season Scouting & Overrides"):
        st.caption(
            "Manual corrections from your own pre-season/friendlies scouting -- these take "
            "precedence over the model's own pre-season fallbacks wherever they disagree."
        )

        all_players = _cached_fetch_players(conn, tuple(sorted(_ensemble_weights().items())))
        if not all_players:
            st.info("No player data cached yet -- sync live data in the sidebar first.")
            return

        player_options = {
            f"{p.web_name} ({p.team_name})": p.id for p in sorted(all_players, key=lambda p: p.web_name)
        }
        selected_label = st.selectbox("Player", list(player_options.keys()), key="preseason_override_player")
        selected_id = player_options[selected_label]
        existing = database.get_preseason_adjustment(conn, selected_id) or {}

        current_status = existing.get("preseason_mins_status") or "-- none --"
        mins_status = st.selectbox(
            "Pre-season minutes status",
            PRESEASON_MINS_STATUS_OPTIONS,
            index=PRESEASON_MINS_STATUS_OPTIONS.index(current_status) if current_status in PRESEASON_MINS_STATUS_OPTIONS else 0,
            key=f"preseason_mins_status_{selected_id}",
        )
        is_oop = st.checkbox(
            "Out of position (e.g. a MID deployed as an auxiliary striker)",
            value=existing.get("is_out_of_position", False),
            key=f"preseason_oop_{selected_id}",
            help=f"+{optimizer.PRESEASON_OOP_ATTACK_BOOST:.0%} to baseline attacking expected points.",
        )
        pens = st.checkbox(
            "Confirmed penalty duty (pre-season/friendlies)",
            value=existing.get("preseason_penalties", False),
            key=f"preseason_pens_{selected_id}",
            help="Overrides the live FPL API's penalty order when it hasn't caught up yet, plus a small xP bump.",
        )
        set_pieces = st.checkbox(
            "Confirmed corner/free-kick duty (pre-season/friendlies)",
            value=existing.get("preseason_set_pieces", False),
            key=f"preseason_setpieces_{selected_id}",
        )

        use_custom_xmins = st.checkbox(
            "Override projected starting minutes (xMins)",
            value=existing.get("custom_xmins_override") is not None,
            key=f"preseason_use_xmins_{selected_id}",
            help="Used directly as the Starting XI minutes-security gatekeeper, bypassing both "
                 "the baseline formula and any uploaded-CSV xMins.",
        )
        custom_xmins = None
        if use_custom_xmins:
            custom_xmins = st.slider(
                "Custom xMins", min_value=0, max_value=90,
                value=int(existing.get("custom_xmins_override") or 60),
                key=f"preseason_xmins_{selected_id}",
            )

        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button("Save override", use_container_width=True, key=f"preseason_save_{selected_id}"):
                database.save_preseason_adjustment(
                    conn, selected_id,
                    preseason_mins_status=None if mins_status == "-- none --" else mins_status,
                    is_out_of_position=is_oop,
                    preseason_penalties=pens,
                    preseason_set_pieces=set_pieces,
                    custom_xmins_override=float(custom_xmins) if use_custom_xmins else None,
                )
                st.cache_data.clear()
                st.success(f"Saved override for {selected_label}.")
        with clear_col:
            if existing and st.button("Clear override", use_container_width=True, key=f"preseason_clear_{selected_id}"):
                database.clear_preseason_adjustment(conn, selected_id)
                st.cache_data.clear()
                st.success(f"Cleared override for {selected_label}.")

        active = database.get_preseason_adjustments(conn)
        if active:
            id_to_name = {p.id: p.web_name for p in all_players}
            names = ", ".join(id_to_name.get(pid, str(pid)) for pid in active)
            st.caption(f"{len(active)} player(s) with an active pre-season override: {names}")


# --- Top header bar: Automated FPL Team ID Sync ---------------------------------

def _sync_squad_with_gw1_fallback(client: FPLClient, team_id: int, event_id: int) -> dict:
    """fpl_api.fetch_squad_state, with one extra safety net for the specific way GW1 squad sync
    can get wrongly blocked: event_id (from _picks_event_id) is already deadline-aware, but if its
    own inputs are stale or missing (gameweeks table not freshly synced, deadline_time absent) it
    can still land on a gameweek whose picks aren't actually live yet. Rather than trust that and
    show a misleading "pre-season" message, a 404 on any event other than 1 is followed by one
    direct live probe of event=1's picks -- if THAT returns 200, the season/GW1 is genuinely
    active regardless of what event_id or FPL's own is_current flag said, and that data is used
    instead. Only re-raises the original 404 if the event-1 probe also fails."""
    try:
        return fpl_api.fetch_squad_state(client, team_id, event_id)
    except FPLAPIError as exc:
        if exc.status_code != 404 or event_id == 1:
            raise
        return fpl_api.fetch_squad_state(client, team_id, 1)



def _apply_synced_team(team: dict, team_id_input: str) -> None:
    """Persists a fpl_api.fetch_squad_state() result into session_state -- shared by every sync
    entry point (top header, sidebar) so a full sync always hydrates the same fields.

    selected_league_id is only defaulted here (setdefault), never overwritten on a re-sync --
    once a manager has picked an Active Mini-League from the dropdown, a later re-sync (e.g.
    after a new gameweek's picks lock in) shouldn't silently reset it back to whichever league
    happens to sort first."""
    st.session_state.manager_id = team_id_input
    st.session_state.squad_ids = team["squad_ids"]
    st.session_state.bank = team["bank"]
    st.session_state.free_transfers = team["free_transfers"]
    st.session_state.team_name = team["team_name"]
    st.session_state.manager_name = team["manager_name"]
    st.session_state.overall_rank = team["overall_rank"]
    st.session_state.total_transfers = team["total_transfers"]
    st.session_state.chip_usage = chip_planner.parse_chip_usage({"chips": team["chips_played"]})
    st.session_state.leagues_classic = team["leagues_classic"]
    st.session_state.leagues_h2h = team["leagues_h2h"]
    if team["leagues_classic"]:
        st.session_state.setdefault("selected_league_id", team["leagues_classic"][0]["id"])
    st.session_state.manager_synced = True


def _render_team_id_header(conn) -> None:
    """Top Header Bar: FPL Team ID + one-click Automated Team ID Sync -- hydrates the active 15,
    exact bank (ITB), team/manager identity, overall rank, season transfer count, chips played,
    joined mini-leagues, and estimated banked Free Transfers in a single
    fpl_api.fetch_squad_state() call, persisting the Team ID across the session. This is the
    full/rich sync; the sidebar's "Sync My Squad" does the same squad+bank fetch for anyone who
    prefers using it from there instead -- both write the same session keys."""
    col1, col2, col3 = st.columns([2, 1, 4])
    with col1:
        team_id_input = st.text_input(
            "FPL Team ID", value=st.session_state.get("manager_id", ""),
            key="header_team_id_input", label_visibility="collapsed", placeholder="FPL Team ID",
        )
    with col2:
        sync_clicked = st.button("\U0001F504 Sync Squad", use_container_width=True, key="header_sync_button")
    with col3:
        if st.session_state.get("manager_synced"):
            team_name = st.session_state.get("team_name")
            rank = st.session_state.get("overall_rank")
            ft = st.session_state.get("free_transfers")
            bank_m = st.session_state.get("bank", 0) / 10
            rank_text = f"Rank #{rank:,}" if rank else "Rank N/A"
            name_text = f"{team_name} · " if team_name else ""
            st.caption(f"✅ {name_text}{rank_text} · Bank £{bank_m:.1f}m · {ft if ft is not None else '?'} FT banked")
        else:
            st.caption("Enter your FPL Team ID and Sync to auto-hydrate squad, bank, rank, transfers, and chips.")

    if sync_clicked and team_id_input:
        try:
            team_id = int(team_id_input)
        except ValueError:
            st.error("FPL Team ID must be a number.")
            return

        event_id = _picks_event_id(conn)
        if event_id is None:
            st.error("No gameweeks found locally -- sync live data first.")
            return

        client = FPLClient()
        try:
            team = _sync_squad_with_gw1_fallback(client, team_id, event_id)
        except FPLAPIError as exc:
            if exc.status_code == 404:
                st.info("Pre-season active: GW1 picks are private until deadline. Use Squad Optimizer to build a draft squad.")
            else:
                st.error(f"Could not sync team: {exc}")
            return

        if len(team["squad_ids"]) != 15:
            st.error("Unexpected squad size returned from the FPL API.")
            return

        _apply_synced_team(team, team_id_input)
        st.success(f"Fully synced Team {team_id}: squad, bank, rank, chips, and free transfers.")


# --- Sidebar -------------------------------------------------------------------

def render_sidebar(conn) -> str:
    _init_persisted_prefs(conn)
    with st.sidebar:
        st.title("⚽ FPL Analytics")
        st.caption(_db_status(conn))

        if st.button("\U0001f504 Sync Live FPL Data", use_container_width=True):
            with st.spinner("Fetching live FPL API & external xP models..."):
                saved_draft = database.load_local_draft(conn)
                database.clear_all_data(conn)
                status = sync_all_with_fallback(conn)
                st.cache_data.clear()

                if status["source"] == "official":
                    gw_note = "fixtures/gameweeks synced" if status["fixtures_synced"] else "fixtures sync failed"
                    st.success(
                        f"Synced {status['players_synced']} players / {status['teams_synced']} teams "
                        f"from the official FPL API ({gw_note})."
                    )
                elif status["source"] == "fallback":
                    st.warning(
                        f"Official FPL API was unreachable; used the community fallback (season "
                        f"{status['season']}) instead: {status['teams_synced']} teams, "
                        f"{status['players_synced']} players. Fixtures/gameweeks weren't updated -- "
                        f"that data isn't available via the fallback -- retry once the API is back."
                    )
                else:
                    st.error(f"Sync failed on both the official API and the community fallback: {status['error']}")

                if saved_draft is not None:
                    try:
                        database.save_local_draft(
                            conn, saved_draft["player_ids"], saved_draft["bank_balance"],
                            saved_draft["captain_id"], saved_draft["vice_id"],
                        )
                    except Exception as exc:
                        st.warning(f"Could not restore your saved draft after the sync: {exc}")

        with st.expander("\U0001F553 Sync Player History (slower, optional)"):
            st.caption(
                "Fetches each player's per-gameweek breakdown for this season and their prior "
                "season's totals -- one request per player, so this can take a few minutes for "
                "the full pool, unlike the fast bulk sync above. Powers two things: a recent-form "
                "rolling window (reacts faster to a genuine role change than the flat "
                "season-to-date average) once there are enough real games banked, and a real "
                "prior-season prior for a true cold-start player instead of guessing off price/"
                "ownership alone. Safe to skip -- everything works without this, just with a "
                "slightly noisier early read."
            )
            if st.button("\U0001F553 Sync Player History", use_container_width=True):
                player_ids = [r["id"] for r in conn.execute("SELECT id FROM players WHERE status != 'u'").fetchall()]
                if not player_ids:
                    st.warning("No players cached locally yet -- run 'Sync Live FPL Data' first.")
                else:
                    progress = st.progress(0.0, text=f"0 / {len(player_ids)} players")

                    def _on_progress(done: int, total: int, _name: str) -> None:
                        progress.progress(done / total, text=f"{done} / {total} players")

                    client = FPLClient()
                    result = fpl_api.sync_player_history(conn, client, player_ids, progress_callback=_on_progress)
                    progress.empty()
                    st.cache_data.clear()
                    if result["failed"]:
                        st.warning(
                            f"Synced history for {result['synced']} players; {len(result['failed'])} "
                            f"couldn't be fetched (no history yet, or a request failed) and were skipped."
                        )
                    else:
                        st.success(f"Synced history for {result['synced']} players.")

        st.divider()
        with st.expander("📄 Projections & Data Source (optional)"):
            st.caption(
                "Upload your own exports from FPL Review and/or FPL Form to blend into this app's "
                "own xP model as a weighted ensemble. Neither has a public/free API, so both are "
                "manual files you export yourself and upload -- nothing here makes a web request."
            )

            numbering_choice = st.radio(
                "Gameweek column numbering (applies to both uploads)",
                ["Relative (col 1 = next gameweek)", "Absolute (col 1 = season GW1)"],
                horizontal=True,
                help="Most planner exports label columns relative to the next gameweek. Switch "
                     "to Absolute only if your file's columns are literally the season's GW numbers.",
            )
            gw_numbering = "relative" if numbering_choice.startswith("Relative") else "absolute"

            upload_col_a, upload_col_b = st.columns(2)
            with upload_col_a:
                st.markdown("**FPL Review**")
                _render_projection_upload_slot(conn, "fpl_review", "FPL Review", "fpl_review", gw_numbering)
            with upload_col_b:
                st.markdown("**FPL Form**")
                _render_projection_upload_slot(conn, "fpl_form", "FPL Form", "fpl_form", gw_numbering)

            review_loaded = database.has_external_projections(conn, source="fpl_review")
            form_loaded = database.has_external_projections(conn, source="fpl_form")

            st.divider()
            review_pct = st.slider(
                "Model Weighting: FPL Review % vs FPL Form %",
                min_value=0, max_value=100, value=st.session_state.get("fpl_review_weight_pct", 50),
                key="fpl_review_weight_pct",
                help="Only matters for a player covered by both sources -- one covered by only one "
                     "source uses that source directly, regardless of this slider. Players covered "
                     "by neither fall back to this app's own model.",
            )
            st.caption(
                f"FPL Review {review_pct}% / FPL Form {100 - review_pct}%"
                + ("" if (review_loaded and form_loaded) else " -- no effect until both sources are loaded")
            )

            if review_loaded and form_loaded:
                divergence_event_id = _current_or_next_event_id(conn)
                if divergence_event_id is not None:
                    table = projections.compute_divergence_table(conn, divergence_event_id)
                    st.caption(
                        f"Model Divergence Table -- next GW, |FPL Review − FPL Form| > "
                        f"{projections.DIVERGENCE_THRESHOLD_DEFAULT} xP"
                    )
                    if table:
                        df = pd.DataFrame(table).rename(
                            columns={
                                "web_name": "Player", "team_name": "Team",
                                "fpl_review": "FPL Review xP", "fpl_form": "FPL Form xP",
                                "diff": "Diff (Review − Form)",
                            }
                        )[["Player", "Team", "FPL Review xP", "FPL Form xP", "Diff (Review − Form)"]]
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No players disagree by more than the threshold for the next gameweek.")

        st.divider()
        with st.expander("\U0001F6E1️ Lineup Security & Squad Locks"):
            st.caption(
                "Starter Security sets a hard floor on projected starting minutes (xMins) for the "
                "Starting XI -- a player below it can't be picked into the 11 no matter how good "
                "their raw xP looks, which is what stops a 2nd/3rd-choice rotation asset (e.g. a "
                "backup goalkeeper) from sneaking in. It doesn't affect the 15-man squad itself, "
                "only who among them can start."
            )
            security_options = list(STARTER_SECURITY_OPTIONS.keys())
            st.radio(
                "Starter security",
                security_options,
                index=security_options.index(st.session_state.get("starter_security_label", STARTER_SECURITY_DEFAULT_LABEL)),
                key="starter_security_label",
                on_change=partial(_persist_pref, conn, "starter_security_label"),
                help="Conservative excludes more players (safer, may cost some xP ceiling); "
                     "Aggressive excludes fewer (higher ceiling, higher rotation risk).",
            )

            st.caption(
                "Formation Lock pins the Starting XI to an exact DEF-MID-FWD shape instead of "
                "letting the solver pick freely within the usual 3-5 DEF / 2-5 MID / 1-3 FWD range."
            )
            st.selectbox(
                "Formation Lock",
                optimizer.FORMATION_CHOICES,
                key="formation_lock_label",
                on_change=partial(_persist_pref, conn, "formation_lock_label"),
                help="Auto (Best xP) lets the solver choose whichever legal shape maximizes "
                     "projected points; any other option forces that exact formation.",
            )

            all_players = _cached_fetch_players(conn, tuple(sorted(_ensemble_weights().items())))
            player_options = {
                f"{p.web_name} ({p.team_name})": p.id for p in sorted(all_players, key=lambda p: p.web_name)
            }

            # Reconstruct the multiselects' default selections from last session's saved player
            # ids (not saved labels -- see _init_persisted_prefs) the first time this widget key
            # exists in session_state each session.
            if "squad_lock_labels" not in st.session_state:
                st.session_state.squad_lock_labels = [
                    label for label, pid in player_options.items() if pid in st.session_state._persisted_lock_ids
                ]
            if "squad_blacklist_labels" not in st.session_state:
                st.session_state.squad_blacklist_labels = [
                    label for label, pid in player_options.items() if pid in st.session_state._persisted_blacklist_ids
                ]

            lock_labels = st.multiselect(
                "Force lock (always include)", list(player_options.keys()), key="squad_lock_labels",
                help="These players are pinned into every generated/optimal squad.",
            )
            blacklist_labels = st.multiselect(
                "Blacklist / ban (always exclude)", list(player_options.keys()), key="squad_blacklist_labels",
                help="These players are excluded from every generated/optimal squad.",
            )
            st.session_state.squad_lock_ids = {player_options[label] for label in lock_labels}
            st.session_state.squad_blacklist_ids = {player_options[label] for label in blacklist_labels}
            database.save_preference(conn, "squad_lock_ids", sorted(st.session_state.squad_lock_ids))
            database.save_preference(conn, "squad_blacklist_ids", sorted(st.session_state.squad_blacklist_ids))

            overlap_ids = st.session_state.squad_lock_ids & st.session_state.squad_blacklist_ids
            if overlap_ids:
                overlap_names = [name for name, pid in player_options.items() if pid in overlap_ids]
                st.warning(f"Locked *and* blacklisted (blacklist wins): {', '.join(overlap_names)}")

        st.divider()
        with st.expander("\U0001F3AF Risk & Ownership Strategy"):
            st.caption(
                "How much the Squad Optimizer and Starting XI solver favor high-ownership/"
                "high-captaincy-likelihood players over pure projected points -- 'shielding' "
                "your rank against a mega-template asset hauling when you don't own it, at the "
                "cost of some raw EV. Applies to squad building, Starting XI selection, and "
                "captain/vice picks everywhere in the app."
            )
            risk_options = list(optimizer.RISK_PROFILE_LAMBDA.keys())
            st.select_slider(
                "Optimization Strategy",
                options=risk_options,
                value=st.session_state.get("risk_profile_label", optimizer.DEFAULT_RISK_PROFILE),
                key="risk_profile_label",
                on_change=partial(_persist_pref, conn, "risk_profile_label"),
                help="Pure Mathematical EV ignores ownership entirely; Conservative Shield most "
                     "strongly favors owning the high-EO template picks.",
            )

        st.divider()
        _render_preseason_scouting_expander(conn)

        st.divider()
        st.subheader("Manager Sync")
        manager_id_input = st.text_input("FPL Manager ID", value=st.session_state.get("manager_id", ""))
        if st.button("Sync My Squad", use_container_width=True) and manager_id_input:
            try:
                manager_id = int(manager_id_input)
                event_id = _picks_event_id(conn)
                if event_id is None:
                    st.error("No gameweeks found locally -- sync live data first.")
                else:
                    client = FPLClient()
                    try:
                        # Same full Automated Team ID Sync as the top header (squad, bank, rank,
                        # identity, joined leagues, chips, estimated banked FTs) -- see
                        # _apply_synced_team.
                        team = _sync_squad_with_gw1_fallback(client, manager_id, event_id)
                    except FPLAPIError as exc:
                        if exc.status_code == 404:
                            st.info(
                                "Pre-season active: GW1 picks are private until deadline. "
                                "Use Squad Optimizer to build a draft squad."
                            )
                        else:
                            st.error(f"Could not sync manager: {exc}")
                        team = None

                    if team is not None:
                        if len(team["squad_ids"]) != 15:
                            st.error("Unexpected squad size returned from the FPL API.")
                        else:
                            _apply_synced_team(team, manager_id_input)
                            st.success(f"Synced squad for manager {manager_id}.")
            except ValueError as exc:
                st.error(f"Could not sync manager: {exc}")

        if st.session_state.get("manager_synced"):
            team_name = st.session_state.get("team_name") or "—"
            manager_name = st.session_state.get("manager_name")
            rank = st.session_state.get("overall_rank")
            rank_text = f"#{rank:,}" if rank else "N/A"
            bank_text = f"£{st.session_state.get('bank', 0) / 10:.1f}m"
            st.caption(f"**{team_name}**" + (f" ({manager_name})" if manager_name else ""))
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Overall rank", rank_text)
            mcol2.metric("Bank", bank_text)

            leagues_classic = st.session_state.get("leagues_classic") or []
            if leagues_classic:
                league_labels = [
                    f"{l['name']} (Rank: {l['entry_rank']})" if l.get("entry_rank") else l["name"]
                    for l in leagues_classic
                ]
                current_league_id = st.session_state.get("selected_league_id")
                default_idx = next(
                    (i for i, l in enumerate(leagues_classic) if l["id"] == current_league_id), 0
                )
                chosen_label = st.selectbox(
                    "Active Mini-League", league_labels, index=default_idx, key="active_league_select",
                    help="Every classic mini-league your synced FPL account has joined -- picking "
                         "one here feeds the Rival Radar tab's league ID automatically.",
                )
                st.session_state.selected_league_id = leagues_classic[league_labels.index(chosen_label)]["id"]
            else:
                st.caption("No classic mini-leagues found on this account.")

        if "squad_ids" not in st.session_state:
            if st.button("Use a sample squad instead", use_container_width=True):
                try:
                    _ensure_sample_squad(conn)
                    st.success("Sample squad generated.")
                except OptimizationError as exc:
                    st.error(str(exc))

        st.session_state.setdefault("free_transfers", 1)
        st.session_state.free_transfers = st.number_input(
            "Free transfers available",
            min_value=0,
            max_value=5,
            value=st.session_state.free_transfers,
            step=1,
            help="The public FPL API doesn't expose this directly -- enter what the FPL website shows you.",
        )

        st.divider()
        nav = st.radio("Navigate", NAV_PAGES)

    return nav


# --- Tab 0: Manager Command Center -----------------------------------------------

# Mini-league objective -> which optimizer.build_optimal_squad mode "optimal" comparisons use.
# Shield Lead mirrors the highest-ownership team (protect a lead by matching the field);
# Differential Chase targets low-ownership players (climb ranks by diverging from the field).
STRATEGY_MODE_OPTIONS = {
    "Balanced": "balanced",
    "Shield Lead": "template",
    "Differential Chase": "differential",
}

def _next_deadline_countdown(conn, event_id) -> Optional[str]:
    if event_id is None:
        return None
    row = conn.execute("SELECT deadline_time FROM gameweeks WHERE id = ?", (event_id,)).fetchone()
    if row is None or not row["deadline_time"]:
        return None
    try:
        deadline = datetime.fromisoformat(row["deadline_time"].replace("Z", "+00:00"))
    except ValueError:
        return None
    delta = deadline - datetime.now(timezone.utc)
    if delta.total_seconds() <= 0:
        return "Deadline passed"
    days, remainder = divmod(int(delta.total_seconds()), 86400)
    hours = remainder // 3600
    return f"{days}d {hours}h"


def _bench_order_check(squad_ids: list, starting_xi: list, bench: list):
    """Compares the saved-squad order (position_in_squad, i.e. the order squad_ids is already
    in) against the ILP's auto-sub-optimal bench order. Returns (is_optimal, user_order,
    ilp_order) where both orders are lists of the 4 bench player ids."""
    starting_ids = {p.id for p in starting_xi}
    user_order = [pid for pid in squad_ids if pid not in starting_ids]
    ilp_order = [p.id for p in bench]
    return user_order == ilp_order, user_order, ilp_order


def _bench_order_message(user_order: list, ilp_order: list, id_to_name: dict) -> str:
    for i, (user_pid, ilp_pid) in enumerate(zip(user_order, ilp_order)):
        if user_pid != ilp_pid:
            correct_current_slot = user_order.index(ilp_pid) + 1
            return (
                f"Swap Sub {correct_current_slot} ({id_to_name.get(ilp_pid, ilp_pid)}) ahead of Sub {i + 1} "
                f"({id_to_name.get(user_pid, user_pid)}) to maximize auto-sub coverage."
            )
    return "Bench order looks fine."


def _benchmark_status(xp: float):
    """FPL gameweek score benchmarks: <48 below average, 48-58 solid, 59-67 high-performing,
    68+ elite. Boundaries are inclusive on the upper end of each tier."""
    if xp < 48:
        return "\U0001F534", "Below average risk"
    if xp <= 58:
        return "\U0001F7E1", "On track / solid"
    if xp <= 67:
        return "\U0001F7E2", "High-performing squad"
    return "\U0001F525", "Elite / optimal peak"


BENCHMARK_BAR_MAX_XP = 80.0  # visual ceiling for the progress bar; comfortably above the elite tier


def _positional_xp_breakdown(starting_xi: list, captain) -> dict:
    breakdown = {"GKP": 0.0, "DEF": 0.0, "MID": 0.0, "FWD": 0.0}
    for p in starting_xi:
        breakdown[p.position] += p.projected_xp
    breakdown["Captain armband"] = captain.projected_xp
    return breakdown


def _render_points_projection_panel(conn, squad_ids: list, starting_xi: list, captain, strategy_mode: str = "balanced") -> None:
    """"Points & Performance Projection" panel: next-GW / 3-GW / 5-GW team xP for the user's
    squad vs. a freshly ILP-optimized squad under the selected strategy mode (one shared
    projections fetch for both squads and both horizons), a benchmark status bar, and a
    positional xP breakdown of the Starting XI.

    Baseline xP source: optimizer.calculate_positional_xp (xG/xAG/CS-odds, fixture-weighted) --
    already the engine-wide default xP, not FPL's own ep_next, per the earlier positional-xP
    upgrade; ep_next remains available on each PlayerRow for reference but doesn't feed this.
    """
    st.subheader("Points & performance projection")

    user_next_gw_xp = optimizer.calculate_team_xp(starting_xi, captain)
    user_3gw = user_5gw = optimal_next_gw_xp = None

    horizon_gws = 5
    event_ids = transfer_planner.get_horizon_event_ids(conn, horizon_gws)
    if event_ids:
        projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids, ensemble_weights=_ensemble_weights())

        user_per_gw = transfer_planner.team_xp_by_gameweek(
            conn, squad_ids, event_ids=event_ids, projections=projections, min_starter_xmins=_min_starter_xmins()
        )
        user_per_gw[event_ids[0]] = user_next_gw_xp  # keep in sync with the exact XI/captain already shown above
        user_3gw = round(sum(user_per_gw.get(eid, 0.0) for eid in event_ids[:3]), 1)
        user_5gw = round(sum(user_per_gw.values()), 1)

        try:
            optimal_squad = optimizer.build_optimal_squad(
                conn, mode=strategy_mode, ensemble_weights=_ensemble_weights(),
                locked_ids=_squad_locks(), excluded_ids=_squad_blacklist(), risk_lambda=_risk_lambda(),
            )
            optimal_per_gw = transfer_planner.team_xp_by_gameweek(
                conn, [p.id for p in optimal_squad], event_ids=event_ids, projections=projections,
                min_starter_xmins=_min_starter_xmins(),
            )
            optimal_next_gw_xp = optimal_per_gw.get(event_ids[0])
        except OptimizationError:
            pass

    with st.container(horizontal=True):
        st.metric("\U0001F3AF Next GW projected points", f"{user_next_gw_xp:.1f} xP", border=True)
        st.metric(
            "\U0001F4C8 3-GW horizon projection", f"{user_3gw:.1f} xP" if user_3gw is not None else "N/A", border=True
        )
        if optimal_next_gw_xp is not None:
            delta = user_next_gw_xp - optimal_next_gw_xp
            st.metric(
                "\U0001F3C6 Optimization delta", f"{delta:+.1f} xP", border=True,
                help=f"User: {user_next_gw_xp:.1f} xP vs. Optimal ({strategy_mode}): {optimal_next_gw_xp:.1f} xP",
            )
        else:
            st.metric("\U0001F3C6 Optimization delta", "N/A", border=True)

    emoji, label = _benchmark_status(user_next_gw_xp)
    st.progress(
        min(user_next_gw_xp / BENCHMARK_BAR_MAX_XP, 1.0),
        text=f"{emoji} {label} -- {user_next_gw_xp:.1f} xP (next GW)",
    )

    st.caption("Positional xP breakdown -- Starting XI (next GW)")
    breakdown = _positional_xp_breakdown(starting_xi, captain)
    for col, (part_name, value) in zip(st.columns(5), breakdown.items()):
        col.metric(part_name, f"{value:.1f}")

    if user_5gw is not None:
        st.caption(f"5-GW horizon projection: {user_5gw:.1f} xP")

    st.divider()


def _render_horizon_planner_widget(conn, squad_ids: list, bank: int, free_transfers: int) -> None:
    """"4-Gameweek Horizon Planner" widget: projected Starting XI points for GW_n..GW_{n+3},
    plus the step-by-step Hold/Transfer roadmap from optimizer.solve_horizon_transfers."""
    st.subheader("4-gameweek horizon planner")

    horizon_gws = 4
    event_ids = transfer_planner.get_horizon_event_ids(conn, horizon_gws)
    if not event_ids:
        st.info("No upcoming gameweeks found -- sync live data in the sidebar.")
        st.divider()
        return

    per_gw = transfer_planner.team_xp_by_gameweek(
        conn, squad_ids, event_ids=event_ids, ensemble_weights=_ensemble_weights(), min_starter_xmins=_min_starter_xmins()
    )
    fig = go.Figure(go.Bar(x=[f"GW{eid}" for eid in event_ids], y=[per_gw.get(eid, 0.0) for eid in event_ids]))
    fig.update_layout(title="Projected Starting XI points", height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    try:
        with st.spinner("Solving transfer roadmap..."):
            steps = optimizer.solve_horizon_transfers(
                conn, squad_ids, bank=bank, free_transfers=free_transfers, horizon_gws=horizon_gws,
                ensemble_weights=_ensemble_weights(), min_starter_xmins=_min_starter_xmins(),
            )
    except OptimizationError as exc:
        st.error(str(exc))
        st.divider()
        return

    for step in steps:
        if step.action == "hold":
            st.caption(step.summary)
        elif step.action == "initial_selection":
            st.info(step.summary)
        else:
            st.write(step.summary)

    st.divider()


# --- Glassmorphism AI action cards (Captain / Transfer) --------------------------

def _captain_card_html(captain_info: dict, vice_captain: Optional[dict]) -> str:
    captain = captain_info["captain"]
    safe = captain_info.get("safe_pick")
    diff = captain_info.get("differential_pick")

    vice_row = ""
    if vice_captain:
        vice_row = f"""<div class="captain-row">
<div class="captain-badge captain-badge-static badge-v">V</div>
<div><div class="captain-name">{html.escape(vice_captain['player'].web_name)}</div>
<div class="captain-sub">Vice-Captain</div></div>
</div>"""

    pick_rows = ""
    if safe:
        pick_rows += (
            '<div class="captain-pick-row"><span>Safe pick</span><span>'
            f"{html.escape(safe['player'].web_name)} · score {safe['captain_score']:.2f} · "
            f"{safe['player'].selected_by_percent:.1f}% owned</span></div>"
        )
    if diff:
        pick_rows += (
            '<div class="captain-pick-row"><span>Differential</span><span>'
            f"{html.escape(diff['player'].web_name)} · score {diff['captain_score']:.2f} · "
            f"{diff['player'].selected_by_percent:.1f}% owned</span></div>"
        )
    else:
        pick_rows += '<div class="captain-pick-row"><span>Differential</span><span>None in this squad</span></div>'

    return f"""<div class="glass-card">
<h4>\U0001F451 Captain Recommendation</h4>
<div class="captain-row">
<div class="captain-badge captain-badge-static badge-c">C</div>
<div><div class="captain-name">{html.escape(captain['player'].web_name)}</div>
<div class="captain-sub">Projected {captain['player'].projected_xp:.1f} xP &times;2 armband &rarr; <strong>{captain['player'].projected_xp * 2:.1f} xP</strong></div></div>
</div>
{vice_row}
{pick_rows}
</div>"""


def _transfer_card_html(suggestions: list) -> str:
    if not suggestions:
        return (
            '<div class="glass-card"><h4>\U0001F501 Suggested Transfers (Next GW)</h4>'
            '<p style="color:var(--text-secondary);font-size:12.5px;margin:0;">'
            "No clear upgrades found -- your squad already looks strong for the next 3 GWs.</p></div>"
        )
    rows = "".join(
        '<div class="transfer-pill-row">'
        f'<span class="transfer-name-out">{html.escape(s["out"].web_name)} ({html.escape(s["out"].position)})</span>'
        '<span class="transfer-arrow">&#10132;</span>'
        f'<span class="transfer-name-in">{html.escape(s["in"].web_name)}</span>'
        f'<span class="transfer-gain-badge">+{s["xp_gain_3gw"]:.1f} xP</span>'
        f'<span class="transfer-meta">Cost delta £{s["cost_delta"] / 10:+.1f}m over 3 GWs</span>'
        "</div>"
        for s in suggestions
    )
    return f'<div class="glass-card"><h4>\U0001F501 Suggested Transfers (Next GW)</h4>{rows}</div>'


# --- Daily Price Change Sentinel: Command Center alert pill ----------------------

def _price_alert_pill_html(alerts: list) -> str:
    """One pill per at-risk squad asset (see fpl_api.compute_price_change_alerts) -- rise
    likelihood in green ('lock in team value'), drop risk in red ('execute the transfer before
    the ~1:30am UK price-change window'). Not a guaranteed predictor even with FPL's own
    price_change_percent factored in -- see compute_price_change_alerts' own docstring."""
    if not alerts:
        return ""
    pills = []
    for a in alerts:
        if a["direction"] == "rise":
            pills.append(
                '<span class="price-alert-pill price-alert-rise">'
                f'\U0001F4C8 {html.escape(a["web_name"])} -- High Rise Likelihood (lock in team value)</span>'
            )
        else:
            pills.append(
                '<span class="price-alert-pill price-alert-drop">'
                f'\U0001F4C9 {html.escape(a["web_name"])} -- Imminent Price Drop Risk '
                "(execute transfer before ~1:30 AM)</span>"
            )
    return f'<div class="price-alert-wrap">{"".join(pills)}</div>'


def _render_price_alert_pill(conn, squad_ids: list) -> None:
    alerts = fpl_api.compute_price_change_alerts(conn, squad_ids=squad_ids)
    if alerts:
        st.markdown(_price_alert_pill_html(alerts), unsafe_allow_html=True)


# --- Tactical Rationale & Strategy card (Squad Optimizer / Transfer Planner) ----

_RATIONALE_SQUAD_SECTIONS = (
    ("captaincy", "\U0001F451 Captaincy Logic"),
    ("tactical_theme", "\U0001F9E9 Tactical Theme & Spending"),
    ("key_players", "⭐ Key Player Highlights"),
    ("bench_strategy", "\U0001FA91 Bench & Enabler Strategy"),
)


def _rationale_bullet_html(bullet) -> str:
    tags_html = ""
    if bullet.tags:
        tag_spans = "".join(f'<span class="rationale-tag">{html.escape(tag)}</span>' for tag in bullet.tags)
        tags_html = f'<div class="rationale-tag-row">{tag_spans}</div>'
    return f'<div class="rationale-bullet">{html.escape(bullet.text)}{tags_html}</div>'


def _rationale_card_html(title: str, body: str) -> str:
    if not body:
        body = '<p class="rationale-empty">No rationale available.</p>'
    return f'<div class="rationale-card"><p class="rationale-card-title">{title}</p>{body}</div>'


def _squad_rationale_card_html(rationale: dict) -> str:
    body = ""
    for key, section_title in _RATIONALE_SQUAD_SECTIONS:
        bullets = rationale.get(key) or []
        if not bullets:
            continue
        body += f'<div class="rationale-section-title">{html.escape(section_title)}</div>'
        body += "".join(_rationale_bullet_html(b) for b in bullets)
    return _rationale_card_html("\U0001F9E0 Tactical Rationale &amp; Strategy", body)


def _transfer_rationale_card_html(rationale: list) -> str:
    if not rationale:
        body = '<p class="rationale-empty">No transfers in this roadmap -- generate one above to see the rationale.</p>'
        return _rationale_card_html("\U0001F9E0 Tactical Rationale &amp; Strategy", body)

    body = ""
    for step in rationale:
        body += f'<div class="rationale-section-title">GW{step["event_id"]}</div>'
        if not step["transfers"]:
            body += _rationale_bullet_html(step["hit_roll_bullet"])
            continue
        for t in step["transfers"]:
            body += (
                '<div class="rationale-transfer-pair">'
                f'<span class="out">{html.escape(t["out"].web_name)}</span> &#10132; '
                f'<span class="in">{html.escape(t["in"].web_name)}</span></div>'
            )
            body += "".join(_rationale_bullet_html(b) for b in t["bullets"])
        body += _rationale_bullet_html(step["hit_roll_bullet"])
    return _rationale_card_html("\U0001F9E0 Tactical Rationale &amp; Strategy", body)


# --- 5-GW fixture ticker (full squad) --------------------------------------------

def _squad_fixture_ticker_data(conn, squad_rows: list, event_ids: list) -> dict:
    """team_id -> {event_id: [(opponent_short, is_home, difficulty), ...]} across event_ids,
    restricted to the given squad's teams. A list of legs per (team_id, event_id) handles
    blank (empty list, via .get() at render time) and double (2+ legs) gameweeks alike."""
    if not event_ids:
        return {}
    team_ids = {p.team_id for p in squad_rows}
    placeholders = ",".join(["?"] * len(event_ids))
    rows = conn.execute(
        f"""
        SELECT f.event, f.team_h, f.team_a, f.team_h_difficulty, f.team_a_difficulty,
               th.short_name AS home_short, ta.short_name AS away_short
        FROM fixtures f
        JOIN teams th ON th.id = f.team_h
        JOIN teams ta ON ta.id = f.team_a
        WHERE f.event IN ({placeholders})
        """,
        event_ids,
    ).fetchall()
    data: dict = {}
    for row in rows:
        if row["team_h"] in team_ids:
            data.setdefault(row["team_h"], {}).setdefault(row["event"], []).append(
                (row["away_short"], True, row["team_h_difficulty"])
            )
        if row["team_a"] in team_ids:
            data.setdefault(row["team_a"], {}).setdefault(row["event"], []).append(
                (row["home_short"], False, row["team_a_difficulty"])
            )
    return data


def _ticker_cell_html(legs: list) -> str:
    if not legs:
        pill_bg, pill_text = _difficulty_color(None)
        return f'<span class="ticker-pill" style="background:{pill_bg};color:{pill_text};">BGW</span>'
    pills = []
    for opponent, is_home, difficulty in legs:
        pill_bg, pill_text = _difficulty_color(difficulty)
        venue = "H" if is_home else "A"
        pills.append(
            f'<span class="ticker-pill" style="background:{pill_bg};color:{pill_text};">'
            f"{html.escape(opponent)} ({venue})</span>"
        )
    return "".join(pills)


def _fixture_ticker_html(squad_rows: list, event_ids: list, ticker_data: dict) -> str:
    """A color-coded FDR strip for every squad member (not just the Starting XI) across
    `event_ids`, sorted GKP->DEF->MID->FWD then name."""
    header_cells = "".join(f"<th>GW{eid}</th>" for eid in event_ids)
    body_rows = []
    for p in sorted(squad_rows, key=lambda p: (p.element_type, p.web_name)):
        team_fixtures = ticker_data.get(p.team_id, {})
        cells = "".join(
            f'<td class="ticker-cell">{_ticker_cell_html(team_fixtures.get(eid, []))}</td>' for eid in event_ids
        )
        body_rows.append(
            '<tr><td class="ticker-player-cell">'
            f'<span class="ticker-player-name">{html.escape(p.web_name)}</span>'
            f'<span class="ticker-player-pos">{html.escape(p.position)}</span></td>{cells}</tr>'
        )
    return (
        '<div class="ticker-wrap"><table class="ticker-table"><thead><tr><th></th>'
        f'{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'
    )


def render_command_center_tab(conn):
    st.header("Manager Command Center")

    if "squad_ids" not in st.session_state:
        st.info("Sync your manager squad, use a sample squad, or save a pre-season draft first.")
        return

    all_players = {p.id: p for p in optimizer.fetch_players(conn, ensemble_weights=_ensemble_weights())}
    squad_ids = st.session_state.squad_ids
    squad_rows = [all_players[pid] for pid in squad_ids if pid in all_players]
    if len(squad_rows) < 11:
        st.error("Not enough squad players resolved locally to build a starting XI.")
        return

    _render_price_alert_pill(conn, squad_ids)

    strategy_label = st.segmented_control(
        "Strategy mode", list(STRATEGY_MODE_OPTIONS.keys()), default="Balanced",
        key="strategy_mode_label", required=True,
        help="Shield Lead benchmarks against the highest-ownership squad (protect a lead by mirroring "
             "the field); Differential Chase benchmarks against a low-ownership squad (climb ranks by "
             "diverging from it). Changes which 'optimal' squad the comparisons below use.",
    )
    strategy_mode = STRATEGY_MODE_OPTIONS.get(strategy_label, "balanced")

    try:
        starting_xi, bench, formation, floor_used, was_relaxed = optimizer.solve_starting_xi_with_fallback(
            squad_rows, min_starter_xmins=_min_starter_xmins(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
        )
        captain_info = optimizer.get_captain_recommendations(
            conn, [p.id for p in squad_rows], ensemble_weights=_ensemble_weights(),
            min_starter_xmins=floor_used, risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
        )
    except OptimizationError as exc:
        st.error(str(exc))
        return

    if was_relaxed:
        st.warning(_starter_floor_relaxed_message(floor_used))

    _render_points_projection_panel(conn, squad_ids, starting_xi, captain_info["captain"]["player"], strategy_mode)

    bank = st.session_state.get("bank", 0)
    free_transfers = st.session_state.get("free_transfers", 1)
    _render_horizon_planner_widget(conn, squad_ids, bank, free_transfers)

    user_xi_xp = sum(p.projected_xp for p in starting_xi)
    optimization_pct = None
    try:
        optimal_squad = optimizer.build_optimal_squad(
            conn, mode=strategy_mode, ensemble_weights=_ensemble_weights(),
            locked_ids=_squad_locks(), excluded_ids=_squad_blacklist(), risk_lambda=_risk_lambda(),
        )
        optimal_xi, _optimal_bench, _optimal_formation, _floor, _relaxed = optimizer.solve_starting_xi_with_fallback(
            optimal_squad, min_starter_xmins=_min_starter_xmins(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
        )
        optimal_xi_xp = sum(p.projected_xp for p in optimal_xi)
        if optimal_xi_xp > 0:
            optimization_pct = user_xi_xp / optimal_xi_xp * 100
    except OptimizationError:
        pass

    squad_cost = sum(p.now_cost for p in squad_rows)
    event_id = _current_or_next_event_id(conn)
    deadline_str = _next_deadline_countdown(conn, event_id)

    with st.container(horizontal=True):
        st.metric(
            "Optimization score", f"{optimization_pct:.0f}%" if optimization_pct is not None else "N/A",
            help=f"Your Starting XI's projected xP as a % of a freshly ILP-optimized ({strategy_label}) squad's Starting XI.",
            border=True,
        )
        st.metric("Squad cost / Bank", f"£{squad_cost / 10:.1f}m / £{bank / 10:.1f}m", border=True)
        st.metric("Next deadline", deadline_str or "Unknown", border=True)

    st.divider()

    # Computed here (before the column split) so both the pitch column's 1-Click Deadline Sheet
    # and the AI action panel's transfer card can share one evaluation instead of solving twice.
    pre_gw1 = optimizer.is_before_gw1_deadline(conn)
    transfer_suggestions: list = []
    if not pre_gw1:
        bank_for_suggestions = max(bank, 0)
        try:
            with st.spinner("Evaluating transfer options..."):
                transfer_suggestions = _suggest_transfers_next_gw(
                    conn, squad_ids, bank_for_suggestions, horizon_gws=3, top_n=2
                )
                _annotate_would_start_after_swap(
                    all_players, squad_rows, transfer_suggestions,
                    _min_starter_xmins(), _risk_lambda(), _formation_lock(),
                )
        except OptimizationError as exc:
            st.error(str(exc))
            transfer_suggestions = []

    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Starting XI ({formation})")
        captain_id = captain_info["captain"]["player"].id
        vice_captain = captain_info.get("vice_captain")
        vice_id = vice_captain["player"].id if vice_captain else None
        opponent_map = _team_opponent_labels(conn, event_id)
        # Suggested Transfers-Out Highlight: transfer_suggestions is keyed by outgoing player id
        # so render_pitch_view can mark the exact starting-XI (or bench) card(s) it names --
        # "the optimal 11 you have right now, with the tool's own suggested changes highlighted
        # on it" rather than the suggestions living only in the separate text sheet below.
        transfer_out_map = {s["out"].id: s for s in transfer_suggestions}
        render_pitch_view(conn, starting_xi, bench, captain_id, vice_id, opponent_map, metric="xp", transfer_out_map=transfer_out_map)

        transfers_summary = "; ".join(
            f"OUT {s['out'].web_name} → IN {s['in'].web_name} (+{s['xp_gain']:.1f} xP)" for s in transfer_suggestions
        ) or None
        with st.expander("\U0001F4CB 1-Click Deadline Sheet", expanded=False):
            st.code(
                _format_deadline_sheet_text(
                    event_id, starting_xi, bench, captain_info["captain"]["player"],
                    vice_captain["player"] if vice_captain else None, formation,
                    bank / 10, free_transfers, transfers_summary,
                ),
                language="text",
            )

    with right:
        st.subheader("AI action panel")

        st.markdown(_captain_card_html(captain_info, vice_captain), unsafe_allow_html=True)

        if pre_gw1:
            # Real FPL rule: squad changes are free/unlimited before the GW1 deadline, so
            # "transfer suggestions" (cost delta, hit accounting) aren't a meaningful concept
            # yet -- see transfer_planner.plan_transfers' initial_selection handling. Point the
            # user at the free-building tools instead of pretending a transfer decision exists.
            st.markdown(
                '<div class="glass-card"><h4>\U0001F195 Pre-Season Squad Building</h4>'
                '<p style="color:var(--text-secondary);font-size:12.5px;margin:0;">'
                "Squad changes are free and unlimited until the GW1 deadline -- there's no "
                "transfer cost or hit to weigh yet. Use the Squad Optimizer or the pitch view "
                "above to keep refining your 15 for free; transfer suggestions with real "
                "free-transfer/hit accounting begin from GW2.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(_transfer_card_html(transfer_suggestions), unsafe_allow_html=True)

        with st.expander("\U0001FA91 Optimal bench order", expanded=True):
            is_optimal, user_order, ilp_order = _bench_order_check(squad_ids, starting_xi, bench)
            if is_optimal:
                st.success("Your bench is already in auto-sub-optimal order.")
            else:
                id_to_name = {p.id: p.web_name for p in squad_rows}
                st.warning(_bench_order_message(user_order, ilp_order, id_to_name))
            for i, p in enumerate(bench, start=1):
                st.caption(f"Sub {i}: {p.web_name} ({p.position})")

    if transfer_suggestions:
        st.divider()
        st.subheader("\U0001F52E Projected Starting XI After Suggested Transfers")
        st.caption(
            "Your Starting XI if you actually made every suggestion above -- rebuilt from scratch "
            "as one 15-man squad, not swap-by-swap, since a suggested-out player who's currently "
            "benched doesn't free a STARTING slot on their own. New signings are marked green; "
            "whoever they actually displace from today's XI is named below, which isn't always "
            "the specific player being sold."
        )
        post_squad = _post_transfer_squad(all_players, squad_rows, transfer_suggestions)
        if len(post_squad) != len(squad_rows):
            st.warning("Couldn't resolve every suggested player in the current pool -- projected XI unavailable.")
        else:
            try:
                new_xi, new_bench, new_formation, new_floor_used, new_was_relaxed = optimizer.solve_starting_xi_with_fallback(
                    post_squad, min_starter_xmins=_min_starter_xmins(), risk_lambda=_risk_lambda(),
                    formation_lock=_formation_lock(),
                )
                new_captain_info = optimizer.get_captain_recommendations(
                    conn, [p.id for p in post_squad], ensemble_weights=_ensemble_weights(),
                    min_starter_xmins=new_floor_used, risk_lambda=_risk_lambda(),
                    formation_lock=_formation_lock(),
                )
            except OptimizationError as exc:
                st.error(f"Couldn't build the projected XI: {exc}")
            else:
                if new_was_relaxed:
                    st.warning(_starter_floor_relaxed_message(new_floor_used))
                old_xi_ids = {p.id for p in starting_xi}
                new_xi_ids = {p.id for p in new_xi}
                dropped_from_xi = [p.web_name for p in starting_xi if p.id not in new_xi_ids]
                entered_xi = [p.web_name for p in new_xi if p.id not in old_xi_ids]
                if dropped_from_xi or entered_xi:
                    st.info(
                        f"**Changes to your Starting XI:** OUT of the 11 -- "
                        f"{', '.join(dropped_from_xi) or 'no one'}. IN to the 11 -- "
                        f"{', '.join(entered_xi) or 'no one'}."
                    )
                else:
                    st.caption("Making these transfers wouldn't actually change your Starting XI's line-up.")

                new_captain = new_captain_info["captain"]["player"]
                new_vice_info = new_captain_info.get("vice_captain")
                if new_captain.id != captain_id:
                    st.caption(f"Captain would also change to **{new_captain.web_name}**.")

                st.markdown(f"**Formation: {new_formation}**")
                new_signing_ids = {s["in"].id for s in transfer_suggestions}
                render_pitch_view(
                    conn, new_xi, new_bench, new_captain.id,
                    new_vice_info["player"].id if new_vice_info else None,
                    opponent_map, metric="xp", new_player_ids=new_signing_ids, key_prefix="proj_",
                )

    st.divider()
    st.subheader("5-GW Fixture Ticker")
    ticker_event_ids = transfer_planner.get_horizon_event_ids(conn, 5)
    if not ticker_event_ids:
        st.info("No upcoming gameweeks found -- sync live data in the sidebar.")
    else:
        ticker_data = _squad_fixture_ticker_data(conn, squad_rows, ticker_event_ids)
        st.markdown(_fixture_ticker_html(squad_rows, ticker_event_ids, ticker_data), unsafe_allow_html=True)


# --- Shared pitch view (used by Tab 1 and Tab 3) --------------------------------
# A styled HTML pitch (turf gradient + markings) with floating player cards, plus a bench strip
# below it -- see assets/style.css for the .pitch-*/.player-card/.bench-* classes. Each card
# lives in its own st.container(key=...) alongside an invisible, same-size st.button (see the
# CSS overlay rules targeting [class*="st-key-pcard_"]/[class*="st-key-bcard_"]): the visual is
# raw HTML, but the actual click is a real native Streamlit widget, so clicking a card opens the
# Player Analysis dialog like any other Streamlit interaction. Player/club names are escaped
# since they're interpolated into raw HTML rendered via unsafe_allow_html.

_PITCH_ROW_ORDER = (4, 3, 2, 1)  # FWD, MID, DEF, GKP -- top to bottom, standard broadcast/tactical orientation

STATUS_LABELS = {
    "a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended", "u": "Unavailable", "n": "Not available",
}


def _transfer_suggestion_note_html(transfer_suggestion: dict) -> str:
    """Shared footer note for a Suggested Transfer-Out card (see _player_card_html/
    _bench_card_html): names the incoming replacement, the projected gain, and -- when
    _annotate_would_start_after_swap has resolved it -- whether that incoming player would
    actually make the rebuilt Starting XI (in_would_start) or just take over this same bench
    slot (in_would_start is False) rather than leaving that ambiguous, which is exactly what
    made a real "shouldn't the incoming player be a starter?" question hard to answer at a
    glance from the pitch view alone."""
    in_name = html.escape(transfer_suggestion["in"].web_name)
    xp_gain = transfer_suggestion["xp_gain"]
    would_start = transfer_suggestion.get("in_would_start")
    if would_start is True:
        start_note = ' <span class="in-would-start">&middot; would START</span>'
    elif would_start is False:
        start_note = ' <span class="in-would-bench">&middot; would stay benched</span>'
    else:
        start_note = ""
    return (
        f'<div class="transfer-suggestion-note">&#8644; <span class="in-name">{in_name}</span> '
        f"(+{xp_gain:.1f} xP){start_note}</div>"
    )


def _player_metric_label(p, metric: str) -> str:
    if metric == "ownership":
        return f"Own {p.selected_by_percent:.1f}%"
    return f"xP {p.projected_xp:.1f}"


def _player_card_html(
    p, is_captain: bool, is_vice: bool, opponent_map: dict, metric: str, transfer_suggestion: Optional[dict] = None,
    is_new_signing: bool = False,
) -> str:
    # Dedicated Captaincy Callout: the badge alone (existing gold/silver C/V) marks WHO has the
    # armband, but doesn't say what it's worth -- this adds the actual doubled points share for
    # the captain, and a "steps in as captain" note for the vice, directly on the pitch card.
    captaincy_row = ""
    if is_captain:
        badge_html = '<div class="captain-badge badge-c" title="Captain">C</div>'
        captaincy_row = (
            '<div class="captaincy-share-row" title="Captaincy doubles this player\'s points">'
            f"2&times; &rarr; {p.projected_xp * 2:.1f} xP</div>"
        )
    elif is_vice:
        badge_html = '<div class="captain-badge badge-v" title="Vice-Captain">V</div>'
        captaincy_row = (
            '<div class="captaincy-share-row captaincy-share-vice" '
            'title="Takes the 2x armband if the captain is withdrawn or blanks on minutes">'
            "Backup 2&times;</div>"
        )
    else:
        badge_html = ""

    # Suggested Transfer-Out Highlight: transfer_suggestion (see render_command_center_tab's
    # transfer_out_map, built from _suggest_transfers_next_gw) marks a starter the tool thinks is
    # worth transferring out this gameweek -- a dashed red border/badge on the card itself (same
    # class carries into the click-through Player Analysis dialog's context, none needed there
    # since the card is self-explanatory) plus a small footer line naming the suggested incoming
    # replacement and the projected gain, so the "who" and "who for" are both visible without
    # leaving the pitch view for the separate 1-Click Deadline Sheet text.
    card_class = "player-card"
    suggestion_note = ""
    if transfer_suggestion is not None:
        card_class += " transfer-out-suggested"
        badge_html += '<div class="transfer-out-badge" title="Suggested transfer-out this gameweek">&#8644;</div>'
        suggestion_note = _transfer_suggestion_note_html(transfer_suggestion)
    if is_new_signing:
        card_class += " new-signing"
        badge_html += '<div class="new-signing-badge" title="New signing from the suggested transfers">NEW</div>'

    opponent = opponent_map.get(p.team_id, "BGW")
    pill_bg, pill_text = _difficulty_color(p.fixture_difficulty if p.has_fixture else None)

    return f"""<div class="{card_class}">
{badge_html}
<span class="player-pos-tag">{html.escape(p.position)}</span>
<div class="player-name" title="{html.escape(p.web_name)}">{html.escape(p.web_name)}</div>
<div class="player-club">{html.escape(p.team_name)}</div>
<span class="fixture-pill" style="background:{pill_bg};color:{pill_text};">{html.escape(opponent)}</span>
<div class="player-stats-row">
<span class="stat-badge">£{p.cost_millions:.1f}m</span>
<span class="stat-badge xp-badge">{html.escape(_player_metric_label(p, metric))}</span>
</div>
{captaincy_row}
{suggestion_note}
</div>"""


def _bench_slot_labels(bench: list) -> list:
    """The Two-Stage Bench Allocation's Step 2 labels, in exact display order (see
    optimizer.order_bench): 'Sub GKP' for the backup goalkeeper (always first), then 'Sub 1'
    (highest projected_xp among the outfield bench), 'Sub 2', 'Sub 3' (lowest -- the pure
    budget enabler) -- a strict descending-xP sort, not a minutes-security ranking."""
    labels, sub_n = [], 0
    for p in bench:
        if p.element_type == 1:
            labels.append("Sub GKP")
        else:
            sub_n += 1
            labels.append(f"Sub {sub_n}")
    return labels


def _format_deadline_sheet_text(
    gw,
    starting_xi: list,
    bench: list,
    captain,
    vice,
    formation: str,
    bank_millions: float,
    banked_ft,
    transfers_summary: Optional[str] = None,
) -> str:
    """1-Click Deadline Sheet: a plain-text pre-deadline reference block -- captain/vice with
    club names, the Starting XI grouped GK/DEF/MID/FWD, the bench in its Two-Stage Bench
    Allocation order (Sub GKP first, Sub 1 flagged as Priority Cover, see order_bench) with club
    names, and the planned transfer action (or an explicit roll fallback) plus bank/banked-FT
    state -- ready to read straight off against the official FPL squad-selection screen.

    gw/banked_ft render as '?' when unknown (e.g. no gameweek resolved locally yet) rather than
    crashing on a None -- everything else here is assumed present since callers only invoke this
    once a Starting XI/bench/captain have already been successfully solved.
    """
    position_order = [("DEF", 2), ("MID", 3), ("FWD", 4)]
    gk = next((p for p in starting_xi if p.element_type == 1), None)

    lines = [
        f"=== FPL DEADLINE SHEET (GW{gw if gw is not None else '?'}) ===",
        f"\U0001F451 Captain: {captain.web_name} ({captain.team_name})",
        f"\U0001F6E1️ Vice-Captain: {vice.web_name} ({vice.team_name})" if vice else "\U0001F6E1️ Vice-Captain: --",
        "",
        f"Starting XI ({formation}):",
        f"GK:  {gk.web_name} ({gk.team_name})" if gk else "GK:  --",
    ]
    for label, element_type in position_order:
        names = [f"{p.web_name} ({p.team_name})" for p in starting_xi if p.element_type == element_type]
        lines.append(f"{label}: {', '.join(names)}")

    lines.append("")
    lines.append("Bench:")
    sub_n = 0
    for p in bench:
        if p.element_type == 1:
            lines.append(f"GKP: {p.web_name} ({p.team_name})")
        else:
            sub_n += 1
            priority_tag = " [Priority Cover]" if sub_n == 1 else ""
            lines.append(f"{sub_n}.   {p.web_name} ({p.team_name}){priority_tag}")

    lines.append("")
    lines.append(f"Transfers Planned: {transfers_summary or 'None (Roll FT)'}")
    lines.append(f"Bank: £{bank_millions:.1f}m | Banked FTs: {banked_ft if banked_ft is not None else '?'}")
    lines.append("===================================")
    return "\n".join(lines)


def _bench_card_html(
    p, slot_label: str, metric: str, opponent_map: dict, highlight_highest_xp: bool = False,
    transfer_suggestion: Optional[dict] = None, is_new_signing: bool = False,
) -> str:
    """highlight_highest_xp marks Sub 1 -- the Two-Stage Bench Allocation's Step 2 always assigns
    it to the outfield bench player with the single highest projected_xp (see optimizer.order_bench).
    transfer_suggestion mirrors _player_card_html's own -- a bench enabler can be a suggested
    transfer-out too, not just a starter. is_new_signing mirrors its own "NEW" highlight too."""
    opponent = opponent_map.get(p.team_id, "BGW")
    pill_bg, pill_text = _difficulty_color(p.fixture_difficulty if p.has_fixture else None)
    slot_tag_html = html.escape(slot_label)
    slot_title = slot_label
    if highlight_highest_xp:
        slot_tag_html += ' <span class="bench-slot-hint">(Highest xP)</span>'
        slot_title = f"{slot_label} (Highest xP)"

    card_class = "bench-card"
    suggestion_note = ""
    new_badge = ""
    if transfer_suggestion is not None:
        card_class += " transfer-out-suggested"
        suggestion_note = _transfer_suggestion_note_html(transfer_suggestion)
    if is_new_signing:
        card_class += " new-signing"
        new_badge = '<div class="new-signing-badge" title="New signing from the suggested transfers">NEW</div>'

    return f"""<div class="{card_class}">
{new_badge}
<span class="bench-slot-tag" title="{html.escape(slot_title)}">{slot_tag_html}</span>
<div class="player-name" title="{html.escape(p.web_name)}">{html.escape(p.web_name)}</div>
<div class="player-club">{html.escape(p.position)} · £{p.cost_millions:.1f}m</div>
<span class="fixture-pill" style="background:{pill_bg};color:{pill_text};">{html.escape(opponent)}</span>
<div><span class="stat-badge xp-badge">{html.escape(_player_metric_label(p, metric))}</span></div>
{suggestion_note}
</div>"""


@st.dialog("Player Analysis")
def _player_detail_dialog(conn, p, opponent_map: dict) -> None:
    """Click-through detail view for a pitch/bench card: identity + price/ownership, minutes
    security (xMins, derived starting probability, live status/news), a personal 5-GW fixture
    ticker (reusing the same ticker the Command Center shows for the whole squad), and the
    underlying attacking/defensive rates behind the headline xP figure."""
    st.subheader(p.web_name)
    st.caption(f"{p.team_name} -- {p.position}")

    with st.container(horizontal=True):
        st.metric("Price", f"£{p.cost_millions:.1f}m", border=True)
        st.metric("Ownership", f"{p.selected_by_percent:.1f}%", border=True)
        st.metric("Projected xP", f"{p.projected_xp:.2f}", border=True)

    st.divider()
    st.markdown("**Minutes & availability**")
    starting_pct = min(100, round(p.xmins / 90 * 100))
    with st.container(horizontal=True):
        st.metric("Projected xMins", f"{p.xmins:.0f}", border=True)
        st.metric("Starting probability", f"{starting_pct}%", border=True)
        st.metric("Status", STATUS_LABELS.get(p.status, p.status), border=True)
    if p.news:
        st.caption(f"\U0001F4F0 {p.news}")
    elif p.chance_of_playing_next_round is not None:
        st.caption(f"Chance of playing next round: {p.chance_of_playing_next_round}%")

    st.divider()
    st.markdown("**Upcoming fixtures**")
    ticker_event_ids = transfer_planner.get_horizon_event_ids(conn, 5)
    if ticker_event_ids:
        ticker_data = _squad_fixture_ticker_data(conn, [p], ticker_event_ids)
        st.markdown(_fixture_ticker_html([p], ticker_event_ids, ticker_data), unsafe_allow_html=True)
    else:
        st.caption("No upcoming gameweeks found.")

    st.divider()
    st.markdown("**Underlying metrics**")
    with st.container(horizontal=True):
        st.metric("xG per 90", f"{p.xg_per_90:.2f}", border=True)
        st.metric("xA per 90", f"{p.xa_per_90:.2f}", border=True)
        if p.xp_breakdown:
            st.metric("Clean sheet prob", f"{p.xp_breakdown.cs_prob * 100:.0f}%", border=True)
            st.metric("DEFCON prob", f"{p.xp_breakdown.defcon_prob * 100:.0f}%", border=True)
    if p.xp_breakdown:
        st.caption(f"Projected bonus xP (2026/27 BPS weights): {p.xp_breakdown.bonus_xp:.2f}")


def _render_clickable_card(conn, card_html: str, p, opponent_map: dict, key_prefix: str) -> None:
    """One player/bench card: the visual HTML plus an invisible overlay button (see the CSS
    rules for [class*="st-key-{key_prefix}_"]) that opens the Player Analysis dialog on click.
    st.button has no label_visibility param -- the CSS overlay rules already make it fully
    transparent, so the label (kept non-empty for accessibility/screen readers) never shows.

    width="content" is load-bearing: st.container defaults to width="stretch", and left at that
    default, sibling cards inside a horizontal=True row fight over the row's full width with
    uneven flex-grow shares (verified live -- e.g. a 5-card row rendering at [142, 142, 142,
    220, 220]px instead of 5 even ~120px slots) even though the .player-card *inside* each one
    is still correctly 112px. Sizing each card's own container to its content is what actually
    fixes the row layout; the CSS in assets/style.css is a belt-and-braces backstop on top."""
    with st.container(key=f"{key_prefix}_{p.id}", width="content"):
        st.markdown(card_html, unsafe_allow_html=True)
        if st.button(p.web_name, key=f"{key_prefix}_btn_{p.id}"):
            _player_detail_dialog(conn, p, opponent_map)


def render_pitch_view(
    conn, starting_xi, bench, captain_id=None, vice_id=None, opponent_map=None, metric: str = "xp",
    transfer_out_map: Optional[dict] = None, new_player_ids: Optional[set] = None, key_prefix: str = "",
):
    """Reusable pitch + bench display: a turf-gradient formation layout (FWD/MID/DEF/GKP rows,
    top to bottom -- standard broadcast/tactical orientation, attacking end up) of floating,
    clickable player cards with glowing gold (C) / silver (V) captain badges, FDR-colored
    fixture pills, and price/xP stat badges, followed by a horizontal bench strip in auto-sub
    order (GKP, Sub 1-3), each sub card carrying the same fixture pill, xP badge, and click-through
    detail dialog as the starting pitch cards. Used by the Command Center, My Squad tab, and
    Squad Optimizer tab.

    transfer_out_map (player_id -> {"in": PlayerRow, "xp_gain": float}) optionally highlights
    Suggested Transfers-Out directly on the pitch/bench cards (dashed red border + badge + the
    suggested incoming replacement's name) -- see render_command_center_tab, the only current
    caller that passes one; every other caller leaves it None and gets the plain card, unchanged.

    new_player_ids optionally highlights a set of player ids as a "NEW" green signing -- used by
    render_command_center_tab's own "Projected Starting XI After Suggested Transfers" section,
    the SECOND render_pitch_view call on that page, to mark exactly who the suggested transfers
    actually brought into the XI/bench. key_prefix disambiguates that second call's Streamlit
    widget keys from the first (both render on the same page; Streamlit requires unique keys)."""
    opponent_map = opponent_map or {}
    transfer_out_map = transfer_out_map or {}
    new_player_ids = new_player_ids or set()
    rows: dict = {1: [], 2: [], 3: [], 4: []}
    for p in starting_xi:
        rows[p.element_type].append(p)

    with st.container(key=f"{key_prefix}pitch_wrap"):
        st.markdown(
            '<div class="pitch-markings"><div class="pitch-box-top"></div><div class="pitch-halfway"></div>'
            '<div class="pitch-circle"></div><div class="pitch-box-bottom"></div></div>',
            unsafe_allow_html=True,
        )
        for element_type in _PITCH_ROW_ORDER:
            players_in_row = rows[element_type]
            if not players_in_row:
                continue
            # horizontal_alignment intentionally omitted: Streamlit sets it via an inline
            # !important justify-content that beats our own CSS (verified live -- it forced
            # space-between regardless of what we set here); assets/style.css owns
            # justify-content for this row exclusively instead.
            with st.container(key=f"{key_prefix}pitch_row_{element_type}", horizontal=True):
                for p in players_in_row:
                    card_html = _player_card_html(
                        p, p.id == captain_id, p.id == vice_id, opponent_map, metric,
                        transfer_suggestion=transfer_out_map.get(p.id), is_new_signing=p.id in new_player_ids,
                    )
                    _render_clickable_card(conn, card_html, p, opponent_map, f"{key_prefix}pcard")

    if transfer_out_map:
        st.caption("⇄ dashed red = a suggested transfer-out this gameweek (see the incoming name on the card)")
    if new_player_ids:
        st.caption("🟢 dashed green = a new signing from the suggested transfers")
    st.caption("Bench (auto-sub order) -- Sub GKP, then Sub 1/2/3 by descending projected xP")
    with st.container(key=f"{key_prefix}bench_row", horizontal=True):
        for i, (p, slot_label) in enumerate(zip(bench, _bench_slot_labels(bench))):
            card_html = _bench_card_html(
                p, slot_label, metric, opponent_map, highlight_highest_xp=(i == 1),
                transfer_suggestion=transfer_out_map.get(p.id), is_new_signing=p.id in new_player_ids,
            )
            _render_clickable_card(conn, card_html, p, opponent_map, f"{key_prefix}bcard")


# --- Tab 1: My Squad & Pitch View ----------------------------------------------

DRAFT_POSITION_SLOTS = [(1, "GKP", 2), (2, "DEF", 5), (3, "MID", 5), (4, "FWD", 3)]


def _greedy_swap_suggestions(out_players: list, in_players: list, xp_lookup: dict, owned_ids: set, bank_units: int, top_n: int) -> list:
    """Up to top_n same-position 1-for-1 swaps (out_players -> in_players), ranked by biggest
    xp_lookup gain first. Each candidate incoming/outgoing player appears in at most one
    suggestion -- a naive "best replacement per outgoing player" search can recommend the same
    incoming player for several different swaps at once, which isn't executable as a real
    transfer plan since you can only own each player once.

    Also enforces MAX_PLAYERS_PER_TEAM across the whole accepted BATCH, not just each swap
    checked in isolation against the original squad -- real bug found live: two independently-
    legal suggestions (each individually taking a club from 2 -> 3 players owned) can still add
    up to 4 of the same club if they happen to share one, which the old per-swap-only check never
    caught. out_players is taken as the full current squad (not just candidates to sell) so the
    starting club counts are real; a candidate whose incoming player would push its club over the
    limit -- given whichever earlier suggestions in this same batch were already accepted -- is
    skipped in favor of the next-best candidate instead.
    """
    candidate_pairs = []
    for out_p in out_players:
        out_xp = xp_lookup.get(out_p.id, out_p.projected_xp)
        for in_p in in_players:
            if in_p.element_type != out_p.element_type or in_p.id in owned_ids:
                continue
            cost_delta = in_p.now_cost - out_p.now_cost
            xp_gain = xp_lookup.get(in_p.id, in_p.projected_xp) - out_xp
            if xp_gain > 0 and cost_delta <= bank_units:
                candidate_pairs.append({"out": out_p, "in": in_p, "xp_gain": xp_gain, "cost_delta": cost_delta})

    candidate_pairs.sort(key=lambda s: s["xp_gain"], reverse=True)

    club_counts = Counter(p.team_id for p in out_players)
    suggestions, used_out_ids, used_in_ids = [], set(), set()
    for pair in candidate_pairs:
        out_p, in_p = pair["out"], pair["in"]
        if out_p.id in used_out_ids or in_p.id in used_in_ids:
            continue
        net_in_club_count = club_counts[in_p.team_id] + (0 if in_p.team_id == out_p.team_id else 1)
        if net_in_club_count > optimizer.MAX_PLAYERS_PER_TEAM:
            continue
        suggestions.append(pair)
        used_out_ids.add(out_p.id)
        used_in_ids.add(in_p.id)
        club_counts[out_p.team_id] -= 1
        club_counts[in_p.team_id] += 1
        if len(suggestions) >= top_n:
            break

    return suggestions


def _draft_upgrade_suggestions(conn, draft_ids: list, bank_units: int) -> list:
    """Up to 5 same-position 1-for-1 swaps from the ILP-optimal squad, ranked by biggest
    projected-xP gain first."""
    all_players = {p.id: p for p in optimizer.fetch_players(conn, ensemble_weights=_ensemble_weights())}
    draft_players = [all_players[pid] for pid in draft_ids if pid in all_players]
    optimal_squad = optimizer.build_optimal_squad(conn, mode="balanced", ensemble_weights=_ensemble_weights(), locked_ids=_squad_locks(), excluded_ids=_squad_blacklist(), risk_lambda=_risk_lambda())
    xp_lookup = {p.id: p.projected_xp for p in all_players.values()}
    return _greedy_swap_suggestions(draft_players, optimal_squad, xp_lookup, set(draft_ids), bank_units, top_n=5)


def _suggest_transfers_next_gw(conn, squad_ids: list, bank_units: int, horizon_gws: int = 3, top_n: int = 2) -> list:
    """Up to top_n same-position 1-for-1 swaps against the full player pool, ranked by
    projected xP gain summed over the next `horizon_gws` gameweeks (not just the next one) --
    a transfer that looks marginal this week can still be the right call for a good fixture
    run starting the week after. Each returned dict also carries 'xp_gain_3gw' as an alias for
    the (horizon-agnostic) 'xp_gain' key, for display clarity when horizon_gws == 3."""
    event_ids = transfer_planner.get_horizon_event_ids(conn, horizon_gws)
    if not event_ids:
        return []
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)
    xp_lookup = {pid: sum(proj["gw_xp"].get(eid, 0.0) for eid in event_ids) for pid, proj in projections.items()}

    squad_rows = [transfer_planner.player_row_for_gw(projections[pid], event_ids[0]) for pid in squad_ids if pid in projections]
    pool_rows = [transfer_planner.player_row_for_gw(proj, event_ids[0]) for proj in projections.values()]

    suggestions = _greedy_swap_suggestions(squad_rows, pool_rows, xp_lookup, set(squad_ids), bank_units, top_n=top_n)
    for s in suggestions:
        s["xp_gain_3gw"] = s["xp_gain"]
    return suggestions


def _annotate_would_start_after_swap(
    all_players: dict, squad_rows: list, suggestions: list,
    min_starter_xmins: Optional[float], risk_lambda: float, formation_lock: Optional[str],
) -> None:
    """Mutates each suggestion dict in place, adding 'in_would_start': bool -- whether the
    suggested incoming player would actually make the REBUILT Starting XI, not just outproject
    the outgoing player in isolation. _suggest_transfers_next_gw's own xp_gain ranking is a plain
    same-position 1-for-1 comparison; it says nothing about where the incoming player would
    actually SIT once they're really in the squad, which can be genuinely surprising when the
    outgoing player is a bench/low-minutes one (see a real case this caught live: a big xP gain
    over a benched player who wasn't costing you anything anyway, with the incoming player easily
    good enough to become an outright starter, not just take over that same bench slot).

    Leaves 'in_would_start' UNSET (key absent, not False) if the incoming player isn't resolvable
    in `all_players` (the current single-gameweek pool) or the rebuild is infeasible for any
    reason -- callers should treat a missing key as "unknown," never as "would not start."
    """
    for s in suggestions:
        in_player = all_players.get(s["in"].id)
        if in_player is None:
            continue
        new_squad = [p for p in squad_rows if p.id != s["out"].id] + [in_player]
        if len(new_squad) != len(squad_rows):
            continue
        try:
            xi, _bench, _formation, _floor, _relaxed = optimizer.solve_starting_xi_with_fallback(
                new_squad, min_starter_xmins=min_starter_xmins, risk_lambda=risk_lambda, formation_lock=formation_lock,
            )
        except OptimizationError:
            continue
        s["in_would_start"] = in_player.id in {p.id for p in xi}


def _post_transfer_squad(all_players: dict, squad_rows: list, suggestions: list) -> list:
    """The hypothetical 15-man squad if every suggestion in `suggestions` were actually made --
    each suggestion's outgoing player removed, its incoming player (looked up fresh in
    all_players, the current single-gameweek pool, for consistency with squad_rows -- not the
    suggestion's own PlayerRow object, which came from a different multi-gw-horizon fetch) added.
    Feeds the "Projected Starting XI After Suggested Transfers" section: rebuilding the WHOLE
    squad at once (not one swap at a time) is what makes that section correct when two
    suggestions are taken together, e.g. the real case that motivated this -- a suggested-out
    player who was already benched freeing a squad slot doesn't free a STARTING slot, so the
    incoming replacement (if good enough to start) actually displaces whichever different player
    was the current XI's weakest link, not the specific player being sold."""
    out_ids = {s["out"].id for s in suggestions}
    new_squad = [p for p in squad_rows if p.id not in out_ids]
    for s in suggestions:
        in_player = all_players.get(s["in"].id)
        if in_player is not None and in_player.id not in {p.id for p in new_squad}:
            new_squad.append(in_player)
    return new_squad


def render_draft_builder(conn):
    st.info(
        "Pre-Season Draft Mode Active (Live FPL API picks locked until GW1 deadline). "
        "Build and test your GW1 draft below."
    )

    all_players = _cached_fetch_players(conn, tuple(sorted(_ensemble_weights().items())))
    if not all_players:
        st.info("No player data cached yet -- sync live data in the sidebar.")
        return

    id_to_player = {p.id: p for p in all_players}
    players_by_position = {
        etype: sorted((p for p in all_players if p.element_type == etype), key=lambda p: -p.projected_xp)
        for etype, _, _ in DRAFT_POSITION_SLOTS
    }

    def _label(pid):
        p = id_to_player[pid]
        return f"{p.web_name} ({p.team_name}) £{p.cost_millions:.1f}m"

    if "draft_selection" not in st.session_state:
        existing_draft = database.load_local_draft(conn)
        initial = {etype: [] for etype, _, _ in DRAFT_POSITION_SLOTS}
        if existing_draft:
            for pid in existing_draft["player_ids"]:
                player = id_to_player.get(pid)
                if player:
                    initial[player.element_type].append(pid)
        st.session_state.draft_selection = initial

    selected_ids = []
    for etype, pos_name, max_count in DRAFT_POSITION_SLOTS:
        options = [p.id for p in players_by_position[etype]]
        previous = [pid for pid in st.session_state.draft_selection.get(etype, []) if pid in options]
        chosen = st.multiselect(
            f"{pos_name} (pick {max_count})",
            options=options,
            default=previous,
            format_func=_label,
            max_selections=max_count,
            key=f"draft_multiselect_{etype}",
        )
        st.session_state.draft_selection[etype] = chosen
        selected_ids.extend(chosen)

    selected_players = [id_to_player[pid] for pid in selected_ids]
    total_cost = sum(p.now_cost for p in selected_players)
    remaining_budget = optimizer.BUDGET_LIMIT - total_cost
    team_counts = Counter(p.team_id for p in selected_players)
    over_limit_teams = sorted(
        {p.team_name for p in selected_players if team_counts[p.team_id] > optimizer.MAX_PLAYERS_PER_TEAM}
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Players selected", f"{len(selected_ids)} / 15")
    c2.metric("Remaining budget", f"£{remaining_budget / 10:.1f}m")
    c3.metric("Club limit violations", len(over_limit_teams))

    is_complete = len(selected_ids) == 15
    is_within_budget = remaining_budget >= 0
    is_within_club_limits = not over_limit_teams

    if not is_complete:
        st.caption(f"Select {15 - len(selected_ids)} more player(s) to complete your draft.")
    if not is_within_budget:
        st.error(f"Over budget by £{-remaining_budget / 10:.1f}m.")
    if over_limit_teams:
        st.error(f"Too many players from: {', '.join(over_limit_teams)} (max 3 per club).")

    captain_id = vice_id = None
    if is_complete:
        cap_col, vice_col = st.columns(2)
        captain_id = cap_col.selectbox(
            "Captain", selected_ids, format_func=lambda pid: id_to_player[pid].web_name, key="draft_captain",
        )
        vice_options = [pid for pid in selected_ids if pid != captain_id]
        vice_id = vice_col.selectbox(
            "Vice-Captain", vice_options, format_func=lambda pid: id_to_player[pid].web_name, key="draft_vice",
        )

    can_save = is_complete and is_within_budget and is_within_club_limits
    save_col, suggest_col = st.columns(2)
    if save_col.button("Save Draft as My Team", disabled=not can_save, width="stretch"):
        database.save_local_draft(conn, selected_ids, remaining_budget, captain_id, vice_id)
        st.session_state.squad_ids = selected_ids
        st.session_state.bank = remaining_budget
        st.session_state.manager_synced = False
        st.session_state.draft_source = "local_draft"
        st.success("Draft saved as My Team -- now used across the Squad Optimizer, Transfer Planner, and Captain Engine.")

    if suggest_col.button("Suggest Draft Upgrades", disabled=not is_complete, width="stretch"):
        try:
            with st.spinner("Comparing against the ILP-optimal squad..."):
                st.session_state.draft_upgrade_suggestions = _draft_upgrade_suggestions(
                    conn, selected_ids, max(remaining_budget, 0)
                )
        except OptimizationError as exc:
            st.error(str(exc))

    suggestions = st.session_state.get("draft_upgrade_suggestions")
    if suggestions is not None:
        if not suggestions:
            st.success("No clear upgrades found within budget -- your draft already looks strong.")
        else:
            st.subheader("Suggested swaps")
            for s in suggestions:
                st.write(
                    f"**OUT:** {s['out'].web_name} ({s['out'].position}, xP {s['out'].projected_xp:.2f}) "
                    f"-> **IN:** {s['in'].web_name} (xP {s['in'].projected_xp:.2f}) "
                    f"-- +{s['xp_gain']:.2f} xP, cost delta £{s['cost_delta'] / 10:+.1f}m"
                )


def render_squad_tab(conn):
    st.header("My Squad & Pitch View")

    with st.expander("Pre-Season Squad Builder", expanded="squad_ids" not in st.session_state):
        render_draft_builder(conn)

    if "squad_ids" not in st.session_state:
        st.info("Enter your FPL Manager ID and click 'Sync My Squad' in the sidebar, or use a sample squad.")
        return

    all_players = {p.id: p for p in optimizer.fetch_players(conn, ensemble_weights=_ensemble_weights())}
    squad_rows = [all_players[pid] for pid in st.session_state.squad_ids if pid in all_players]
    missing = len(st.session_state.squad_ids) - len(squad_rows)
    if missing:
        st.warning(f"{missing} squad player(s) not found locally -- try 'Sync Live FPL Data' in the sidebar.")
    if len(squad_rows) < 11:
        st.error("Not enough squad players resolved locally to build a starting XI.")
        return

    try:
        starting_xi, bench, formation, floor_used, was_relaxed = optimizer.solve_starting_xi_with_fallback(
            squad_rows, min_starter_xmins=_min_starter_xmins(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
        )
        captain_info = optimizer.get_captain_recommendations(
            conn, [p.id for p in squad_rows], ensemble_weights=_ensemble_weights(),
            min_starter_xmins=floor_used, risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
        )
    except OptimizationError as exc:
        st.error(str(exc))
        return

    if was_relaxed:
        st.warning(_starter_floor_relaxed_message(floor_used))

    captain_id = captain_info["captain"]["player"].id
    vice_captain = captain_info.get("vice_captain")
    vice_id = vice_captain["player"].id if vice_captain else None

    event_id = _current_or_next_event_id(conn)
    opponent_map = _team_opponent_labels(conn, event_id)

    st.subheader(f"Formation: {formation}")
    render_pitch_view(conn, starting_xi, bench, captain_id, vice_id, opponent_map, metric="xp")

    col1, col2 = st.columns(2)
    col1.metric("Bank", f"£{st.session_state.get('bank', 0) / 10:.1f}m")
    col2.metric("Squad value", f"£{sum(p.now_cost for p in squad_rows) / 10:.1f}m")


# --- Tab 2: Horizon Transfer Planner --------------------------------------------

def render_transfer_planner_tab(conn):
    st.header("Horizon Transfer Planner")

    if "squad_ids" not in st.session_state:
        st.info("Sync your manager squad (or use a sample squad) in the sidebar first.")
        return

    horizon = st.slider("Planning horizon (GWs)", min_value=3, max_value=6, value=4)
    allow_hits = st.toggle("Allow points hits (-4 per extra transfer)", value=True)
    bank_millions = st.number_input(
        "Bank (£m)", value=st.session_state.get("bank", 0) / 10, step=0.1, format="%.1f"
    )
    freeze_gkp = st.checkbox(
        "Freeze Goalkeeper Transfers (Set-and-Forget)", value=True,
        help="Locks the squad's goalkeeper(s) out of routine transfers unless injured/suspended "
             "(chance of playing < 50%).",
    )
    transfer_hurdle_xp = st.slider(
        "Minimum Transfer Gain Hurdle (xP)", min_value=0.5, max_value=3.0, value=1.5, step=0.5,
        help="A transfer only executes when its net projected gain over holding clears this "
             "amount -- otherwise the free transfer rolls instead of churning for marginal gains.",
    )

    if st.button("Generate Roadmap"):
        bank_units = round(bank_millions * 10)
        try:
            with st.spinner("Solving transfer roadmap..."):
                st.session_state.roadmap = transfer_planner.plan_transfers(
                    conn,
                    st.session_state.squad_ids,
                    bank=bank_units,
                    free_transfers=st.session_state.free_transfers,
                    horizon_gws=horizon,
                    allow_hits=allow_hits,
                    ensemble_weights=_ensemble_weights(),
                    min_starter_xmins=_min_starter_xmins(),
                    freeze_gkp_transfers=freeze_gkp,
                    transfer_hurdle_xp=transfer_hurdle_xp,
                )
        except OptimizationError as exc:
            st.error(str(exc))

    roadmap = st.session_state.get("roadmap")
    if not roadmap:
        return

    col1, col2 = st.columns([65, 35])

    with col1:
        fig = go.Figure(
            data=[
                go.Bar(name="Net points", x=[f"GW{p.event_id}" for p in roadmap], y=[p.net_points for p in roadmap]),
                go.Bar(name="Hit cost", x=[f"GW{p.event_id}" for p in roadmap], y=[-p.hit_cost for p in roadmap]),
            ]
        )
        fig.update_layout(title="Projected points by gameweek", barmode="relative", height=320)
        st.plotly_chart(fig, use_container_width=True)

        for plan in roadmap:
            title = f"GW {plan.event_id} -- {plan.formation} -- Net {plan.net_points:.1f} pts"
            with st.expander(title):
                if plan.transfers_in:
                    st.write(f"**IN:** {', '.join(plan.transfers_in)}")
                    st.write(f"**OUT:** {', '.join(plan.transfers_out)}")
                else:
                    st.write("HOLD (bank free transfer)")

                if plan.hit_cost:
                    st.warning(f"Hit taken: -{plan.hit_cost} pts ({plan.transfers_made} transfers made, "
                               f"{plan.free_transfers_before} FT available)")
                else:
                    st.caption(f"{plan.transfers_made} transfer(s) made, {plan.free_transfers_before} FT available")

                c1, c2, c3 = st.columns(3)
                c1.metric("Bank remaining", f"£{plan.bank_remaining / 10:.1f}m")
                c2.metric("Gross points", f"{plan.gross_points:.1f}")
                c3.metric("Net points", f"{plan.net_points:.1f}")
                st.write(f"Captain: **{plan.captain.web_name}** | Vice: {plan.vice_captain.web_name}")

        total_net = sum(p.net_points for p in roadmap)
        st.success(f"Total projected net points over {len(roadmap)} GWs: {total_net:.1f}")

    with col2:
        all_players = {p.id: p for p in optimizer.fetch_players(conn, ensemble_weights=_ensemble_weights())}
        current_squad = [all_players[pid] for pid in st.session_state.squad_ids if pid in all_players]
        transfer_rationale = transfer_planner.generate_transfer_rationale(conn, roadmap, current_squad, horizon_gws=horizon)
        st.markdown(_transfer_rationale_card_html(transfer_rationale), unsafe_allow_html=True)


# --- Tab 3: Squad Optimizer & Generators ----------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch_players(_conn, ensemble_weights_key: tuple = ()):
    """ensemble_weights_key: sorted (source, weight) tuples -- Streamlit's cache keys on
    hashable arguments, and this needs to bust the cache when the sidebar's weighting slider
    moves, not just when the underlying data changes (which callers already handle via an
    explicit st.cache_data.clear() after every sync/ingest)."""
    return optimizer.fetch_players(_conn, ensemble_weights=dict(ensemble_weights_key) or None)


def render_optimizer_tab(conn):
    st.header("Squad Optimizer & Custom Generators")

    all_players = _cached_fetch_players(conn, tuple(sorted(_ensemble_weights().items())))
    if all_players and max(p.defensive_contribution_per_90 for p in all_players) == 0.0:
        st.info(
            "Pre-season active: Using historical DEFCON position baselines until live "
            "2026/27 match stats accumulate."
        )

    col1, col2, col3 = st.columns(3)
    try:
        if col1.button("Optimal £100m Squad", use_container_width=True):
            with st.spinner("Solving..."):
                st.session_state.generated_squad = ("Optimal Squad (Balanced xP)", optimizer.build_optimal_squad(conn, mode="balanced", ensemble_weights=_ensemble_weights(), locked_ids=_squad_locks(), excluded_ids=_squad_blacklist(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock()))
        if col2.button("Template Team (Highest Ownership)", use_container_width=True):
            with st.spinner("Solving..."):
                st.session_state.generated_squad = ("Template Squad (Highest Ownership)", optimizer.build_optimal_squad(conn, mode="template", ensemble_weights=_ensemble_weights(), locked_ids=_squad_locks(), excluded_ids=_squad_blacklist()))
        if col3.button("Differential Radar (<10% owned)", use_container_width=True):
            with st.spinner("Solving..."):
                st.session_state.generated_squad = ("Differential Squad (<10% Ownership)", optimizer.build_optimal_squad(conn, mode="differential", ensemble_weights=_ensemble_weights(), locked_ids=_squad_locks(), excluded_ids=_squad_blacklist()))
    except OptimizationError as exc:
        st.error(str(exc))

    if "generated_squad" in st.session_state:
        label, squad = st.session_state.generated_squad
        st.subheader(label)

        pitch_tab, table_tab = st.tabs(["Pitch View (Visual Layout)", "Data Table"])

        with table_tab:
            st.dataframe(_squad_to_dataframe(squad), use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            c1.metric("Total cost", f"£{sum(p.now_cost for p in squad) / 10:.1f}m")
            c2.metric("Total projected xP", f"{sum(p.projected_xp for p in squad):.1f}")

        with pitch_tab:
            try:
                starting_xi, bench, formation, floor_used, was_relaxed = optimizer.solve_starting_xi_with_fallback(
                    squad, min_starter_xmins=_min_starter_xmins(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
                )
                captain_info = optimizer.get_captain_recommendations(
                    conn, [p.id for p in squad], ensemble_weights=_ensemble_weights(),
                    min_starter_xmins=floor_used, risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
                )
            except OptimizationError as exc:
                st.error(f"Could not build pitch view: {exc}")
            else:
                if was_relaxed:
                    st.warning(_starter_floor_relaxed_message(floor_used))
                captain_id = captain_info["captain"]["player"].id
                vice_captain = captain_info.get("vice_captain")
                vice_id = vice_captain["player"].id if vice_captain else None
                event_id = _current_or_next_event_id(conn)
                opponent_map = _team_opponent_labels(conn, event_id)
                metric = "ownership" if label.startswith("Template") else "xp"

                col1, col2 = st.columns([65, 35])
                with col1:
                    st.caption(f"Formation: {formation}")
                    render_pitch_view(conn, starting_xi, bench, captain_id, vice_id, opponent_map, metric=metric)
                with col2:
                    rationale = optimizer.generate_squad_rationale(
                        conn, squad, starting_xi, bench,
                        captain_info["captain"]["player"],
                        vice_captain["player"] if vice_captain else None,
                    )
                    st.markdown(_squad_rationale_card_html(rationale), unsafe_allow_html=True)

                with st.expander("\U0001F4CB 1-Click Deadline Sheet", expanded=False):
                    st.code(
                        _format_deadline_sheet_text(
                            event_id, starting_xi, bench, captain_info["captain"]["player"],
                            vice_captain["player"] if vice_captain else None, formation,
                            st.session_state.get("bank", 0) / 10, st.session_state.get("free_transfers", 1),
                        ),
                        language="text",
                    )

    st.divider()
    st.subheader("Player Explorer")
    st.caption("Browse and filter the full player pool -- useful for manual scouting alongside the generators above.")

    if not all_players:
        st.info("No player data cached yet -- sync live data in the sidebar.")
        return

    positions = st.multiselect("Position", ["GKP", "DEF", "MID", "FWD"], default=["GKP", "DEF", "MID", "FWD"])
    teams_available = sorted({p.team_name for p in all_players})
    teams_selected = st.multiselect("Team", teams_available, default=teams_available)
    min_price = min(p.cost_millions for p in all_players)
    max_price = max(p.cost_millions for p in all_players)
    price_range = st.slider("Price (£m)", min_price, max_price, (min_price, max_price), step=0.5)

    filtered = [
        p for p in all_players
        if p.position in positions and p.team_name in teams_selected
        and price_range[0] <= p.cost_millions <= price_range[1]
    ]
    st.dataframe(_squad_to_dataframe(filtered), use_container_width=True, hide_index=True)


# --- Tab 4: Fixture Difficulty Matrix --------------------------------------------

def _difficulty_color(diff):
    if diff is None:
        return "#9e9e9e", "black"
    frac = max(0.0, min(1.0, (diff - 1) / 4))
    if frac <= 0.5:
        t = frac / 0.5
        r, g, b = 46 + t * (255 - 46), 125 + t * (235 - 125), 50 + t * (59 - 50)
    else:
        t = (frac - 0.5) / 0.5
        r, g, b = 255 + t * (198 - 255), 235 + t * (40 - 235), 59 + t * (40 - 59)
    text_color = "white" if frac < 0.15 or frac > 0.85 else "black"
    return f"rgb({int(r)},{int(g)},{int(b)})", text_color


def render_fixture_matrix_tab(conn):
    st.header("Fixture Difficulty Matrix")

    horizon = st.slider("Gameweeks ahead", min_value=5, max_value=8, value=6)

    row = conn.execute("SELECT id FROM gameweeks WHERE is_next = 1 ORDER BY id LIMIT 1").fetchone()
    if row is None:
        row = conn.execute("SELECT id FROM gameweeks WHERE is_current = 1 ORDER BY id LIMIT 1").fetchone()
    if row is None:
        st.info("No gameweeks cached yet -- sync live data in the sidebar.")
        return
    event_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM gameweeks WHERE id >= ? ORDER BY id LIMIT ?", (row["id"], horizon)
    ).fetchall()]

    teams = conn.execute("SELECT id, short_name FROM teams ORDER BY short_name").fetchall()
    placeholders = ",".join(["?"] * len(event_ids))
    fixture_rows = conn.execute(
        f"""
        SELECT f.event, f.team_h, f.team_a, f.team_h_difficulty, f.team_a_difficulty,
               th.short_name AS home_short, ta.short_name AS away_short
        FROM fixtures f
        JOIN teams th ON th.id = f.team_h
        JOIN teams ta ON ta.id = f.team_a
        WHERE f.event IN ({placeholders})
        """,
        event_ids,
    ).fetchall()

    by_team_event: dict = {}
    for r in fixture_rows:
        by_team_event.setdefault((r["team_h"], r["event"]), []).append(
            (r["away_short"], True, r["team_h_difficulty"] or 3)
        )
        by_team_event.setdefault((r["team_a"], r["event"]), []).append(
            (r["home_short"], False, r["team_a_difficulty"] or 3)
        )

    gw_labels = [f"GW{eid}" for eid in event_ids]
    display_rows, color_rows, text_color_rows, team_names = [], [], [], []
    for team in teams:
        team_names.append(team["short_name"])
        display_row, color_row, text_color_row = [], [], []
        for eid in event_ids:
            legs = by_team_event.get((team["id"], eid), [])
            if not legs:
                display_row.append("BGW")
                color, text_color = _difficulty_color(None)
            elif len(legs) == 1:
                opp, is_home, diff = legs[0]
                display_row.append(f'{opp} ({"H" if is_home else "A"})')
                color, text_color = _difficulty_color(diff)
            else:
                labels = [f'{opp} ({"H" if is_home else "A"})' for opp, is_home, diff in legs]
                display_row.append("DGW: " + " + ".join(labels))
                avg_diff = sum(d for _, _, d in legs) / len(legs)
                color, text_color = _difficulty_color(avg_diff)
            color_row.append(color)
            text_color_row.append(text_color)
        display_rows.append(display_row)
        color_rows.append(color_row)
        text_color_rows.append(text_color_row)

    df = pd.DataFrame(display_rows, columns=gw_labels, index=team_names)
    color_df = pd.DataFrame(color_rows, columns=gw_labels, index=team_names)
    text_color_df = pd.DataFrame(text_color_rows, columns=gw_labels, index=team_names)

    def _style(_):
        return pd.DataFrame(
            [
                [f"background-color: {color_df.loc[t, c]}; color: {text_color_df.loc[t, c]}" for c in df.columns]
                for t in df.index
            ],
            columns=df.columns, index=df.index,
        )

    st.dataframe(df.style.apply(_style, axis=None), use_container_width=True)
    st.caption("Green = easy fixture, red = hard. Grey BGW = blank gameweek. DGW rows list both fixtures.")


# --- Tab 5: Rival Radar & Mini-League Analysis ----------------------------------

def render_rival_radar_tab(conn):
    st.header("Rival Radar & Mini-League Analysis")

    if "squad_ids" not in st.session_state:
        st.info("Sync your manager squad (or use a sample squad) in the sidebar first.")
        return

    synced_league_id = st.session_state.get("selected_league_id")
    league_id_input = st.text_input(
        "FPL Mini-League ID",
        value=str(synced_league_id) if synced_league_id else "",
        help="Auto-filled from the sidebar's Active Mini-League once a team is synced -- "
             "override it here to check a different league instead." if synced_league_id else None,
    )
    num_rivals = st.slider("Rivals to compare", min_value=2, max_value=10, value=5)

    if st.button("Fetch League & Compare") and league_id_input:
        try:
            league_id = int(league_id_input)
        except ValueError:
            st.error("League ID must be a number.")
            return

        target_event = _current_or_next_event_id(conn)
        if target_event is None:
            st.error("No gameweeks found locally -- sync live data in the sidebar first.")
            return

        client = FPLClient()
        try:
            with st.spinner("Fetching mini-league standings & rival squads..."):
                minileague_squads = fpl_api.fetch_minileague_squads(
                    client, league_id, target_event, max_managers=num_rivals
                )
        except FPLAPIError as exc:
            st.error(f"Could not fetch league: {exc}")
            return

        if not minileague_squads:
            st.warning("No rival squads found for this league ID (private league, no picks set yet, or API issue).")
            return

        # Persisted so the widgets below survive Streamlit's rerun-on-any-interaction model
        # (e.g. clicking into a player detail dialog) without re-fetching the whole league.
        st.session_state.minileague_squads = minileague_squads

    minileague_squads = st.session_state.get("minileague_squads")
    if not minileague_squads:
        return

    st.subheader("Standings")
    st.dataframe(
        pd.DataFrame(
            [{"Rank": s["rank"], "Team": s["entry_name"], "Manager": s["player_name"]} for s in minileague_squads]
        ),
        use_container_width=True, hide_index=True,
    )

    players = {p.id: p for p in optimizer.fetch_players(conn, ensemble_weights=_ensemble_weights())}
    my_squad_ids = set(st.session_state.squad_ids)
    n_rivals = len(minileague_squads)

    leo = live_tracker.compute_local_effective_ownership(minileague_squads)
    classification = live_tracker.classify_shield_and_differential(minileague_squads, my_squad_ids, leo=leo)

    st.subheader("\U0001F6E1️ Mini-League LEO Breakdown")
    st.caption(
        f"Local Effective Ownership (LEO = started% + captained%, among rivals' STARTING XIs -- "
        f"bench slots don't count) across the {n_rivals} fetched rival{'s' if n_rivals != 1 else ''} "
        f"-- shields at LEO ≥ {live_tracker.SHIELD_LEO_THRESHOLD:.0f}%, differentials at "
        f"LEO ≤ {live_tracker.DIFFERENTIAL_LEO_THRESHOLD:.0f}%."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**\U0001F6E1️ Shield Assets (High Rival Ownership)** -- critical locks; missing out risks falling behind the pack.")
        shield_ids = classification["shield_owned"] + classification["shield_missing"]
        if shield_ids:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Player": players[pid].web_name if pid in players else str(pid),
                            "LEO %": leo[pid],
                            "You own": "✅" if pid in classification["shield_owned"] else "—",
                        }
                        for pid in shield_ids
                    ]
                ),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No player clears the Shield LEO threshold in this league yet.")

    with col2:
        st.markdown("**⚔️ Differential Opportunities (Low Rival Ownership)** -- high-xP weapons few rivals are exposed to.")
        diff_rows = sorted(
            (
                {
                    "Player": players[pid].web_name,
                    "xP": players[pid].projected_xp,
                    "LEO %": leo.get(pid, 0.0),
                }
                for pid in classification["differential_candidates"] if pid in players
            ),
            key=lambda row: row["xP"], reverse=True,
        )[:10]
        if diff_rows:
            st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No differential candidates found under the LEO threshold.")


# --- Tab 6: Chip Strategy & Tactics ---------------------------------------------

CHIP_TIMELINE_ICONS = {"TC": "⚡", "WC": "\U0001F0CF", "BB": "\U0001FA91", "FH": "\U0001F6E1️"}
CHIP_TIMELINE_COLORS = {
    "TC": ("rgba(255, 193, 7, 0.22)", "#FFC107"),
    "WC": ("rgba(121, 40, 202, 0.28)", "#B388FF"),
    "BB": ("rgba(0, 229, 255, 0.18)", "#00E5FF"),
    "FH": ("rgba(0, 255, 135, 0.18)", "#00FF87"),
}


def _chip_timeline_html(recommendations: list, gw_start: int, gw_end: int) -> str:
    """A gw_start-gw_end cell strip (GW1-19 for Set 1, GW20-38 for Set 2); a plain cell for every
    gameweek with no chip planned, a colour-coded, wider cell (icon + chip name, hover title = the
    reasoning) for the gameweek(s) a chip is actually recommended for."""
    by_gw = {r.event_id: r for r in recommendations}
    cells = []
    for gw in range(gw_start, gw_end + 1):
        rec = by_gw.get(gw)
        if rec is None:
            cells.append(f'<div class="chip-timeline-cell"><span class="chip-timeline-gw">GW{gw}</span></div>')
            continue
        bg, border = CHIP_TIMELINE_COLORS.get(rec.chip, ("rgba(255,255,255,0.08)", "var(--border-strong)"))
        icon = CHIP_TIMELINE_ICONS.get(rec.chip, "")
        label = chip_planner.CHIP_NAMES.get(rec.chip, rec.chip)
        cells.append(
            '<div class="chip-timeline-cell chip-timeline-cell-active" '
            f'style="background:{bg};border-color:{border};" title="{html.escape(rec.reasoning)}">'
            f'<span class="chip-timeline-gw">GW{gw}</span>'
            f'<span class="chip-timeline-icon">{icon}</span>'
            f'<span class="chip-timeline-label">{html.escape(label)}</span>'
            "</div>"
        )
    return f'<div class="chip-timeline-wrap">{"".join(cells)}</div>'


def render_chip_strategy_tab(conn):
    st.header("Chip Strategy & Tactics")

    if "squad_ids" not in st.session_state:
        st.info("Sync your manager squad (or use a sample squad) in the sidebar first.")
        return

    squad_ids = st.session_state.squad_ids
    current_event = _current_or_next_event_id(conn)
    current_set = chip_planner.chip_set_for_event(current_event)
    set_key = "set1" if current_set == 1 else "set2"
    set_label = "Set 1 (GW1-19)" if current_set == 1 else "Set 2 (GW20-38)"

    # --- Active chip tracker & expiry warning ---
    st.subheader("Active Chip Tracker")

    if st.session_state.get("manager_synced") and st.session_state.get("manager_id"):
        if st.button("Refresh chip usage from FPL"):
            try:
                client = FPLClient()
                history = client.get_manager_history(int(st.session_state.manager_id))
                st.session_state.chip_usage = chip_planner.parse_chip_usage(history)
                st.success("Chip usage refreshed.")
            except (FPLAPIError, ValueError) as exc:
                st.error(f"Could not fetch chip history: {exc}")

    auto_used_this_set = st.session_state.get("chip_usage", {}).get(set_key, {})

    cols = st.columns(4)
    used_flags = {}
    for col, code in zip(cols, chip_planner.CHIP_CODES):
        state_key = f"chip_used_{set_key}_{code}"
        st.session_state.setdefault(state_key, auto_used_this_set.get(code) is not None)
        with col:
            used_flags[code] = st.checkbox(f"{chip_planner.CHIP_NAMES[code]} used", key=state_key)

    remaining = [chip_planner.CHIP_NAMES[c] for c in chip_planner.CHIP_CODES if not used_flags[c]]
    st.caption(f"Currently in **{set_label}**. Remaining this set: {', '.join(remaining) if remaining else 'none'}")

    if current_set == 1:
        expiry = chip_planner.get_set1_expiry(conn)
        days_left = (expiry - datetime.now(timezone.utc)).days
        unused = [c for c in chip_planner.CHIP_CODES if not used_flags[c]]
        if unused:
            banner = st.error if days_left <= 14 else (st.warning if days_left <= 30 else st.info)
            banner(
                f"⏳ {len(unused)} Set 1 chip(s) unused ({', '.join(chip_planner.CHIP_NAMES[c] for c in unused)}) "
                f"-- expires {expiry:%Y-%m-%d}, {max(days_left, 0)} day(s) remaining."
            )

    st.divider()

    # --- Season Strategy & Chip Roadmap (Set 1: GW1-19, Set 2: GW20-38) ---------------
    # Which set is ACTIVE (see current_set above, from the manager's real current gameweek)
    # decides which half's planner/window/session-state key this section uses -- crossing GW19/20
    # switches the whole section over rather than just leaving a stale GW1-19 view up with a "kept
    # for reference only" note, as Set 1 alone used to do before Set 2 had its own planner.
    if current_set == 1:
        roadmap_gw_start, roadmap_gw_end = 2, chip_planner.CHIP_SET_1_LAST_GW
        roadmap_window_label = f"GW{roadmap_gw_start}–{roadmap_gw_end}"
        roadmap_solver = chip_planner.solve_season_half_chip_strategy
        roadmap_spinner_text = f"Planning chip windows across {roadmap_window_label}..."
    else:
        roadmap_gw_start, roadmap_gw_end = chip_planner.SECOND_HALF_START_GW, chip_planner.CHIP_SET_2_LAST_GW
        roadmap_window_label = f"GW{roadmap_gw_start}–{roadmap_gw_end}"
        roadmap_solver = chip_planner.solve_second_half_chip_strategy
        roadmap_spinner_text = f"Planning chip windows across {roadmap_window_label} (Bench Boost first, Wildcard timed ahead of it)..."
    roadmap_state_key = f"macro_chip_roadmap_{set_key}"
    roadmap_set_name = "Set 1" if current_set == 1 else "Set 2"

    st.subheader(f"Season Strategy & Chip Roadmap ({roadmap_window_label})")
    st.caption(
        f"A full {roadmap_set_name} view spreading Wildcard, Bench Boost, Triple Captain, and Free Hit "
        f"across {roadmap_window_label} at once -- distinct from the rolling-horizon roadmap below, which "
        "only looks a few gameweeks ahead. Each chip targets its own natural window (fixture swings, "
        "squad freshness, blank/double detection) rather than being crammed into the opening gameweeks; "
        "the toggles above (which chips are already used) decide what's still in play."
    )

    macro_available_chips = [code for code in chip_planner.CHIP_CODES if not used_flags[code]]
    if not macro_available_chips:
        st.caption(f"All {roadmap_set_name} chips are marked used above -- nothing left to plan.")
    elif st.button("Build Season Roadmap", key=f"build_roadmap_{set_key}"):
        try:
            with st.spinner(roadmap_spinner_text):
                st.session_state[roadmap_state_key] = roadmap_solver(
                    conn, squad_ids, available_chips=macro_available_chips,
                )
        except OptimizationError as exc:
            st.error(str(exc))

    macro_roadmap = st.session_state.get(roadmap_state_key)
    if macro_roadmap:
        st.markdown(_chip_timeline_html(macro_roadmap, roadmap_gw_start, roadmap_gw_end), unsafe_allow_html=True)
        for rec in macro_roadmap:
            icon = CHIP_TIMELINE_ICONS.get(rec.chip, "")
            confidence_note = (
                "" if rec.data_driven
                else " *(calendar-based placeholder -- not yet confirmed by the published fixture list)*"
            )
            st.markdown(f"{icon} **GW{rec.event_id}: {chip_planner.CHIP_NAMES[rec.chip]}** -- {rec.reasoning}{confidence_note}")
        planned_chips = {r.chip for r in macro_roadmap}
        missing = [code for code in macro_available_chips if code not in planned_chips]
        if missing:
            st.caption(
                f"No confident {roadmap_window_label} window found for: "
                f"{', '.join(chip_planner.CHIP_NAMES[c] for c in missing)}."
            )

    st.divider()

    # --- Recommended chip deployment roadmap ---
    st.subheader("Recommended Chip Deployment Roadmap")
    horizon = st.slider("Roadmap horizon (GWs)", min_value=5, max_value=10, value=8, key="chip_roadmap_horizon")

    if st.button("Build Roadmap"):
        try:
            with st.spinner("Evaluating chips across the horizon..."):
                st.session_state.chip_roadmap = chip_planner.build_chip_roadmap(conn, squad_ids, horizon_gws=horizon)
        except OptimizationError as exc:
            st.error(str(exc))

    roadmap = st.session_state.get("chip_roadmap")
    if roadmap:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Chip": chip_planner.CHIP_NAMES[r.chip],
                        "Target GW": r.event_id,
                        "Projected Boost": round(r.projected_boost, 1),
                        "Justification": r.justification,
                    }
                    for r in roadmap
                ]
            ),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # --- "What-if" chip simulator ---
    st.subheader('"What-If" Chip Simulator')
    chip_choice = st.radio(
        "Chip to simulate", chip_planner.CHIP_CODES, format_func=lambda c: chip_planner.CHIP_NAMES[c], horizontal=True,
    )

    if st.button("Simulate"):
        try:
            with st.spinner("Simulating..."):
                st.session_state.chip_sim = chip_planner.simulate_chip(conn, squad_ids, chip_choice)
        except OptimizationError as exc:
            st.error(str(exc))

    sim = st.session_state.get("chip_sim")
    if sim and sim["chip"] == chip_choice:
        if sim["basis"] == "single_gw":
            label_standard = f"Standard Starting XI (GW{sim['event_id']})"
            label_chip = f"{chip_planner.CHIP_NAMES[sim['chip']]}-Boosted Score"
        else:
            label_standard = f"Current Squad (next {sim['horizon_gws']} GWs)"
            label_chip = f"Optimal Rebuilt Squad (next {sim['horizon_gws']} GWs)"
            st.caption(
                "Wildcard has no single-week scoring effect -- its value is the squad upgrade "
                "persisting across future weeks, so this compares projected totals over a short horizon."
            )

        c1, c2, c3 = st.columns(3)
        c1.metric(label_standard, f"{sim['standard_score']:.1f}")
        c2.metric(label_chip, f"{sim['chip_score']:.1f}")
        c3.metric("Net Projected Advantage", f"{sim['net_advantage']:+.1f}")


# --- Tab: Live Gameweek Radar -----------------------------------------------------

def _live_status_dataframe(status_by_id: dict, ordered_ids: list, bench_ids: list) -> pd.DataFrame:
    bench_set = set(bench_ids)
    rows = []
    for pid in ordered_ids:
        st_row = status_by_id.get(pid)
        if st_row is None:
            continue
        rows.append({
            "": st_row.status_icon,
            "Player": st_row.web_name,
            "Pos": optimizer.POSITION_NAMES.get(st_row.element_type, "?"),
            "Role": "Bench" if pid in bench_set else "Starting XI",
            "Minutes": st_row.minutes,
            "Status": st_row.status_label,
            "Live pts": st_row.base_points,
            "Bonus (prov.)": st_row.provisional_bonus if st_row.official_bonus == 0 else st_row.official_bonus,
            "Total": st_row.live_points,
        })
    return pd.DataFrame(rows)


def render_live_radar_tab(conn) -> None:
    st.header("Live Gameweek Radar")
    st.caption(
        "Real-time provisional points, in-play status, and a simulated auto-sub/captaincy "
        "preview -- built from FPL's live matchday feed, not the season-long projection engine "
        "used elsewhere in this app. Only meaningful once a gameweek's matches have kicked off."
    )

    if "squad_ids" not in st.session_state:
        st.info("Sync your manager squad (or use a sample squad) in the sidebar first.")
        return

    live_event_id = _current_live_event_id(conn)
    if live_event_id is None:
        st.info(
            "No gameweek is currently live (pre-deadline, or between gameweeks) -- check back "
            "once the next gameweek's matches have kicked off. In the meantime, Replay Mode below "
            "runs this same engine against a real, already-finished past gameweek."
        )
        st.divider()
        _render_replay_mode_expander(conn)
        return

    squad_ids = st.session_state.squad_ids
    manager_id = st.session_state.get("manager_id")
    using_real_picks = bool(st.session_state.get("manager_synced") and manager_id)

    client = FPLClient()
    captain_id = vice_id = None
    starting_xi_ids: list = []
    bench_ids: list = []

    if using_real_picks:
        # The one place in the app that tracks a manager's ACTUAL live FPL selections instead of
        # this app's own optimizer-computed "best possible" XI -- see
        # live_tracker.calculate_live_gameweek_points's docstring for why that's the right call
        # specifically for a live scorecard.
        try:
            with st.spinner("Fetching live matchday data..."):
                live = live_tracker.calculate_live_gameweek_points(conn, client, int(manager_id), live_event_id)
        except FPLAPIError as exc:
            st.info(
                f"Could not fetch your real picks for GW{live_event_id} ({exc}) -- falling back "
                "to this app's own optimal Starting XI below."
            )
            using_real_picks = False
        except ValueError:
            using_real_picks = False

    if not using_real_picks:
        try:
            squad_rows = [p for p in optimizer.fetch_players(conn, ensemble_weights=_ensemble_weights()) if p.id in squad_ids]
            starting_xi, bench, _formation, floor_used, was_relaxed = optimizer.solve_starting_xi_with_fallback(
                squad_rows, min_starter_xmins=_min_starter_xmins(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
            )
            captain_info = optimizer.get_captain_recommendations(
                conn, squad_ids, ensemble_weights=_ensemble_weights(),
                min_starter_xmins=floor_used, risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
            )
        except OptimizationError as exc:
            st.error(str(exc))
            return

        if was_relaxed:
            st.warning(_starter_floor_relaxed_message(floor_used))

        captain_id = captain_info["captain"]["player"].id
        vice_captain = captain_info.get("vice_captain")
        vice_id = vice_captain["player"].id if vice_captain else None
        starting_xi_ids = [p.id for p in starting_xi]
        bench_ids = [p.id for p in bench]
        st.caption(
            "⚠️ Showing this app's own optimal Starting XI (not your real FPL picks) -- sync a "
            "real FPL Team ID in the sidebar to track your actual live team instead."
        )

        if st.button("\U0001F504 Refresh Live Data", use_container_width=False):
            st.cache_data.clear()

        try:
            with st.spinner("Fetching live matchday data..."):
                live = live_tracker.get_live_gameweek_status(
                    conn, client, squad_ids, live_event_id, starting_xi_ids, bench_ids, captain_id, vice_id,
                )
        except FPLAPIError as exc:
            st.error(f"Could not fetch live data: {exc}")
            return

    if using_real_picks:
        # For the dataframe's Role column: the EFFECTIVE (post-auto-sub) XI/bench split, not the
        # nominal one FPL started the day with -- a subbed-in player reads as "Starting XI" here,
        # which is the more meaningful live-scorecard read.
        squad_ids = list(live["player_status"].keys())
        starting_xi_ids = list(live["effective_starting_xi_ids"])
        bench_ids = [pid for pid in squad_ids if pid not in starting_xi_ids]

    # captain_id is already None when using_real_picks (set at the top of this function) --
    # nominal_captain_id only matters for the non-real-picks vice-promotion callout wording.
    _render_live_scorecard(live, starting_xi_ids, bench_ids, captain_id)

    st.divider()
    _render_replay_mode_expander(conn)


def _render_live_scorecard(live: dict, starting_xi_ids: list, bench_ids: list, nominal_captain_id=None) -> None:
    """Shared rendering for one live_tracker.get_live_gameweek_status-shaped result -- used by
    both the real/optimizer-XI live paths above and Replay Mode below, so a real gameweek and a
    replayed historical one look and behave identically on screen (same metrics, same auto-sub/
    captain callouts, same table). nominal_captain_id is only used to phrase the vice-promotion
    callout correctly when the caller wasn't using real FPL picks (see the call site above); pass
    None to skip that distinction (Replay Mode has no "real picks" concept to contrast against)."""
    status_by_id = live["player_status"]
    id_to_name = {pid: st_row.web_name for pid, st_row in status_by_id.items()}

    # Live Matchday Scorecard: how many of the effective XI are currently on the pitch, done, or
    # still to come -- using the spec's own icon set for this one summary line (distinct from the
    # per-player 🟢/🔴/🟡 status column in the table below, which predates this scorecard).
    live_count = done_count = upcoming_count = 0
    for pid in live["effective_starting_xi_ids"]:
        st_row = status_by_id.get(pid)
        if st_row is None:
            continue
        if st_row.fixture_finished:
            done_count += 1
        elif st_row.minutes > 0:
            live_count += 1
        else:
            upcoming_count += 1

    col1, col2, col3 = st.columns(3)
    col1.metric("\U0001F3AF Provisional Total", f"{live['provisional_total_points']:.1f} pts")
    active_cap_name = id_to_name.get(live["active_captain_id"], "None")
    col2.metric("\U0001F451 Active Captain (2x)", active_cap_name, f"+{live['captain_doubled_points']:.1f} pts")
    col3.metric("\U0001F504 Auto-Subs Applied", len(live["auto_sub_moves"]))

    st.caption(f"🟢 {live_count} Live · ⚪ {done_count} Done · ⏳ {upcoming_count} Upcoming (effective Starting XI)")

    if nominal_captain_id is not None and live["active_captain_id"] not in (nominal_captain_id, None):
        st.info(f"Vice-Captain **{active_cap_name}** is active -- the captain didn't feature this gameweek.")
    elif live["active_captain_id"] is None:
        st.warning("Armband wasted this gameweek -- neither the captain nor vice-captain featured.")

    if live["auto_sub_moves"]:
        moves_text = "; ".join(
            f"{id_to_name.get(m['out'], m['out'])} → {id_to_name.get(m['in'], m['in'])}"
            for m in live["auto_sub_moves"]
        )
        st.success(f"\U0001F504 **Auto-Sub Applied:** {moves_text}")

    formation_counts = Counter(
        status_by_id[pid].element_type for pid in live["effective_starting_xi_ids"] if pid in status_by_id
    )
    st.subheader(f"Formation: {formation_counts.get(2, 0)}-{formation_counts.get(3, 0)}-{formation_counts.get(4, 0)}")
    df = _live_status_dataframe(status_by_id, starting_xi_ids + bench_ids, bench_ids)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_replay_mode_expander(conn) -> None:
    """Historical Gameweek Replay Mode: runs a real, finished past-season gameweek through the
    exact same live-tracking engine as the section above (see src.replay), against this app's own
    optimizer-computed Starting XI for the CURRENT squad -- there's no "real past picks" to use
    here, so unlike the live section this always uses the optimal-XI path. The point isn't
    predicting the past; it's proving the auto-sub/bonus/captaincy pipeline against real results
    without waiting for the current season's first live gameweek."""
    with st.expander("\U0001F501 Replay a past gameweek", expanded=False):
        st.caption(
            "Runs a real, finished gameweek from a past season through this app's own live "
            "auto-sub/bonus/captaincy engine -- a way to sanity-check that logic against real "
            "results before the current season's own first gameweek goes live."
        )
        if "squad_ids" not in st.session_state:
            st.info("Sync your manager squad (or use a sample squad) in the sidebar first.")
            return

        col1, col2 = st.columns(2)
        season = col1.text_input("Season", value=fpl_api.current_fpl_season(), help="e.g. 2025-26")
        gw = col2.number_input("Gameweek", min_value=1, max_value=38, value=1, step=1)

        if not st.button("\U0001F501 Run Replay"):
            return

        squad_ids = st.session_state.squad_ids
        try:
            squad_rows = [p for p in optimizer.fetch_players(conn, ensemble_weights=_ensemble_weights()) if p.id in squad_ids]
            starting_xi, bench, _formation, floor_used, was_relaxed = optimizer.solve_starting_xi_with_fallback(
                squad_rows, min_starter_xmins=_min_starter_xmins(), risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
            )
            captain_info = optimizer.get_captain_recommendations(
                conn, squad_ids, ensemble_weights=_ensemble_weights(),
                min_starter_xmins=floor_used, risk_lambda=_risk_lambda(), formation_lock=_formation_lock(),
            )
        except OptimizationError as exc:
            st.error(str(exc))
            return

        if was_relaxed:
            st.warning(_starter_floor_relaxed_message(floor_used))

        captain_id = captain_info["captain"]["player"].id
        vice_captain = captain_info.get("vice_captain")
        vice_id = vice_captain["player"].id if vice_captain else None
        starting_xi_ids = [p.id for p in starting_xi]
        bench_ids = [p.id for p in bench]

        try:
            with st.spinner(f"Replaying {season} GW{gw}..."):
                replay_result = replay.replay_gameweek(
                    conn, season, int(gw), squad_ids, starting_xi_ids, bench_ids, captain_id, vice_id,
                )
        except FPLAPIError as exc:
            st.error(f"Could not fetch {season} GW{gw}: {exc}")
            return

        match_pct = replay_result["match_rate"] * 100
        coverage_note = (
            "high -- most of your current squad has real historical stats for this gameweek"
            if match_pct >= 60 else
            "partial -- some current squad members weren't in the league that season (transfers, "
            "promotions/relegations) and are shown as 0/blank below"
        )
        st.caption(f"Match coverage: {match_pct:.0f}% of {season} GW{gw}'s players resolved to a current player ({coverage_note}).")

        _render_live_scorecard(replay_result, starting_xi_ids, bench_ids)


# --- Entrypoint ------------------------------------------------------------------

def main():
    _inject_pwa_meta()
    _inject_global_css()
    database.init_db()
    conn = database.get_connection()
    try:
        _render_team_id_header(conn)
        nav = render_sidebar(conn)
        if nav == "Manager Command Center":
            render_command_center_tab(conn)
        elif nav == "My Squad & Pitch View":
            render_squad_tab(conn)
        elif nav == "Horizon Transfer Planner":
            render_transfer_planner_tab(conn)
        elif nav == "Squad Optimizer & Generators":
            render_optimizer_tab(conn)
        elif nav == "Live Gameweek Radar":
            render_live_radar_tab(conn)
        elif nav == "Fixture Difficulty Matrix":
            render_fixture_matrix_tab(conn)
        elif nav == "Rival Radar & Mini-League":
            render_rival_radar_tab(conn)
        elif nav == "Chip Strategy & Tactics":
            render_chip_strategy_tab(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

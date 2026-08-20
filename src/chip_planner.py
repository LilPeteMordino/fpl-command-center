"""Chip strategy engine: evaluates Triple Captain, Bench Boost, Free Hit, and Wildcard
across a rolling gameweek horizon, tracks chip-set usage/expiry under the 2026/27 rules
(two chips of each kind -- one usable in GW1-19, one in GW20-38), and powers a single-week
"what-if" simulator.

Reuses transfer_planner's per-gameweek projection engine (fixture-aware xP for GW N..N+H)
and optimizer's ILP solvers, rather than re-deriving projections independently.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src import optimizer, transfer_planner
from src.optimizer import OptimizationError

CHIP_CODES = ("WC", "FH", "BB", "TC")
CHIP_NAMES = {"WC": "Wildcard", "FH": "Free Hit", "BB": "Bench Boost", "TC": "Triple Captain"}
CHIP_API_NAME_MAP = {"wildcard": "WC", "freehit": "FH", "bboost": "BB", "3xc": "TC"}

CHIP_SET_1_LAST_GW = 19  # last gameweek Set 1 chips are valid for
CHIP_SET_1_EXPIRY_FALLBACK = datetime.fromisoformat("2027-01-02T00:00:00+00:00")

BENCH_BOOST_XP_THRESHOLD = 12.0
WILDCARD_XP_GAP_THRESHOLD = 15.0
FAVORABLE_FIXTURE_DIFFICULTY = 2  # FDR at or below this counts as a soft/favorable fixture

DEFAULT_ROADMAP_HORIZON_GWS = 8
DEFAULT_WILDCARD_HORIZON_GWS = 5


@dataclass
class ChipRecommendation:
    chip: str
    event_id: Optional[int]
    projected_boost: float
    justification: str


# --- Chip-set awareness --------------------------------------------------------

def chip_set_for_event(event_id: Optional[int]) -> int:
    """1 if the gameweek falls in Set 1 (GW1-19), else 2 (GW20-38)."""
    if event_id is None:
        return 1
    return 1 if event_id <= CHIP_SET_1_LAST_GW else 2


def parse_chip_usage(history_payload: Optional[dict]) -> dict:
    """From the /entry/{id}/history/ payload's 'chips' list, return which chips have been
    played in each set: {"set1": {"WC": event_or_None, ...}, "set2": {...}}."""
    used = {"set1": {code: None for code in CHIP_CODES}, "set2": {code: None for code in CHIP_CODES}}
    if not history_payload:
        return used
    for entry in history_payload.get("chips", []):
        code = CHIP_API_NAME_MAP.get(entry.get("name"))
        if not code:
            continue
        event = entry.get("event")
        set_key = "set1" if event and event <= CHIP_SET_1_LAST_GW else "set2"
        used[set_key][code] = event
    return used


def get_set1_expiry(conn) -> datetime:
    """The real GW19 deadline from the synced data if available, else the known fallback date."""
    row = conn.execute("SELECT deadline_time FROM gameweeks WHERE id = ?", (CHIP_SET_1_LAST_GW,)).fetchone()
    if row and row["deadline_time"]:
        try:
            return datetime.fromisoformat(row["deadline_time"].replace("Z", "+00:00"))
        except ValueError:
            pass
    return CHIP_SET_1_EXPIRY_FALLBACK


# --- Per-chip evaluation over a horizon ----------------------------------------

def evaluate_triple_captain(projections: dict, squad_ids: list, event_ids: list) -> list:
    """Ranks gameweeks by the projected gain of tripling (rather than doubling) your best
    captain candidate that week: the marginal gain is exactly that player's raw projected xP."""
    results = []
    for event_id in event_ids:
        squad_rows = [transfer_planner.player_row_for_gw(projections[pid], event_id) for pid in squad_ids if pid in projections]
        if len(squad_rows) < 11:
            continue
        try:
            starting_xi, _bench, _formation = optimizer.solve_starting_xi(squad_rows)
        except OptimizationError:
            continue
        captain, _vice = transfer_planner.captain_pick_for_gw(starting_xi)
        if not captain.has_fixture or captain.projected_xp <= 0:
            continue

        fixture_count = projections[captain.id]["gw_fixture_count"].get(event_id, 1)
        flags = []
        if fixture_count >= 2:
            flags.append("Double Gameweek")
        if captain.fixture_difficulty <= FAVORABLE_FIXTURE_DIFFICULTY:
            flags.append(f"Favorable fixture (FDR {captain.fixture_difficulty:.0f})")
        if not flags:
            flags.append("Best available captain option")

        results.append({
            "event_id": event_id,
            "player": captain,
            "projected_boost": round(captain.projected_xp, 3),
            "flags": flags,
        })
    results.sort(key=lambda r: r["projected_boost"], reverse=True)
    return results


def evaluate_bench_boost(projections: dict, squad_ids: list, event_ids: list) -> list:
    """Ranks gameweeks by the combined projected xP of the 4 bench players that week."""
    results = []
    for event_id in event_ids:
        squad_rows = [transfer_planner.player_row_for_gw(projections[pid], event_id) for pid in squad_ids if pid in projections]
        if len(squad_rows) < 15:
            continue
        try:
            _starting_xi, bench, _formation = optimizer.solve_starting_xi(squad_rows)
        except OptimizationError:
            continue
        bench_xp = round(sum(p.projected_xp for p in bench), 3)
        results.append({
            "event_id": event_id,
            "bench_xp": bench_xp,
            "bench": bench,
            "recommended": bench_xp > BENCH_BOOST_XP_THRESHOLD,
        })
    results.sort(key=lambda r: r["bench_xp"], reverse=True)
    return results


def evaluate_free_hit(projections: dict, squad_ids: list, event_ids: list) -> list:
    """Ranks gameweeks by how much a freely-rebuilt one-week squad would outscore your actual
    XI -- driven by blank gameweeks (0-fixture squad players) or bad fixture swings."""
    results = []
    for event_id in event_ids:
        squad_rows = [transfer_planner.player_row_for_gw(projections[pid], event_id) for pid in squad_ids if pid in projections]
        if len(squad_rows) < 11:
            continue
        try:
            starting_xi, _bench, _formation = optimizer.solve_starting_xi(squad_rows)
        except OptimizationError:
            continue
        captain, _vice = transfer_planner.captain_pick_for_gw(starting_xi)
        current_score = sum(p.projected_xp for p in starting_xi) + captain.projected_xp
        blank_count = sum(1 for p in squad_rows if not p.has_fixture)

        pool_rows = [transfer_planner.player_row_for_gw(proj, event_id) for proj in projections.values()]
        try:
            fh_squad = optimizer.solve_squad(pool_rows, objective_attr="projected_xp")
            fh_xi, _fh_bench, _fh_formation = optimizer.solve_starting_xi(fh_squad)
            fh_captain, _fh_vice = transfer_planner.captain_pick_for_gw(fh_xi)
            fh_score = sum(p.projected_xp for p in fh_xi) + fh_captain.projected_xp
        except OptimizationError:
            fh_score = current_score

        results.append({
            "event_id": event_id,
            "blank_count": blank_count,
            "current_score": round(current_score, 3),
            "free_hit_score": round(fh_score, 3),
            "projected_gain": round(fh_score - current_score, 3),
        })
    results.sort(key=lambda r: r["projected_gain"], reverse=True)
    return results


def evaluate_wildcard(conn, squad_ids: list, horizon_gws: int = DEFAULT_WILDCARD_HORIZON_GWS) -> dict:
    """Compares your current squad's total projected XI+captain points over the horizon against
    a freshly ILP-optimized squad held fixed across the same weeks -- the "decay gap" a wildcard
    would close. Assumes a standard GBP100m budget rather than your actual bank + squad value,
    since chip planning here is squad-agnostic of a specific manager's finances."""
    event_ids = transfer_planner.get_horizon_event_ids(conn, horizon_gws)
    if not event_ids:
        raise OptimizationError("No upcoming gameweeks found; sync live data first.")
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    def _total_for(ids) -> float:
        total = 0.0
        for event_id in event_ids:
            rows = [transfer_planner.player_row_for_gw(projections[pid], event_id) for pid in ids if pid in projections]
            if len(rows) < 11:
                continue
            try:
                starting_xi, _bench, _formation = optimizer.solve_starting_xi(rows)
            except OptimizationError:
                continue
            captain, _vice = transfer_planner.captain_pick_for_gw(starting_xi)
            total += sum(p.projected_xp for p in starting_xi) + captain.projected_xp
        return round(total, 3)

    current_total = _total_for(squad_ids)

    pool_rows = [transfer_planner.player_row_for_gw(proj, event_ids[0]) for proj in projections.values()]
    optimal_squad = optimizer.solve_squad(pool_rows, objective_attr="projected_xp")
    optimal_total = _total_for({p.id for p in optimal_squad})

    gap = round(optimal_total - current_total, 3)
    return {
        "horizon_gws": len(event_ids),
        "current_total": current_total,
        "optimal_total": optimal_total,
        "gap": gap,
        "recommended": gap > WILDCARD_XP_GAP_THRESHOLD,
    }


# --- Unified roadmap ------------------------------------------------------------

def build_chip_roadmap(conn, squad_ids: list, horizon_gws: int = DEFAULT_ROADMAP_HORIZON_GWS) -> list:
    """One top recommendation per chip type across the horizon, each with a tactical
    justification string, sorted by projected boost (highest first)."""
    event_ids = transfer_planner.get_horizon_event_ids(conn, horizon_gws)
    if not event_ids:
        raise OptimizationError("No upcoming gameweeks found; sync live data first.")
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    recs = []

    tc_results = evaluate_triple_captain(projections, squad_ids, event_ids)
    if tc_results:
        best = tc_results[0]
        flags = ", ".join(best["flags"])
        recs.append(ChipRecommendation(
            chip="TC", event_id=best["event_id"], projected_boost=best["projected_boost"],
            justification=(
                f'GW{best["event_id"]} Triple Captain {best["player"].web_name}: {flags}. '
                f'Projected boost: +{best["projected_boost"]:.1f} pts.'
            ),
        ))

    bb_results = evaluate_bench_boost(projections, squad_ids, event_ids)
    if bb_results:
        best = bb_results[0]
        note = "exceeds the strong-bench threshold" if best["recommended"] else "below the usual strong-bench bar"
        recs.append(ChipRecommendation(
            chip="BB", event_id=best["event_id"], projected_boost=best["bench_xp"],
            justification=f'GW{best["event_id"]} Bench Boost: bench projected {best["bench_xp"]:.1f} pts ({note}).',
        ))

    fh_results = evaluate_free_hit(projections, squad_ids, event_ids)
    if fh_results:
        best = fh_results[0]
        recs.append(ChipRecommendation(
            chip="FH", event_id=best["event_id"], projected_boost=best["projected_gain"],
            justification=(
                f'GW{best["event_id"]} Free Hit: {best["blank_count"]} squad player(s) blank -- '
                f'a rebuilt one-week squad projects +{best["projected_gain"]:.1f} pts over your actual XI.'
            ),
        ))

    try:
        wc_result = evaluate_wildcard(conn, squad_ids, horizon_gws=min(horizon_gws, DEFAULT_WILDCARD_HORIZON_GWS))
        note = "rebuild recommended" if wc_result["recommended"] else "squad still competitive"
        recs.append(ChipRecommendation(
            chip="WC", event_id=event_ids[0], projected_boost=wc_result["gap"],
            justification=(
                f'Wildcard: current squad trails the optimal ILP squad by {wc_result["gap"]:.1f} pts '
                f'over the next {wc_result["horizon_gws"]} GWs ({note}).'
            ),
        ))
    except OptimizationError:
        pass

    for rec in recs:
        if rec.event_id is not None and CHIP_SET_1_LAST_GW - 1 <= rec.event_id <= CHIP_SET_1_LAST_GW:
            rec.justification += " Near Set 1 expiry -- use before the GW19 deadline or lose it."

    recs.sort(key=lambda r: r.projected_boost, reverse=True)
    return recs


# --- First-half (GW1-19) macro chip roadmap --------------------------------------
# Unlike build_chip_roadmap above (single best pick per chip type over a short rolling horizon,
# with no awareness of the other chips), this spreads all four chips across the *entire* Set 1
# window at once: each chip gets its own target-window logic (see the four _*_macro_candidates
# functions), then a greedy priority pass resolves any gameweek collisions (max one chip per GW)
# by moving a chip to its next-best candidate rather than silently overlapping. Chips are
# deliberately NOT force-placed into the opening gameweeks: each one's own candidate logic
# already targets its natural window (Wildcard around the GW6-8 international break unless an
# early rotation-risk trigger fires, Bench Boost split between an early-GW and a post-Wildcard
# path, Free Hit toward the back of the window), so the roadmap spreads out on its own rather
# than needing an artificial anti-clustering penalty.
#
# GW1 is always excluded from every chip's candidates: chips apply to a live, locked-in squad,
# and GW1 itself is still the free/unlimited pre-deadline selection window (see
# optimizer.is_before_gw1_deadline / transfer_planner.plan_transfers) -- there's no "chip
# decision" to make before you've even confirmed a squad.

FIRST_HALF_START_GW = 2  # chips need a live, locked-in squad -- GW1 is free/unlimited squad building, not a chip decision
TC_MACRO_XP_THRESHOLD = 9.0  # "extreme single-fixture ceiling" per spec
WC_MACRO_WINDOW = (6, 8)  # typical autumn-international-break macro fixture swing
WC_LOOKAHEAD_GWS = 3
ROTATION_RISK_TRIGGER_COUNT = 3  # squad players below the floor before an early WC trigger fires -- see _rotation_risk_count
BB_EARLY_WINDOW_GWS = 2  # first two gameweeks of the window -- "GW1/2" per spec, minus the excluded GW1
BB_POST_WC_OFFSETS = (1, 2)  # a post-Wildcard Bench Boost needs the rebuilt squad to bed in, not fire immediately
FH_MIN_BLANK_OR_DOUBLE = 2  # squad players affected before a blank/double gameweek is worth flagging for Free Hit


@dataclass
class MacroChipRecommendation:
    chip: str
    event_id: int
    reasoning: str
    data_driven: bool  # False only for the Free Hit calendar-based fallback -- see _fh_macro_candidates


def _squad_avg_difficulty(squad_ids: list, projections: dict, event_ids: list) -> Optional[float]:
    vals = []
    for pid in squad_ids:
        proj = projections.get(pid)
        if not proj:
            continue
        for event_id in event_ids:
            vals.append(proj["gw_difficulty"].get(event_id, optimizer.NEUTRAL_FIXTURE_DIFFICULTY))
    return sum(vals) / len(vals) if vals else None


def _rotation_risk_count(squad_ids: list, projections: dict, event_id: int) -> int:
    floor = optimizer.STARTER_SECURITY_PROFILES["balanced"]
    count = 0
    for pid in squad_ids:
        proj = projections.get(pid)
        if proj and proj["gw_xmins"].get(event_id, 90.0) < floor:
            count += 1
    return count


def _wc_macro_candidates(squad_ids: list, projections: dict, event_ids: list) -> list:
    """Ranked candidates for the first-half Wildcard: an early rotation-risk trigger (>= 3
    current squad members already projecting below a 'balanced' starting-minutes floor) takes
    priority over the default autumn-international-break fixture-swing window."""
    first_gw = event_ids[0] if event_ids else None
    if first_gw is not None:
        risk_count = _rotation_risk_count(squad_ids, projections, first_gw)
        if risk_count >= ROTATION_RISK_TRIGGER_COUNT:
            return [{
                "event_id": first_gw,
                "reason": (
                    f"{risk_count} current squad player(s) already project as rotation/injury "
                    "risks -- early Wildcard trigger rather than waiting for the usual "
                    "fixture-swing window."
                ),
                "score": float("inf"),
                "data_driven": True,
            }]

    candidates = []
    window_start, window_end = WC_MACRO_WINDOW
    for gw in range(window_start, window_end + 1):
        if gw not in event_ids:
            continue
        lookahead = [e for e in range(gw, gw + WC_LOOKAHEAD_GWS) if e in event_ids]
        if not lookahead:
            continue
        avg_difficulty = _squad_avg_difficulty(squad_ids, projections, lookahead)
        if avg_difficulty is None:
            continue
        candidates.append({
            "event_id": gw,
            "reason": (
                f"Favorable macro fixture swing starting GW{gw} (avg squad FDR {avg_difficulty:.2f} "
                f"across the following {len(lookahead)} GW(s) -- the typical autumn "
                "international-break window)."
            ),
            "score": -avg_difficulty,
            "data_driven": True,
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _tc_macro_candidates(squad_ids: list, projections: dict, event_ids: list) -> list:
    """Any current squad player's single-gameweek projected xP clearing TC_MACRO_XP_THRESHOLD --
    an extreme single-fixture ceiling, or a double gameweek (summed-leg xP already reflects
    that automatically, no separate double-gameweek logic needed)."""
    candidates = []
    for pid in squad_ids:
        proj = projections.get(pid)
        if not proj:
            continue
        for event_id in event_ids:
            xp = proj["gw_xp"].get(event_id, 0.0)
            if xp < TC_MACRO_XP_THRESHOLD:
                continue
            is_double = proj["gw_fixture_count"].get(event_id, 1) >= 2
            note = "double gameweek" if is_double else "single-fixture ceiling"
            candidates.append({
                "event_id": event_id,
                "reason": (
                    f"{proj['web_name']} projects {xp:.1f} xP in GW{event_id} ({note}) -- a "
                    "captain ceiling extreme enough to triple rather than just double."
                ),
                "score": xp,
                "data_driven": True,
            })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _bench_strength(squad_ids: list, projections: dict, event_id: int) -> Optional[float]:
    rows = [transfer_planner.player_row_for_gw(projections[pid], event_id) for pid in squad_ids if pid in projections]
    if len(rows) < 15:
        return None
    try:
        _starting_xi, bench, _formation = optimizer.solve_starting_xi(rows)
    except OptimizationError:
        return None
    return round(sum(p.projected_xp for p in bench), 3)


def _bb_macro_candidates(squad_ids: list, projections: dict, event_ids: list, wc_event_id: Optional[int]) -> list:
    """Two paths per spec: (A) early, while the initial 15 are all fresh/fully fit; (B) paired a
    gameweek or two after a Wildcard, once the rebuilt squad has bedded in (not immediately --
    new signings need to prove their fixture run first). Path B is necessarily a proxy: it scores
    against the *current* squad's bench, since the actual post-Wildcard squad doesn't exist yet
    to evaluate -- treat it as a placeholder timing signal, not a bench-strength guarantee."""
    candidates = []
    for event_id in event_ids[:BB_EARLY_WINDOW_GWS]:
        strength = _bench_strength(squad_ids, projections, event_id)
        if strength is None:
            continue
        candidates.append({
            "event_id": event_id,
            "reason": (
                f"Early Bench Boost (GW{event_id}): all 15 initial picks still fresh/fully fit, "
                f"bench projects {strength:.1f} xP."
            ),
            "score": strength,
            "data_driven": True,
        })
    if wc_event_id is not None:
        for offset in BB_POST_WC_OFFSETS:
            event_id = wc_event_id + offset
            if event_id not in event_ids:
                continue
            strength = _bench_strength(squad_ids, projections, event_id)
            if strength is None:
                continue
            candidates.append({
                "event_id": event_id,
                "reason": (
                    f"Post-Wildcard Bench Boost (GW{event_id}, {offset} GW(s) after the planned "
                    f"GW{wc_event_id} Wildcard): current squad depth projects {strength:.1f} xP "
                    "as a timing proxy for the rebuilt squad."
                ),
                "score": strength,
                "data_driven": True,
            })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _fh_macro_candidates(squad_ids: list, projections: dict, event_ids: list) -> list:
    """Real blank/double detection from the currently-published fixture schedule; falls back to
    a calendar-based winter-congestion placeholder when none is visible yet. Finding nothing is
    the expected, honest outcome this early in the season -- genuine blank/double gameweeks are
    usually only confirmed once cup-replay/European scheduling is known, well after GW19."""
    candidates = []
    for event_id in event_ids:
        blanks = sum(
            1 for pid in squad_ids
            if pid in projections and not projections[pid]["gw_has_fixture"].get(event_id, True)
        )
        doubles = sum(
            1 for pid in squad_ids
            if pid in projections and projections[pid]["gw_fixture_count"].get(event_id, 1) >= 2
        )
        if blanks + doubles >= FH_MIN_BLANK_OR_DOUBLE:
            candidates.append({
                "event_id": event_id,
                "reason": f"GW{event_id}: {blanks} squad blank(s) / {doubles} squad double(s) in the currently-published schedule.",
                "score": blanks + doubles,
                "data_driven": True,
            })
    if not candidates and event_ids:
        fallback_gw = event_ids[-1]
        candidates.append({
            "event_id": fallback_gw,
            "reason": (
                "No blank or double gameweek is visible yet in the currently-published fixture "
                f"list -- reserved at GW{fallback_gw} (end of the first-half window) as a "
                "winter-congestion placeholder; revisit once the schedule firms up."
            ),
            "score": 0,
            "data_driven": False,
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def solve_season_half_chip_strategy(
    conn,
    squad_ids: list,
    gw_start: int = 1,
    gw_end: int = CHIP_SET_1_LAST_GW,
    available_chips: Optional[list] = None,
) -> list:
    """Assigns each available chip to a single non-conflicting gameweek across the first-half
    (Set 1, GW1-19) window -- max one chip per gameweek -- distributing them across the full
    range rather than clustering everything into the opening gameweeks (see the module comment
    above for how). GW1 itself is always excluded (see FIRST_HALF_START_GW).

    available_chips: which chips to plan for, e.g. omit any already used this season -- pass a
    plain list of the still-available codes from CHIP_CODES ("WC", "FH", "BB", "TC").

    Returns a list of MacroChipRecommendation, one per chip that found any viable gameweek in the
    window (a chip with no qualifying candidate anywhere is simply omitted rather than forcing a
    low-confidence pick), sorted by event_id.
    """
    available_chips = list(available_chips) if available_chips is not None else list(CHIP_CODES)
    start = max(gw_start, FIRST_HALF_START_GW)
    end = min(gw_end, CHIP_SET_1_LAST_GW)
    if start > end:
        return []

    event_ids = [
        row["id"] for row in conn.execute(
            "SELECT id FROM gameweeks WHERE id >= ? AND id <= ? ORDER BY id", (start, end)
        ).fetchall()
    ]
    if not event_ids:
        return []
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    candidate_lists = {}
    if "WC" in available_chips:
        candidate_lists["WC"] = _wc_macro_candidates(squad_ids, projections, event_ids)
    if "TC" in available_chips:
        candidate_lists["TC"] = _tc_macro_candidates(squad_ids, projections, event_ids)
    if "FH" in available_chips:
        candidate_lists["FH"] = _fh_macro_candidates(squad_ids, projections, event_ids)

    taken_gws: set = set()
    recommendations = []

    def assign(chip: str, candidates: list) -> Optional[int]:
        for candidate in candidates:
            event_id = candidate["event_id"]
            if event_id in taken_gws:
                continue
            taken_gws.add(event_id)
            recommendations.append(
                MacroChipRecommendation(
                    chip=chip, event_id=event_id, reasoning=candidate["reason"],
                    data_driven=candidate.get("data_driven", True),
                )
            )
            return event_id
        return None

    wc_event_id = assign("WC", candidate_lists["WC"]) if "WC" in candidate_lists else None

    if "BB" in available_chips:
        candidate_lists["BB"] = _bb_macro_candidates(squad_ids, projections, event_ids, wc_event_id)
        assign("BB", candidate_lists["BB"])

    if "TC" in candidate_lists:
        assign("TC", candidate_lists["TC"])
    if "FH" in candidate_lists:
        assign("FH", candidate_lists["FH"])

    recommendations.sort(key=lambda r: r.event_id)
    return recommendations


# --- "What-if" single-chip simulator --------------------------------------------

def simulate_chip(conn, squad_ids: list, chip: str, event_id: Optional[int] = None) -> dict:
    """Standard vs chip-boosted projected score for one chip.

    TC/BB/FH are genuinely single-gameweek effects, so their comparison is for that one gw.
    Wildcard has no single-week scoring effect of its own -- its value is the squad upgrade
    persisting across future weeks -- so its comparison instead spans a short horizon
    (basis='multi_gw') rather than forcing an artificial one-week number.
    """
    if chip not in CHIP_CODES:
        raise ValueError(f"Unknown chip: {chip!r}")

    if chip == "WC":
        result = evaluate_wildcard(conn, squad_ids, horizon_gws=DEFAULT_WILDCARD_HORIZON_GWS)
        return {
            "chip": chip,
            "basis": "multi_gw",
            "horizon_gws": result["horizon_gws"],
            "standard_score": result["current_total"],
            "chip_score": result["optimal_total"],
            "net_advantage": result["gap"],
        }

    if event_id is None:
        horizon = transfer_planner.get_horizon_event_ids(conn, 1)
        event_id = horizon[0] if horizon else None
    if event_id is None:
        raise OptimizationError("No upcoming gameweeks found; sync live data first.")

    projections = transfer_planner.fetch_multi_gw_projections(conn, [event_id])
    squad_rows = [transfer_planner.player_row_for_gw(projections[pid], event_id) for pid in squad_ids if pid in projections]
    if len(squad_rows) < 11:
        raise OptimizationError("Not enough squad players resolved locally for this gameweek.")

    starting_xi, bench, formation = optimizer.solve_starting_xi(squad_rows)
    captain, vice = transfer_planner.captain_pick_for_gw(starting_xi)
    standard_score = round(sum(p.projected_xp for p in starting_xi) + captain.projected_xp, 3)

    if chip == "TC":
        chip_score = round(standard_score + captain.projected_xp, 3)
    elif chip == "BB":
        chip_score = round(standard_score + sum(p.projected_xp for p in bench), 3)
    else:  # FH
        pool_rows = [transfer_planner.player_row_for_gw(proj, event_id) for proj in projections.values()]
        fh_squad = optimizer.solve_squad(pool_rows, objective_attr="projected_xp")
        fh_xi, _fh_bench, _fh_formation = optimizer.solve_starting_xi(fh_squad)
        fh_captain, _fh_vice = transfer_planner.captain_pick_for_gw(fh_xi)
        chip_score = round(sum(p.projected_xp for p in fh_xi) + fh_captain.projected_xp, 3)

    return {
        "chip": chip,
        "basis": "single_gw",
        "event_id": event_id,
        "formation": formation,
        "captain": captain,
        "standard_score": standard_score,
        "chip_score": chip_score,
        "net_advantage": round(chip_score - standard_score, 3),
    }

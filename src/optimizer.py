"""ILP-based FPL squad optimization: 15-man squad builder, starting XI/formation
selection, template vs differential squad generation, and captaincy ranking.

Reads from the local SQLite database populated by sync_data.py (Sprint 1).
"""
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Optional

import pulp

from src import config, database

BUDGET_LIMIT = 1000  # integer cost units == GBP 100.0m
SQUAD_POSITION_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}  # element_type -> required count (GKP/DEF/MID/FWD)
MAX_PLAYERS_PER_TEAM = 3
POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

DIFFERENTIAL_OWNERSHIP_THRESHOLD = 10.0
NEUTRAL_FIXTURE_DIFFICULTY = 3.0  # used when a team has no fixture data for the target gameweek
MAX_FIXTURE_DIFFICULTY = 5

# Generic venue advantage, independent of opponent strength (which the FDR-based
# fixture_difficulty_multiplier already captures via team_h_difficulty/team_a_difficulty).
HOME_FIXTURE_MULTIPLIER = 1.05
AWAY_FIXTURE_MULTIPLIER = 0.95

# --- Positional xP model: real FPL scoring rates, applied per position archetype -----------
# Explicit for all 4 positions (GKP/DEF=6pts/goal, MID=5, FWD=4 -- real FPL scoring). Previously
# only FWD/MID were keyed here and DEF silently fell through to a "5" default via .get(), which
# undercounted defender goals by 1pt each -- see calculate_positional_xp's DEF branch.
GOAL_POINTS_BY_POSITION = {1: 6, 2: 6, 3: 5, 4: 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS = 4  # GKP/DEF clean sheet points (real FPL rule)
MID_CLEAN_SHEET_POINTS = 1.0  # real FPL: MID clean sheet is worth 1pt, not the GKP/DEF 4pts
SAVE_POINTS_PER_SAVE = 1 / 3  # real FPL: 1pt per 3 saves (continuous rate-based proxy for the
# real floor(saves/3) rule -- see calculate_positional_xp's docstring for why a literal floor()
# isn't used on what is otherwise a continuous expected-value model)
APPEARANCE_POINTS_FULL = 2  # proxy for the real 2pts-for-60+-minutes appearance rule
DEFCON_POINTS = 2.0
DEFCON_THRESHOLD = 10  # combined CBIT (clearances+blocks+interceptions+tackles) actions needed

# Real FPL: GKP/DEF lose 1pt per 2 goals their team concedes. Modeled the same way as the save-rate
# proxy above -- a continuous expected-value stand-in for the real floor(goals_conceded/2) rule --
# using an xGA (expected goals against) proxy derived from fixture difficulty the same way
# _team_cs_probability derives clean-sheet chance from it (see _team_xga_proxy).
GOALS_CONCEDED_PENALTY_PER_2 = 1.0
XGA_AT_EASIEST_FIXTURE = 0.75  # FDR 1 -- expect to concede less against a weak attack
XGA_AT_HARDEST_FIXTURE = 2.1  # FDR 5 -- expect to concede more against a strong attack

# Forwards have no dedicated shots/box-touches field in the data model, so xG_per_90 -- already
# the best single proxy for shot volume and box presence we have -- doubles as the input for their
# small "Bonus Point Conversion" term (see calculate_positional_xp's FWD branch).
FWD_BONUS_XP_PER_XG = 0.25

# A MID whose defensive_contribution_per_90 clears this rate is treated as playing a
# defensive/holding role. This no longer gates whether a MID gets ANY defensive credit at all
# (every MID's DEFCON/clean-sheet terms now scale continuously off their own rates -- see
# calculate_positional_xp) -- it only selects which pre-season DEFCON fallback baseline applies
# (see _fallback_defcon_prob) while defensive_contribution_per_90 is still 0.0 league-wide.
MID_DEFENSIVE_CONTRIBUTION_THRESHOLD = 6.0

# Linear proxy: an easier fixture (low FDR) implies a higher clean sheet chance for the team.
CS_PROB_AT_EASIEST_FIXTURE = 0.50  # FDR 1
CS_PROB_AT_HARDEST_FIXTURE = 0.14  # FDR 5

# --- Pre-season DEFCON fallback --------------------------------------------------------------
# defensive_contribution_per_90 sits at 0.0 for the entire player pool until enough of the new
# season's matches have been played, since it's a rate stat with no prior-season carryover. Below
# that, fall back to a coarse position/role baseline instead of projecting a 0% DEFCON chance for
# every defender; any player whose live rate turns positive bypasses this automatically (it's
# a plain per-player "> 0.0" check, not a global season/date switch).
MID_LOW_ATTACK_FALLBACK_THRESHOLD = 0.15  # xG90+xAG90 below this (pre-season only) -> defensive-leaning MID
DEFCON_FALLBACK_CS_PROB_THRESHOLD = 0.35  # at/below this CS prob, a defense is under enough pressure to rack up actions
DEFCON_FALLBACK_DEF_COST_MIN = 50  # GBP 5.0m
DEFCON_FALLBACK_DEF_COST_MAX = 60  # GBP 6.0m
DEFCON_FALLBACK_CENTRAL_DEF = 0.35
DEFCON_FALLBACK_DEFENSIVE_MID = 0.25
DEFCON_FALLBACK_ATTACKING = 0.08  # attacking full-backs, wingers, forwards (0.05-0.10 band)

# --- 2026/27 BPS (bonus points system) tweaks -------------------------------------------------
# Real FPL's bonus points are a whole-match, relative ranking across all 22 players (top 3 BPS
# scores get 3/2/1 extra points) -- a fundamentally different computation from the rest of this
# per-player formula, which is why it isn't modeled in full here. The 2026/27 tweaks below are
# captured as small additive "expected bonus points" terms at the new (reduced) conversion rates,
# layered on top of -- not replacing -- the flat DEFCON threshold bonus above. Keeping the two
# mechanics separate/additive is exactly what avoids the CBI "double-dipping" the rule change
# targets: the flat 10-action DEFCON bonus is untouched, only the smaller incremental-bonus-odds
# contribution from defensive actions is reweighted.
#
# "Remove tackle BPS deductions for wingers/attacking full-backs" has no code-level effect here:
# this formula has never modeled BPS deductions for any position (there's nothing to subtract),
# so there's nothing to remove. Noted for completeness rather than silently skipped.
CBI_BPS_ACTIONS_PER_BONUS_UNIT = 3  # 2026/27: was 1 bonus-unit per 2 CBI actions, now per 3 (reduced weight)
BONUS_XP_PER_CBI_UNIT = 0.15  # small expected match-bonus-points contribution per unit
GKP_BONUS_XP_PER_SAVE = 0.04  # 2026/27: increased BPS expectation for busy shot-stoppers

# --- External xP blending (optional, user-uploaded CSV) ---------------------------------------
# When a player+gameweek has a figure from an uploaded external projections CSV (see
# src/projections.py -- e.g. a manual FPL Review export; there is no live web fetch here), it's
# blended with this module's own positional xP rather than overriding it, so a bad or partial
# upload can't silently replace the whole engine. 0.5 weights the two sources equally; a player
# with no external figure for that gameweek is untouched (pure internal model).
#
# Superseded, for players who have one of the *named* ensemble sources below, by
# get_ensemble_xp/ensemble_from_sources -- this stays available as a simpler single-figure
# blend-with-internal utility (e.g. for a one-off "custom" source), but fetch_players and
# transfer_planner.fetch_multi_gw_projections use the ensemble path, not this one.
EXTERNAL_XP_BLEND_WEIGHT = 0.5

# --- Multi-source ensemble xP (optional, user-uploaded CSVs) --------------------------------
# Unlike blend_external_xp above, this combines *named* external sources against each other --
# weighted-averaging whichever of them have an uploaded figure for a player+gameweek -- and only
# drops back to this module's own positional xP as a last resort, for players neither source
# covers. It deliberately does NOT blend the internal model in when external data exists: the
# whole point of an ensemble across two independent projection sources is to compare/combine
# them, not to dilute them with a third (our own) opinion they didn't ask to be weighted against.
DEFAULT_ENSEMBLE_WEIGHTS = {"fpl_review": 0.5, "fpl_form": 0.5}

# --- Built-in ep_next fallback blend (no upload needed) ------------------------------------------
# FPL's own bootstrap-static already carries `ep_next` -- their official "expected points next
# round" consensus figure -- for every player, always, with no CSV upload required. The internal
# model deliberately does NOT lean on it once it has real season data to work with (a fixture-
# and-position-aware xG/xA breakdown is a richer, more transparent signal than one opaque number),
# but early in a player's OWN season -- true cold start (0 real starts, see the GW1 Pre-Season
# Cold-Start Anchor block below) or just a handful of games in -- the internal per-90 rates are
# either exactly 0.0 or built from a tiny, noisy sample. ep_next is FPL's own best guess at
# exactly that same moment, informed by signal this model doesn't have (their own team-news/
# rotation read, last season's role), so blending it in -- weighted down as this player's own
# `starts` count grows -- tempers early-season noise without ever overriding an increasingly
# well-evidenced internal projection. Only applied when no uploaded ensemble source already covers
# this player+gameweek (an explicit human-curated projection always wins outright, unchanged).
EP_NEXT_BLEND_MAX_WEIGHT = 0.5  # ep_next's blend weight at 0 real starts this season (matches
# EXTERNAL_XP_BLEND_WEIGHT's own 50/50 split above, for the same "co-equal opinion" reasoning)
EP_NEXT_BLEND_FADE_OUT_STARTS = 6  # by this many real starts, ep_next's weight has faded to 0.0


def ep_next_blend_weight(starts: int) -> float:
    """Linear fade from EP_NEXT_BLEND_MAX_WEIGHT at starts=0 down to 0.0 at
    starts>=EP_NEXT_BLEND_FADE_OUT_STARTS -- see the block comment above."""
    if starts >= EP_NEXT_BLEND_FADE_OUT_STARTS:
        return 0.0
    return EP_NEXT_BLEND_MAX_WEIGHT * (1.0 - starts / EP_NEXT_BLEND_FADE_OUT_STARTS)


def blend_ep_next_fallback(breakdown: "XPBreakdown", ep_next: Optional[float], starts: int) -> "XPBreakdown":
    """Blends FPL's own ep_next into breakdown.total at ep_next_blend_weight(starts) -- see the
    block comment above. Only `total` changes (same convention as blend_external_xp): the
    attack/defensive/saves/bonus/appearance sub-components stay the internal model's own numbers,
    since there's no equivalent breakdown for FPL's single ep_next figure to blend into. Returns
    `breakdown` unchanged when ep_next is unavailable (None) or its weight has already faded to 0."""
    if ep_next is None:
        return breakdown
    weight = ep_next_blend_weight(starts)
    if weight <= 0.0:
        return breakdown
    blended_total = round(weight * ep_next + (1 - weight) * breakdown.total, 3)
    return replace(breakdown, total=blended_total, external_xp=round(ep_next, 3), blended=True)


# --- Lineup security (xMins) -------------------------------------------------------------------
# Projected starting minutes for the target gameweek, used two ways: (1) every player's xP is
# scaled by xmins/90 ("Effective xP"), so a rotation-risk player's raw model/ensemble score gets
# discounted whether or not the squad/XI solvers below also exclude them outright, and (2) the
# Starting XI solver hard-excludes anyone below a configurable floor, so the optimizer can't pick
# a 2nd/3rd-choice player (e.g. a backup GKP) into the 11 just because their per-90 rates look
# good in a tiny minutes sample.
#
# Baseline formula (used whenever a player has no uploaded external xMins for this gameweek):
#   xMins = min(90, (starts / max(team_games_played, 1)) * 90) * (chance_of_playing_next_round / 100)
# team_games_played is 0 before a team's first fixture of the season, which -- since `starts` is
# then also necessarily 0 (bootstrap-static has no prior-season carryover) -- would divide 0/0.
# In that pre-season window, starts_rate instead falls back to a cost/ownership proxy (see
# _preseason_starts_rate_fallback): the same "price/ownership band as a nailed-vs-fringe signal"
# approach already used by the pre-season DEFCON fallback elsewhere in this module. This is
# exactly what catches a £4.0-4.5m backup goalkeeper before any real minutes data exists yet.
STARTER_SECURITY_PROFILES = {
    "conservative": 75.0,
    "balanced": 60.0,
    "aggressive": 45.0,
}
# The system-level fallback floor for any caller that doesn't explicitly choose a Starter Security
# profile (app.py's sidebar always does -- see its STARTER_SECURITY_OPTIONS/_min_starter_xmins --
# so this only actually governs callers like src/backtest.py that have no user-facing profile
# control of their own). Deliberately pinned to "conservative" (75 mins), not "balanced": a caller
# with no explicit risk tolerance of its own should default to the safer, lower-rotation-risk
# reading rather than a middling one.
DEFAULT_STARTER_XMINS_FLOOR = STARTER_SECURITY_PROFILES["conservative"]
SUB1_XMINS_FLOOR = 60.0  # the bench's first outfield sub should be a real, minutes-secure player

CHANCE_OF_PLAYING_DEFAULT = 100  # null in the API means "no fitness doubt", i.e. fully available

XMINS_FALLBACK_BACKUP_COST_MAX = 45  # GBP 4.5m -- at/below this price, likely a backup/enabler
XMINS_FALLBACK_BACKUP_OWNERSHIP = 3.0  # % selected -- below this, a budget-priced player reads as a clear backup
XMINS_FALLBACK_NAILED_OWNERSHIP = 10.0  # % selected -- high enough to call a *budget-priced* player nailed anyway
XMINS_FALLBACK_STARTS_RATE_NAILED = 0.95
XMINS_FALLBACK_STARTS_RATE_ROTATION = 0.55
XMINS_FALLBACK_STARTS_RATE_BACKUP = 0.10

# --- Pre-season scouting & tactical overrides ------------------------------------------------
# Manual, per-player corrections layered on top of the model's own pre-season fallbacks -- see
# database.preseason_adjustments and apply_preseason_adjustment below. Unlike the DEFCON/xMins
# fallbacks elsewhere in this module (which infer a nailed-vs-fringe read from price/ownership
# bands), these come directly from the user's own scouting of friendlies/pre-season fixtures,
# so they take precedence over the model's own guesses wherever the two would disagree.
PRESEASON_OOP_ATTACK_BOOST = 0.10  # +10% to attack_xp for a confirmed out-of-position deployment
# (e.g. a MID being used as an auxiliary striker) -- extra attacking opportunity a player's own
# season-long xG/xA rate (recorded in their *normal* position) wouldn't yet reflect.
PRESEASON_PENALTY_XP_BOOST = 0.35  # flat xP added when scouting confirms penalty duty the live
# FPL API's penalties_order hasn't caught up to yet -- a rough single-penalty expected-value
# uplift (roughly a fractional penalty's worth of goal probability over a normal match).

# --- Joint squad/lineup/captaincy MILP (real FPL Expected Value formulation) ------------------
# Squad and Starting XI selection used to each maximize a flat sum(xP) -- with captain/vice then
# picked as a separate post-hoc "top-2 by xP within the XI" step. That flat objective has no way
# to know that whichever player becomes captain gets their points doubled, so it has no reason to
# specially value a lower-flat-xP-but-elite-ceiling asset (e.g. Haaland/Salah) purely for the
# captaincy option owning them opens up -- a few cheaper, marginally-higher-combined-xP players
# could out-compete them on a naive per-budget comparison. _solve_lineup_milp fixes this by
# solving Starting XI (s), Captain (c), and Vice-Captain (v) -- and, for squad-building, squad
# membership (x) itself -- ALL IN ONE MILP, so the captaincy value of including a candidate feeds
# the selection directly, not just the display afterward.
#
# Bench allocation is deliberately NOT part of this joint objective -- see the Two-Stage Bench
# Allocation design at order_bench/_solve_lineup_milp's "Sub 1 Security Constraint": Step 1 (in
# this MILP) only guarantees at least one secure outfield bench option exists; Step 2
# (order_bench, run after the solve) does the actual Sub 1/2/3 labeling by a plain xP sort.
CAPTAIN_XP_BONUS_MULTIPLIER = 1.0  # the captain's own Starting XI term already counts their xP
# once; this contributes +1.0x more on top, yielding the real FPL 2x captaincy score.
VICE_XP_INSURANCE_WEIGHT = 0.05  # small extra credit for held-in-reserve vice-captain value
# (rough proxy for captain non-start probability -- the vice's own score is NOT doubled).

# Effective Ownership (EO) risk-profile term: at risk_lambda=0.0 the model is pure mathematical
# EV (captaincy-aware, but ownership-blind). At risk_lambda > 0.0 it increasingly favors including
# high-ownership/high-captain-likelihood players in the Starting XI even when a lower-owned
# alternative scores marginally higher on xP alone -- i.e. "template shielding": protecting
# against the rank damage of NOT owning a mega-template asset when it hauls (everyone who owns it
# gains together; missing out on it specifically is what costs rank, not owning it). The spec
# calls this an "EO_Penalty" even though its sign here is a bonus for inclusion, not a deduction
# -- "penalty" names the downside it's guarding against (being exposed without the template pick),
# not the arithmetic sign of the term itself.
RISK_PROFILE_LAMBDA = {
    "Pure Mathematical EV": 0.0,
    "Balanced Rank Protection": 0.4,
    "Conservative Shield (High EO Lock)": 1.0,
}
DEFAULT_RISK_PROFILE = "Balanced Rank Protection"

# Status codes from the FPL API and how much of a player's expected minutes they imply.
STATUS_MINUTES_MULTIPLIER = {
    "a": 1.0,   # available
    "d": 0.75,  # doubtful
    "i": 0.1,   # injured
    "s": 0.0,   # suspended
    "u": 0.0,   # unavailable (e.g. left the club)
    "n": 0.0,   # not available (e.g. on loan elsewhere)
}

# --- Press Conference / Injury Flag Gatekeeper -------------------------------------------------
# Unconditional hard rules layered on top of the existing xMins-based discounting/floors -- these
# apply regardless of the user's own Starter Security profile (min_starter_xmins), unlike
# STARTER_SECURITY_PROFILES which the user can loosen or tighten.
HARD_EXCLUDE_STATUSES = ("i", "s", "u")  # injured, suspended, unavailable (e.g. left the club)

# Vice-Captain Lock: the assigned vice must be an obviously undoubted, high-minutes safe pick.
# xMins >= 85 is already a demanding bar on its own -- only genuinely nailed regular starters
# clear it (live-data check: ~55% of the player pool, entirely players who'd already pass on
# minutes alone). The chance_of_playing_next_round condition treats null the same as
# CHANCE_OF_PLAYING_DEFAULT (100), unlike _is_minutes_secure's stricter equality-only check:
# chance is *exactly* 100 for only a tiny handful of players in practice (it's reserved for "just
# confirmed recovered from a specific injury scare," not "normal undoubted fitness," which is what
# null means for the vast majority of healthy regular starters) -- a strict-only reading would
# leave almost no real vice candidates ever eligible, which isn't a usable outcome.
VICE_CAPTAIN_XMINS_FLOOR = 85.0

# --- Captaincy Position & Talisman Filtering ---------------------------------------------------
# Real managers essentially never hand the armband to a goalkeeper or a purely defensive
# centre-back -- both have a far lower scoring ceiling than a MID/FWD, and the rare "huge
# defensive haul" upside doesn't come close to compensating on expectation. This is a hard
# eligibility gate on the armband itself (captain AND vice -- see is_vice_eligible), layered on
# TOP OF (not instead of) is_vice_eligible's own xMins/fitness floor: a player can clear that
# floor and still never be a captaincy CANDIDATE at all if they fail this gate.
CAPTAINCY_ELIGIBLE_POSITIONS = (3, 4)  # MID, FWD -- always eligible.
# A DEF only clears the gate as an "attacking wing-back with a set-piece monopoly" proxy -- the
# data model has no direct wing-back/attacking-role field, so confirmed primary penalty or
# corner/free-kick duty stands in for it (a defender entrusted with either is, by definition, part
# of their side's primary attacking unit, not a pure stopper). GKP is never eligible under any
# condition.
CAPTAINCY_ELIGIBLE_DEF_REQUIRES_SET_PIECE_DUTY = True

# Talisman Penalty-Taker boost: a confirmed primary penalty taker (penalties_order == 1) gets a
# ranking bonus for the CAPTAINCY decision specifically -- a strict tie-breaker/nudge applied only
# to c[]/v[] scoring (mirrors CAPTAIN_XP_BONUS_MULTIPLIER/VICE_XP_INSURANCE_WEIGHT's own scope),
# never to squad or Starting XI selection itself, and never a hard eligibility requirement on its
# own (a non-penalty-taker MID/FWD remains fully eligible, just unboosted).
TALISMAN_PENALTY_BOOST = 0.15
TALISMAN_FAVORABLE_FDR = 2  # FDR <= this (or a home fixture) counts as "favorable" for the boost.

# --- GW1 Pre-Season Cold-Start Anchor -----------------------------------------------------------
# Before a single gameweek of real season data exists, every per-90 rate is 0.0 league-wide (see
# the "Pre-season DEFCON fallback" section above) -- projected_xp at that point is driven almost
# entirely by the flat appearance-floor term plus whatever fallback heuristics apply, a much
# weaker signal than a real in-season projection. Left alone, this can let a cheap defensive-role
# punt's fallback score edge out a genuine big-name attacker purely on noise. During that window
# (see is_cold_start_pool), captaincy_candidates additionally restricts the field to premium-priced
# or standout-projection MID/FWD only -- but ONLY as a post-hoc restriction on an already-decided
# Starting XI (get_captain_recommendations/transfer_planner.captain_pick_for_gw), never as a hard
# MILP constraint: the XI is only 11 players and always contains >= 3 base-eligible MID/FWD (the
# formation floor guarantees it), so the restriction can never be infeasible there. Applying it
# inside _solve_lineup_milp itself -- where the *candidate pool* can be the full 600+ player
# universe or an as-yet-undecided squad -- would risk zeroing out c[]/v[] for literally every
# player who ends up actually selected/started, since "top-3 by score across the whole universe"
# has no guarantee of overlapping with a budget-constrained optimal squad at all. The MILP gets
# only a SOFT nudge instead (GW1_COLD_START_PRICE_BONUS, folded into captaincy_score) -- enough to
# meaningfully steer the joint solve toward a premium pick without ever risking infeasibility.
GW1_CAPTAIN_MIN_COST = 100  # GBP 10.0m -- the "premium talisman" price floor.
GW1_COLD_START_PRICE_BONUS = 0.20  # soft captaincy-ranking nudge for a premium-priced candidate
# during the cold-start window (used by captaincy_score, NOT a hard MILP gate -- see above).
GW1_COLD_START_TOP_N = 3  # size of the "standout pre-season projection" fallback pool alongside
# the price floor, when captaincy_candidates applies the hard post-hoc restriction.


def is_captaincy_eligible(player) -> bool:
    """Position/talisman gate for the armband (captain AND vice) -- see the constant block above.
    GKP is never eligible. MID/FWD are always eligible. DEF is eligible only with confirmed
    primary penalty or corner/free-kick duty."""
    if player.element_type in CAPTAINCY_ELIGIBLE_POSITIONS:
        return True
    if player.element_type == 2 and CAPTAINCY_ELIGIBLE_DEF_REQUIRES_SET_PIECE_DUTY:
        return player.penalties_order == 1 or player.corners_order == 1
    return False


def _is_talisman_boost_favorable(player) -> bool:
    """True for a confirmed primary penalty taker at home or facing an easy fixture (FDR <=
    TALISMAN_FAVORABLE_FDR) -- see captaincy_score. "Home" reads PlayerRow.is_home when the caller
    has populated it (fetch_players does); FDR alone still applies when it hasn't (None)."""
    if player.penalties_order != 1:
        return False
    return bool(player.is_home) or player.fixture_difficulty <= TALISMAN_FAVORABLE_FDR


def is_cold_start_pool(pool: list) -> bool:
    """True when literally every candidate in `pool` has zero cumulative starts -- the state
    that's only ever true before a single gameweek of this season has been played (`starts`
    accumulates from GW1 onward; any nonzero value anywhere signals real season data already
    exists). Detected from the candidates' own PlayerRow.starts rather than needing DB/gameweek
    context threaded in, so this stays a pure function of whatever pool is passed to it -- see the
    GW1 Pre-Season Cold-Start Anchor block above. Public (not underscore-prefixed) since
    transfer_planner.captain_pick_for_gw also needs it."""
    return bool(pool) and all(p.starts == 0 for p in pool)


def captaincy_score(player, cold_start: bool = False) -> float:
    """projected_xp, boosted for a favorably-placed primary penalty taker (TALISMAN_PENALTY_BOOST)
    and, when `cold_start` is True, for a premium-priced candidate (GW1_COLD_START_PRICE_BONUS) --
    used to RANK captaincy candidates only, never to rank squad/XI selection itself and never to
    gate eligibility on its own (see is_captaincy_eligible/captaincy_candidates for the actual
    gates). Public since transfer_planner.captain_pick_for_gw also needs it."""
    boost = TALISMAN_PENALTY_BOOST if _is_talisman_boost_favorable(player) else 0.0
    if cold_start and player.now_cost >= GW1_CAPTAIN_MIN_COST:
        boost += GW1_COLD_START_PRICE_BONUS
    return round(player.projected_xp * (1 + boost), 4)


def captaincy_candidates(pool: list) -> set:
    """The set of player ids eligible to hold the captain OR vice-captain armband from `pool` --
    is_captaincy_eligible's base position/talisman gate, always applied first. When `pool` is
    itself a GW1 Pre-Season Cold-Start pool (is_cold_start_pool), the gate additionally narrows to
    premium-priced candidates (now_cost >= GW1_CAPTAIN_MIN_COST) OR the top GW1_COLD_START_TOP_N by
    captaincy_score among the base-eligible pool -- see the GW1 Pre-Season Cold-Start Anchor block
    above for why this is only ever applied post-hoc to an already-decided Starting XI (safe: a
    legal XI always has >= 3 base-eligible MID/FWD), never as a hard _solve_lineup_milp constraint.
    Public since transfer_planner.captain_pick_for_gw also needs it."""
    base_eligible = [p for p in pool if is_captaincy_eligible(p)]
    if not is_cold_start_pool(pool):
        return {p.id for p in base_eligible}
    premium = {p.id for p in base_eligible if p.now_cost >= GW1_CAPTAIN_MIN_COST}
    top_n = sorted(base_eligible, key=lambda p: captaincy_score(p, cold_start=True), reverse=True)[:GW1_COLD_START_TOP_N]
    return premium | {p.id for p in top_n}


class OptimizationError(RuntimeError):
    """Raised when an ILP model is infeasible or the solver fails to find an optimal solution."""


@dataclass
class XPBreakdown:
    """Transparency for calculate_positional_xp: what the total is actually made of, so the UI
    can show a player's attacking (xG/xAG) score alongside their defensive/DEFCON floor."""
    total: float
    attack_xp: float
    defensive_xp: float  # clean sheet + DEFCON combined
    saves_xp: float
    bonus_xp: float  # expected match bonus points from BPS-relevant actions (2026/27 weights)
    appearance_xp: float
    cs_prob: float
    defcon_prob: float
    minutes_factor: float
    external_xp: Optional[float] = None  # the uploaded-CSV figure blended in, if any (see blend_external_xp)
    blended: bool = False


@dataclass
class PlayerRow:
    id: int
    web_name: str
    team_id: int
    team_name: str
    element_type: int
    now_cost: int
    selected_by_percent: float
    form: float
    total_points: int
    ep_next: Optional[float]
    xg_per_90: float
    xa_per_90: float
    saves_per_90: float
    defensive_contribution_per_90: float
    starts_per_90: float
    status: str
    fixture_difficulty: float
    has_fixture: bool
    projected_xp: float
    xp_breakdown: Optional[XPBreakdown] = None
    starts: int = 0  # cumulative starts this season -- xMins lineup-security baseline input
    chance_of_playing_next_round: Optional[int] = None
    xmins: float = 90.0  # projected starting minutes; see calculate_baseline_xmins/effective xP scaling
    news: str = ""  # FPL's own free-text status note, e.g. "Knee injury - Expected back 01 Sep"
    penalties_order: Optional[int] = None  # 1 = primary penalty taker, None = not on the list
    corners_order: Optional[int] = None  # 1 = primary corners/indirect-FK taker, None = not on the list
    is_home: Optional[bool] = None  # True/False for the target gameweek's fixture (None = unknown/
    # no fixture) -- feeds the Talisman Penalty-Taker captaincy boost's "home fixture" condition
    # (see _is_talisman_boost_favorable); not used anywhere else in the model.
    expected_goals_conceded_per_90: float = 0.0  # this player's own real xGC rate -- see
    # _blend_player_xga (0.0 means no real minutes-backed figure yet, same meaning as elsewhere)

    @property
    def cost_millions(self) -> float:
        return self.now_cost / config.PRICE_DIVISOR

    @property
    def position(self) -> str:
        return POSITION_NAMES[self.element_type]


# --- Data loading & xP calculation ------------------------------------------

def _get_target_event_id(conn) -> Optional[int]:
    """The gameweek to project fixture difficulty for: the next one, falling back to current."""
    row = conn.execute("SELECT id FROM gameweeks WHERE is_next = 1 ORDER BY id LIMIT 1").fetchone()
    if row is None:
        row = conn.execute("SELECT id FROM gameweeks WHERE is_current = 1 ORDER BY id LIMIT 1").fetchone()
    return row["id"] if row else None


def is_before_gw1_deadline(conn) -> bool:
    """True until the real Gameweek 1 deadline passes. Real FPL rule: squad changes before that
    moment are the free, unlimited initial-squad-selection window, not "transfers" in the FT/hit
    sense -- that economy (free transfers, -4 hits) only starts applying from GW2 onward. False
    once GW1 locks, even though GW1 itself won't be marked `finished` until its matches complete
    -- it's the deadline that matters here, not full-time. False (not an error) if GW1's deadline
    isn't in the local data yet (e.g. before any sync), since there's nothing to be "before" of."""
    row = conn.execute("SELECT deadline_time FROM gameweeks WHERE id = 1").fetchone()
    if row is None or not row["deadline_time"]:
        return False
    try:
        deadline = datetime.fromisoformat(row["deadline_time"].replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.now(timezone.utc) < deadline


def _team_fixture_difficulties(conn, event_id: Optional[int]) -> dict:
    """team_id -> list of difficulties faced in the target gameweek (handles blank/double gameweeks)."""
    difficulties: dict = {}
    if event_id is None:
        return difficulties
    rows = conn.execute(
        "SELECT team_h, team_a, team_h_difficulty, team_a_difficulty FROM fixtures WHERE event = ?",
        (event_id,),
    ).fetchall()
    for row in rows:
        difficulties.setdefault(row["team_h"], []).append(row["team_h_difficulty"] or NEUTRAL_FIXTURE_DIFFICULTY)
        difficulties.setdefault(row["team_a"], []).append(row["team_a_difficulty"] or NEUTRAL_FIXTURE_DIFFICULTY)
    return difficulties


def _team_home_fixture(conn, event_id: Optional[int]) -> dict:
    """team_id -> True if ANY of that team's fixture(s) this gameweek is at home (double gameweeks
    OR their legs together, so a team with one home and one away leg still reads as home -- a
    talisman's home-fixture captaincy boost shouldn't be denied just because their away leg is
    listed second). Feeds PlayerRow.is_home / _is_talisman_boost_favorable; {} for no fixture
    data at all for the target gameweek."""
    home_by_team: dict = {}
    if event_id is None:
        return home_by_team
    rows = conn.execute("SELECT team_h, team_a FROM fixtures WHERE event = ?", (event_id,)).fetchall()
    for row in rows:
        home_by_team[row["team_h"]] = True
        if row["team_a"] not in home_by_team:
            home_by_team[row["team_a"]] = False
    return home_by_team


def ensemble_from_sources(present: dict, weights: dict, baseline: Optional[float] = None) -> Optional[float]:
    """Weighted average of whichever of `present`'s sources also carry a positive weight in
    `weights`; falls back to `baseline` when none do (covers both "no upload at all" and "the
    only upload we have for this player+gameweek is a source weighted at 0"). Shared by
    get_ensemble_xp, fetch_players, and transfer_planner.fetch_multi_gw_projections so the same
    weighting math runs everywhere -- including a single-source case, which naturally reduces to
    "use that source directly" once the weights are renormalized over just what's present."""
    usable = {source: xp for source, xp in present.items() if weights.get(source, 0) > 0}
    if not usable:
        return baseline
    total_weight = sum(weights[source] for source in usable)
    if total_weight <= 0:
        return baseline
    return round(sum(weights[source] * xp for source, xp in usable.items()) / total_weight, 3)


def get_ensemble_xp(
    conn, player_id: int, event_id: Optional[int], weights: Optional[dict] = None, baseline: Optional[float] = None
) -> Optional[float]:
    """Single player+gameweek ensemble lookup (see ensemble_from_sources) -- convenient for
    one-off queries (e.g. the Model Divergence Table), but fetch_players uses the bulk
    _ensemble_xp_lookup below instead of calling this once per player."""
    if event_id is None:
        return baseline
    weights = weights or DEFAULT_ENSEMBLE_WEIGHTS
    from src import database  # deferred: keeps optimizer.py's import surface minimal for callers that don't need it

    rows = database.get_external_projections(conn, event_ids=[event_id], source=list(weights.keys()))
    present = {
        source: vals["xp"] for (pid, _event, source), vals in rows.items() if pid == player_id
    }
    return ensemble_from_sources(present, weights, baseline)


def _ensemble_xp_lookup(conn, event_id: Optional[int], weights: dict) -> dict:
    """player_id -> ensemble xP (see ensemble_from_sources), for every player who has at least
    one weighted source uploaded for this gameweek; {} if none do (or there's no target
    gameweek). Bulk equivalent of get_ensemble_xp -- one query for the whole player pool rather
    than one per player."""
    if event_id is None:
        return {}
    from src import database  # deferred: keeps optimizer.py's import surface minimal for callers that don't need it

    rows = database.get_external_projections(conn, event_ids=[event_id], source=list(weights.keys()))
    by_player: dict = {}
    for (player_id, _event, source), vals in rows.items():
        by_player.setdefault(player_id, {})[source] = vals["xp"]
    return {
        player_id: ensemble_from_sources(present, weights)
        for player_id, present in by_player.items()
    }


def _chance_of_playing_fraction(chance_of_playing_next_round: Optional[int]) -> float:
    pct = CHANCE_OF_PLAYING_DEFAULT if chance_of_playing_next_round is None else chance_of_playing_next_round
    return max(0.0, min(100.0, pct)) / 100.0


def _preseason_starts_rate_fallback(player) -> float:
    """Starts-rate proxy for when there's no in-season starts data yet (team_games_played == 0,
    so `starts` is necessarily also 0 -- bootstrap-static carries no prior-season history). Uses
    price/ownership bands as a nailed-vs-fringe signal, the same approach the pre-season DEFCON
    fallback elsewhere in this module already takes.

    Price above the budget band is decisive on its own (real starting players are essentially
    never parked at the absolute price floor). At/below it, ownership alone only promotes a
    player to "nailed" past a much higher bar (XMINS_FALLBACK_NAILED_OWNERSHIP) -- a modest 3-8%
    on a £4.0-4.5m player reads as a popular budget *enabler* pick, not proof they start every
    week, and lands in the middle "rotation" tier rather than being waved through as safe. This
    is exactly what's needed to flag a cheap, low-to-modestly-owned backup goalkeeper as a
    minutes risk before a ball has been kicked, without over-trusting small ownership blips."""
    if player.now_cost > XMINS_FALLBACK_BACKUP_COST_MAX:
        return XMINS_FALLBACK_STARTS_RATE_NAILED
    if player.selected_by_percent >= XMINS_FALLBACK_NAILED_OWNERSHIP:
        return XMINS_FALLBACK_STARTS_RATE_NAILED
    if player.selected_by_percent < XMINS_FALLBACK_BACKUP_OWNERSHIP:
        return XMINS_FALLBACK_STARTS_RATE_BACKUP
    return XMINS_FALLBACK_STARTS_RATE_ROTATION


def team_games_played(conn) -> dict:
    """team_id -> count of finished fixtures this season. 0 for a team with none played yet
    (pre-season), which calculate_baseline_xmins treats as its own case rather than a 0/0 divide."""
    rows = conn.execute(
        """
        SELECT team_id, COUNT(*) AS n FROM (
            SELECT team_h AS team_id FROM fixtures WHERE finished = 1
            UNION ALL
            SELECT team_a AS team_id FROM fixtures WHERE finished = 1
        ) GROUP BY team_id
        """
    ).fetchall()
    return {row["team_id"]: row["n"] for row in rows}


def calculate_baseline_xmins(player, games_played: int) -> float:
    """Projected starting minutes for the target gameweek, from the player's own starts record
    (or the pre-season fallback proxy) and their live fitness status. See the module-level
    comment above STARTER_SECURITY_PROFILES for the formula and rationale."""
    if games_played > 0:
        starts_rate = min(1.0, (player.starts or 0) / games_played)
    else:
        starts_rate = _preseason_starts_rate_fallback(player)
    return round(min(90.0, starts_rate * 90.0) * _chance_of_playing_fraction(player.chance_of_playing_next_round), 1)


def _xmins_ensemble_lookup(conn, event_id: Optional[int], weights: dict) -> dict:
    """player_id -> uploaded-CSV xMins ensemble (see ensemble_from_sources) for every player
    covered by at least one weighted source for this gameweek; {} if none are (or there's no
    target gameweek). Callers fall back to calculate_baseline_xmins for players not in this dict."""
    if event_id is None:
        return {}
    from src import database  # deferred: keeps optimizer.py's import surface minimal for callers that don't need it

    rows = database.get_external_projections(conn, event_ids=[event_id], source=list(weights.keys()))
    by_player: dict = {}
    for (player_id, _event, source), vals in rows.items():
        if vals["xmins"] is not None:
            by_player.setdefault(player_id, {})[source] = vals["xmins"]
    return {player_id: ensemble_from_sources(present, weights) for player_id, present in by_player.items()}


def fixture_difficulty_multiplier(avg_difficulty: float) -> float:
    """Maps FPL's 1 (easy) - 5 (hard) FDR scale to a scoring multiplier: 1.2x at difficulty 1,
    0.8x at difficulty 5. Shared by the single-gw fallback formula and the multi-gw planner."""
    return 1.3 - 0.1 * avg_difficulty


def _team_cs_probability(fixture_difficulty: float) -> float:
    """Linear proxy from FDR (1=easy..5=hard) to a team's clean-sheet chance that match."""
    span = CS_PROB_AT_EASIEST_FIXTURE - CS_PROB_AT_HARDEST_FIXTURE
    frac = max(0.0, min(1.0, (fixture_difficulty - 1) / (MAX_FIXTURE_DIFFICULTY - 1)))
    return CS_PROB_AT_EASIEST_FIXTURE - span * frac


def _team_xga_proxy(fixture_difficulty: float) -> float:
    """Linear proxy from FDR (1=easy..5=hard) to a team's expected goals conceded that match --
    same interpolation shape as _team_cs_probability, just increasing instead of decreasing with
    difficulty. Feeds the GKP/DEF goals-conceded penalty (real FPL: -1pt per 2 goals conceded)."""
    span = XGA_AT_HARDEST_FIXTURE - XGA_AT_EASIEST_FIXTURE
    frac = max(0.0, min(1.0, (fixture_difficulty - 1) / (MAX_FIXTURE_DIFFICULTY - 1)))
    return XGA_AT_EASIEST_FIXTURE + span * frac


PLAYER_XGA_BLEND_WEIGHT = 0.5  # how much a player's own real expected_goals_conceded_per_90
# blends into the fixture-difficulty-only team proxy, once they actually have any (see
# _blend_player_xga) -- co-equal with the fixture read, same reasoning as EP_NEXT_BLEND_MAX_WEIGHT.


def _blend_player_xga(player_xga_per_90: float, team_xga_proxy: float) -> float:
    """Blends a player's own real expected_goals_conceded_per_90 (their actual on-pitch defensive
    exposure this season -- e.g. a low-possession, deep-blocking side's fullback genuinely faces
    more shots than a dominant side's, regardless of any one match's FDR) with the fixture-
    difficulty-only team proxy (_team_xga_proxy) -- the player figure is real but doesn't know
    anything about the SPECIFIC upcoming opponent; the team proxy knows the fixture but not this
    player's real personal exposure. Falls back to the team proxy alone (identical to the old
    behavior) when the player has no real minutes-backed xGC yet (0.0 -- a genuine cold start for
    this specific signal, distinct from the general games_played one)."""
    if player_xga_per_90 <= 0.0:
        return team_xga_proxy
    return PLAYER_XGA_BLEND_WEIGHT * player_xga_per_90 + (1 - PLAYER_XGA_BLEND_WEIGHT) * team_xga_proxy


def _poisson_at_least(mu: float, threshold: int) -> float:
    """P(X >= threshold) for X ~ Poisson(mu): converts a per-90 action rate (e.g. defensive
    contributions) into the probability of clearing a fixed count in a single match."""
    if mu <= 0:
        return 0.0
    cdf = term = math.exp(-mu)
    for k in range(1, threshold):
        term *= mu / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def _projected_minutes_fraction(player, games_played: int) -> float:
    """Rough probability of a meaningful appearance this gameweek: how often they start (their own
    starts_per_90 rate, once their team has actually played games this season), tempered by
    fitness/rotation risk (the existing status-based minutes multiplier).

    Pre-season (games_played == 0, so starts_per_90 is necessarily also 0.0 -- bootstrap-static
    carries no prior-season history), falls back to the same price/ownership starts-rate proxy
    calculate_baseline_xmins already uses (_preseason_starts_rate_fallback) instead of leaving this
    at a hard 0.0. Before this fallback existed here, appearance_xp/attack_xp/defensive_xp/bonus_xp
    all being scaled by an unconditional 0.0 meant projected_xp came out exactly 0.0 for literally
    every player before a ball had been kicked -- including established, obviously-starting
    superstars -- even though calculate_baseline_xmins's OWN parallel "minutes" figure (this
    player's displayed xmins) already correctly used this same fallback and looked perfectly
    reasonable. That mismatch made the model's own captaincy/Starting-XI ranking pure noise at
    exactly the highest-stakes moment of the season (the real GW1 deadline), independent of and in
    addition to GW1_COLD_START_PRICE_BONUS's own multiplicative-on-zero bug below."""
    if games_played > 0:
        start_rate = max(0.0, min(1.0, player.starts_per_90 or 0.0))
    else:
        start_rate = _preseason_starts_rate_fallback(player)
    return start_rate * minutes_security_multiplier(player.status)


def _is_defensive_mid(player) -> bool:
    """True for a MID playing a defensive/holding role. Normally judged by their live
    defensive_contribution_per_90 rate; pre-season, when that rate is still 0.0 league-wide,
    falls back to attacking output instead (a MID with little xG/xAG is more likely playing
    deep) so the defensive-MID DEFCON baseline below has any midfielders to apply to at all."""
    if player.element_type != 3:
        return False
    if player.defensive_contribution_per_90 > 0.0:
        return player.defensive_contribution_per_90 >= MID_DEFENSIVE_CONTRIBUTION_THRESHOLD
    return (player.xg_per_90 + player.xa_per_90) < MID_LOW_ATTACK_FALLBACK_THRESHOLD


def _fallback_defcon_prob(player, cs_prob: float, is_defensive_mid: bool) -> float:
    """Pre-season DEFCON baseline for a DEF or defensive MID, used only while
    defensive_contribution_per_90 is still 0.0 (see the module-level comment above)."""
    if is_defensive_mid:
        return DEFCON_FALLBACK_DEFENSIVE_MID
    is_central_defender_profile = (
        cs_prob <= DEFCON_FALLBACK_CS_PROB_THRESHOLD
        or DEFCON_FALLBACK_DEF_COST_MIN <= player.now_cost <= DEFCON_FALLBACK_DEF_COST_MAX
    )
    return DEFCON_FALLBACK_CENTRAL_DEF if is_central_defender_profile else DEFCON_FALLBACK_ATTACKING


def calculate_positional_xp(player, fixture_difficulty: float, games_played: int) -> XPBreakdown:
    """Position-specific projected points: four dedicated formulas (one per real FPL scoring
    archetype), not one generalized formula reused across positions.

    - GKP: team clean-sheet prob x 4, + expected saves x 1/3, - goals-conceded penalty
      (xGA-proxy/2 x 1pt). Outfield xG/xAG is never read here -- attack_xp is always exactly
      0.0 for a goalkeeper, deliberately, so GKPs are never ranked/compared on xGI.
    - DEF: the same clean-sheet + goals-conceded terms as GKP, + attacking returns
      (xG_90 x 6pts/goal + xAG_90 x 3pts/assist, unscaled by fixture -- the CS-prob and xGA-proxy
      terms are already fixture-aware) + DEFCON bonus prob x 2 (P(CBIT actions >= 10), a Poisson
      tail on defensive_contribution_per_90, or a coarse position/cost baseline pre-season while
      that rate is still 0.0 league-wide -- see _fallback_defcon_prob).
    - MID: attacking returns (xG_90 x 5pts/goal + xAG_90 x 3pts/assist, scaled by fixture ease)
      + a small clean-sheet bonus (cs_prob x 1pt, the real MID clean-sheet rate) + the same
      DEFCON-bonus-probability mechanic as DEF, scaled continuously off their own
      defensive_contribution_per_90 rather than gated behind an attacking-vs-defensive role
      split -- every MID gets defensive credit proportional to their own rate now, not only
      players crossing MID_DEFENSIVE_CONTRIBUTION_THRESHOLD.
    - FWD: pure attacking returns (xG_90 x 4pts/goal + xAG_90 x 3pts/assist, scaled by fixture
      ease). No clean-sheet or DEFCON credit at all, per real FPL forward scoring.

    On top of the above, a small "bonus_xp" term (2026/27 BPS reweighting) adds expected match
    bonus points from BPS-relevant actions: CBI-derived for DEF/MID at the new, reduced
    per-action weight, save-derived for GKP at an increased weight for busy shot-stoppers, and
    (new) a shot-volume proxy for FWD using xG_per_90 itself, since there's no dedicated
    shots/box-touches field in the data model to derive a truer figure from. See
    CBI_BPS_ACTIONS_PER_BONUS_UNIT/GKP_BONUS_XP_PER_SAVE/FWD_BONUS_XP_PER_XG.

    Every component (other than the flat appearance points) is scaled by a projected-minutes
    fraction: a clean-sheet or DEFCON bonus can't be banked by a player who doesn't play, and
    xG90/xAG90 are per-90 rates that need a probability-of-actually-getting-90 applied to them.
    Note this uses the player's own defensive_contribution_per_90 rate directly rather than a
    separate "low-possession team" bonus -- that rate already organically reflects it, since
    defenders on deeper, lower-possession sides empirically rack up more CBIT actions.

    Save/goals-conceded penalty note: real FPL applies floor(saves/3) and floor(conceded/2) --
    integer, discrete rules. This model keeps both as continuous rate-based proxies (dividing the
    expected rate directly) rather than literally flooring an expected value, consistent with how
    APPEARANCE_POINTS_FULL already approximates the real 60-minutes discrete threshold with a
    continuous minutes fraction -- flooring a *mean* would understate the true expectation
    (E[floor(X/k)] != floor(E[X]/k) in general) rather than more faithfully model the real rule.

    games_played is the player's TEAM's finished-fixture count this season (see
    team_games_played) -- 0 pre-season, engaging _projected_minutes_fraction's price/ownership
    starts-rate fallback for every component below instead of hard-zeroing them all.
    """
    minutes_fraction = _projected_minutes_fraction(player, games_played)
    appearance_xp = minutes_fraction * APPEARANCE_POINTS_FULL
    is_def_mid = _is_defensive_mid(player)  # only used pre-season, to pick MID's DEFCON fallback baseline

    if player.element_type == 1:  # GKP
        cs_prob = _team_cs_probability(fixture_difficulty)
        xga = _blend_player_xga(player.expected_goals_conceded_per_90, _team_xga_proxy(fixture_difficulty))
        defcon_prob = 0.0
        attack_xp = 0.0  # deliberately always 0.0 -- GKPs are never scored/ranked on outfield xGI
        saves_xp = player.saves_per_90 * SAVE_POINTS_PER_SAVE
        bonus_xp = player.saves_per_90 * GKP_BONUS_XP_PER_SAVE
        conceded_penalty = (xga / 2) * GOALS_CONCEDED_PENALTY_PER_2
        defensive_xp = cs_prob * CLEAN_SHEET_POINTS - conceded_penalty
        total = minutes_fraction * (defensive_xp + saves_xp + bonus_xp) + appearance_xp

    elif player.element_type == 2:  # DEF
        cs_prob = _team_cs_probability(fixture_difficulty)
        xga = _blend_player_xga(player.expected_goals_conceded_per_90, _team_xga_proxy(fixture_difficulty))
        if player.defensive_contribution_per_90 > 0.0:
            defcon_prob = _poisson_at_least(player.defensive_contribution_per_90, DEFCON_THRESHOLD)
        else:
            defcon_prob = _fallback_defcon_prob(player, cs_prob, is_defensive_mid=False)
        goal_pts = GOAL_POINTS_BY_POSITION[2]
        attack_xp = player.xg_per_90 * goal_pts + player.xa_per_90 * ASSIST_POINTS
        saves_xp = 0.0
        bonus_xp = (player.defensive_contribution_per_90 / CBI_BPS_ACTIONS_PER_BONUS_UNIT) * BONUS_XP_PER_CBI_UNIT
        conceded_penalty = (xga / 2) * GOALS_CONCEDED_PENALTY_PER_2
        defensive_xp = cs_prob * CLEAN_SHEET_POINTS + defcon_prob * DEFCON_POINTS - conceded_penalty
        total = minutes_fraction * (defensive_xp + attack_xp + bonus_xp) + appearance_xp

    elif player.element_type == 3:  # MID
        cs_prob = _team_cs_probability(fixture_difficulty)
        if player.defensive_contribution_per_90 > 0.0:
            defcon_prob = _poisson_at_least(player.defensive_contribution_per_90, DEFCON_THRESHOLD)
        else:
            defcon_prob = _fallback_defcon_prob(player, cs_prob, is_def_mid)
        goal_pts = GOAL_POINTS_BY_POSITION[3]
        attack_xp = (player.xg_per_90 * goal_pts + player.xa_per_90 * ASSIST_POINTS) * fixture_difficulty_multiplier(fixture_difficulty)
        saves_xp = 0.0
        bonus_xp = (player.defensive_contribution_per_90 / CBI_BPS_ACTIONS_PER_BONUS_UNIT) * BONUS_XP_PER_CBI_UNIT
        defensive_xp = cs_prob * MID_CLEAN_SHEET_POINTS + defcon_prob * DEFCON_POINTS
        total = minutes_fraction * (defensive_xp + attack_xp + bonus_xp) + appearance_xp

    else:  # FWD
        cs_prob = 0.0
        defcon_prob = 0.0
        saves_xp = 0.0
        goal_pts = GOAL_POINTS_BY_POSITION[4]
        attack_xp = (player.xg_per_90 * goal_pts + player.xa_per_90 * ASSIST_POINTS) * fixture_difficulty_multiplier(fixture_difficulty)
        bonus_xp = player.xg_per_90 * FWD_BONUS_XP_PER_XG
        defensive_xp = 0.0
        total = minutes_fraction * (attack_xp + bonus_xp) + appearance_xp

    return XPBreakdown(
        total=round(total, 3),
        attack_xp=round(attack_xp, 3),
        defensive_xp=round(defensive_xp, 3),
        saves_xp=round(saves_xp, 3),
        bonus_xp=round(bonus_xp, 3),
        appearance_xp=round(appearance_xp, 3),
        cs_prob=round(cs_prob, 3),
        defcon_prob=round(defcon_prob, 3),
        minutes_factor=round(minutes_fraction, 3),
    )


def apply_preseason_adjustment(player, breakdown: XPBreakdown, adjustment: Optional[dict]):
    """Layers a saved Pre-Season Scouting & Overrides entry (see database.preseason_adjustments
    and the app.py sidebar of the same name) on top of the model's own positional xP -- mirroring
    how ensemble/external xP is layered on top of calculate_positional_xp's raw output elsewhere
    in this module (fetch_players, transfer_planner.fetch_multi_gw_projections), so this runs
    AFTER any ensemble override and adjusts whatever total is currently in `breakdown`, not only
    the internal model's own figure.

    Returns (player, breakdown) since the penalty-duty override also corrects player.penalties_order
    itself so has_set_piece_duty/set_piece_label rationale and UI read the right role, not just an
    adjusted xP total. No-ops entirely (returns the inputs unchanged) when there's no saved
    adjustment for this player -- the common case for the vast majority of the player pool.
    """
    if not adjustment:
        return player, breakdown

    total = breakdown.total
    attack_xp = breakdown.attack_xp

    if adjustment.get("is_out_of_position"):
        boost = attack_xp * PRESEASON_OOP_ATTACK_BOOST
        attack_xp += boost
        total += boost

    if adjustment.get("preseason_penalties") and player.penalties_order != 1:
        player = replace(player, penalties_order=1)
        total += PRESEASON_PENALTY_XP_BOOST

    if adjustment.get("preseason_set_pieces") and player.corners_order != 1:
        # Role-only correction for has_set_piece_duty/set_piece_label rationale text -- no direct
        # xP boost here, unlike penalties: a corner/free-kick taker's extra assist upside is
        # already reflected in xA_per_90 once real season data exists, so there's no clean,
        # non-double-counting flat figure to add pre-season the way there is for penalties.
        player = replace(player, corners_order=1)

    if total != breakdown.total or attack_xp != breakdown.attack_xp:
        breakdown = replace(breakdown, total=round(total, 3), attack_xp=round(attack_xp, 3))

    return player, breakdown


def blend_external_xp(
    breakdown: XPBreakdown, external_xp: Optional[float], weight: float = EXTERNAL_XP_BLEND_WEIGHT
) -> XPBreakdown:
    """Blends an uploaded-CSV xP figure into this module's own positional xP total. Only `total`
    changes -- the attack/defensive/saves/bonus/appearance sub-components stay the internal
    model's own numbers, since we have no equivalent breakdown for the external figure to blend
    into. Returns `breakdown` unchanged (external_xp=None, blended=False) when there's nothing
    to blend, e.g. no upload, or no row for this player+gameweek."""
    if external_xp is None:
        return breakdown
    blended_total = round(weight * external_xp + (1 - weight) * breakdown.total, 3)
    return replace(breakdown, total=blended_total, external_xp=round(external_xp, 3), blended=True)


# --- Recent-form rolling window & prior-season cold-start rate ------------------------------
# Both read from the optional player_gw_history/player_season_history tables (see
# fpl_api.sync_player_history -- a separate, opt-in sync step, since it's one HTTP request per
# player with no bulk equivalent). Both tables being empty (sync never run) is the default,
# always-safe state: every function below degrades to returning nothing, leaving fetch_players'
# existing flat cumulative-season xg_per_90/xa_per_90 exactly as it always was.

RECENT_FORM_WINDOW_GAMES = 5  # rolling window size for recent_form_rate
RECENT_FORM_MIN_GAMES = 3  # fewer real-minutes games than this in the window is itself too thin
# a sample to trust over the flat season-long cumulative rate -- falls back to that instead.


def recent_form_rate(gw_rows: list, window: int = RECENT_FORM_WINDOW_GAMES) -> Optional[tuple]:
    """From a player's this-season per-gameweek rows (each a dict with 'minutes',
    'expected_goals', 'expected_assists', in ascending round order), computes (xg_per_90,
    xa_per_90) over just the last `window` games with real minutes -- a recency-weighted
    alternative to the flat season-long cumulative average, which dilutes a genuine recent role
    change (a new penalty taker, a formation switch, returning sharper after injury) by however
    many earlier games don't reflect it. Returns None when fewer than RECENT_FORM_MIN_GAMES real
    games exist in that window, so callers know to keep the flat season-long rate instead."""
    real_games = [r for r in gw_rows if (r.get("minutes") or 0) > 0][-window:]
    if len(real_games) < RECENT_FORM_MIN_GAMES:
        return None
    total_minutes = sum(r["minutes"] for r in real_games)
    if total_minutes <= 0:
        return None
    total_xg = sum(r.get("expected_goals") or 0.0 for r in real_games)
    total_xa = sum(r.get("expected_assists") or 0.0 for r in real_games)
    return round(total_xg / total_minutes * 90, 4), round(total_xa / total_minutes * 90, 4)


def recent_form_by_player_lookup(conn) -> dict:
    """player_id -> (xg_per_90, xa_per_90) rolling-window rate (see recent_form_rate), for every
    player with enough recent real-minutes games in player_gw_history -- {} (empty) when that
    table hasn't been populated yet (sync_player_history never run)."""
    rows = conn.execute(
        "SELECT player_id, round, minutes, expected_goals, expected_assists "
        "FROM player_gw_history ORDER BY player_id, round"
    ).fetchall()
    by_player: dict = {}
    for row in rows:
        by_player.setdefault(row["player_id"], []).append(dict(row))
    result = {}
    for player_id, gw_rows in by_player.items():
        rate = recent_form_rate(gw_rows)
        if rate is not None:
            result[player_id] = rate
    return result


def last_season_rate_by_player_lookup(conn) -> dict:
    """player_id -> (xg_per_90, xa_per_90) from player_season_history's stored prior season -- a
    genuine, personal cold-start prior for calculate_positional_xp's games_played==0 branch,
    instead of the flat 0.0 that branch would otherwise see. {} (empty) when that table hasn't
    been populated yet (sync_player_history never run)."""
    rows = conn.execute("SELECT player_id, minutes, expected_goals, expected_assists FROM player_season_history").fetchall()
    result = {}
    for row in rows:
        minutes = row["minutes"] or 0
        if minutes <= 0:
            continue
        result[row["player_id"]] = (
            round((row["expected_goals"] or 0.0) / minutes * 90, 4),
            round((row["expected_assists"] or 0.0) / minutes * 90, 4),
        )
    return result


def fetch_players(conn, ensemble_weights: Optional[dict] = None) -> list:
    """Load all selectable players with team info and a positional xP projection for the target
    gameweek (see calculate_positional_xp). For any player covered by an uploaded external CSV
    projection (see src/projections.py) for that same gameweek, the internal figure is replaced
    by the weighted ensemble across whichever sources are available (see ensemble_from_sources);
    ensemble_weights defaults to DEFAULT_ENSEMBLE_WEIGHTS when not given.

    Every player's total is then scaled to an "Effective xP" by their projected starting minutes
    for the gameweek (xmins/90 -- see calculate_baseline_xmins), whether that xmins figure comes
    from an uploaded CSV or the internal baseline. This is what keeps a 2nd/3rd-choice rotation
    asset's raw score from looking better than it should; solve_starting_xi separately applies a
    hard floor so one can't be picked into the XI outright regardless of its (now-discounted) xP.
    """
    ensemble_weights = ensemble_weights or DEFAULT_ENSEMBLE_WEIGHTS
    event_id = _get_target_event_id(conn)
    difficulties = _team_fixture_difficulties(conn, event_id)
    home_by_team = _team_home_fixture(conn, event_id)
    ensemble_xp_by_player = _ensemble_xp_lookup(conn, event_id, ensemble_weights)
    xmins_ensemble_by_player = _xmins_ensemble_lookup(conn, event_id, ensemble_weights)
    games_played_by_team = team_games_played(conn)
    preseason_by_player = database.get_preseason_adjustments(conn)
    recent_form_by_player = recent_form_by_player_lookup(conn)
    last_season_rate_by_player = last_season_rate_by_player_lookup(conn)

    rows = conn.execute(
        """
        SELECT p.id, p.web_name, p.team_id, t.name AS team_name, p.element_type, p.now_cost,
               p.selected_by_percent, p.form, p.total_points, p.ep_next, p.status, p.news,
               p.xg_per_90, p.xa_per_90, p.saves_per_90, p.defensive_contribution_per_90, p.starts_per_90,
               p.starts, p.chance_of_playing_next_round, p.penalties_order, p.corners_order,
               p.expected_goals_conceded_per_90
        FROM players p
        JOIN teams t ON t.id = p.team_id
        WHERE p.status != 'u'
        """
    ).fetchall()

    players = []
    for row in rows:
        team_difficulties = difficulties.get(row["team_id"])
        has_fixture = bool(team_difficulties)
        avg_difficulty = (
            sum(team_difficulties) / len(team_difficulties) if has_fixture else NEUTRAL_FIXTURE_DIFFICULTY
        )

        # Recent-form rolling window / prior-season cold-start prior -- both optional (empty
        # dicts, i.e. today's behavior, whenever sync_player_history hasn't been run): a true
        # cold start (this player's TEAM hasn't played yet) prefers their own real last-season
        # rate over the flat 0.0 the cumulative-season columns would otherwise carry; once real
        # in-season games exist, a recent-games rolling window (if there's enough of a sample)
        # takes over from the flat season-long cumulative average instead -- see
        # recent_form_rate/last_season_rate_by_player_lookup's own docstrings for why either wins out.
        team_games = games_played_by_team.get(row["team_id"], 0)
        xg_per_90, xa_per_90 = row["xg_per_90"] or 0.0, row["xa_per_90"] or 0.0
        if team_games == 0 and row["id"] in last_season_rate_by_player:
            xg_per_90, xa_per_90 = last_season_rate_by_player[row["id"]]
        elif row["id"] in recent_form_by_player:
            xg_per_90, xa_per_90 = recent_form_by_player[row["id"]]

        player = PlayerRow(
            id=row["id"],
            web_name=row["web_name"],
            team_id=row["team_id"],
            team_name=row["team_name"],
            element_type=row["element_type"],
            now_cost=row["now_cost"],
            selected_by_percent=row["selected_by_percent"] or 0.0,
            form=row["form"] or 0.0,
            total_points=row["total_points"] or 0,
            ep_next=row["ep_next"],
            xg_per_90=xg_per_90,
            xa_per_90=xa_per_90,
            saves_per_90=row["saves_per_90"] or 0.0,
            defensive_contribution_per_90=row["defensive_contribution_per_90"] or 0.0,
            starts_per_90=row["starts_per_90"] or 0.0,
            status=row["status"],
            fixture_difficulty=avg_difficulty,
            has_fixture=has_fixture,
            projected_xp=0.0,
            starts=row["starts"] or 0,
            chance_of_playing_next_round=row["chance_of_playing_next_round"],
            news=row["news"] or "",
            penalties_order=row["penalties_order"],
            corners_order=row["corners_order"],
            expected_goals_conceded_per_90=row["expected_goals_conceded_per_90"] or 0.0,
            is_home=home_by_team.get(row["team_id"]),
        )
        breakdown = calculate_positional_xp(
            player, avg_difficulty if has_fixture else NEUTRAL_FIXTURE_DIFFICULTY,
            games_played_by_team.get(player.team_id, 0),
        )
        ensemble_xp = ensemble_xp_by_player.get(player.id)
        if ensemble_xp is not None:
            breakdown = replace(breakdown, total=ensemble_xp, external_xp=ensemble_xp, blended=True)
        else:
            breakdown = blend_ep_next_fallback(breakdown, player.ep_next, player.starts)

        adjustment = preseason_by_player.get(player.id)
        player, breakdown = apply_preseason_adjustment(player, breakdown, adjustment)

        baseline_xmins = calculate_baseline_xmins(player, games_played_by_team.get(player.team_id, 0))
        custom_xmins = adjustment.get("custom_xmins_override") if adjustment else None
        if custom_xmins is not None:
            player.xmins = custom_xmins  # pre-season manual override wins over both baseline and ensemble xmins
        else:
            player.xmins = xmins_ensemble_by_player.get(player.id, baseline_xmins)

        effective_total = round(breakdown.total * (player.xmins / 90.0), 3)
        player.projected_xp = effective_total
        player.xp_breakdown = replace(breakdown, total=effective_total)
        players.append(player)
    return players


# --- 15-man squad builder ----------------------------------------------------

def solve_squad(
    players: list,
    budget: int = BUDGET_LIMIT,
    max_per_team: int = MAX_PLAYERS_PER_TEAM,
    position_counts: Optional[dict] = None,
    objective_attr: str = "projected_xp",
    locked_ids: Optional[set] = None,
    excluded_ids: Optional[set] = None,
) -> list:
    """Solve for the 15-man squad that maximizes `objective_attr` subject to budget,
    exact position counts, and a per-team cap."""
    position_counts = position_counts or SQUAD_POSITION_COUNTS
    locked_ids = locked_ids or set()
    excluded_ids = excluded_ids or set()

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    pick = {p.id: pulp.LpVariable(f"pick_{p.id}", cat="Binary") for p in players}

    prob += pulp.lpSum(pick[p.id] * getattr(p, objective_attr) for p in players)

    prob += pulp.lpSum(pick[p.id] * p.now_cost for p in players) <= budget

    for element_type, count in position_counts.items():
        prob += pulp.lpSum(pick[p.id] for p in players if p.element_type == element_type) == count

    team_ids = {p.team_id for p in players}
    for team_id in team_ids:
        prob += pulp.lpSum(pick[p.id] for p in players if p.team_id == team_id) <= max_per_team

    for pid in locked_ids:
        if pid in pick:
            prob += pick[pid] == 1
    for pid in excluded_ids:
        if pid in pick:
            prob += pick[pid] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError(f"Squad solver returned status: {pulp.LpStatus[status]}")

    return [p for p in players if pulp.value(pick[p.id]) > 0.5]


def solve_template_squad(players: list, **kwargs) -> list:
    """Highest-ownership legal squad within budget/position/club constraints."""
    return solve_squad(players, objective_attr="selected_by_percent", **kwargs)


def solve_differential_squad(
    players: list, ownership_threshold: float = DIFFERENTIAL_OWNERSHIP_THRESHOLD, **kwargs
) -> list:
    """Max-xP legal squad restricted to low-ownership (differential) players."""
    pool = [p for p in players if p.selected_by_percent < ownership_threshold]
    return solve_squad(pool, objective_attr="projected_xp", **kwargs)


def solve_squad_with_captaincy(
    players: list,
    budget: int = BUDGET_LIMIT,
    max_per_team: int = MAX_PLAYERS_PER_TEAM,
    position_counts: Optional[dict] = None,
    locked_ids: Optional[set] = None,
    excluded_ids: Optional[set] = None,
    risk_lambda: float = 0.0,
    formation_lock: Optional[str] = None,
) -> list:
    """The 15-man squad that maximizes real FPL Expected Value -- Starting XI xP, the captaincy
    doubling bonus, vice-captain insurance, and weighted bench auto-sub value, all solved jointly
    (see _solve_lineup_milp) -- instead of the old flat sum(xP) objective, which had no way to
    recognize that a lower-flat-xP-but-elite-ceiling asset (e.g. Haaland/Salah) is worth rostering
    specifically for the captaincy option it opens up.

    formation_lock (see FORMATION_CHOICES) pins the resulting Starting XI's outfield shape to an
    exact DEF/MID/FWD split instead of the default flexible bounds -- since squad and Starting XI
    are chosen jointly here, a lock can change which 15 are worth rostering at all, not just which
    11 of an already-fixed 15 start.

    Used for build_optimal_squad's 'balanced' mode; solve_template_squad/solve_differential_squad
    keep the plain flat-objective solve_squad, since their objectives (max ownership / max xP
    restricted to low-ownership players) aren't about captaincy-driven squad value in the same way
    a general-purpose "best squad" build is.
    """
    result = _solve_lineup_milp(
        players, squad_ids=None, budget=budget, max_per_team=max_per_team,
        position_counts=position_counts, locked_ids=locked_ids, excluded_ids=excluded_ids,
        risk_lambda=risk_lambda, formation_lock=formation_lock,
    )
    return result["squad"]


def build_optimal_squad(
    conn, mode: str = "balanced", ensemble_weights: Optional[dict] = None, risk_lambda: float = 0.0, **kwargs
) -> list:
    """mode: 'balanced' (real FPL EV -- captaincy/bench/EO-aware, see solve_squad_with_captaincy),
    'template' (max ownership), 'differential' (max xP, low ownership). risk_lambda (see
    RISK_PROFILE_LAMBDA) only affects 'balanced' mode. A formation_lock kwarg (see
    FORMATION_CHOICES) is only valid for 'balanced' mode too -- solve_template_squad/
    solve_differential_squad use the plain flat-objective solve_squad, which has no Starting XI
    concept (and so no formation) at all; pass formation_lock only when mode='balanced'."""
    players = fetch_players(conn, ensemble_weights=ensemble_weights)
    if mode == "balanced":
        return solve_squad_with_captaincy(players, risk_lambda=risk_lambda, **kwargs)
    if mode == "template":
        return solve_template_squad(players, **kwargs)
    if mode == "differential":
        return solve_differential_squad(players, **kwargs)
    raise ValueError(f"Unknown mode: {mode!r}")


# --- Starting XI, formation, and bench ordering ------------------------------

def _is_minutes_secure(p, floor: float = SUB1_XMINS_FLOOR) -> bool:
    """True if a player clears the Sub 1 Security bar (see _solve_lineup_milp's Step 1
    constraint): either their projected starting minutes (xMins) are at/above `floor`, or the
    live FPL API explicitly reports chance_of_playing_next_round == 100.

    Deliberately a strict equality, NOT also treating a null chance_of_playing_next_round as
    matching (unlike calculate_baseline_xmins' own fallback, which does -- see
    CHANCE_OF_PLAYING_DEFAULT): null just means "no active fitness doubt," which is true of most
    of the player pool including plenty of fringe/rotation players, so treating it as satisfying
    a "high minutes security" bar here would make this check nearly a no-op. xMins is already the
    more complete signal (it factors null-chance in via the baseline formula's own starts-rate
    calculation); this OR-clause exists only to catch the narrower case of an explicit 100%
    fitness confirmation from the API standing in for a still-thin xMins sample.
    """
    xmins = getattr(p, "xmins", 90.0)
    chance = getattr(p, "chance_of_playing_next_round", None)
    return xmins >= floor or chance == CHANCE_OF_PLAYING_DEFAULT


def _is_hard_excluded(p) -> bool:
    """True if the Press Conference / Injury Flag Gatekeeper hard-excludes this player from the
    Starting XI and from ever being labeled Sub 1: chance_of_playing_next_round == 0 (officially
    ruled out) or an unavailable status (HARD_EXCLUDE_STATUSES). Unconditional -- unlike
    min_starter_xmins, this doesn't depend on the user's Starter Security profile at all."""
    chance = getattr(p, "chance_of_playing_next_round", None)
    status = getattr(p, "status", "a")
    return chance == 0 or status in HARD_EXCLUDE_STATUSES


def is_vice_eligible(p) -> bool:
    """True if a player satisfies the Vice-Captain Lock -- see the constant block above for why
    null is treated as equivalent to chance_of_playing_next_round == 100 here. Also requires
    is_captaincy_eligible (the Captaincy Position & Talisman gate): the armband -- captain AND
    vice alike -- never goes to a GKP or a non-set-piece-duty DEF, no matter how safe their
    minutes are. Public (not underscore-prefixed) since transfer_planner.captain_pick_for_gw also
    needs it -- matching the established convention of promoting a helper to a public name once a
    second module needs it (see xgi_per_90, has_set_piece_duty, set_piece_label)."""
    if not is_captaincy_eligible(p):
        return False
    xmins = getattr(p, "xmins", 90.0)
    chance = getattr(p, "chance_of_playing_next_round", None)
    return xmins >= VICE_CAPTAIN_XMINS_FLOOR and (chance is None or chance == CHANCE_OF_PLAYING_DEFAULT)


def order_bench(bench: list) -> list:
    """Step 2 of the Two-Stage Bench Allocation (Step 1 is _solve_lineup_milp's Sub 1 Security
    Constraint, which runs *before* this and guarantees at least one outfield bench player clears
    _is_minutes_secure): post-solve bench auto-ordering. The backup goalkeeper always fills Sub
    GKP; the remaining outfield bench players are sorted strictly by descending projected_xp --
    Sub 1 highest, Sub 3 lowest (the pure budget enabler) -- with no further minutes-security
    tie-break here, since that's already been guaranteed upstream in Step 1.

    The one exception: a player the Injury Flag Gatekeeper hard-excludes (_is_hard_excluded) is
    always sorted to the back regardless of their (should-be-near-zero, but not guaranteed) xP --
    Sub 1 must never be someone who's been officially ruled out or is injured/suspended."""
    goalkeepers = [p for p in bench if p.element_type == 1]
    outfield_sorted = sorted(
        (p for p in bench if p.element_type != 1),
        key=lambda p: (not _is_hard_excluded(p), p.projected_xp),
        reverse=True,
    )
    return goalkeepers + outfield_sorted


def _formation_label(starting_xi: list) -> str:
    counts = Counter(p.element_type for p in starting_xi)
    return f"{counts[2]}-{counts[3]}-{counts[4]}"


# --- Formation Selector & Hard Shape Lock -------------------------------------------------------

AUTO_FORMATION_LABEL = "Auto (Best xP)"

# Every legal outfield shape (DEF-MID-FWD, summing to 10) real FPL allows for a Starting XI,
# alongside the default flexible-bounds "Auto" option -- exposed as-is to app.py's sidebar
# selectbox rather than deriving it from DEF/MID/FWD min/max, since not every combination within
# those flexible bounds (e.g. 5-5-... isn't possible at all, only 10 outfield slots exist) is a
# formation real managers actually pick between.
FORMATION_CHOICES = [
    AUTO_FORMATION_LABEL, "3-5-2", "3-4-3", "4-4-2", "4-3-3", "4-5-1", "5-3-2", "5-4-1", "5-2-3",
]


def parse_formation_lock(formation_choice: Optional[str]) -> Optional[tuple]:
    """Parses a 'DEF-MID-FWD' formation label (e.g. '3-5-2') into an exact (def, mid, fwd)
    starting-count tuple for _solve_lineup_milp's Hard Shape Lock. Returns None for
    AUTO_FORMATION_LABEL (or anything falsy) -- the caller's default flexible bounds (DEF 3-5,
    MID 2-5, FWD 1-3) apply instead. GKP is always exactly 1 and isn't part of the label.

    Raises OptimizationError on a malformed label rather than silently falling back to Auto,
    since a typo'd formation string should surface immediately, not quietly solve as if no lock
    were requested at all."""
    if not formation_choice or formation_choice == AUTO_FORMATION_LABEL:
        return None
    parts = formation_choice.split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise OptimizationError(f"Invalid formation lock: {formation_choice!r}")
    def_n, mid_n, fwd_n = (int(part) for part in parts)
    return def_n, mid_n, fwd_n


def _effective_ownership_scores(pool: list) -> dict:
    """EO_Score_i = (selected_by_percent_i / 100) x Captain_Likelihood_i -- feeds the risk_lambda
    rank-protection term in _solve_lineup_milp. Captain_Likelihood_i is proxied as this
    candidate's projected_xp relative to the single highest projected_xp in the same pool (1.0
    for the pool's most likely captaincy pick, scaling down for lesser options): there's no real
    "% of managers who captained X" field anywhere in the data model, and the pool's own xP
    ranking is the most defensible in-pool stand-in for it -- real-world captaincy choices are
    themselves overwhelmingly driven by projected returns."""
    max_xp = max((p.projected_xp for p in pool), default=0.0)
    if max_xp <= 0:
        return {p.id: 0.0 for p in pool}
    return {p.id: (p.selected_by_percent / 100.0) * (p.projected_xp / max_xp) for p in pool}


def _solve_lineup_milp(
    pool: list,
    *,
    squad_ids: Optional[set] = None,
    budget: int = BUDGET_LIMIT,
    max_per_team: int = MAX_PLAYERS_PER_TEAM,
    position_counts: Optional[dict] = None,
    locked_ids: Optional[set] = None,
    excluded_ids: Optional[set] = None,
    min_starter_xmins: Optional[float] = None,
    risk_lambda: float = 0.0,
    formation_lock: Optional[str] = None,
) -> dict:
    """The joint MILP behind both solve_squad_with_captaincy (squad_ids=None -- squad membership
    x is a free decision variable, budget/position/club-constrained) and solve_starting_xi
    (squad_ids given -- x is fixed to an already-owned 15). One combined objective picks Starting
    XI (s), Captain (c), and Vice-Captain (v) together, instead of each in a separate post-hoc
    stage -- see CAPTAIN_XP_BONUS_MULTIPLIER / VICE_XP_INSURANCE_WEIGHT / RISK_PROFILE_LAMBDA for
    the exact formulation.

    c_i/v_i are real decision variables during the solve -- their bonus contribution has to be
    able to influence which players get selected as starters in the first place, which is the
    entire point of this rewrite -- but aren't returned: given the OPTIMAL Starting XI, the
    optimal captain/vice are provably just the top-2 by captaincy_score among is_captaincy_eligible
    candidates within it (the objective's c/v terms are monotonic in captaincy_score, gated by the
    same hard eligibility constraint applied here, with no coupling to anything else), so
    get_captain_recommendations/captain_pick_for_gw's own equivalent eligible-and-scored ranking
    already recovers them correctly with no further changes needed there. The one acknowledged
    asymmetry: the GW1 Cold-Start Anchor's "premium OR top-N" refinement (captaincy_candidates) is
    a pool-composition-dependent computation this function only ever applies as a SOFT scoring
    nudge (via cold_start/c_score above), while get_captain_recommendations/captain_pick_for_gw
    apply it as a hard restriction -- scoped to the already-decided Starting XI there, where (unlike
    here, pre-solve, over a squad/universe that hasn't been chosen yet) it's provably always
    non-empty. See captaincy_candidates' own docstring.

    Bench allocation is a deliberate Two-Stage design, not part of this objective:
      - Step 1 (here): a Sub 1 Security Constraint requires at least one outfield squad member
        NOT in the Starting XI to clear _is_minutes_secure (xMins >= SUB1_XMINS_FLOOR, or no
        fitness doubt at all) -- guaranteeing a real, minutes-secure auto-sub option exists on the
        bench somewhere, without dictating which bench slot they end up labeled as. Silently
        skipped (not enforced) when no candidate in the whole pool could ever satisfy it, rather
        than forcing an unsatisfiable constraint.
      - Step 2 (order_bench, called by the callers below): post-solve, sorts the resulting bench
        purely by descending projected_xp -- see order_bench's own docstring.

    formation_lock (see FORMATION_CHOICES/parse_formation_lock): a 'DEF-MID-FWD' label like
    '3-5-2' pins the Starting XI's outfield shape to that exact split instead of the default
    flexible bounds (DEF 3-5, MID 2-5, FWD 1-3) -- None or AUTO_FORMATION_LABEL leaves the
    flexible bounds in place.

    Returns {"squad": list[PlayerRow] (only when squad_ids is None), "starting_xi": [...],
    "bench": [...], "formation": str}.
    """
    position_counts = position_counts or SQUAD_POSITION_COUNTS
    locked_ids = locked_ids or set()
    excluded_ids = excluded_ids or set()
    squad_fixed = squad_ids is not None

    prob = pulp.LpProblem("fpl_joint_lineup", pulp.LpMaximize)
    x = {} if squad_fixed else {p.id: pulp.LpVariable(f"x_{p.id}", cat="Binary") for p in pool}
    s = {p.id: pulp.LpVariable(f"s_{p.id}", cat="Binary") for p in pool}
    c = {p.id: pulp.LpVariable(f"c_{p.id}", cat="Binary") for p in pool}
    v = {p.id: pulp.LpVariable(f"v_{p.id}", cat="Binary") for p in pool}
    outfield = [p for p in pool if p.element_type != 1]

    def x_of(pid):
        return 1 if squad_fixed else x[pid]

    eo_scores = _effective_ownership_scores(pool)
    # Captaincy Position & Talisman Filtering / GW1 Pre-Season Cold-Start Anchor (see the constant
    # block above is_captaincy_eligible): cold_start is a soft signal here (folds into
    # captaincy_score's price nudge, via c_score/v_score below) -- the harder "premium OR top-N"
    # cold-start restriction only ever applies post-hoc, to an already-decided Starting XI (see
    # captaincy_candidates' docstring for why doing it here would risk infeasibility).
    cold_start = is_cold_start_pool(pool)
    c_score = {p.id: captaincy_score(p, cold_start) for p in pool}

    objective = pulp.lpSum(s[p.id] * p.projected_xp for p in pool)
    objective += CAPTAIN_XP_BONUS_MULTIPLIER * pulp.lpSum(c[p.id] * c_score[p.id] for p in pool)
    objective += VICE_XP_INSURANCE_WEIGHT * pulp.lpSum(v[p.id] * c_score[p.id] for p in pool)
    if risk_lambda:
        objective += risk_lambda * pulp.lpSum(s[p.id] * eo_scores.get(p.id, 0.0) for p in pool)
    prob += objective

    if squad_fixed:
        for p in pool:
            if p.id not in squad_ids:
                raise OptimizationError(f"Player id {p.id} passed to a fixed-squad lineup solve isn't in squad_ids.")
    else:
        prob += pulp.lpSum(x[p.id] * p.now_cost for p in pool) <= budget
        for element_type, count in position_counts.items():
            prob += pulp.lpSum(x[p.id] for p in pool if p.element_type == element_type) == count
        team_ids = {p.team_id for p in pool}
        for team_id in team_ids:
            prob += pulp.lpSum(x[p.id] for p in pool if p.team_id == team_id) <= max_per_team
        for pid in locked_ids:
            if pid in x:
                prob += x[pid] == 1
        for pid in excluded_ids:
            if pid in x:
                prob += x[pid] == 0
        for p in pool:
            prob += s[p.id] <= x[p.id]

    for p in pool:
        prob += c[p.id] <= s[p.id]
        prob += v[p.id] <= s[p.id]
        prob += c[p.id] + v[p.id] <= 1
        # Captaincy Position & Talisman Filtering: the BASE position/set-piece gate is a hard
        # constraint here (always feasible -- a legal Starting XI's formation floor guarantees
        # >= 3 base-eligible MID/FWD candidates regardless of formation_lock). The stricter
        # cold-start-only "premium or top-N" restriction is deliberately NOT enforced here -- see
        # the cold_start/c_score comment above and captaincy_candidates' docstring.
        if not is_captaincy_eligible(p):
            prob += c[p.id] == 0
            prob += v[p.id] == 0

    # Starting XI formation: 1 GKP always, plus either the default flexible outfield bounds
    # (3-5 DEF, 2-5 MID, 1-3 FWD) or, when formation_lock pins a specific shape (e.g. "3-5-2"),
    # an exact DEF/MID/FWD split instead -- see parse_formation_lock/FORMATION_CHOICES.
    def_count = pulp.lpSum(s[p.id] for p in pool if p.element_type == 2)
    mid_count = pulp.lpSum(s[p.id] for p in pool if p.element_type == 3)
    fwd_count = pulp.lpSum(s[p.id] for p in pool if p.element_type == 4)
    prob += pulp.lpSum(s[p.id] for p in pool) == 11
    prob += pulp.lpSum(s[p.id] for p in pool if p.element_type == 1) == 1
    locked_shape = parse_formation_lock(formation_lock)
    if locked_shape is not None:
        locked_def, locked_mid, locked_fwd = locked_shape
        prob += def_count == locked_def
        prob += mid_count == locked_mid
        prob += fwd_count == locked_fwd
    else:
        prob += def_count >= 3
        prob += def_count <= 5
        prob += mid_count >= 2
        prob += mid_count <= 5
        prob += fwd_count >= 1
        prob += fwd_count <= 3
    prob += pulp.lpSum(c[p.id] for p in pool) == 1
    prob += pulp.lpSum(v[p.id] for p in pool) == 1

    # Step 1 -- Sub 1 Security Constraint: at least one benched outfield player must clear
    # _is_minutes_secure (and not also be hard-excluded -- see below). Skipped entirely if no
    # candidate anywhere in the pool could ever satisfy it (an empty sum >= 1 is unsatisfiable by
    # construction).
    secure_outfield = [p for p in outfield if _is_minutes_secure(p) and not _is_hard_excluded(p)]
    if secure_outfield:
        prob += pulp.lpSum(x_of(p.id) - s[p.id] for p in secure_outfield) >= 1

    # Press Conference / Injury Flag Gatekeeper: unconditional, independent of min_starter_xmins
    # and the user's own Starter Security profile -- an officially-ruled-out or
    # injured/suspended/unavailable player is never a Starting XI candidate.
    for p in pool:
        if _is_hard_excluded(p):
            prob += s[p.id] == 0

    if min_starter_xmins is not None:
        for p in pool:
            if getattr(p, "xmins", 90.0) < min_starter_xmins:
                prob += s[p.id] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        floor_note = (
            f" -- try a lower Starter Security floor (currently {min_starter_xmins} xMins)."
            if min_starter_xmins else ""
        )
        formation_note = (
            f" -- try Auto (Best xP) instead of the {formation_lock} Formation Lock."
            if locked_shape is not None else ""
        )
        kind = "Starting XI" if squad_fixed else "Squad"
        raise OptimizationError(f"{kind} solver returned status: {pulp.LpStatus[status]}{floor_note}{formation_note}")

    squad_result = list(pool) if squad_fixed else [p for p in pool if pulp.value(x[p.id]) > 0.5]
    starting_ids = {p.id for p in pool if pulp.value(s[p.id]) > 0.5}
    starting_xi = [p for p in squad_result if p.id in starting_ids]
    bench_pool = [p for p in squad_result if p.id not in starting_ids]

    # Step 2 -- post-solve bench auto-ordering (see order_bench).
    bench = order_bench(bench_pool)

    result = {"starting_xi": starting_xi, "bench": bench, "formation": _formation_label(starting_xi)}
    if not squad_fixed:
        result["squad"] = squad_result
    return result


def solve_starting_xi(
    squad: list,
    min_starter_xmins: Optional[float] = None,
    risk_lambda: float = 0.0,
    formation_lock: Optional[str] = None,
):
    """From a fixed 15-man squad, jointly picks the Starting XI and (internally) Captain/
    Vice-Captain in one combined MILP (see _solve_lineup_milp), then applies the Two-Stage Bench
    Allocation's Step 2 (order_bench) to label the bench. Captain/vice aren't returned by this
    function -- get_captain_recommendations/captain_pick_for_gw recover them from the returned
    starting_xi via a simple top-2-by-xP sort, which is provably identical to the joint solve's
    own choice (see _solve_lineup_milp).

    When min_starter_xmins is given (see STARTER_SECURITY_PROFILES), any squad member projected
    below that many starting minutes is hard-excluded from the XI -- this is what stops a
    2nd/3rd-choice rotation asset (e.g. a backup GKP) from being picked into the 11 outright, on
    top of their xP already being discounted by the same xmins figure (see fetch_players'
    "Effective xP" scaling). risk_lambda (see RISK_PROFILE_LAMBDA) trades some pure xP for
    high-ownership/high-captain-likelihood template protection in which 11 get selected.
    formation_lock (see FORMATION_CHOICES) pins the outfield shape to an exact DEF/MID/FWD split
    (e.g. "3-5-2") instead of letting the solver pick freely within the default flexible bounds.

    Raises OptimizationError if the floor/lock makes the squad's formation infeasible (e.g. every
    rostered defender is currently a minutes risk, or the squad simply doesn't have enough fit
    players in one position to fill a locked shape) -- the message says so explicitly rather than
    just reporting a bare solver status, since the fix (lower the floor, or pick Auto) is specific
    and actionable.

    Returns (starting_xi, ordered_bench, formation_label).
    """
    result = _solve_lineup_milp(
        squad, squad_ids={p.id for p in squad}, min_starter_xmins=min_starter_xmins, risk_lambda=risk_lambda,
        formation_lock=formation_lock,
    )
    return result["starting_xi"], result["bench"], result["formation"]


# Every STARTER_SECURITY_PROFILES tier, strictest first, plus a final "no floor" resort -- see
# solve_starting_xi_with_fallback. src/backtest.py has its own equivalent chain (anchored at
# DEFAULT_STARTER_XMINS_FLOOR, since a 38-gameweek unattended run has no user to ask); this one is
# anchored at whatever floor the CALLER actually requested, since here there IS a human on the
# other end who chose a specific Starter Security profile on purpose.
_FULL_STARTER_FLOOR_CHAIN = (
    STARTER_SECURITY_PROFILES["conservative"], STARTER_SECURITY_PROFILES["balanced"],
    STARTER_SECURITY_PROFILES["aggressive"], None,
)


def solve_starting_xi_with_fallback(
    squad: list, min_starter_xmins: Optional[float] = None, risk_lambda: float = 0.0,
    formation_lock: Optional[str] = None,
):
    """solve_starting_xi, but degrades the minutes-security floor automatically instead of
    raising OptimizationError outright when the requested one makes the squad's XI infeasible --
    real case this fixes: several squad members genuinely below 60 xMins at once (thin/no recent
    minutes) can leave fewer than 11 players clearing even the "Balanced" floor, which used to
    just dead-end the whole page with a bare error and no way forward short of the user manually
    finding the sidebar's Starter Security control themselves.

    Tries min_starter_xmins first, then each STARTER_SECURITY_PROFILES tier strictly BELOW it (so
    a caller who already asked for "Aggressive" doesn't retry stricter floors that would only
    fail the same way), then finally no floor at all -- the same last-resort backtest.py's own
    fallback chain uses. formation_lock is intentionally NOT relaxed by this fallback (only the
    minutes floor is) -- an explicit formation choice should fail loudly, not silently change shape.

    Returns (starting_xi, ordered_bench, formation_label, floor_used, was_relaxed) -- floor_used
    is whichever floor actually worked (None if no floor was needed at all), and was_relaxed is
    True whenever that isn't the same floor the caller originally asked for, so callers can show a
    "had to relax your Starter Security setting" notice rather than silently ignoring it. Raises
    OptimizationError (same as solve_starting_xi) only if even no floor at all is infeasible --
    a genuine squad-construction problem (e.g. a formation_lock no legal 11 can satisfy), not a
    minutes-security one.
    """
    if min_starter_xmins is None:
        chain = [None]  # already the most permissive floor -- nothing stricter would help, nothing looser exists
    else:
        chain = [min_starter_xmins] + [
            floor for floor in _FULL_STARTER_FLOOR_CHAIN if floor is None or floor < min_starter_xmins
        ]
    last_error: Optional[OptimizationError] = None
    for floor in chain:
        try:
            starting_xi, bench, formation = solve_starting_xi(
                squad, min_starter_xmins=floor, risk_lambda=risk_lambda, formation_lock=formation_lock,
            )
            return starting_xi, bench, formation, floor, floor != min_starter_xmins
        except OptimizationError as exc:
            last_error = exc
            continue
    raise last_error or OptimizationError("Starting XI solver infeasible even with no minutes-security floor at all.")


def calculate_team_xp(starting_xi: list, captain: PlayerRow) -> float:
    """Team Starting XI xP = sum(starter xP) + the captain's xP again -- captaincy doubles
    their points, so one copy is already in the starters' sum and this is the second copy."""
    return round(sum(p.projected_xp for p in starting_xi) + captain.projected_xp, 3)


# --- Captain / vice-captain recommendations ----------------------------------

def minutes_security_multiplier(status: str) -> float:
    return STATUS_MINUTES_MULTIPLIER.get(status, 1.0)


def get_captain_recommendations(
    conn,
    squad_ids: list,
    ensemble_weights: Optional[dict] = None,
    min_starter_xmins: Optional[float] = None,
    risk_lambda: float = 0.0,
    formation_lock: Optional[str] = None,
) -> dict:
    """Captain = the highest captaincy_score among captaincy-eligible Starting XI players (see
    is_captaincy_eligible: MID/FWD always, DEF only with confirmed penalty/corner duty, GKP never
    -- and, during a GW1 Pre-Season Cold-Start window, captaincy_candidates further restricts the
    field to premium-priced or standout-projection candidates only, see its own docstring). Vice =
    the highest-scoring eligible Starting XI player (other than the captain) that ALSO clears the
    Vice-Captain Lock (is_vice_eligible: xMins >= VICE_CAPTAIN_XMINS_FLOOR and
    chance_of_playing_next_round in (None, 100)) -- falling back to the plain runner-up if nobody
    in the XI clears that bar, so a vice pick is still always returned. captaincy_score is plain
    projected_xp plus a talisman boost for a favorably-placed confirmed penalty taker (and, during
    a cold start, a premium-price nudge) -- minutes/fixture risk are already priced in upstream of
    projected_xp itself (the xMins "Effective xP" scaling and the min_starter_xmins hard floor --
    see fetch_players/solve_starting_xi -- and calculate_positional_xp's own fixture-aware terms),
    so no separate multiplier for either is layered on again here. risk_lambda and formation_lock
    (see RISK_PROFILE_LAMBDA/FORMATION_CHOICES) are passed straight through to solve_starting_xi --
    either can change WHICH 11 are in the XI (and so who's eligible here), but the top-scoring
    captain pick within that XI is unaffected by either directly (see _solve_lineup_milp's
    docstring for why).

    Only Starting XI players are eligible at all: you can't captain someone who isn't playing, so
    a squad member left on the bench -- whether by formation choice or a minutes-security
    exclusion -- is never a candidate here (an earlier version of this function scored the whole
    15-man squad, including the bench, which could recommend captaining someone who wasn't even
    starting that gameweek). A legal Starting XI always has >= 3 base-eligible MID/FWD (the
    formation floor guarantees it), so the eligible pool here can never come back empty.

    Also returns a high-ownership 'safe' pick and a low-ownership 'differential' pick, both still
    ranked by the same captaincy_score, for extra context beyond the raw top two.
    """
    players = fetch_players(conn, ensemble_weights=ensemble_weights)
    squad = [p for p in players if p.id in squad_ids]
    if not squad:
        raise OptimizationError("No matching players found for the given squad_ids.")

    starting_xi, _bench, _formation = solve_starting_xi(
        squad, min_starter_xmins=min_starter_xmins, risk_lambda=risk_lambda, formation_lock=formation_lock,
    )

    cold_start = is_cold_start_pool(starting_xi)
    eligible_ids = captaincy_candidates(starting_xi)
    scored = [
        {"player": p, "captain_score": captaincy_score(p, cold_start)}
        for p in starting_xi if p.id in eligible_ids
    ]
    scored.sort(key=lambda c: c["captain_score"], reverse=True)

    overall_best = scored[0]
    safe_candidates = [c for c in scored if c["player"].selected_by_percent >= DIFFERENTIAL_OWNERSHIP_THRESHOLD]
    diff_candidates = [c for c in scored if c["player"].selected_by_percent < DIFFERENTIAL_OWNERSHIP_THRESHOLD]
    safe_pick = safe_candidates[0] if safe_candidates else overall_best
    differential_pick = diff_candidates[0] if diff_candidates else None

    top_picks = []
    seen_ids = set()
    for candidate in (overall_best, safe_pick, differential_pick):
        if candidate is None or candidate["player"].id in seen_ids:
            continue
        top_picks.append(candidate)
        seen_ids.add(candidate["player"].id)
    for candidate in scored:
        if len(top_picks) >= 3:
            break
        if candidate["player"].id not in seen_ids:
            top_picks.append(candidate)
            seen_ids.add(candidate["player"].id)

    # Vice-Captain Lock: prefer the highest-scoring eligible non-captain starter; fall back to the
    # plain runner-up if nobody clears the (xMins + captaincy-position) eligibility bar (see
    # is_vice_eligible).
    eligible_vice = [
        c for c in scored if c["player"].id != overall_best["player"].id and is_vice_eligible(c["player"])
    ]
    if eligible_vice:
        vice_candidate = eligible_vice[0]
    else:
        vice_candidate = scored[1] if len(scored) > 1 else None

    return {
        "top_picks": top_picks,
        "captain": overall_best,
        "safe_pick": safe_pick,
        "differential_pick": differential_pick,
        "vice_captain": vice_candidate,
    }


# --- Strategy rationale generation -------------------------------------------

RATIONALE_KEY_PLAYER_COUNT = 3
RATIONALE_FIXTURE_RUN_FAVORABLE = 2.4  # avg fixture difficulty over the horizon at/below this reads as a good run
RATIONALE_FIXTURE_RUN_TOUGH = 3.6  # avg fixture difficulty over the horizon at/above this reads as a tough run
RATIONALE_HIGH_XGI_PER_90 = 0.5  # combined xG90+xA90 (xGI proxy) at/above this reads as elite underlying output
RATIONALE_MID_HEAVY_SPEND_PCT = 40.0  # % of squad budget in MID at/above this reads as a midfield-heavy build
RATIONALE_PREMIUM_COST_MIN = 90  # GBP 9.0m -- price floor for a "premium" attacking asset
RATIONALE_DOUBLE_PREMIUM_COUNT = 2  # >= this many premium MID/FWD reads as a "Double Premium Attack"
RATIONALE_DEF_VALUE_SPEND_PCT = 26.0  # % of squad budget in DEF at/above this reads as a defense-led value build
RATIONALE_ENABLER_COST_MAX = 45  # GBP 4.5m -- at/below this price, a bench player reads as a pure budget enabler


@dataclass
class RationaleBullet:
    """One line of generated strategy rationale, plus optional UI badge tags for app.py's
    Tactical Rationale sidebar (spec badges: "Captain", "Set Pieces", "Fixture Swing", "High xGI")."""
    text: str
    tags: list = field(default_factory=list)


def xgi_per_90(player) -> float:
    """xG + xA per 90 -- mathematically FPL's own xGI definition; there's no dedicated
    xgi_per_90 field on Player/PlayerRow, so this is computed rather than duplicated."""
    return round(player.xg_per_90 + player.xa_per_90, 3)


def has_set_piece_duty(player) -> bool:
    return player.penalties_order == 1 or player.corners_order == 1


def set_piece_label(player) -> str:
    parts = []
    if player.penalties_order == 1:
        parts.append("penalties")
    if player.corners_order == 1:
        parts.append("corners/indirect free-kicks")
    return " & ".join(parts)


def _multi_gw_fixture_difficulty(conn, player_ids: list, horizon_gws: int) -> dict:
    """Average fixture difficulty per player over the next `horizon_gws` gameweeks (only counting
    gameweeks the player actually has a fixture in), for genuine multi-GW fixture-run rationale
    rather than a single-gameweek proxy. Deferred import: see solve_horizon_transfers's docstring
    for why optimizer<->transfer_planner imports must happen inside the function body."""
    from src import transfer_planner  # deferred: see docstring

    event_ids = transfer_planner.get_horizon_event_ids(conn, horizon_gws)
    if not event_ids:
        return {}
    projections = transfer_planner.fetch_multi_gw_projections(conn, event_ids)

    result = {}
    for pid in player_ids:
        proj = projections.get(pid)
        if proj is None:
            continue
        difficulties = [proj["gw_difficulty"][eid] for eid in event_ids if proj["gw_has_fixture"].get(eid)]
        if difficulties:
            result[pid] = round(sum(difficulties) / len(difficulties), 2)
    return result


def _captaincy_rationale(captain, vice) -> list:
    reasons = [f"the highest projected ceiling in the Starting XI at {captain.projected_xp:.2f} xP"]
    if captain.fixture_difficulty <= RATIONALE_FIXTURE_RUN_FAVORABLE:
        reasons.append(f"a favorable fixture (FDR {captain.fixture_difficulty:.1f})")
    xgi = xgi_per_90(captain)
    if xgi >= RATIONALE_HIGH_XGI_PER_90:
        reasons.append(f"an elite underlying rate ({xgi:.2f} xGI/90)")
    if has_set_piece_duty(captain):
        reasons.append(f"first-choice {set_piece_label(captain)} duty")

    tags = ["👑 Captain"]
    if has_set_piece_duty(captain):
        tags.append("🎯 Set Pieces")
    if xgi >= RATIONALE_HIGH_XGI_PER_90:
        tags.append("⚡ High xGI")

    bullets = [RationaleBullet(text=f"{captain.web_name} gets the armband: {', '.join(reasons)}.", tags=tags)]

    if vice is not None:
        bullets.append(RationaleBullet(
            text=(
                f"{vice.web_name} is vice-captain as the second-highest projected scorer in the "
                f"Starting XI ({vice.projected_xp:.2f} xP), covering the armband if {captain.web_name} "
                f"is withdrawn or blanks on minutes."
            ),
            tags=["👑 Captain"],
        ))
    return bullets


def _tactical_theme_rationale(squad: list) -> list:
    total_cost = sum(p.cost_millions for p in squad) or 1.0
    spend_by_position: dict = {}
    for p in squad:
        spend_by_position[p.position] = spend_by_position.get(p.position, 0.0) + p.cost_millions
    pct = {pos: (cost / total_cost) * 100 for pos, cost in spend_by_position.items()}
    breakdown_text = ", ".join(f"{pos} {pct.get(pos, 0.0):.0f}%" for pos in ("GKP", "DEF", "MID", "FWD"))

    premium_attackers = [
        p for p in squad if p.position in ("MID", "FWD") and p.now_cost >= RATIONALE_PREMIUM_COST_MIN
    ]
    if pct.get("MID", 0.0) >= RATIONALE_MID_HEAVY_SPEND_PCT:
        profile = "Heavy Midfield Powerhouse"
    elif len(premium_attackers) >= RATIONALE_DOUBLE_PREMIUM_COUNT:
        profile = "Double Premium Attack"
    elif pct.get("DEF", 0.0) >= RATIONALE_DEF_VALUE_SPEND_PCT:
        profile = "Defense-Led Value Build"
    else:
        profile = "Balanced Spread"

    bullets = [
        RationaleBullet(
            text=f"£{total_cost:.1f}m split {breakdown_text} across the squad -- a '{profile}' tactical profile.",
            tags=[],
        )
    ]
    if premium_attackers and profile == "Double Premium Attack":
        names = ", ".join(f"{p.web_name} (£{p.cost_millions:.1f}m)" for p in premium_attackers)
        bullets.append(RationaleBullet(
            text=f"Premium attacking spend is concentrated in {names}, betting on ceiling over squad depth.",
            tags=[],
        ))
    return bullets


def _key_player_rationale(conn, squad: list, horizon_gws: int) -> list:
    difficulty_by_id = _multi_gw_fixture_difficulty(conn, [p.id for p in squad], horizon_gws)

    fixture_candidates = sorted(
        (p for p in squad if difficulty_by_id.get(p.id, RATIONALE_FIXTURE_RUN_FAVORABLE + 1) <= RATIONALE_FIXTURE_RUN_FAVORABLE),
        key=lambda p: difficulty_by_id[p.id],
    )
    set_piece_candidates = sorted((p for p in squad if has_set_piece_duty(p)), key=lambda p: p.projected_xp, reverse=True)
    value_candidates = sorted((p for p in squad if p.cost_millions > 0), key=lambda p: p.projected_xp / p.cost_millions, reverse=True)

    bullets = []
    seen_ids = set()

    if fixture_candidates:
        p = fixture_candidates[0]
        avg_fdr = difficulty_by_id[p.id]
        bullets.append(RationaleBullet(
            text=(
                f"{p.web_name} ({p.position}) has a favorable {horizon_gws}-GW fixture run "
                f"(avg FDR {avg_fdr:.1f}), projected at {p.projected_xp:.2f} xP this gameweek."
            ),
            tags=["📅 Fixture Swing"],
        ))
        seen_ids.add(p.id)

    for p in set_piece_candidates:
        if p.id in seen_ids:
            continue
        bullets.append(RationaleBullet(
            text=(
                f"{p.web_name} ({p.position}) holds first-choice {set_piece_label(p)} duty, "
                f"adding a scoring floor beyond open play."
            ),
            tags=["🎯 Set Pieces"],
        ))
        seen_ids.add(p.id)
        break

    for p in value_candidates:
        if p.id in seen_ids:
            continue
        xgi = xgi_per_90(p)
        tags = ["⚡ High xGI"] if xgi >= RATIONALE_HIGH_XGI_PER_90 else []
        bullets.append(RationaleBullet(
            text=(
                f"{p.web_name} ({p.position}) offers {p.projected_xp / p.cost_millions:.2f} xP per £m "
                f"at £{p.cost_millions:.1f}m -- strong baseline value efficiency."
            ),
            tags=tags,
        ))
        seen_ids.add(p.id)
        break

    # Backfill to RATIONALE_KEY_PLAYER_COUNT highlights if the criteria above didn't surface enough.
    for p in sorted(squad, key=lambda p: p.projected_xp, reverse=True):
        if len(bullets) >= RATIONALE_KEY_PLAYER_COUNT:
            break
        if p.id in seen_ids:
            continue
        bullets.append(RationaleBullet(text=f"{p.web_name} ({p.position}) is a top squad projection at {p.projected_xp:.2f} xP.", tags=[]))
        seen_ids.add(p.id)

    return bullets[:RATIONALE_KEY_PLAYER_COUNT]


def _bench_rationale(starting_xi: list, bench: list) -> list:
    if not bench:
        return [RationaleBullet(text="No bench players supplied.", tags=[])]

    bench_cost = sum(p.cost_millions for p in bench)
    starters_cost = sum(p.cost_millions for p in starting_xi)
    enablers = [p for p in bench if p.now_cost <= RATIONALE_ENABLER_COST_MAX]

    if enablers:
        names = ", ".join(f"{p.web_name} (£{p.cost_millions:.1f}m)" for p in enablers)
        text = (
            f"{names} sit on the bench as budget enablers, keeping bench spend to £{bench_cost:.1f}m "
            f"so the £{starters_cost:.1f}m Starting XI can carry the squad's premium assets within the £100.0m cap."
        )
    else:
        text = (
            f"The bench costs £{bench_cost:.1f}m total, balancing the £{starters_cost:.1f}m committed "
            f"to the Starting XI within the £100.0m cap."
        )
    return [RationaleBullet(text=text, tags=[])]


def generate_squad_rationale(
    conn, squad: list, starting_xi: list, bench: list, captain, vice, horizon_gws: int = 4
) -> dict:
    """Metric-driven, plain-English explanation of a built squad: why the captain was picked, the
    squad's tactical/spending shape, 2-3 standout picks, and the bench strategy.

    Deviates from the spec's literal `(squad_df, starting_xi_df, bench_df, captain_row, vice_row)`
    pandas-DataFrame signature: every other function in optimizer.py/transfer_planner.py works on
    `list[PlayerRow]` exclusively -- pandas only appears in app.py for st.dataframe display, built
    from PlayerRow lists via `_squad_to_dataframe`. This keeps that convention (list[PlayerRow]
    squad/starting_xi/bench, single PlayerRow captain/vice) instead of introducing a second data
    representation into the engine layer. `conn` is added (not in the literal spec signature) so
    this can pull real multi-GW fixture-run data via transfer_planner instead of a single-GW proxy.

    Returns {"captaincy": [...], "tactical_theme": [...], "key_players": [...], "bench_strategy": [...]},
    each a list of RationaleBullet carrying UI badge `tags` for app.py's Tactical Rationale sidebar.
    """
    return {
        "captaincy": _captaincy_rationale(captain, vice),
        "tactical_theme": _tactical_theme_rationale(squad),
        "key_players": _key_player_rationale(conn, squad, horizon_gws),
        "bench_strategy": _bench_rationale(starting_xi, bench),
    }


# --- Multi-gameweek horizon transfer roadmap ----------------------------------

@dataclass
class HorizonStep:
    """One gameweek's entry in a solve_horizon_transfers roadmap."""
    event_id: int
    action: str  # "hold", "transfer", or "initial_selection" (free/unlimited GW1 -- see plan_transfers)
    players_out: list
    players_in: list
    hit_cost: int
    cumulative_xp: float
    summary: str


def solve_horizon_transfers(
    conn,
    squad_ids: list,
    bank: int,
    free_transfers: int,
    horizon_gws: int = 4,
    allow_hits: bool = True,
    ensemble_weights: Optional[dict] = None,
    min_starter_xmins: Optional[float] = None,
) -> list:
    """Step-by-step transfer roadmap over a rolling horizon (default 4 GWs: GW_n..GW_{n+3}),
    accounting for banked free transfers (capped at 5) and only taking a -4 hit when the
    ILP finds it nets a positive return -- e.g. "GW1: Hold / Roll Transfer" or "GW2: Sell
    Player A (GBP6.0m) -> Buy Player B (GBP6.5m) [+3.2 cumulative xP]".

    This wraps transfer_planner.plan_transfers (the actual per-gameweek ILP) and reshapes its
    output into readable roadmap steps. The import is deferred to inside the function body,
    not at module level: transfer_planner imports from optimizer, so an unconditional
    module-level import here would be a circular import. By the time anything actually calls
    this function, both modules are already fully loaded, so the deferred import is safe.
    """
    from src import transfer_planner  # deferred: see docstring

    roadmap = transfer_planner.plan_transfers(
        conn, squad_ids, bank=bank, free_transfers=free_transfers, horizon_gws=horizon_gws, allow_hits=allow_hits,
        ensemble_weights=ensemble_weights, min_starter_xmins=min_starter_xmins,
    )

    steps = []
    cumulative_xp = 0.0
    for plan in roadmap:
        cumulative_xp += plan.net_points

        if plan.initial_selection:
            # Free/unlimited pre-GW1-deadline window (see transfer_planner.plan_transfers) --
            # never phrased as a "transfer" or a hold/roll decision, since neither concept
            # applies yet: this is the very first squad pick, not a change from a prior one.
            action = "initial_selection"
            if not plan.transfers_in:
                summary = (
                    f"GW{plan.event_id}: Initial squad confirmed as-is (still free/unlimited "
                    f"until the GW1 deadline) [{cumulative_xp:+.1f} cumulative xP]"
                )
            else:
                n = len(plan.transfers_in)
                summary = (
                    f"GW{plan.event_id}: Initial squad -- {n} change(s) before the free/unlimited "
                    f"GW1 deadline (no hit) [{cumulative_xp:+.1f} cumulative xP]"
                )
        elif not plan.transfers_in:
            action = "hold"
            summary = f"GW{plan.event_id}: Hold / Roll Transfer [{cumulative_xp:+.1f} cumulative xP]"
        else:
            action = "transfer"
            hit_note = f", -{plan.hit_cost}pt hit" if plan.hit_cost else ""
            if len(plan.transfers_out) == 1 and len(plan.transfers_in) == 1:
                out_desc = f"{plan.transfers_out[0]} (£{plan.transfers_out_cost[0] / 10:.1f}m)"
                in_desc = f"{plan.transfers_in[0]} (£{plan.transfers_in_cost[0] / 10:.1f}m)"
                summary = f"GW{plan.event_id}: Sell {out_desc} -> Buy {in_desc}{hit_note} [{cumulative_xp:+.1f} cumulative xP]"
            else:
                out_desc = ", ".join(
                    f"{name} (£{cost / 10:.1f}m)" for name, cost in zip(plan.transfers_out, plan.transfers_out_cost)
                )
                in_desc = ", ".join(
                    f"{name} (£{cost / 10:.1f}m)" for name, cost in zip(plan.transfers_in, plan.transfers_in_cost)
                )
                summary = f"GW{plan.event_id}: Sell [{out_desc}] -> Buy [{in_desc}]{hit_note} [{cumulative_xp:+.1f} cumulative xP]"

        steps.append(
            HorizonStep(
                event_id=plan.event_id,
                action=action,
                players_out=plan.transfers_out,
                players_in=plan.transfers_in,
                hit_cost=plan.hit_cost,
                cumulative_xp=round(cumulative_xp, 3),
                summary=summary,
            )
        )

    return steps

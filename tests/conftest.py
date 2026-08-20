"""Shared fixtures for the real (assertion-based) regression suite.

Deliberately built on hand-constructed, deterministic PlayerRow objects rather than a synced
SQLite DB or live API data -- solve_squad/solve_starting_xi/_solve_lineup_milp all operate on
plain player lists with no DB dependency, so testing them this way is both faster and, more
importantly, actually deterministic: a test asserting "the hard-excluded player never starts"
needs to know for certain that player WOULD otherwise have been picked (highest xP in their
position), which is only guaranteed with a fixture we built ourselves, not real live data that
changes daily.

This intentionally does NOT replace test_optimizer.py/test_transfer_planner.py at the repo root --
those remain useful as manual, real-data smoke checks (do the numbers look sane against this
week's actual player pool). This suite exists to catch a REGRESSION mechanically, on every push,
which print-and-eyeball scripts can't do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.optimizer import PlayerRow

TEAM_IDS = list(range(1, 9))  # 8 synthetic teams -- enough spread that MAX_PLAYERS_PER_TEAM (3)
# never accidentally makes the whole fixture infeasible (15 picks / 3 per team needs >= 5 teams).


def make_player(id: int, element_type: int, team_id: int, cost: int, xp: float, **overrides) -> PlayerRow:
    """One synthetic PlayerRow with sane defaults for every field _solve_lineup_milp's constraints
    touch -- override just the field(s) a given test cares about (status, chance_of_playing_next_round,
    xmins, ...).

    starts defaults to 10 (NOT PlayerRow's own dataclass default of 0) specifically so a test
    doesn't accidentally read as a GW1 Pre-Season Cold-Start pool (optimizer.is_cold_start_pool --
    True only when EVERY candidate has starts == 0) just because it never bothered setting starts
    at all -- a test that specifically wants to exercise cold-start behavior should override
    starts=0 explicitly, making that intent visible at the call site rather than accidental."""
    defaults = dict(
        id=id,
        web_name=f"Player{id}",
        team_id=team_id,
        team_name=f"Team{team_id}",
        element_type=element_type,
        now_cost=cost,
        selected_by_percent=5.0,
        form=3.0,
        total_points=20,
        ep_next=3.0,
        xg_per_90=0.1,
        xa_per_90=0.1,
        saves_per_90=0.0,
        defensive_contribution_per_90=0.0,
        starts_per_90=0.8,
        starts=10,
        status="a",
        fixture_difficulty=3.0,
        has_fixture=True,
        projected_xp=xp,
        xmins=80.0,
        chance_of_playing_next_round=None,
    )
    defaults.update(overrides)
    return PlayerRow(**defaults)


@pytest.fixture
def synthetic_pool() -> list:
    """44 synthetic players -- 6 GKP, 14 DEF, 14 MID, 10 FWD -- generously exceeding
    SQUAD_POSITION_COUNTS (2/5/5/3) in every position so solve_squad has a real optimization
    choice, not just one feasible combination, and every one of FORMATION_CHOICES' 8 shapes is
    satisfiable from within any resulting 15-man squad (which always has exactly 5 DEF/5 MID/3
    FWD -- enough for even the most DEF- or FWD-heavy formation). xP and cost both increase with
    each player's index within their position, so "highest xP" and "most expensive" are
    predictable, nameable individual players for assertions."""
    players = []
    pid = 1
    for i in range(6):
        players.append(make_player(pid, 1, TEAM_IDS[i % 8], cost=40 + i * 3, xp=3.0 + i * 0.2))
        pid += 1
    for i in range(14):
        players.append(make_player(pid, 2, TEAM_IDS[i % 8], cost=40 + i * 3, xp=3.5 + i * 0.15))
        pid += 1
    for i in range(14):
        players.append(make_player(pid, 3, TEAM_IDS[i % 8], cost=45 + i * 3, xp=4.0 + i * 0.15))
        pid += 1
    for i in range(10):
        players.append(make_player(pid, 4, TEAM_IDS[i % 8], cost=45 + i * 3, xp=4.2 + i * 0.2))
        pid += 1
    return players


@pytest.fixture
def synthetic_squad(synthetic_pool) -> list:
    """A real, ILP-optimized 15-man squad built from synthetic_pool -- the shared starting point
    for every Starting XI/formation/captaincy test below, so each test isn't re-solving the squad
    step itself."""
    from src.optimizer import solve_squad

    return solve_squad(synthetic_pool)

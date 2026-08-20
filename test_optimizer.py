"""Manual verification script for the Sprint 2 optimization engine.

Run with: python test_optimizer.py
Requires data/fpl_data.db to already be populated (run sync_data.py first).
"""
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles default to cp1252, which can't encode every player name (accents, etc).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src import database
from src.optimizer import (
    BUDGET_LIMIT,
    MAX_PLAYERS_PER_TEAM,
    SQUAD_POSITION_COUNTS,
    OptimizationError,
    build_optimal_squad,
    fetch_players,
    get_captain_recommendations,
    solve_starting_xi,
)


def print_squad(title: str, squad: list) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for p in sorted(squad, key=lambda p: p.element_type):
        print(f"  [{p.position}] {p.web_name:<20} {p.team_name:<15} £{p.cost_millions:.1f}m  xP={p.projected_xp:.2f}")
    total_cost = sum(p.now_cost for p in squad)
    total_xp = sum(p.projected_xp for p in squad)
    print(f"  Total cost: £{total_cost / 10:.1f}m / £{BUDGET_LIMIT / 10:.1f}m")
    print(f"  Total projected xP: {total_xp:.2f}")


def verify_squad_constraints(squad: list) -> None:
    total_cost = sum(p.now_cost for p in squad)
    assert total_cost <= BUDGET_LIMIT, f"Budget exceeded: {total_cost} > {BUDGET_LIMIT}"

    team_counts = Counter(p.team_id for p in squad)
    over_limit = {tid: n for tid, n in team_counts.items() if n > MAX_PLAYERS_PER_TEAM}
    assert not over_limit, f"Club limit exceeded: {over_limit}"

    position_counts = Counter(p.element_type for p in squad)
    assert position_counts == Counter(SQUAD_POSITION_COUNTS), f"Bad position counts: {dict(position_counts)}"

    print("\nConstraint checks passed: budget, club limit (<=3/team), and position counts all OK.")


def main() -> int:
    database.init_db()
    with database.db_connection() as conn:
        players = fetch_players(conn)
        if not players:
            print("No players found in data/fpl_data.db. Run sync_data.py first.", file=sys.stderr)
            return 1

        try:
            squad = build_optimal_squad(conn, mode="balanced")
        except OptimizationError as exc:
            print(f"Squad optimization failed: {exc}", file=sys.stderr)
            return 1

        print_squad("Optimal 15-Man Squad (balanced xP)", squad)
        verify_squad_constraints(squad)

        starting_xi, bench, formation = solve_starting_xi(squad)
        print_squad(f"Starting XI - Formation {formation}", starting_xi)

        print("\nBench (auto-sub order):")
        for i, p in enumerate(bench, start=1):
            print(f"  {i}. [{p.position}] {p.web_name:<20} xP={p.projected_xp:.2f} "
                  f"fixture_difficulty={p.fixture_difficulty:.1f}")

        squad_ids = [p.id for p in squad]
        captain_info = get_captain_recommendations(conn, squad_ids)

        print("\nCaptaincy recommendations (top 3):")
        for c in captain_info["top_picks"]:
            print(f"  {c['player'].web_name:<20} score={c['captain_score']:.2f} "
                  f"ownership={c['player'].selected_by_percent:.1f}%")

        print(f"\n  Captain:      {captain_info['captain']['player'].web_name}")
        if captain_info["vice_captain"]:
            print(f"  Vice-Captain: {captain_info['vice_captain']['player'].web_name}")
        if captain_info["differential_pick"]:
            print(f"  Differential option: {captain_info['differential_pick']['player'].web_name}")

        try:
            template_squad = build_optimal_squad(conn, mode="template")
            print_squad("Template (highest-ownership) Squad", template_squad)
            verify_squad_constraints(template_squad)
        except OptimizationError as exc:
            print(f"Template squad failed: {exc}", file=sys.stderr)

        try:
            differential_squad = build_optimal_squad(conn, mode="differential")
            print_squad("Differential (<10% ownership) Squad", differential_squad)
            verify_squad_constraints(differential_squad)
        except OptimizationError as exc:
            print(f"Differential squad failed (pool may be too thin): {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

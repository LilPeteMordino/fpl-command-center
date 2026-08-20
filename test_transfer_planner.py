"""Verification script for the Sprint 3 multi-gameweek transfer planner.

Run with: python test_transfer_planner.py
Requires data/fpl_data.db to be populated (run sync_data.py first) with enough
upcoming fixtures loaded to cover the requested horizon.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles default to cp1252, which can't encode every player name (accents, etc).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src import database
from src.optimizer import OptimizationError, build_optimal_squad
from src.transfer_planner import plan_transfers

BANK = 5  # integer cost units == GBP 0.5m
FREE_TRANSFERS = 1
HORIZON_GWS = 4


def main() -> int:
    database.init_db()
    with database.db_connection() as conn:
        try:
            sample_squad = build_optimal_squad(conn, mode="balanced")
        except OptimizationError as exc:
            print(f"Could not build a sample starting squad: {exc}", file=sys.stderr)
            return 1

        squad_ids = [p.id for p in sample_squad]
        print("Sample starting 15-man squad:")
        for p in sample_squad:
            print(f"  [{p.position}] {p.web_name:<20} {p.team_name:<15} £{p.cost_millions:.1f}m")

        try:
            roadmap = plan_transfers(
                conn, squad_ids, bank=BANK, free_transfers=FREE_TRANSFERS, horizon_gws=HORIZON_GWS
            )
        except OptimizationError as exc:
            print(f"Transfer planning failed: {exc}", file=sys.stderr)
            return 1

        header = (
            f"TRANSFER ROADMAP ({HORIZON_GWS}-GW horizon, starting bank "
            f"£{BANK / 10:.1f}m, {FREE_TRANSFERS} FT)"
        )
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")

        for plan in roadmap:
            print(f"\nGW {plan.event_id}")
            print("-" * 20)

            if plan.transfers_in:
                print(f"  OUT: {', '.join(plan.transfers_out)}")
                print(f"  IN:  {', '.join(plan.transfers_in)}")
            else:
                print("  HOLD (bank free transfer)")

            print(f"  Transfers made: {plan.transfers_made}  (had {plan.free_transfers_before} FT available)")
            if plan.hit_cost:
                print(f"  Hit taken: -{plan.hit_cost} pts")
            print(f"  Bank remaining: £{plan.bank_remaining / 10:.1f}m")
            print(f"  Starting XI formation: {plan.formation}")
            print(f"  Captain: {plan.captain.web_name} ({plan.captain.position})  |  "
                  f"Vice: {plan.vice_captain.web_name} ({plan.vice_captain.position})")
            print(f"  Projected GW points: {plan.gross_points:.1f} gross, {plan.net_points:.1f} net of hits")

        total_net = sum(plan.net_points for plan in roadmap)
        print(f"\nTotal projected net points over {HORIZON_GWS} GWs: {total_net:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

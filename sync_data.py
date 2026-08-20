"""Clears the local FPL cache and re-syncs fresh data from the official FPL API, automatically
falling back to a community mirror (github.com/vaastav/Fantasy-Premier-League) for teams/players
if the official API is unreachable. Any locally saved pre-season draft is preserved across the
resync either way.

Usage:
    python sync_data.py
    python sync_data.py --manager-id 1234567 --event 1
    python sync_data.py --force-fallback   # use the community fallback directly, skipping the API
"""
import argparse
import sys

from src import database
from src.fpl_api import (
    FPLAPIError,
    current_fpl_season,
    sync_all_with_fallback,
    sync_players_from_vaastav_fallback,
    sync_teams_from_vaastav_fallback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manager-id", type=int, default=None,
        help="FPL manager (entry) ID to also sync a squad for.",
    )
    parser.add_argument(
        "--event", type=int, default=None,
        help="Gameweek number to pull the manager's squad for (requires --manager-id).",
    )
    parser.add_argument(
        "--force-fallback", action="store_true",
        help="Skip the official API and use the community (vaastav) fallback for teams/players "
             "directly -- useful to verify the fallback path, or if you already know the official "
             "API is down. Fixtures/gameweeks are not available via this path.",
    )
    return parser.parse_args()


def _log_result(status: dict) -> None:
    """Prints an accurate summary of what sync_all_with_fallback actually did -- never a fixed
    success string, since that could claim things (e.g. fixtures/gameweeks) that didn't happen."""
    if status["source"] == "official":
        gw_note = "fixtures/gameweeks synced" if status["fixtures_synced"] else "fixtures sync FAILED"
        print(
            f"Successfully fetched official FPL data for {status['players_synced']} players and "
            f"{status['teams_synced']} teams ({gw_note}); on-demand xP projections are available "
            f"for every synced gameweek."
        )
    elif status["source"] == "fallback":
        print(
            f"Official FPL API was unreachable ({status['error']}); used the community fallback "
            f"(vaastav, season {status['season']}) instead: {status['teams_synced']} teams, "
            f"{status['players_synced']} players. Fixtures/gameweeks were NOT updated -- that data "
            f"isn't available via the fallback -- rerun once the official API is back."
        )
    else:
        print(f"Sync failed on both the official API and the community fallback: {status['error']}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if (args.manager_id is None) != (args.event is None):
        print("Error: --manager-id and --event must be provided together.", file=sys.stderr)
        return 1

    database.init_db()
    with database.db_connection() as conn:
        saved_draft = database.load_local_draft(conn)

        print("Clearing cached data...")
        database.clear_all_data(conn)

        if args.force_fallback:
            print("Using the community (vaastav) fallback for teams/players...")
            season = current_fpl_season()
            try:
                teams_n = sync_teams_from_vaastav_fallback(conn, season)
                players_n = sync_players_from_vaastav_fallback(conn, season)
            except FPLAPIError as exc:
                print(f"Fallback sync failed: {exc}", file=sys.stderr)
                return 1
            print(
                f"Used the community fallback (season {season}): {teams_n} teams, {players_n} players. "
                f"Fixtures/gameweeks were NOT updated -- not available via this path."
            )
        else:
            print("Syncing FPL data (official API, with automatic community fallback if unreachable)...")
            status = sync_all_with_fallback(conn, manager_id=args.manager_id, event=args.event)
            _log_result(status)
            if status["source"] == "failed":
                return 1

        if saved_draft is not None:
            try:
                database.save_local_draft(
                    conn, saved_draft["player_ids"], saved_draft["bank_balance"],
                    saved_draft["captain_id"], saved_draft["vice_id"],
                )
                print("Restored your locally saved pre-season draft.")
            except Exception as exc:
                print(f"Could not restore your saved draft after the sync: {exc}", file=sys.stderr)

    print(f"Sync complete. Database at {database.config.DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

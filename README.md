# FPL Analytics Engine

A local-first Fantasy Premier League analytics dashboard: a real xP (expected points) model built
from xG/xA/DEFCON/minutes-security data, a joint squad + Starting XI MILP optimizer, a multi-
gameweek transfer planner with anti-churn rules, a chip-strategy roadmap, and a live gameweek
radar with auto-substitution simulation and mini-league comparison -- all running against a local
SQLite cache of the official FPL API.

## Stack

- **Backend**: Python, SQLite, [PuLP](https://coin-or.github.io/pulp/) (CBC solver) for the MILP
  optimization core.
- **Frontend**: [Streamlit](https://streamlit.io/), Plotly.
- **Data**: the official [Fantasy Premier League API](https://fantasy.premierleague.com/api/),
  with a [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
  community-mirror fallback when the official API is unreachable.

## Features

- **Manager Command Center** -- synced squad snapshot, captaincy recommendation, points
  projection, and a 4-GW horizon transfer planner in one view.
- **Squad Optimizer & Generators** -- joint squad/lineup MILP with Formation Lock, Starter
  Security floor, Risk Profile (EV vs template-shielding), and manual player locks/blacklist.
- **Horizon Transfer Planner** -- multi-gameweek ILP roadmap with a hurdle-rate gate, GKP freeze
  ("Set-and-Forget"), and rebuy-prevention anti-churn rules.
- **Live Gameweek Radar** -- real-time points, auto-substitution simulation, captain doubling, and
  a Historical Gameweek Replay mode (exercises the exact same engine against a finished past
  gameweek, for testing before a real one is live).
- **Chip Strategy & Tactics** -- season-half Wildcard/Bench Boost/Triple Captain/Free Hit roadmap.
- **Rival Radar & Mini-League** -- LEO (Live Effective Ownership) and Shield/Sword classification
  against a linked mini-league.
- **Fixture Difficulty Matrix** -- rolling difficulty ticker for the full squad.
- Optional CSV ensemble blending against FPL Review / FPL Form exports, and a pre-season scouting
  overrides drawer for manual xMins/penalty/set-piece corrections before real season data exists.

## Local setup

Requires Python 3.13 (see `runtime.txt`).

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app creates `data/fpl_data.db` on first run. Use the sidebar's **"Sync Live FPL Data"** button
to populate it from the official API (falls back to the community mirror if that's unreachable).

### Running the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
```

`tests/` is a real assertion-based regression suite (MILP constraints, transfer-planner anti-churn
rules) that runs deterministically against synthetic/in-memory fixtures -- no network or synced
database required. It also runs automatically on every push/PR via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml).

The two root-level scripts, `test_optimizer.py` and `test_transfer_planner.py`, are older manual
print-and-eyeball smoke scripts kept for interactive spot-checking against a real synced database
-- they're not part of the automated suite and aren't picked up by pytest's default discovery from
`tests/`.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo,
   branch `main`, main file `app.py`.
3. No secrets are required -- the FPL API is public and the app takes your Team ID as plain
   sidebar/URL input, never a credential. `.streamlit/secrets.toml` stays gitignored and unused
   unless you later add something that genuinely needs one.
4. `runtime.txt` pins the deploy to Python 3.13; `requirements.txt` is fully version-pinned for a
   reproducible `pip install` on every reboot -- see the comment at the top of that file before
   loosening it.
5. First boot has an empty `data/` -- use **"Sync Live FPL Data"** in the sidebar to populate it.
   The synced database is a local/per-instance build artifact (gitignored), not something you
   commit or need to seed.

### Notes on Streamlit Community Cloud's storage model

The free tier's filesystem is ephemeral -- an app reboot (inactivity sleep, redeploy, or platform
maintenance) wipes `data/fpl_data.db` and any saved local draft/preferences along with it. That's
consistent with the rest of the app's design (a "Sync Live FPL Data" click rebuilds it from the
API in seconds) but is worth knowing before relying on a deployed instance to remember state
between sessions for longer than that.

## Project layout

```
app.py                   Streamlit dashboard (all tabs/sidebar UI)
src/
  database.py             SQLite schema, connection handling, CRUD helpers
  fpl_api.py               Official FPL API client + community-mirror fallback
  optimizer.py             xP model + squad/lineup MILP solvers
  transfer_planner.py       Multi-gameweek transfer ILP roadmap
  chip_planner.py           Season-half chip strategy roadmap
  live_tracker.py           Live points, auto-subs, LEO/mini-league engine
  replay.py                 Historical Gameweek Replay Mode
  projections.py            External CSV ensemble ingestion/matching
  models.py                 Pydantic data models
  config.py                 Endpoints, paths, shared constants
sync_data.py               Standalone CLI sync script
tests/                     Automated pytest regression suite (see above)
assets/, static/           Dark theme CSS, PWA manifest
```

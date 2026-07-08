# 08 - Flask API and UI

## What You Will Understand After This Lesson

- How Flask starts.
- How blueprints organize API and page routes.
- What every endpoint does.
- How templates call the API with browser-side JavaScript.
- Why this is not a full SPA frontend.
- How Gunicorn reaches the Flask app through `wsgi.py`.
- How route-level predictor refresh keeps API responses aligned with changed artifacts.

## First Principles

A web app usually has:

- Server routes that receive HTTP requests.
- Business logic that does the work.
- Responses, often JSON or HTML.
- Frontend code that calls server endpoints.

Flask is a lightweight Python framework for this.

## Project-Specific Startup Flow

File: `src/api/app.py`

```text
local development: python -m src.api.app
  -> set thread-limiting environment variables
  -> create_app()
  -> configure Flask
  -> add CORS headers
  -> import route blueprints
  -> register blueprints
  -> app.run()
```

The app factory is `create_app`.

Container/production-style startup:

```text
gunicorn ... wsgi:app
  -> import wsgi.py
  -> from src.api.app import create_app
  -> app = create_app()
  -> Gunicorn workers serve requests through that app object
```

Interview answer:

> `src/api/app.py` owns app creation. `wsgi.py` exposes the concrete app object. Gunicorn imports that object and handles the HTTP serving process, workers, timeouts, and logs.

## Blueprints

File: `src/api/routes.py`

```python
api_bp = Blueprint("api", __name__)
pages_bp = Blueprint("pages", __name__, template_folder="templates")
```

`api_bp` handles JSON endpoints under `/api`.

`pages_bp` renders HTML pages such as `/predict` and `/simulate`.

## Runtime Globals

`routes.py` creates:

```python
predictor = MatchPredictor()
simulator = TournamentSimulator()
```

This loads models once at startup, which makes requests faster than loading artifacts per request.

Trade-off:

Startup is heavier, and global state needs care when retraining/model files change.

Current mitigation:

`routes.py` defines `refresh_predictor_if_needed()`, which calls the predictor's internal hot-reload check before routes that depend on fresh predictor state. The current route layer calls it before:

- `/api/teams`
- `/api/predict`
- `/api/team-details/<team_name>`
- `/api/team-matches/<team_name>`

This does not make artifact swaps fully transactional, but it does let the API notice changed model or feature files without restarting the container.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Basic health check. |
| `/api/teams` | GET | Sorted known team list. |
| `/api/predict` | POST | Single-match prediction. |
| `/api/simulate` | POST | Monte Carlo champion probabilities. |
| `/api/simulate-detailed` | POST | One detailed tournament simulation. |
| `/api/team-details/<team_name>` | GET | Team state, radar stats, win ratio. |
| `/api/team-matches/<team_name>` | GET | Last five matches for a team. |
| `/api/visualisations/<filename>` | GET | Serves generated PNG plots. |
| `/api/model-meta` | GET | Returns `models/registry/meta.json`. |
| `/api/live/standings` | GET | Returns flattened standings list. |
| `/api/live/fixtures` | GET | Returns flattened fixtures list. |

## Templates

| Template | What it does |
|---|---|
| `home.html` | Landing/dashboard overview, fetches model metadata. |
| `predict.html` | Team selectors and prediction result UI. |
| `analytics.html` | Team analytics and radar view. |
| `insights.html` | Model metrics and plot display. |
| `live.html` | Standings and fixtures tabs. |
| `simulate.html` | Monte Carlo controls, champion table, interactive bracket. |
| `about.html` | Static project/methodology page. |
| `privacy.html` | Static privacy page. |
| `terms.html` | Static terms page. |

The frontend uses inline JavaScript and `fetch`. There is no React, Vue, Next.js, build pipeline, or frontend package manager.

## Browser Fetch Flow

Example prediction flow:

```text
User selects teams
  -> predict.html JavaScript builds JSON
  -> fetch('/api/predict', { method: 'POST', body: ... })
  -> Flask route validates required teams
  -> route accepts optional match_date, home_rest_days, away_rest_days, match_stake
  -> MatchPredictor infers missing rest-day context from feature history
  -> MatchPredictor returns probabilities
  -> route jsonify(...)
  -> browser updates bars, labels, expected goals, scorelines, and context display
```

Current `predict.html` behavior:

The browser no longer sends hardcoded rest-day overrides by default. It lets the backend infer rest days and then displays the returned `context.rest_days` values. This is better for explainability because the UI shows what the model actually used.

## Live Data Flow

`live.html` calls:

- `/api/live/standings`
- `/api/live/fixtures`

The routes call `src/utils/api_football.py`, which uses:

```text
derive standings from fixtures if possible
  -> valid local cache
  -> Football-Data.org if configured
  -> API-Football if configured
  -> expired cache
  -> mock fallback
```

Fixture-derived standings:

`get_standings()` now attempts to build standings from finished fixtures first. It initializes group tables from hardcoded World Cup groups, applies finished results (`FT`, `AET`, `PEN`), updates wins/draws/losses/goals/points/form, and sorts by points, goal difference, goals for, then team name.

Why this exists:

If fixture results are available but a standings endpoint is stale or unavailable, the app can still show a coherent group table derived from match results. It is not a full replacement for an official standings API, but it is more useful than showing old standings.

## Common Interview Questions

| Question | Strong answer |
|---|---|
| Why use Flask blueprints? | They separate API routes from page routes and keep route registration organized. |
| Is this a single-page app? | No. Flask serves HTML pages, and each page has inline JavaScript that calls JSON endpoints. |
| Why instantiate predictor globally? | Model loading is expensive, so loading once improves request latency. |
| What API validation exists? | `/api/predict` checks required teams, `/api/simulate` validates positive integer and caps runs. Validation is limited and could be stronger. |
| What changed in `/api/predict`? | It now passes optional match date and rest-day overrides to the predictor. Missing rest days are inferred and returned in the response context. |
| How are live standings produced now? | The app first tries to derive them from finished fixtures, then falls back to valid cache, configured external APIs, expired cache, and mock data. |
| Why does the route layer call predictor refresh? | The predictor is global, so the route must give it a chance to reload changed model or feature files before serving data-dependent responses. |

## Rebuild Exercise

Build a tiny Flask app with:

1. `/api/health`
2. `/api/predict` that returns dummy probabilities
3. `/predict` page with a button that calls the endpoint

Then replace dummy logic with your predictor class.

## Self-Check Quiz

1. Which file has `create_app`?
2. Which file has `/api/predict`?
3. Which template calls `/api/simulate`?
4. Does the frontend use React?

Answers:

1. `src/api/app.py`
2. `src/api/routes.py`
3. `src/api/templates/simulate.html`
4. No.

## External Links

- Flask app factories: https://flask.palletsprojects.com/en/stable/patterns/appfactories/
- Flask blueprints: https://flask.palletsprojects.com/blueprints/
- Flask Gunicorn deployment: https://flask.palletsprojects.com/en/stable/deploying/gunicorn/
- Real Python Flask tutorials: https://realpython.com/tutorials/flask/
- Fetch API: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API

# 11 - Current Changes Deep Dive

## What You Will Understand After This Lesson

- What changed in the latest project state.
- Which files were affected by the recent commits.
- Why each change was made.
- How to explain the new runtime, deployment, and data-update behavior in an interview.
- How to rebuild the same changes in a similar project.

## Evidence Used

This lesson is based on read-only inspection of:

- `git log --reverse --name-status d85750b..HEAD`
- current tracked source/config/test files
- current local data artifact shapes
- current ignored `models/registry/meta.json`

The latest commit observed during the update was:

```text
d6821e5 Refit production model on completed matches
```

## Chronological Timeline Of Recent Changes

| Commit | Main idea | Files affected |
|---|---|---|
| `d6b5070` | Prediction requests gained optional `match_date` and automatic rest-day inference. | `README.md`, `src/api/routes.py`, `src/api/templates/predict.html`, `src/models/predict.py`, `tests/test_predict.py` |
| `387e301` | Added June 2026 international match results to data. | `data/raw/matches.csv` |
| `cb3f915` / `da8b739` | Improved recent-match imports and live-data behavior. | `scripts/fetch_recent_matches.py`, `src/utils/api_football.py`, `src/api/routes.py`, `data/live_cache/*.json`, visualizations |
| `e3b1b9e` | Fixed analytics reload after match imports. | `src/models/predict.py`, `tests/test_predict.py`, `data/live_cache/standings.json` |
| `6ddc0e5` / `62dd5ee` | Removed large model artifacts from git. | `models/registry/*`, `.gitignore` |
| `8797cad` | Fixed pytest and Docker Compose behavior. | `tests/conftest.py`, `docker-compose.yaml`, related test/runtime files |
| `c40464a` | Switched Docker web service to Gunicorn and added WSGI entry point. | `Dockerfile`, `requirements.txt`, `README.md`, `wsgi.py`, `src/models/poisson_model.py`, `tests/test_poisson_model.py` |
| `d6821e5` | Refit production model on completed matches and separated production artifacts from evaluation artifacts. | `config.yaml`, `src/models/train.py`, `src/models/evaluate.py`, `data/**/*.csv`, `data/live_cache/*.json`, `visualisations/*.png` |

## Change 1: Gunicorn And `wsgi.py`

### First Principles

Flask provides the application framework: routing, request objects, templates, and JSON responses. A WSGI server provides the process that accepts HTTP traffic and calls the Flask app.

In local development, this project can still run with:

```text
python -m src.api.app
```

In Docker, the current project runs:

```text
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --access-logfile - --error-logfile - wsgi:app
```

### Project-Specific Files

`wsgi.py`:

```python
from src.api.app import create_app

app = create_app()
```

This file exists because Gunicorn needs a concrete object to import. `wsgi:app` means "import the module named `wsgi` and use the variable named `app` as the WSGI application."

`Dockerfile`:

- installs dependencies with retries/timeouts
- runs `python -m pytest tests/ -v`
- starts Gunicorn instead of `python -m src.api.app`

`requirements.txt`:

- now includes `gunicorn==23.0.0`
- uses exact pins for reproducibility

### Interview Explanation

> I kept the Flask factory pattern for clean app construction and tests, but added `wsgi.py` as the production-style entry point. Docker now starts Gunicorn with two workers and container-friendly logging, so the app is served by a WSGI server instead of Flask's development server.

### Common Mistake

Do not say Gunicorn replaces Flask. Gunicorn serves the Flask app; Flask still handles routing and response generation.

## Change 2: Prediction Context And Rest-Day Inference

### First Principles

Rest days are a contextual feature. They depend on:

- the team
- the date of the match being predicted
- the team's latest known previous match

Hardcoding a default like `7` is simple, but it hides an assumption. Inferring rest days makes the prediction more explainable and closer to the available match history.

### Project-Specific Files

`src/models/predict.py` now includes these responsibilities:

- resolves a prediction date from optional `match_date`
- loads feature-matrix state through `_load_feature_matrix_state()`
- builds `default_prediction_date` from current date and latest feature date
- infers rest days from team match history
- includes context in the prediction cache key
- returns a `context` object in prediction output

Core flow:

```text
predict_match(...)
  -> _check_and_reload()
  -> _resolve_prediction_date(match_date)
  -> _resolve_rest_context(...)
  -> _construct_features_numpy(...)
  -> model.predict_proba(...)
  -> predict_scoreline(...)
  -> return probabilities + scoreline + context
```

The returned context has this shape:

```json
{
  "match_date": "2026-06-24",
  "rest_days": {
    "home": 5,
    "away": 7,
    "source": "inferred"
  },
  "match_stake": "group"
}
```

`src/api/routes.py` now passes optional values from `/api/predict`:

- `match_date`
- `home_rest_days`
- `away_rest_days`
- `match_stake`

`src/api/templates/predict.html` now lets the backend infer rest days by default and displays the returned rest context.

### Interview Explanation

> The predictor now treats rest days as runtime context. If the request supplies rest-day overrides, it uses them. Otherwise, it looks at the feature matrix, finds each team's latest known match before the prediction date, caps the rest gap at 30 days, and returns the values it used. That makes the API output easier to explain and avoids hidden hardcoded rest assumptions.

### Edge Cases

| Case | Behavior |
|---|---|
| Missing `match_date` | Uses `default_prediction_date`. |
| Invalid `match_date` | Falls back to `default_prediction_date`. |
| No previous match found | Uses 30 rest days. |
| Very long gap | Caps at 30 rest days. |
| Explicit rest override supplied | Marks rest source as `override`. |

## Change 3: Feature-Matrix Hot Reload

### First Principles

The model artifact is not the only runtime input. The predictor also needs the latest feature matrix because it derives:

- latest team states
- latest feature date
- inferred rest days

If `feature_matrix.csv` changes but the model pickle does not, a predictor that only watches model files can keep serving stale team state.

### Project-Specific Behavior

`src/models/predict.py` now stores:

```text
last_model_loaded_time
last_feature_matrix_loaded_time
```

`_check_and_reload()` now checks both:

- model artifact modification time
- `feature_matrix.csv` modification time

If only the feature matrix changed, it reloads feature-matrix state and clears prediction cache without requiring a full model reload.

`src/api/routes.py` calls `refresh_predictor_if_needed()` before endpoints that depend on fresh predictor state.

### Interview Explanation

> The feature matrix is a serving-time dependency, not just a training artifact. After recent match imports, team state and rest-day inference should update even if the trained model did not change. The route layer asks the global predictor to refresh before data-dependent responses.

## Change 4: Recent-Match Upserts

### First Principles

Appending is not enough when a future fixture already exists in a CSV with blank scores. When the result arrives, the correct action is to update that row, not create a duplicate.

### Project-Specific File

`scripts/fetch_recent_matches.py` now:

- reads existing rows as dictionaries
- detects missing scores
- matches existing World Cup rows by teams, tournament, and date
- allows a one-day tolerance for date mismatches
- updates blank-score rows
- appends genuinely new matches
- skips already scored matches
- retrains only if something changed

### Interview Explanation

> I changed the importer from append-style behavior to upsert-style behavior. It avoids duplicate World Cup matches when fixtures already exist and only triggers retraining when the raw match file actually changed.

## Change 5: Fixture-Derived Standings

### First Principles

Standings are derived data. Given fixtures and final scores, you can compute:

- played
- wins
- draws
- losses
- goals for
- goals against
- goal difference
- points
- recent form

### Project-Specific File

`src/utils/api_football.py` now attempts:

```text
generate standings from fixtures
  -> valid local cache
  -> Football-Data.org
  -> API-Football
  -> expired cache
  -> mock fallback
```

The derived standings logic:

- initializes rows from hardcoded World Cup groups
- processes finished fixtures with statuses such as `FT`, `AET`, and `PEN`
- updates table statistics
- sorts by points, goal difference, goals for, then name

### Interview Explanation

> The live UI can now derive group standings from fixture results before depending on an external standings endpoint. That makes the display more resilient when fixtures are fresher than standings data, while still keeping API/cache/mock fallbacks.

## Change 6: Model Artifact Cleanup

### First Principles

ML artifacts are generated outputs. They can be large, binary, dependency-sensitive, and hard to review in git diffs.

### Project-Specific State

Current `.gitignore` ignores:

- `models/`
- `wc2026-files/`
- local environments
- logs
- caches

Current tracked data snapshots include:

- `data/raw/matches.csv`
- `data/raw/fifa_rankings.csv`
- `data/raw/elo_ratings.csv`
- `data/processed/matches_clean.csv`
- `data/features/feature_matrix.csv`

Local ignored `models/registry/meta.json` currently reports:

| Field | Value |
|---|---:|
| Active model | `Stacking Ensemble` |
| Selected by | `log_loss` |
| Artifact role | `production_refit` |
| Holdout accuracy | `0.6100374064837906` |
| Holdout log loss | `0.866081836103257` |
| Holdout Brier score | `0.1693144308142882` |
| Holdout samples | `3208` |
| Production refit samples | `15647` |

### Interview Explanation

> The code and data snapshots are enough to study and rebuild the pipeline, but the pickled model artifacts are local generated outputs. In production I would use an artifact registry or versioned artifact directory instead of relying on untracked local pickle files.

## Change 7: Poisson Numerical Stability

### First Principles

Scoreline probabilities must be finite and normalized. Invalid expected goals can create invalid probability matrices.

Bad outputs include:

- `NaN`
- `inf`
- all-zero probability mass
- probability rows that do not sum to 1

### Project-Specific File

`src/models/poisson_model.py` now:

- clips expected goals to a bounded range
- converts non-finite lambdas to safe defaults
- checks scoreline matrix mass before normalizing
- falls back to a safe single-bucket distribution if mass is invalid

`tests/test_poisson_model.py` adds a regression test for extreme lambdas.

### Interview Explanation

> The scoreline model sits on a probability grid used by the UI and simulator. Defensive clipping and normalization protect downstream code from invalid floating-point outputs.

## Change 8: Production Refit After Completed Matches

### First Principles

Model training has two different goals:

- **Evaluation**: estimate how well the model performs on future-like unseen data.
- **Production serving**: use as much completed, trustworthy data as possible for current predictions.

These goals can conflict if you use the same artifact for both. If you train on all data and then report metrics on that same data, the metric is not a clean holdout estimate. If you never refit after evaluation, the served model ignores newer completed matches.

The current project solves this by separating artifact roles.

### Project-Specific Files

`config.yaml` now includes:

```yaml
model:
  production_cv_splits: 3
```

`src/models/train.py` now:

- imports `clone` from `sklearn.base`
- imports `StratifiedKFold`
- centralizes calibration choice in `calibration_method_for_model`
- adds `fit_production_calibrated_model`
- saves holdout evaluation artifacts
- refits production artifacts on all completed feature rows
- records artifact roles in `meta.json`

Artifact roles:

| Artifact | Role |
|---|---|
| `evaluation_model.pkl` | Calibrated model used for holdout metrics and plots. |
| `evaluation_scaler.pkl` | Scaler paired with the evaluation model. |
| `evaluation_score_model.pkl` | Score model from the temporal train/calibration setup. |
| `best_model.pkl` | Production calibrated model refit on all completed feature rows. |
| `scaler.pkl` | Production scaler fit on all completed feature rows. |
| `score_model.pkl` | Production score model refit on all completed feature rows. |

`src/models/evaluate.py` now reads the `meta["artifacts"]` mapping:

```text
meta["artifacts"]["evaluation_model"]
meta["artifacts"]["evaluation_scaler"]
```

If those files are missing, it falls back to `best_model.pkl` and `scaler.pkl`. This fallback keeps older artifact sets usable, but the intended current path is evaluation artifacts for reported metrics.

### Latest Local Metadata

Current ignored `models/registry/meta.json` reports:

| Field | Value |
|---|---:|
| Active model | `Stacking Ensemble` |
| Selected by | `log_loss` |
| Artifact role | `production_refit` |
| Holdout accuracy | `0.6100374064837906` |
| Holdout log loss | `0.866081836103257` |
| Holdout Brier score | `0.1693144308142882` |
| Holdout samples | `3208` |
| Production refit samples | `15647` |
| Latest production match date | `2026-06-23` |
| Production CV splits | `3` |

The holdout report still shows draw recall of `0.00`, so draw prediction remains a known weakness.

### Completed Match Data Update

The refit followed four completed June 23, 2026 World Cup rows:

| Match | Score |
|---|---|
| Portugal vs Uzbekistan | `5-0` |
| Colombia vs DR Congo | `1-0` |
| England vs Ghana | `0-0` |
| Panama vs Croatia | `0-1` |

This moved:

- `matches_clean.csv` to 49,406 rows through `2026-06-23`
- `feature_matrix.csv` to 15,647 rows through `2026-06-23`

### Interview Explanation

> The newest training architecture separates evaluation from serving. I still evaluate on a temporal holdout using `evaluation_model.pkl`, but after choosing the best model family I refit production artifacts on all completed rows. That gives the live predictor the freshest completed data while preserving honest holdout metrics.

### Common Mistakes

- Saying the production-refit model's metrics are directly holdout metrics.
- Forgetting that `best_model.pkl` is now the serving artifact, not the evaluation artifact.
- Ignoring `meta["artifacts"]` and hardcoding evaluation to `best_model.pkl`.
- Claiming the draw problem is fixed. It is not; the current holdout report still has zero draw recall.

## New Tests To Understand

| Test area | Why it matters |
|---|---|
| Rest-day inference by default | Proves the new default path no longer depends on hardcoded rest values. |
| Rest-day override | Proves explicit request context still works. |
| Rest-day cap | Prevents unrealistic long gaps from dominating predictions. |
| Feature-matrix reload | Proves data-only updates refresh team state without model pickle changes. |
| Extreme Poisson lambdas | Proves scoreline matrices stay finite and normalized. |

## Rebuild Exercise

Add these changes to a smaller Flask ML project:

1. Add `wsgi.py` with a module-level `app`.
2. Add `gunicorn` to requirements.
3. Change Docker `CMD` to `gunicorn ... wsgi:app`.
4. Add optional request context to your prediction endpoint.
5. Infer missing context from historical data.
6. Include a `context` object in the JSON response.
7. Track modification time for a serving-time data artifact.
8. Reload that artifact when it changes.
9. Add tests for inferred context, explicit override, and data reload.

## Self-Check Quiz

1. What does `gunicorn wsgi:app` mean?
2. Why is `feature_matrix.csv` a runtime dependency?
3. Why is upsert better than append for recent World Cup match imports?
4. What does `context.rest_days.source` tell the UI?
5. Why guard the Poisson scoreline matrix against invalid probability mass?
6. Why should holdout metrics come from `evaluation_model.pkl` instead of the production-refit `best_model.pkl`?

Answers:

1. Import module `wsgi` and use its variable `app` as the WSGI callable.
2. Latest team states, prediction date defaults, and inferred rest days are derived from it.
3. It updates existing blank-score fixture rows instead of duplicating matches.
4. Whether rest days were inferred from history or supplied as overrides.
5. Invalid probability mass can break JSON responses and simulator outcomes.
6. `evaluation_model.pkl` preserves the temporal holdout contract; `best_model.pkl` is refit on all completed rows for serving freshness.

## External Links

- Flask Gunicorn deployment: https://flask.palletsprojects.com/en/stable/deploying/gunicorn/
- Gunicorn official site: https://gunicorn.org/
- Flask app factories: https://flask.palletsprojects.com/en/stable/patterns/appfactories/
- scikit-learn model persistence: https://scikit-learn.org/stable/model_persistence.html
- SciPy Poisson distribution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html

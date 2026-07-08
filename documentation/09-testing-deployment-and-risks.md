# 09 - Testing, Deployment, and Risks

## What You Will Understand After This Lesson

- What the tests prove and what they do not prove.
- Why dummy model artifacts exist for tests.
- How Docker and Compose run the app.
- What the known engineering risks are.
- Which new tests protect rest-day inference, feature reload, and Poisson stability.

## First Principles

Tests should protect behavior. Deployment should make the app reproducible. Risk analysis should be honest about what can fail.

For ML apps, there are two kinds of correctness:

- Code correctness: functions behave as expected.
- Model quality: predictions are good.
- Evaluation integrity: reported metrics must come from artifacts that preserve the intended train/calibration/test split.

The current tests mostly check code correctness, not model quality.

## Test Files

### `tests/conftest.py`

This file defines an autouse pytest fixture. It runs automatically before tests.

Purpose:

If real model artifacts are missing, it creates toy artifacts so tests can still exercise prediction code.

Why:

`models/registry` is ignored by git, but Docker runs tests during image build. Without dummy artifacts, clean builds would fail.

### `tests/test_clean.py`

Checks:

- `USA -> United States`
- `Korea Republic -> South Korea`
- whitespace/non-breaking-space cleanup
- result labels
- competitive flag

### `tests/test_features.py`

Checks:

- Elo expected score.
- K-factor tiers.
- Goal margin multiplier.
- EWMA form.
- EWMA goals.
- Output columns from Elo/form/goals feature builders.

### `tests/test_model_selection.py`

Checks:

- Lower log loss beats higher accuracy when selection metric is log loss.
- Brier score and accuracy tie-breakers work.

### `tests/test_poisson_model.py`

Checks:

- Scoreline matrices sum to 1.
- Probabilities are non-negative.
- H/D/A probability rows sum to 1.
- Rho tuning returns a value from the grid.
- Extreme/non-finite expected-goal inputs do not produce `NaN` scoreline probabilities.

### `tests/test_predict.py`

Checks:

- Predictor loads artifacts.
- Prediction output has expected keys.
- Probabilities sum to 1.
- Neutral predictions are symmetric.
- Non-neutral predictions are asymmetric.
- Prediction context infers rest days by default.
- Explicit rest-day overrides are still honored.
- `infer_rest_days` uses match history and caps long gaps.
- Feature-matrix changes reload team state even without a model pickle change.

### `tests/test_simulate.py`

Checks:

- Away `goal_diff_avg` is computed as away scored average minus away conceded average after state update.

## Deployment Files

### `Dockerfile`

Builds a Python 3.12 image:

- sets environment variables
- installs `libgomp1`
- installs `requirements.txt`
- copies app source
- runs pytest
- exposes port 5000
- starts Gunicorn with `wsgi:app`

Current command:

```text
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --access-logfile - --error-logfile - wsgi:app
```

What each important part means:

| Piece | Meaning |
|---|---|
| `--bind 0.0.0.0:5000` | Listen on all container interfaces at port 5000. |
| `--workers 2` | Run two worker processes. |
| `--timeout 120` | Kill/restart workers that hang longer than 120 seconds. |
| `--access-logfile -` | Send access logs to stdout. |
| `--error-logfile -` | Send errors to stderr/stdout-style container logs. |
| `wsgi:app` | Import `app` from `wsgi.py`. |

### `docker-compose.yaml`

Defines:

- `web`: Flask app on port 5000.
- `scheduler`: background retraining/fetch loop.

Both mount local `src`, `scripts`, `data`, `models`, `visualisations`, and `config.yaml`.

## Known Risks

| Risk | Why it matters | How to explain it |
|---|---|---|
| No database | CSV/JSON files are simple but not transactional. | Good for portfolio/prototype, not ideal for concurrent production writes. |
| No auth | Any exposed endpoint could be called publicly. | Fine locally, risky publicly. |
| CORS allows all origins | Browser clients from any origin can call the API. | Should be restricted in production. |
| Model artifacts are file-based | Retraining can overwrite files while app reads them. | Use atomic artifact swaps in production. |
| Feature data is file-based | `feature_matrix.csv` can change while a request is reading runtime state. | Current hot reload helps freshness, but production should use atomic writes/versioned data. |
| Production and evaluation artifacts differ | `best_model.pkl` is refit on all completed rows, while holdout metrics come from `evaluation_model.pkl`. | This is useful, but docs/interviews must not claim production artifact metrics are directly holdout metrics. |
| Draw recall is poor | Current metadata shows zero draw recall. | Improve threshold/objective/evaluation before claiming strong draw predictions. |
| Feature schema duplication | Feature names appear in multiple modules. | Centralize feature schema to avoid drift. |
| Hardcoded groups | Groups/bracket are embedded in code. | Move to config/data for maintainability. |
| `_simulate_match_fast` unreachable code | The fast path currently returns before optimized code runs. | Document and refactor before performance claims. |
| Empty notebooks | They do not support methodology claims. | Say implementation is in source modules. |
| `.dockerignore` includes `Dockerfile` | Could cause build-context confusion depending on builder behavior. | Test Docker build and remove if problematic. |
| `wc2026-files/` is ignored | Curriculum edits are local study material unless intentionally force-added or moved. | Mention this when expecting docs to appear in `git status`. |
| Gunicorn on Windows | Gunicorn is meant for Unix-style environments. | Use Docker/WSL/Linux for this deployment path; use Flask dev server locally on native Windows if needed. |

## Common Interview Questions

| Question | Strong answer |
|---|---|
| What do your tests prove? | They prove deterministic code behavior and prediction contract basics, not real-world model quality. |
| Why create dummy models in tests? | Real model artifacts are ignored, so tests need lightweight artifacts in clean environments. |
| How would you make deployment safer? | Use versioned artifacts, atomic swaps, one retrain job at a time, stricter API validation, and monitoring. |
| What is the biggest production blocker? | File-based artifact/data writes without locking or versioning. |
| What recent behaviors are covered by tests? | Rest-day inference/overrides, feature-matrix reload without model changes, and Poisson scoreline stability under extreme lambdas. |
| Why run Gunicorn in Docker? | It gives a production-style WSGI server with workers/timeouts/logging instead of relying on Flask's development server. |
| Why separate production and evaluation artifacts? | The evaluation artifact preserves honest holdout metrics, while the production artifact is refit on all completed rows for fresher serving. |

## Rebuild Exercise

Add tests for a new simple feature:

1. Create a function that computes `rank_diff`.
2. Write a pytest test for normal and missing values.
3. Run only that test file.

Then explain whether the test checks model quality or code correctness.

## Self-Check Quiz

1. Which file creates dummy artifacts?
2. Which file runs tests during Docker build?
3. Does the project have CI/CD config?
4. What endpoint can check basic API health?

Answers:

1. `tests/conftest.py`
2. `Dockerfile`
3. No.
4. `/api/health`

## External Links

- pytest fixtures: https://docs.pytest.org/en/stable/reference/fixtures.html
- Dockerfile docs: https://docs.docker.com/build/concepts/dockerfile/
- Docker Compose quickstart: https://docs.docker.com/compose/gettingstarted/
- Flask Gunicorn deployment: https://flask.palletsprojects.com/en/stable/deploying/gunicorn/

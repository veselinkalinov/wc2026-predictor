# 01 - Folder and File Map

## What You Will Understand After This Lesson

- What every tracked file and important local folder does.
- Which files are source, generated artifacts, docs, tests, or local-only runtime data.
- How to answer "walk me through your repo" in an interview.

## First Principles

A repo is easier to understand when you classify each file by responsibility:

- Source code: code you maintain.
- Configuration: values that change behavior without editing code.
- Artifacts: generated outputs, model files, plots, data cache.
- Tests: executable checks.
- Documentation: explanations and usage instructions.
- Infrastructure: container/build/runtime files.

## Project-Specific Walkthrough

The tracked project is mostly Python source, HTML templates, tests, configuration, and data snapshots. The local folders such as `venv/`, `logs/`, `catboost_info/`, `models/`, and `wc2026-files/` are environment, generated artifact, or documentation folders ignored by git in the current project state.

## Folder Map

| Folder | Source or generated | Runtime role |
|---|---|---|
| `.git/` | Git metadata, local-only | Version history. Do not edit manually. |
| `.pytest_cache/` | Generated | pytest cache. Safe to delete. |
| `.vscode/` | Editor config, local | VS Code workspace settings. Ignored by git. |
| `catboost_info/` | Generated | CatBoost training logs. Ignored by git. |
| `data/raw/` | Tracked data snapshots | Input CSVs used by validation, cleaning, feature engineering, and retraining. |
| `data/processed/` | Tracked generated snapshot | Cleaned match table produced by `src/data/clean.py`. |
| `data/features/` | Tracked generated snapshot | Model-ready feature matrix produced by `src/features/build.py` and consumed by training/prediction. |
| `data/live_cache/` | Tracked cache examples | Cached standings and fixtures JSON. |
| `models/registry/` | Generated/local artifacts | Pickled models, scaler, metadata. Ignored by git. |
| `notebooks/` | Tracked docs/exploration placeholder | Current notebooks are empty or zero-cell. |
| `scripts/` | Source | Command-line pipeline and maintenance workflows. |
| `src/` | Source | Importable app package. |
| `tests/` | Source tests | pytest suite. |
| `venv/` | Local environment | Installed Python dependencies. Ignored by git. |
| `visualisations/` | Generated plus tracked plots | Evaluation PNGs served by the app. |
| `wc2026-files/` | Local docs/project materials | Master doc, transcript/materials, curriculum. Ignored by git in the current project state. |

## Root File Coverage

| File | Purpose | Interview angle |
|---|---|---|
| `.dockerignore` | Excludes local/generated files from Docker build context. | Know why `venv`, `.git`, `.env`, caches, and model artifacts are excluded. |
| `.gitignore` | Excludes local environments, logs, model artifacts, `wc2026-files/`, caches, and other generated noise. Data CSV snapshots are no longer ignored. | Explain artifact reproducibility vs source control. |
| `Dockerfile` | Builds Python image, installs deps, runs tests, and starts Gunicorn with `wsgi:app`. | Explain image build vs runtime container and why production Flask apps use WSGI servers. |
| `LICENSE` | MIT license. | Lets others reuse the code with license notice. |
| `README.md` | Main project overview and usage documentation. | Public entry point for the project. |
| `config.yaml` | Central paths, feature params, model params, API params, data date range ending on `2026-07-19`, and production calibration split count. | Single source of truth for pipeline behavior. |
| `docker-compose.yaml` | Defines `web` and `scheduler` services. | Explain multi-service local deployment. |
| `requirements.txt` | Python dependency manifest. | Explain why pinned versions matter for ML artifacts. |
| `wsgi.py` | Imports `create_app()` and exposes a module-level Flask app for Gunicorn. | Explain the WSGI callable pattern: `gunicorn wsgi:app`. |

## Data and Artifact File Coverage

| File | Purpose |
|---|---|
| `data/features/.gitkeep` | Keeps the feature artifact folder visible even when generated CSVs are absent. |
| `data/features/feature_matrix.csv` | Model-ready training/prediction table. Current local snapshot has 15,647 rows and 37 columns, dated from `2010-01-02` to `2026-06-23`. |
| `data/live_cache/fixtures.json` | Cached or mock fixture payload for `/api/live/fixtures`; also feeds derived standings logic. |
| `data/live_cache/standings.json` | Cached or generated standings payload for `/api/live/standings`. |
| `data/processed/.gitkeep` | Keeps the processed-data folder visible. |
| `data/processed/matches_clean.csv` | Cleaned match table. Current local snapshot has 49,406 rows and 15 columns, dated from `1872-11-30` to `2026-06-23`. |
| `data/raw/.gitkeep` | Keeps the raw-data folder visible. |
| `data/raw/elo_ratings.csv` | Raw Elo rating time series used for historical strength features. Current local snapshot has 6,678 rows and 4 columns. |
| `data/raw/fifa_rankings.csv` | Raw FIFA ranking history used by cleaning and model features. Current local snapshot has 13,130 rows and 8 columns. |
| `data/raw/matches.csv` | Raw international match history. Current local snapshot has 49,430 rows and 9 columns, dated from `1872-11-30` to `2026-06-27`. |
| `visualisations/.gitkeep` | Keeps folder in git when empty. |
| `visualisations/calibration_curve.png` | Generated reliability plot. |
| `visualisations/confusion_matrix.png` | Generated classification confusion matrix. |
| `visualisations/feature_importance.png` | Generated model importance/contribution plot. |

Local but not tracked artifacts:

- `models/registry/*.pkl`
- `models/registry/meta.json`
- `logs/*.log`
- `wc2026-files/**`

These are documented because the app or study process depends on them, but they are not tracked source files in the current git state.

Current local model registry roles:

| Artifact | Role |
|---|---|
| `best_model.pkl` | Production calibrated model refit on all completed feature rows. Used by prediction runtime. |
| `scaler.pkl` | Production scaler fit on all completed feature rows. Used by prediction runtime. |
| `score_model.pkl` | Production Poisson score model refit on all completed feature rows. |
| `evaluation_model.pkl` | Holdout-selected calibrated model used for evaluation plots/metrics. |
| `evaluation_scaler.pkl` | Scaler fit only on the training split and used with `evaluation_model.pkl`. |
| `evaluation_score_model.pkl` | Holdout/evaluation score model. |
| `meta.json` | Contract describing model type, features, comparison metrics, production refit metadata, temporal split, and artifact filenames. |

Why this matters:

The project now separates "what we report as holdout quality" from "what we serve in production." Holdout metrics come from evaluation artifacts; runtime predictions use production-refit artifacts.

## Source Package Coverage

| File | Purpose |
|---|---|
| `src/__init__.py` | Marks `src` as a package. |
| `src/api/__init__.py` | Marks API folder as a package. |
| `src/api/app.py` | Flask app factory, CORS headers, blueprint registration. |
| `src/api/routes.py` | JSON API endpoints and HTML page routes; refreshes the predictor before relevant API reads and accepts optional prediction context. |
| `src/api/templates/about.html` | Static about/methodology page. |
| `src/api/templates/analytics.html` | Team analytics page with browser-side fetch calls. |
| `src/api/templates/home.html` | Landing/dashboard overview page. |
| `src/api/templates/insights.html` | Model metrics and visualization page. |
| `src/api/templates/live.html` | Live standings and fixtures page. |
| `src/api/templates/predict.html` | Single-match prediction UI. |
| `src/api/templates/privacy.html` | Static privacy page. |
| `src/api/templates/simulate.html` | Monte Carlo and interactive tournament UI. |
| `src/api/templates/terms.html` | Static terms page. |
| `src/data/__init__.py` | Marks data folder as a package. |
| `src/data/clean.py` | Cleans matches, normalizes names, merges FIFA rankings. |
| `src/data/fetch.py` | Checks required raw CSV files exist and are non-empty. |
| `src/data/validate.py` | Validates raw schemas, row counts, and basic constraints. |
| `src/features/__init__.py` | Marks feature folder as a package. |
| `src/features/build.py` | Runs all feature builders and saves `feature_matrix.csv`. |
| `src/features/elo.py` | Computes chronological Elo ratings. |
| `src/features/form.py` | Computes opponent-adjusted EWMA form features. |
| `src/features/goals.py` | Computes EWMA goal scoring/conceding features. |
| `src/models/__init__.py` | Marks model folder as a package. |
| `src/models/baseline.py` | Evaluates random, most-frequent, and Elo heuristic baselines. |
| `src/models/evaluate.py` | Generates metrics and evaluation plots from evaluation artifacts when present, falling back to production artifacts if needed. |
| `src/models/poisson_model.py` | Scikit-learn-compatible Poisson/Dixon-Coles score model with guards for non-finite expected goals and scoreline matrices. |
| `src/models/predict.py` | Loads artifacts, builds feature vectors, infers rest context, hot-reloads model/feature artifacts, predicts matches. |
| `src/models/simulate.py` | Simulates group and knockout tournament flows. |
| `src/models/train.py` | Trains, tunes, calibrates, compares, creates holdout evaluation artifacts, refits production artifacts on all completed rows, and serializes metadata. |
| `src/utils/__init__.py` | Marks utility folder as a package. |
| `src/utils/api_football.py` | Live data API client, fixture-derived standings generator, cache reader/writer, mock fallback. |
| `src/utils/config.py` | Loads `config.yaml` and exposes `PROJECT_ROOT`. |
| `src/utils/logger.py` | Central console/file logger factory. |

## Script Coverage

| File | Purpose |
|---|---|
| `scripts/diagnostics.py` | Prints raw data shape, overlap, Elo coverage, tournament distribution. |
| `scripts/fetch_recent_matches.py` | Fetches finished World Cup matches, upserts missing scores/new rows, and skips retraining when no rows changed. |
| `scripts/optimize_elo.py` | Grid-searches Elo/home/form/goals parameters and edits config. |
| `scripts/retrain.py` | Fast clean -> features -> train -> evaluate loop. |
| `scripts/run_pipeline.py` | Full raw check -> validate -> clean -> features -> train -> evaluate loop. |
| `scripts/scheduler.py` | Infinite interval scheduler for Docker Compose service. |

## Test Coverage

| File | Purpose |
|---|---|
| `tests/conftest.py` | Creates dummy model artifacts if real artifacts are missing. |
| `tests/test_clean.py` | Tests team normalization, result labels, competitive flag logic. |
| `tests/test_features.py` | Tests Elo, form, and goals math. |
| `tests/test_model_selection.py` | Tests probability-first model selection tie-breakers. |
| `tests/test_poisson_model.py` | Tests Poisson scoreline grids, rho tuning, and extreme-lambda numerical stability. |
| `tests/test_predict.py` | Tests predictor loading, output shape, neutral symmetry, non-neutral asymmetry, rest-day inference/override, and feature-matrix reload. |
| `tests/test_simulate.py` | Regression test for away goal-diff update. |

## Notebook Coverage

| File | Current state |
|---|---|
| `notebooks/01_eda.ipynb` | Valid notebook JSON with zero cells in current workspace. |
| `notebooks/02_feature_engineering.ipynb` | Empty file in current workspace. |
| `notebooks/03_model_evaluation.ipynb` | Empty file in current workspace. |

Do not claim notebook insights in an interview. Say the implementation is in source modules, not notebooks.

## Interview Questions

| Question | Strong answer |
|---|---|
| Why split into `src/data`, `src/features`, and `src/models`? | It separates pipeline stages: input validation/cleaning, feature construction, and ML training/serving. |
| Why are model files ignored? | They are generated artifacts and can be large/version-sensitive. The source defines how to rebuild them. |
| Why are CSV data snapshots tracked but model files ignored? | The CSVs make the pipeline and tests reproducible enough to study and rerun. The pickled models are larger, dependency-sensitive generated artifacts, so they are kept local and rebuilt from source. |
| Why have both production and evaluation model artifacts? | Evaluation artifacts preserve the temporal holdout contract for honest metrics. Production artifacts are refit on all completed rows so serving uses the freshest completed data. |
| Why keep `data/live_cache` tracked? | It gives the live UI a working cached/mock shape even without API credentials. |
| Why does the repo have `wsgi.py`? | Gunicorn needs a module-level WSGI callable. `wsgi.py` imports the Flask factory, creates the app once, and exposes it as `app` for `gunicorn wsgi:app`. |
| What would you improve? | Add artifact versioning, a single feature-schema module, and production-grade deployment/monitoring. |

## Rebuild Exercise

Create an empty folder tree for a similar project:

```text
src/data
src/features
src/models
src/api
src/utils
scripts
tests
data/raw
data/processed
data/features
models/registry
visualisations
```

Then write one sentence describing each folder.

## Self-Check Quiz

1. Which file creates the Flask app?
2. Which file defines endpoints?
3. Which script runs the full pipeline?
4. Which file loads YAML config?
5. Which tests create dummy models when needed?

Answers:

1. `src/api/app.py`
2. `src/api/routes.py`
3. `scripts/run_pipeline.py`
4. `src/utils/config.py`
5. `tests/conftest.py`

## External Links

- Python modules: https://docs.python.org/3/tutorial/modules.html
- Docker Compose concepts: https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-docker-compose/

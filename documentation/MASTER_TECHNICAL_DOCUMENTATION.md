# WC 2026 Predictor - Master Technical Documentation

> Curriculum note: this file is a reference-style snapshot generated earlier. For the current study path and the newest changes, start with `README.md` and especially `11-current-changes.md`. The updated curriculum lessons document the current Gunicorn/`wsgi.py` runtime, feature-matrix hot reload, rest-day inference, fixture-derived standings, recent-match upserts, ignored model artifacts, production-refit vs evaluation artifacts, and Poisson stability changes.

Generated from local project evidence in `C:\Users\vesko\Documents\Projects\wc2026-predictor` on 2026-06-07.

Verification performed:

- Repository files listed with `git ls-files` and `rg --files`.
- Git chronology reconstructed from `git log --reverse --name-status`.
- Data shapes sampled with the project virtual environment.
- Tests run with `.\venv\Scripts\python.exe -m pytest tests -q`: 20 passed, 4 SciPy deprecation warnings.

Important evidence boundaries:

- Historical intent is inferred only from commit messages, file diffs, README text, and current code. Personal design discussions are not present in the repository.
- `.env` exists locally but is intentionally not documented with values because it may contain secrets.
- `venv/`, `.git/`, `.pytest_cache/`, `logs/`, `catboost_info/`, model pickle internals, and generated binary plots are environment or generated artifacts. Their role is documented, but their third-party/generated internal files are not analyzed file-by-file.
- The notebooks are effectively empty in the current workspace: `notebooks/01_eda.ipynb` has zero cells, and `notebooks/02_feature_engineering.ipynb` plus `notebooks/03_model_evaluation.ipynb` are empty files.

External technical references used for framework/library behavior:

- pandas `merge_asof`: https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
- Flask application factories: https://flask.palletsprojects.com/en/stable/patterns/appfactories/
- Flask blueprints: https://flask.palletsprojects.com/blueprints/
- scikit-learn `CalibratedClassifierCV`: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- scikit-learn probability calibration guide: https://sklearn.org/stable/modules/calibration.html
- scikit-learn `GridSearchCV`: https://sklearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- CatBoost `CatBoostClassifier`: https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier
- LightGBM `LGBMClassifier`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
- Requests quickstart and timeout behavior: https://requests.readthedocs.io/en/master/user/quickstart/
- joblib persistence: https://joblib.readthedocs.io/en/latest/persistence.html
- pytest fixtures: https://docs.pytest.org/en/stable/reference/fixtures.html
- Dockerfile reference: https://docs.docker.com/reference/dockerfile

## 1. Executive Summary

WC 2026 Predictor is a Python machine-learning and Flask web application that predicts international football match outcomes and simulates a 48-team World Cup 2026 tournament. The current codebase is not a simple notebook experiment: it has a pipeline, reusable modules, trained artifact registry, evaluation plots, Flask JSON endpoints, Tailwind-powered HTML templates, live-data cache utilities, Docker packaging, and pytest coverage.

The core runtime path is:

```text
raw CSVs
  -> src.data.fetch checks file presence
  -> src.data.validate checks structure
  -> src.data.clean normalizes names and merges FIFA rankings
  -> src.features builds Elo, form, goals, rest, continent, stake features
  -> src.models.train trains/calibrates/selects models
  -> src.models.evaluate writes metrics and plots
  -> src.models.predict serves single-match predictions
  -> src.models.simulate runs tournament simulations
  -> src.api serves REST endpoints and HTML pages
```

The active model metadata in `models/registry/meta.json` says the selected champion is a calibrated Stacking Ensemble chosen by lowest log loss. Holdout metrics in that file are accuracy `0.6075665277332478`, log loss `0.8669866215910907`, Brier score `0.16950624198749228`, and `3119` holdout samples. The same metadata shows a major modeling weakness: draw recall is `0.00` in the current classification report even though draw probability is still exposed as risk/context.

The strongest architecture decision is temporal discipline: feature generation processes matches chronologically, model splitting uses time cutoffs, and prediction states use the latest known team snapshots. The weakest current implementation areas are duplicated feature-column definitions, unreachable "fast" simulator code, hardcoded World Cup groups, no real database layer, no authentication, limited API input validation, and generated/model artifacts that are not reproducible without local data.

## 2. Project Overview

### Project Purpose

The project predicts football match outcomes for Home Win, Draw, or Away Win and uses those probabilities plus scoreline modeling to simulate a World Cup tournament. Evidence:

- `README.md` describes a machine-learning pipeline for FIFA World Cup 2026 match outcomes.
- `src/models/predict.py` returns probabilities for `home_win`, `draw`, and `away_win`.
- `src/models/simulate.py` simulates group-stage and knockout tournament progression.

### Business Objective

The repository evidence supports an educational/portfolio/analytics objective: demonstrate an end-to-end ML system with data processing, model training, evaluation, dashboard UI, and REST API. There is no project file proving monetization, subscriptions, paid users, or commercial deployment.

### Target Users

Supported by UI and API structure:

- Football fans who want match forecasts: `src/api/templates/predict.html`.
- Users exploring team profiles and model metrics: `analytics.html`, `insights.html`.
- Technical reviewers/interviewers evaluating a full-stack ML project: README, tests, Dockerfile, modular source layout.
- Developers consuming API endpoints: `src/api/routes.py`.

### Main Features

| Feature | Evidence | Runtime role |
|---|---|---|
| Raw file checks | `src/data/fetch.py` | Prevents pipeline from running with missing CSVs. |
| Structural validation | `src/data/validate.py` | Checks row counts, required columns, known teams, positive Elo ratings. |
| Team normalization | `src/data/clean.py` | Aligns naming across matches and ranking data. |
| FIFA ranking merge | `src/data/clean.py` uses `pd.merge_asof` | Adds latest prior ranking snapshot per team. |
| Elo features | `src/features/elo.py` | Chronological ratings, home-field adjustment, dynamic K-factor. |
| Form features | `src/features/form.py` | Opponent-adjusted EWMA form. |
| Goal features | `src/features/goals.py` | EWMA goals scored/conceded/difference. |
| Advanced features | `src/features/build.py` | Rest days, continent advantage, match stake. |
| Multi-model training | `src/models/train.py` | Logistic Regression, Random Forest, HGB, LightGBM, CatBoost, XGBoost, Poisson, Stacking. |
| Calibration | `src/models/train.py` | `CalibratedClassifierCV` wraps models before holdout evaluation. |
| Scoreline model | `src/models/poisson_model.py` | Predicts expected goals and scoreline grids. |
| Prediction service | `src/models/predict.py` | Loads artifacts, builds feature vectors, predicts match probabilities. |
| Tournament simulator | `src/models/simulate.py` | Monte Carlo tournament progression with dynamic state updates. |
| Flask API/UI | `src/api/app.py`, `src/api/routes.py`, templates | Serves JSON endpoints and pages. |
| Live data cache | `src/utils/api_football.py` | API-Football/Football-Data.org with cache and mock fallbacks. |
| Docker deployment | `Dockerfile`, `docker-compose.yaml` | Web and scheduler services. |
| Tests | `tests/` | Unit coverage for cleaning, features, model selection, Poisson model, prediction, simulation bug regression. |

### Technology Stack

| Layer | Technologies | Evidence |
|---|---|---|
| Language | Python | `.py` source files, `requirements.txt`. |
| Data processing | pandas, NumPy | `src/data`, `src/features`, `src/models`. |
| ML | scikit-learn, LightGBM, CatBoost, XGBoost, SciPy Poisson | `requirements.txt`, `src/models/train.py`, `src/models/poisson_model.py`. |
| Serialization | joblib, JSON | `models/registry/*.pkl`, `meta.json`, `src/models/predict.py`. |
| Visualization | matplotlib, seaborn | `src/models/evaluate.py`, `visualisations/*.png`. |
| Web | Flask, HTML templates, Tailwind CDN, Material Symbols | `src/api/app.py`, templates. |
| HTTP/API integrations | requests, python-dotenv | `src/utils/api_football.py`, `scripts/fetch_recent_matches.py`. |
| Config | YAML via PyYAML | `config.yaml`, `src/utils/config.py`. |
| Tests | pytest | `tests/`, `requirements.txt`. |
| Packaging/runtime | Docker, Docker Compose | `Dockerfile`, `docker-compose.yaml`. |

### High-Level Architecture

```text
                 +------------------------+
                 | config.yaml            |
                 +-----------+------------+
                             |
          +------------------+------------------+
          |                                     |
  +-------v-------+                     +-------v-------+
  | Data pipeline |                     | Flask runtime |
  +-------+-------+                     +-------+-------+
          |                                     |
 raw -> clean -> features -> train -> artifacts |
          |                                     |
          +--------------+----------------------+
                         |
               +---------v----------+
               | models/registry    |
               | best_model.pkl     |
               | scaler.pkl         |
               | score_model.pkl    |
               | meta.json          |
               +---------+----------+
                         |
          +--------------+--------------+
          |                             |
 +--------v---------+          +--------v---------+
 | MatchPredictor   |          | TournamentSim    |
 | single matches   |          | Monte Carlo      |
 +--------+---------+          +--------+---------+
          |                             |
          +--------------+--------------+
                         |
                 +-------v-------+
                 | Flask routes  |
                 | JSON + pages  |
                 +---------------+
```

### System Design Decisions

| Decision | Evidence | Why it exists | Trade-off |
|---|---|---|---|
| Central YAML config | `src/utils/config.py`, `config.yaml` | Keeps path/model/features/API parameters in one place. | Config is loaded once at import time; runtime config edits are not automatically reloaded. |
| Chronological features | `compute_elo_ratings`, `compute_form_features`, `compute_goal_features` sort by date | Avoids using future matches for pre-match features. | Slower than vectorized static transforms, but more correct for time-series sports data. |
| Deferred filtering | comment and logic in `src/data/clean.py`, `src/features/build.py` | Uses older history to warm Elo/form/goals before filtering training dates. | More complex pipeline split. |
| Temporal train/cal/test | `src/models/train.py` | Simulates future prediction better than random split. | Less data in calibration/test; class drift can hurt draw learning. |
| Probability-first selection | `select_best_model` in `src/models/train.py` | Match/tournament simulations depend on calibrated probabilities, not only hard labels. | Best log-loss model may not maximize accuracy or draw recall. |
| Symmetric neutral prediction | `src/models/predict.py` | Removes artificial home/away order bias for neutral venues. | Doubles prediction calls for neutral matches. |
| Global Flask predictor/simulator | `src/api/routes.py` | Loads models once for faster requests. | Startup is heavier and globals make tests/concurrency more delicate. |
| Cache fallback for live data | `src/utils/api_football.py` | Keeps UI usable without API keys or during provider failure. | Mock data may look real unless clearly labeled in UI. |

### Project Structure Overview

The repository is organized into `src` modules by responsibility, `scripts` for runnable workflows, `tests` for pytest coverage, `data` for raw/processed/features/cache files, `models/registry` for serialized artifacts, `visualisations` for generated plots, and root-level config/container/docs files.

## 3. Chronological Development Timeline

This timeline is reconstructed from git evidence only.

| Date | Commit | What changed | Interpretation from evidence |
|---|---|---|---|
| 2026-05-30 | `facf6af milestone-0` | Scaffold, config, logger, notebooks, scripts, data/features/models modules, initial tests. | Project started as a modular ML pipeline, not a single notebook. |
| 2026-06-01 | `c7b95a4 milestone-1` | Updated `fetch.py` and `validate.py`. | Raw data presence and structural validation became an explicit pipeline gate. |
| 2026-06-02 | `7d76a09` | Cleaning, feature engineering, diagnostics, tests improved. | Core data preparation and feature generation matured. |
| 2026-06-02 | `ee81a09` | Removed misspelled `requirments.txt`. | Dependency manifest cleanup. |
| 2026-06-02 | `341b84c` | Added MIT license and detailed README. | Documentation/license milestone. |
| 2026-06-03 | `e6a9aa1` | Baseline evaluation, model training, visualizations. | Shift from feature prep to ML evaluation. |
| 2026-06-03 | `043ac47` | Flask API, prediction endpoint, simulation model. | Project became service-oriented. |
| 2026-06-03 | `b77ac1a` | README usage and retraining scripts. | Operational workflow documented. |
| 2026-06-03 | `91aa895` | Multi-model training and hyperparameter tuning. | Model comparison expanded beyond basic classifier. |
| 2026-06-03 | `15e5da3` | Dashboard templates and route updates. | UI layer introduced. |
| 2026-06-03 | `495e1a0` | README update for dashboard/model roadmap. | Documentation synced to new UI/model scope. |
| 2026-06-03 | `9609607` | Dynamic XGBoost import fallback. | Dependency robustness improved. |
| 2026-06-04 | `d9a02b0` | Dashboard layout refinements, searchable analytics combobox, privacy/terms. | Frontend interaction and legal pages expanded. |
| 2026-06-04 | `9911b94` | Docker, compose, live page, API client, fetch script. | Deployment and live data integration introduced. |
| 2026-06-04 | `3a2f264` | Scheduler, UI and live API improvements, prediction/simulation changes. | Background retraining and runtime integration expanded. |
| 2026-06-04 | `a366eee` | README update. | Documentation refresh. |
| 2026-06-05 | `efccd56` | Ensembling, Platt/isotonic calibration, advanced rolling features, CatBoost artifacts, calibration plot. | Probability calibration and richer ML pipeline introduced. |
| 2026-06-05 | `ed1210a` | Template refactor. | UI maintainability cleanup. |
| 2026-06-05 | `84815c0` | Prediction/simulation efficiency improvements. | Runtime performance work began. |
| 2026-06-05 | `ae34ea5` | Simulate import/readability refactor. | Internal cleanup. |
| 2026-06-05 | `4063297` | Training/evaluation enhancements, dummy model fixture, more artifact churn. | Build/test reliability improved; artifacts later removed from git. |
| 2026-06-05 | `5acc123` | Removed obsolete CatBoost training files. | Generated CatBoost files cleaned from repository history. |
| 2026-06-06 | `bde54a6`, `51ba37b` | Visualization and `.gitignore` updates. | Generated outputs refreshed; ignored directories expanded. |
| 2026-06-06 | `c2139ab` | Prediction/simulation refactor and visualizations. | Runtime code cleanup. |
| 2026-06-06 | `b98b550` | Draw threshold, Poisson model, Elo optimizer, feature extensions. | Scoreline modeling and draw-risk logic added. |
| 2026-06-06 | `2b30090`, `1a27e90` | Model initialization, threshold loading, fallback loading, hot-reload safety. | Prediction runtime made more resilient. |
| 2026-06-07 | `d85750b` | Poisson goal model and prediction system, tests added. | Current state emphasizes calibrated model plus scoreline model. |

Unavailable historical information:

- The repository does not contain issue tickets, design docs, PR reviews, or meeting notes explaining personal reasoning behind each commit.
- The actual original sources and download dates for raw datasets are described in code comments but not proven by embedded source metadata.

## 4. File-by-File Analysis

### Root Configuration and Project Files

#### `.gitignore`

Purpose: prevents committing caches, virtualenvs, raw/processed/features data, model registry, logs, and CatBoost training folders.

Why it exists: the repo contains large/generated artifacts locally. Git history shows data/model/cache paths were added or removed over time. Ignoring them keeps version control focused on source.

Interactions: affects whether `data/raw`, `data/processed`, `data/features`, `models/registry`, and `logs` are tracked. Tests compensate for missing model artifacts through `tests/conftest.py`.

Interview points:

- Question: Why are data and model files ignored?
- Strong answer: They are generated or large artifacts. The reproducible contract is the pipeline plus config, while artifacts can be regenerated locally if raw data and dependencies are present.
- Common mistake: Treating ignored models as always present in deployment. This project partially mitigates tests with dummy models, but production runtime still needs real artifacts.

#### `.dockerignore`

Purpose: reduces Docker build context by excluding virtualenv, git data, caches, editor files, logs, bytecode, `.env`, and `models/registry`.

Important risk: it also contains `Dockerfile`. The Dockerfile reference treats Dockerfile as the build instructions; excluding it from context can be risky depending on how the builder resolves `-f Dockerfile`. In this repo `docker-compose.yaml` sets `dockerfile: Dockerfile`, so this should be tested with Docker locally.

#### `.env`

Purpose: local environment variables for API keys/settings. Evidence from code: `src/utils/api_football.py` reads `RAPIDAPI_KEY` and `FOOTBALL_DATA_API_KEY`; `scripts/scheduler.py` reads `RETRAINING_INTERVAL_HOURS`.

Security: values are intentionally not documented. `.gitignore` and `.dockerignore` both exclude `.env`.

#### `LICENSE`

Purpose: MIT license. Allows use, modification, distribution, sublicensing, and sale with copyright/license notice retained.

#### `README.md`

Purpose: project-facing documentation. It describes overview, features, project structure, setup, API endpoints, model performance, config, notebooks, tech stack, roadmap, and license.

Important evidence: README model performance aligns with `models/registry/meta.json`, including Stacking Ensemble as active best.

Risk: terminal output showed encoding artifacts for emojis/accented characters. The file itself may be UTF-8, but Windows terminal display can be misconfigured.

#### `requirements.txt`

Purpose: dependency manifest for local install and Docker image. It pins core versions for pandas, SciPy, scikit-learn, seaborn, matplotlib, requests, Flask, PyYAML, Jupyter, notebook, ipykernel, python-dotenv, joblib, and minimum versions for pytest/LightGBM/CatBoost/XGBoost.

Why exact pins matter: ML artifacts serialized with joblib are often sensitive to package versions. Pinning reduces load-time incompatibility.

Risk: Python version is not pinned here; Docker uses Python 3.12 while README says Python 3.10+. Current tests passed in the local venv.

#### `config.yaml`

Purpose: central source of truth.

Key blocks:

- `paths`: raw, processed, features, models, visualisations.
- `data`: source filenames, date range, minimum team match count.
- `features`: form/goals windows, Elo initial/HFA/K, EWMA alphas.
- `model`: train/calibration cutoffs, selection metric, thresholds, random state, CV splits.
- `score_model`: Poisson/Dixon-Coles parameters and rho grid.
- `api`: host, port, debug, cache TTL.

Runtime: `src/utils/config.py` loads it once at import time into global `config`.

Risk: because config is imported as a dictionary singleton, scripts like `scripts/optimize_elo.py` mutate it in memory before writing YAML. Long-running processes will not reload edits automatically unless code reloads modules or is restarted.

#### `Dockerfile`

Purpose: builds a Python 3.12 slim container, installs dependencies, copies the repo, runs tests, exposes port 5000, starts `python -m src.api.app`.

Runtime flow:

```text
FROM python:3.12-slim
  -> set Python/threading env vars
  -> WORKDIR /app
  -> install libgomp1 for LightGBM/OpenMP
  -> pip install requirements
  -> COPY repository
  -> python -m pytest tests/ -v
  -> CMD Flask app
```

Risk: `.dockerignore` excludes `models/registry`, but `tests/conftest.py` generates dummy artifacts if core files are missing. That makes image build pass but may produce a container with toy models unless real artifacts are mounted at runtime. `docker-compose.yaml` mounts `./models:/app/models`, which restores local real artifacts if present.

#### `docker-compose.yaml`

Purpose: defines two services:

- `web`: builds image, maps `5000:5000`, loads `.env`, mounts source/data/models/visualisations/config.
- `scheduler`: same image, runs `python -m scripts.scheduler`, loads `.env`, mounts same project folders.

Design: split web serving from periodic data refresh/retrain loop.

Risk: both services mount and may mutate the same `models` and `data` directories. The predictor supports hot reload, but concurrent writes to pickle files are still a deployment concern.

### Data and Artifact Files

#### `data/raw/matches.csv`

Purpose: raw international match history. Local shape: `49387` rows, `9` columns. Columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral`.

Data flow: read by `src/data/validate.py`, `src/data/clean.py`, `scripts/diagnostics.py`, and appended by `scripts/fetch_recent_matches.py`.

#### `data/raw/fifa_rankings.csv`

Purpose: FIFA ranking snapshots. Local shape: `13130` rows, `8` columns. Required columns include `date`, `semester`, `rank`, `team`, `total.points`.

Data flow: read by `clean.py`, transformed into Jan/Jul snapshot dates, merged twice into matches as home and away ranking data.

#### `data/raw/elo_ratings.csv`

Purpose: externally sourced Elo data used only for validation/diagnostics in current source. Local shape: `6678` rows, `4` columns. Current feature generation computes Elo from match history instead of merging this file.

Interview point: If asked "Do you use raw Elo ratings for model features?", the correct answer is no in current code; `src/features/elo.py` computes Elo from `matches_clean.csv`.

#### `data/processed/matches_clean.csv`

Purpose: cleaned, normalized, ranking-merged match table. Local shape: `49315` rows, `15` columns.

Created by: `src/data/clean.py`.

Consumed by: `src/features/build.py` and `scripts/optimize_elo.py`.

#### `data/features/feature_matrix.csv`

Purpose: final model-ready feature table. Local shape: `15558` rows, `37` columns.

Created by: `src/features/build.py`.

Consumed by: `src/models/train.py`, `src/models/evaluate.py`, `src/models/predict.py`, tests, and API team-detail endpoints.

Key model input columns: `models/registry/meta.json` lists 27 active features.

#### `data/live_cache/fixtures.json` and `standings.json`

Purpose: cached API-Football-compatible payloads for live fixtures and standings. Current local fixtures cache has `104` responses; standings cache has `12` groups of `4` teams.

Created/updated by: `src/utils/api_football.py`.

Consumed by: `/api/live/fixtures` and `/api/live/standings`.

#### `models/registry/*.pkl`

Purpose: serialized model artifacts. Current local registry includes calibrated model files, uncalibrated best model, scaler, score model, and individual calibrated model variants.

Key artifacts:

- `best_model.pkl`: active calibrated classifier.
- `best_model_uncalibrated.pkl`: raw champion base model.
- `scaler.pkl`: StandardScaler parameters.
- `score_model.pkl`: Poisson goal model used for expected goals/scoreline grids.
- `meta.json`: model metadata and metrics.

Security note: joblib loading uses pickle-style persistence. joblib documentation warns that loading arbitrary pickles can execute code. This project only loads local registry files.

#### `models/registry/meta.json`

Purpose: runtime contract between training and prediction/evaluation.

Important fields:

- `model_type`: `Stacking Ensemble`.
- `selected_by`: `log_loss`.
- `features`: 27 ordered feature names. Prediction must build vectors in this exact order.
- `classes`: `["H", "D", "A"]`.
- `draw_threshold`: `1.0`.
- `draw_risk_threshold`: `0.3`.
- `score_model`: Poisson model metadata, including `rho: 0.03`.
- `comparison`: per-model holdout metrics.

Risk: the classification report shows `D (Draw)` precision/recall/f1 all `0.00`, meaning the current balanced threshold still does not predict draws on holdout when `draw_threshold` is `1.0`.

#### `visualisations/*.png`

Purpose: generated evaluation outputs served by `/api/visualisations/<filename>` and shown in `insights.html`.

Created by: `src/models/evaluate.py`.

#### `logs/*.log`

Purpose: runtime log files created by `src/utils/logger.py` and Flask runs. Ignored by git.

### Python Package Initializers

Files: `src/__init__.py`, `src/api/__init__.py`, `src/data/__init__.py`, `src/features/__init__.py`, `src/models/__init__.py`, `src/utils/__init__.py`.

Purpose: mark folders as Python packages so imports like `from src.models.predict import MatchPredictor` work.

Current content: empty.

### Shared Utilities

#### `src/utils/config.py`

Imports: `yaml`, `Path`.

Exports: `PROJECT_ROOT`, `load_config`, `config`.

Execution:

1. Calculates project root as two parents above `src/utils/config.py`.
2. `load_config()` opens `config.yaml`.
3. `yaml.safe_load` parses YAML into a dictionary.
4. `config = load_config()` runs at import time.

Why it exists: every module needs consistent paths and parameters without hardcoding root-relative paths repeatedly.

Edge cases:

- Missing or invalid `config.yaml` raises during import.
- Changes to YAML after import are not reflected unless modules reload or mutate `config`.

#### `src/utils/logger.py`

Imports: `logging`, `sys`, `Path`, `PROJECT_ROOT`.

Exports: `get_logger(name)`.

Execution:

1. Gets logger by name.
2. If handlers already exist, returns it to avoid duplicate logs.
3. Sets DEBUG level.
4. Adds INFO console handler to stdout.
5. Creates `logs/`.
6. Adds DEBUG file handler writing `logs/app.log`.

Why it exists: provides consistent logs across pipeline, API, and scripts.

Edge cases:

- File handler writes with default platform encoding; non-ASCII logs may be terminal/encoding-sensitive.
- Handler check is per logger name, not global root.

#### `src/utils/api_football.py`

Purpose: live data integration with cache and fallback.

Imports: `os`, `json`, `time`, `Path`, `datetime`, `requests`, `load_dotenv`, logger, project root, team cleaner.

Key globals:

- `RAPIDAPI_KEY`, `FOOTBALL_DATA_API_KEY`.
- `CACHE_DIR`, `FIXTURES_CACHE`, `STANDINGS_CACHE`.
- hardcoded 12 World Cup groups.

Important functions:

- `_ensure_cache_dir`: creates `data/live_cache`.
- `_is_cache_valid`: checks cache file age against `config["api"]["cache_ttl_hours"]`.
- `_generate_mock_standings`: builds API-Football-style zero-point standings for the hardcoded groups.
- `_generate_mock_fixtures`: builds a mock group-stage fixture schedule.
- `fetch_from_api`: calls RapidAPI API-Football with timeout and error handling.
- `fetch_from_football_data`: calls Football-Data.org with token and timeout.
- `map_football_data_standings`: converts Football-Data.org standings shape into API-Football-like shape.
- `map_football_data_fixtures`: converts Football-Data.org matches shape into API-Football-like shape.
- `get_standings`: cache -> Football-Data.org -> API-Football -> expired cache -> mock.
- `get_fixtures`: cache -> Football-Data.org -> API-Football -> expired cache -> mock.

Interview points:

- Question: Why have two provider formats?
- Strong answer: The frontend and routes expect one normalized schema. Provider-specific mapping isolates API differences from UI code.
- Question: What is the downside of mock fallback?
- Strong answer: Availability improves, but stale/mock data can be mistaken for real unless response metadata identifies source.

### Data Pipeline

#### `src/data/fetch.py`

Purpose: confirms raw files exist and are non-empty. It does not download files.

Dependencies: `Path`, `config`, logger.

Runtime:

1. Reads raw path from config.
2. Defines required files and human download instructions.
3. Checks existence and byte size.
4. Logs success/failure.
5. Raises `FileNotFoundError` if any file is missing/empty.

Why it exists: prevents later pandas errors from missing raw inputs.

#### `src/data/validate.py`

Purpose: validates raw data schema and sanity before cleaning.

Key constants:

- `MIN_ROWS`: conservative lower bounds.
- `REQUIRED_COLUMNS`: schema expectations.
- `KNOWN_TEAMS`: teams that must appear in match data.

Functions:

- `_load`: reads from raw data path.
- `validate_matches`: row count, required columns, no null identity fields, known teams, boolean-like `neutral`.
- `validate_rankings`: row count, required columns, rank >= 1.
- `validate_elo`: row count, required columns, no negative ratings.
- `run_all_validations`: executes all validators.

Edge cases:

- `neutral` must be actual `True`/`False`; string values such as `"TRUE"` from appended data could fail if present in raw CSV parsing.
- It validates raw Elo file even though model features compute Elo internally.

#### `src/data/clean.py`

Purpose: creates `matches_clean.csv`.

Key logic:

- `TEAM_MAPPING`: canonicalizes variants such as `USA -> United States`, `Korea Republic -> South Korea`, `Congo DR -> DR Congo`.
- `clean_team_name`: handles non-strings, strips non-breaking spaces, applies mapping.
- `run_cleaning`: loads matches/rankings, parses dates, normalizes names, drops null scores, creates `result`, creates `is_competitive`, merges ranking snapshots, fills missing rankings, saves processed CSV.

Ranking merge:

- Rankings semester is converted to month 1 or 7.
- Home and away ranking tables are separately renamed.
- `pd.merge_asof(..., by="home_team", direction="backward")` assigns the latest ranking snapshot at or before match date. pandas requires sorted merge keys for this operation.

Why filtering is deferred: comments state date/min-match filtering is delayed until `build.py` so Elo/form/goal calculations have historical warm-up.

Common mistakes:

- Filtering too early would make first included matches look like teams have no history.
- Exact merge on ranking date would miss most matches because rankings are snapshots, not daily.

### Feature Engineering

#### `src/features/elo.py`

Purpose: computes chronological Elo ratings from match history.

Functions:

- `calculate_expected_score(rating_a, rating_b)`: standard Elo logistic curve.
- `get_k_factor(tournament)`: World Cup 60, major continental/confederations 50, qualification/nations league 40, other 20.
- `goal_margin_multiplier(goal_diff)`: 1.0 for margin <= 1, 1.5 for margin 2, `(11 + abs_diff) / 8` for larger margins.
- `compute_elo_ratings(matches_df)`: processes matches in date order, records pre-match Elo columns, then updates team ratings.

Runtime data flow:

```text
matches_clean rows
  -> sort by date
  -> initialize unseen teams at config elo_initial
  -> append pre-match home_elo/away_elo
  -> apply home-field adjustment for expected score if not neutral
  -> update ratings by K * margin_multiplier * (actual - expected)
  -> output home_elo, away_elo, elo_diff
```

Interview answer: The model uses pre-match Elo as a feature, not post-match Elo, because the code appends ratings before updating them for the current row.

#### `src/features/form.py`

Purpose: computes recent form as an opponent-adjusted EWMA.

Functions:

- `compute_ewma_form(history, alpha)`: returns 0.5 cold-start, otherwise weighted average of adjusted points scaled by 3.
- `compute_form_features(matches_df)`: for each match, records form before the match, then appends opponent-adjusted points to team histories.

Opponent adjustment: points are multiplied by `1 + (opponent_elo - 1500) / 1000`, clamped to minimum `0.5`.

Edge case: form is clamped to `[0, 1]`, limiting outlier effects.

#### `src/features/goals.py`

Purpose: computes rolling scoring/defensive strength.

Functions:

- `compute_ewma_goals(history, alpha, default)`: returns default for no history.
- `compute_goal_features(matches_df)`: records pre-match EWMA goals scored/conceded for both teams, then updates histories with current score.

Default goals: `1.2`.

Output columns: scored average, conceded average, and goal-diff average for home and away teams.

#### `src/features/build.py`

Purpose: orchestrates feature construction and saves `feature_matrix.csv`.

Functions:

- `compute_advanced_features`: rest days, continent/home-continent indicators, match stake.
- `build_feature_matrix`: loads cleaned data, runs Elo/form/goals/advanced features, computes ranking diffs, filters date/team counts, drops null rows, saves output.

Advanced feature details:

- Rest days: last match date per team, default 30, capped at 30.
- Continent features: hardcoded `CONTINENT_MAP`; compares team continent to match country mapped through the same dictionary.
- Match stake: World Cup 4, major tournament 3, qualifiers/nations league 2, otherwise 1.

Risk: `CONTINENT_MAP` is duplicated in `src/models/predict.py`; future changes can drift.

### Model Training, Evaluation, and Prediction

#### `src/models/baseline.py`

Purpose: evaluates simple baselines on the post-cutoff test set.

Baselines:

- Uniform random probabilities.
- Most frequent class with training base-rate probabilities.
- Elo heuristic: higher Elo predicts win, no draw predictions.

Why it exists: gives performance floor for ML models.

#### `src/models/poisson_model.py`

Purpose: scikit-learn-compatible goal-count model.

Class: `PoissonGoalModel(BaseEstimator, ClassifierMixin)`.

Internal models:

- `home_regressor`: `PoissonRegressor`.
- `away_regressor`: `PoissonRegressor`.

Important methods:

- `fit`: fits regressors on actual goal pairs when `y` is 2D; otherwise uses dummy goals inferred from H/D/A class.
- `predict_expected_goals`: returns positive lambdas, floored at `0.1`.
- `_dixon_coles_tau`: low-score correction factor for 0-0, 0-1, 1-0, 1-1.
- `scoreline_matrix_for_lambdas`: builds truncated Poisson home/away score grid, applies correction, normalizes.
- `predict_proba`: sums grid lower triangle for home wins, diagonal for draws, upper triangle for away wins.
- `tune_rho`: grid-searches rho by negative log likelihood on calibration goal scores.
- `scoreline_dict`: returns expected goals and top scorelines.

Interview point: This bridges score modeling and classification by deriving H/D/A probabilities from a scoreline distribution.

#### `src/models/train.py`

Purpose: train, tune, calibrate, compare, select, and serialize models.

Key constants:

- `FEATURE_COLUMNS`: 27 input features. Must match `meta.json` and prediction feature construction.

Functions:

- `select_best_model`: chooses model by accuracy or by log-loss/Brier/accuracy tie-breakers.
- `find_optimal_draw_threshold`: tries thresholds `0.15` to `0.44` to improve calibration-set accuracy for draw decisions.
- `predict_classes_with_threshold`: if draw probability exceeds threshold, predicts draw; otherwise chooses home/away by larger probability.
- `train_model`: full training workflow.

Training workflow:

```text
load feature_matrix
  -> map H/D/A to 0/1/2
  -> split by train_cutoff and calibration_cutoff
  -> fit StandardScaler on train only
  -> GridSearchCV with TimeSeriesSplit on train
  -> fit PoissonGoalModel on score targets
  -> optionally build StackingClassifier
  -> calibrate each model on calibration set
  -> tune draw threshold per model
  -> evaluate on holdout test set
  -> select champion
  -> dump artifacts and meta.json
```

Why calibration exists: tournament simulation and UI probabilities depend on truthful probabilities, not just predicted labels. scikit-learn supports sigmoid and isotonic calibration through `CalibratedClassifierCV`.

Risk:

- `cv="prefit"` in `CalibratedClassifierCV` is used. This pattern depends on scikit-learn version behavior and can be deprecated/changed over versions.
- Feature list is duplicated in `evaluate.py` and indirectly in `predict.py` through `meta.json`.
- Heavy grid search can be slow because Random Forest and boosters run multiple parameter combinations.

#### `src/models/evaluate.py`

Purpose: loads trained artifacts, evaluates on holdout, writes plots and updates metadata.

Flow:

1. Load `best_model.pkl`, `scaler.pkl`, `meta.json`.
2. Load `feature_matrix.csv`.
3. Use calibration cutoff as test boundary.
4. Transform features with scaler.
5. Predict probabilities and thresholded classes.
6. Log classification report, accuracy, log loss, Brier.
7. Save confusion matrix.
8. Save feature importance chart using native importances, coefficients, stacking meta weights, or permutation fallback.
9. Save calibration curve.
10. Update `meta["evaluation"]`.

Why `matplotlib.use("Agg")`: enables non-interactive plot rendering in scripts/containers.

#### `src/models/predict.py`

Purpose: runtime predictor for single matches.

Class: `MatchPredictor`.

Initialization:

1. Resolve model/scaler/meta/score-model/feature-matrix paths.
2. Load scaler.
3. Load requested model with fallback to `histgradientboosting.pkl`, `logistic_regression.pkl`, or `best_model.pkl` if package imports fail.
4. Load metadata/classes/features.
5. Load score model.
6. Determine draw threshold for current model filename.
7. Load feature matrix and build latest team states.
8. Store model modification time and initialize prediction cache.

State model:

- `_build_team_state` extracts the latest pre-match features for each team depending on whether it appeared home or away in its latest row.
- `get_team_state` returns default state for unknown teams: Elo 1500, form 0.5, goals 1.2, rank 211, rank points 0.

Feature construction:

- `_construct_features_numpy` builds an array in `self.features` order.
- `_construct_features` builds an equivalent DataFrame but is not used in current prediction hot path.
- `_scale_features` manually applies `(features - scaler.mean_) / scaler.scale_`.

Prediction:

- `predict_scoreline` uses the Poisson score model. Neutral matches average the forward grid with transposed reverse grid.
- `predict_match` checks hot reload, builds a rounded cache key, predicts forward direction, optionally predicts reverse direction for neutral venues, averages probabilities, applies draw threshold, computes draw risk, adds expected goals/top scorelines, returns JSON-ready dict.

Important behavior:

- Neutral symmetric averaging is directly tested in `tests/test_predict.py`.
- Non-neutral asymmetry is directly tested.
- Cache key rounds state deltas, so similar states may reuse predictions. This improves speed but can hide small state changes.

#### `src/models/simulate.py`

Purpose: Monte Carlo World Cup simulation.

Globals:

- Thread-related environment variables limit BLAS/joblib multiprocessing.
- `GROUPS`: hardcoded 12 groups of 4 teams.
- `HOSTS`: United States, Mexico, Canada.

Class: `TournamentSimulator`.

Initialization:

- Creates `MatchPredictor`.
- Stores baseline team states as deep-copy JSON and fast dict copies.

State updates:

- `_update_stats_after_match` updates Elo, form, and goal averages after each simulated match. It mirrors feature logic using World Cup K-factor and EWMA updates.

Match simulation:

- `_simulate_match` assigns host as home when one team is a host; otherwise neutral.
- Uses `predictor.predict_scoreline`.
- Samples a scoreline from the score grid when available; otherwise samples independent Poisson goals.
- Determines H/D/A result.
- Knockout draws go to extra time and then Elo-weighted shootout.
- Updates team state.
- Returns scores in original `team_a`, `team_b` order.

Tournament simulation:

- `simulate_group_stage`: round-robin groups, ranks by points, goal difference, goals for.
- `_get_knockout_bracket_teams`: selects top two plus best eight third-place teams and maps them into a hardcoded Round of 32 bracket.
- `simulate_tournament`: group -> R32 -> R16 -> QF -> SF -> final.
- `simulate_detailed_tournament`: returns detailed group matches, standings, knockout rounds, champion.
- `run_monte_carlo`: uses multiprocessing pool to run many tournaments and returns champion probabilities.

Maintenance risk:

- `_simulate_match_fast` immediately returns `_simulate_match(...)`; all optimized NumPy code below that statement is unreachable. Therefore `simulate_tournament_fast` and `fast_simulate_group_stage` are not truly fast in current runtime.

### Flask API and Pages

#### `src/api/app.py`

Purpose: Flask application factory.

Runtime:

1. Sets environment variables to limit numeric library threads.
2. `create_app` constructs Flask app.
3. Copies API config from YAML.
4. Registers `after_request` CORS headers.
5. Imports and registers API and page blueprints.
6. `__main__` runs app on configured host/port.

Why factory pattern: Flask docs recommend app factories for deferred app creation and cleaner registration patterns. This project uses it even though the route module still has global model objects.

#### `src/api/routes.py`

Purpose: JSON API endpoints and page rendering.

Globals:

- `api_bp`: `/api` blueprint.
- `pages_bp`: page blueprint.
- `predictor = MatchPredictor()`.
- `simulator = TournamentSimulator()`.

API endpoints:

| Endpoint | Method | Purpose | Calls |
|---|---|---|---|
| `/api/health` | GET | Health check | none |
| `/api/teams` | GET | List known teams | `predictor.team_states` |
| `/api/predict` | POST | Single match forecast | `predictor.predict_match` |
| `/api/simulate` | POST | Monte Carlo champion probabilities | `simulator.run_monte_carlo` |
| `/api/simulate-detailed` | POST | One detailed tournament run | `simulator.simulate_detailed_tournament` |
| `/api/team-details/<team>` | GET | Team profile/radar stats | predictor state and feature matrix |
| `/api/team-matches/<team>` | GET | Recent five matches | feature matrix |
| `/api/visualisations/<filename>` | GET | Serve plots | filesystem |
| `/api/model-meta` | GET | Return `meta.json` | filesystem |
| `/api/live/standings` | GET | Flatten live standings | `get_standings` |
| `/api/live/fixtures` | GET | Flatten fixtures | `get_fixtures` |

Pages:

- `/`, `/predict`, `/analytics`, `/insights`, `/about`, `/simulate`, `/live`, `/privacy`, `/terms`.

Risks:

- No authentication/authorization.
- CORS allows all origins.
- `/api/visualisations/<filename>` serves requested filenames from visualisations directory; Flask's `send_from_directory` helps constrain path traversal, but file allow-listing would be stronger.
- `n_sims` is capped at `5000`; still potentially expensive because multiprocessing loads simulators per worker.

### Templates

All templates use Tailwind via CDN and Material Symbols. There is no separate frontend build system.

| Template | Purpose | API calls/functions |
|---|---|---|
| `home.html` | Landing/dashboard overview | fetches `/api/model-meta` to populate model labels/metrics. |
| `predict.html` | Match prediction UI | fetches `/api/teams`, `/api/model-meta`, posts `/api/predict`; functions include combobox handling and result rendering. |
| `analytics.html` | Team analytics UI | fetches `/api/teams` and team-specific endpoints; renders radar/team stats. |
| `insights.html` | Model metrics and plots | fetches `/api/model-meta`; embeds `/api/visualisations/*.png`. |
| `live.html` | Live standings/fixtures | fetches `/api/live/standings` and `/api/live/fixtures`. |
| `simulate.html` | Monte Carlo and interactive tournament UI | calls `/api/simulate`, `/api/simulate-detailed`, `/api/teams`, team detail/match endpoints. |
| `about.html` | Static methodology/about page | no API calls. |
| `privacy.html` | Static privacy page | no API calls. |
| `terms.html` | Static terms page | no API calls. |

Frontend architecture:

- Server-side route renders static HTML templates.
- Browser-side JavaScript directly calls Flask JSON endpoints with `fetch`.
- State lives in DOM elements and page-level JS variables, not React/Vue/store libraries.

### Scripts

#### `scripts/run_pipeline.py`

Purpose: full six-step pipeline: file check, validation, cleaning, feature engineering, training, evaluation.

Failure behavior: logs critical error and exits with code 1.

#### `scripts/retrain.py`

Purpose: faster four-step pipeline: cleaning, feature building, training, evaluation. Skips raw checks and validations.

Use case: when raw files are already trusted.

#### `scripts/fetch_recent_matches.py`

Purpose: fetch finished World Cup matches from Football-Data.org, append new matches to raw CSV, retrain.

Flow:

1. Load `.env`.
2. Read existing match keys `(date, home_team, away_team)`.
3. Query `competitions/WC/matches`.
4. Parse finished matches only.
5. Normalize team names.
6. Append deduplicated new records.
7. Run cleaning, feature building, training, evaluation.

Risk: `neutral` is written as `"TRUE"`/`"FALSE"` strings, while validation expects boolean-like values in raw data. Since this script does not run validation before retraining, the pipeline may still work if pandas parses strings acceptably downstream, but the full validation step may fail depending on CSV parsing.

#### `scripts/scheduler.py`

Purpose: infinite loop scheduler for Docker Compose `scheduler` service.

Config: `RETRAINING_INTERVAL_HOURS`, default `24`.

Risk: no distributed lock. Running multiple scheduler instances could retrain and write artifacts concurrently.

#### `scripts/diagnostics.py`

Purpose: one-off diagnostic print script for data shapes, team overlap, Elo coverage, ranking date format, tournament distribution.

Architecture role: investigation tool, not part of normal runtime.

#### `scripts/optimize_elo.py`

Purpose: grid-searches feature hyperparameters for Elo home advantage, form alpha, and goals alpha using Logistic Regression validation log loss, then writes best values back to `config.yaml`.

Risk: directly edits `config.yaml`; current best params in config are an output of prior optimization but the script itself does not preserve a backup.

### Tests

#### `tests/conftest.py`

Purpose: session autouse fixture creates dummy model artifacts when core registry files are missing. This makes tests pass in clean Docker builds where real models are git-ignored.

Artifacts generated: scaler, HGB model, calibrated best model, logistic model, Poisson score model, metadata.

Interview point: This is a testability workaround. It validates code paths, not production model quality.

#### `tests/test_clean.py`

Covers `clean_team_name`, outcome derivation, competitive flag logic.

#### `tests/test_features.py`

Covers expected score, K-factor, goal margin multiplier, EWMA form/goals, Elo rating outputs, form features, goal features.

#### `tests/test_model_selection.py`

Covers probability-first model selection and tie-breakers.

#### `tests/test_poisson_model.py`

Covers scoreline grid normalization and rho tuning.

#### `tests/test_predict.py`

Covers predictor initialization, output shape/probability sum, neutral symmetry, non-neutral asymmetry, default context equivalence.

#### `tests/test_simulate.py`

Regression test proving away `goal_diff_avg` equals away scored average minus away conceded average after simulated state update.

### Notebooks

Current state:

- `notebooks/01_eda.ipynb`: valid notebook JSON with zero cells.
- `notebooks/02_feature_engineering.ipynb`: empty file.
- `notebooks/03_model_evaluation.ipynb`: empty file.

Therefore, no notebook-based methodology can be reconstructed from current notebook contents.

## 5. Architecture Deep Dive

### Frontend Architecture

Routing is Flask server-rendered:

```text
GET /           -> home.html
GET /predict    -> predict.html
GET /analytics  -> analytics.html
GET /insights   -> insights.html
GET /simulate   -> simulate.html
GET /live       -> live.html
GET /about      -> about.html
GET /privacy    -> privacy.html
GET /terms      -> terms.html
```

State management is page-local JavaScript and DOM state. There is no SPA router, no component framework, no client-side store, and no package bundler in the repo.

Data flow:

```text
Template rendered by Flask
  -> Browser JS calls /api/*
  -> Flask returns JSON
  -> JS mutates DOM
```

Rendering lifecycle:

1. Flask returns HTML.
2. Tailwind CDN and Material Symbols load from external URLs.
3. Inline JS initializes controls.
4. JS fetches API data.
5. DOM updates with model/team/simulation results.

### Backend Architecture

Backend is Flask plus local Python service classes, not a separate API/controller/service framework.

Routes are in `src/api/routes.py`. Business logic mostly lives outside routes:

- `MatchPredictor` handles prediction.
- `TournamentSimulator` handles simulation.
- `api_football.py` handles live provider/cache logic.

Authentication/authorization: none in current code.

Middleware: only `after_request` CORS header injection.

Error handling: route-level try/except returning `500` JSON for prediction/simulation/live failures, explicit `400` for missing prediction fields or invalid simulation count.

### Database Architecture

There is no database. CSV and JSON files act as storage:

- Raw source data: CSV.
- Processed/features: CSV.
- Model registry: pickle and JSON.
- Live cache: JSON.
- Logs: text.

No schema migrations, indexes, SQL constraints, ORM models, or database transactions exist.

### Infrastructure

Build:

- Docker builds Python image, installs dependencies, runs tests.

Deployment:

- Docker Compose defines web and scheduler.
- No cloud deployment config is present.

Environment:

- `.env` for API keys and scheduler interval.
- `config.yaml` for application/model settings.

CI/CD:

- No GitHub Actions, GitLab CI, or other CI config exists in the current tracked files.

Monitoring:

- Logging to stdout and `logs/app.log`.
- No metrics, tracing, alerting, Sentry, Prometheus, or health monitoring beyond `/api/health`.

## 6. Dependency Analysis

| Dependency | Used in project | Why it exists | Alternatives/trade-offs |
|---|---|---|---|
| pandas | Data CSV loading, cleaning, merging, feature tables | Tabular data manipulation | Polars is faster but would require API rewrites. |
| numpy | Arrays, probabilities, score grids, random sampling | Numeric operations | Pure Python slower and less concise. |
| scipy | Poisson PMF | Goal scoreline probabilities | Manual PMF possible but less tested. |
| scikit-learn | Models, scalers, metrics, calibration, CV | Core ML training/evaluation | More flexible than hand-coded models; artifact version sensitivity. |
| seaborn | Confusion matrix heatmap | Higher-level statistical plotting | matplotlib-only is more verbose. |
| matplotlib | Plot generation | Saves evaluation PNGs | Plotly could make interactive plots but adds frontend complexity. |
| requests | HTTP API calls | Football providers | httpx supports async; current app is sync Flask. |
| Flask | API and pages | Lightweight web server | FastAPI would provide validation/OpenAPI but require migration. |
| PyYAML | Config loading | YAML config | TOML/JSON are alternatives; YAML is human-friendly but indentation-sensitive. |
| jupyter/notebook/ipykernel | Notebook environment | Intended exploratory analysis | Current notebooks are empty. |
| python-dotenv | `.env` loading | Local secrets/config | Docker/OS env variables without dotenv are simpler in production. |
| joblib | Model serialization | Stores scikit-learn-like objects | pickle equivalent but joblib is common for NumPy-heavy objects; same untrusted-load risk. |
| pytest | Unit tests | Test runner and fixtures | unittest built-in but less ergonomic. |
| lightgbm | Gradient boosting model | Model comparison | Extra native dependency; Docker installs `libgomp1`. |
| catboost | Gradient boosting model | Model comparison | Produces training artifacts; can be heavier. |
| xgboost | Gradient boosting model | Model comparison | Extra dependency; code includes fallback for missing package. |

## 7. Runtime Execution Flow

### Full Pipeline Startup

```text
python -m scripts.run_pipeline
  -> import config/logger/modules
  -> check_raw_files
  -> run_all_validations
  -> run_cleaning
  -> build_feature_matrix
  -> train_model
  -> generate_evaluation_report
```

### Flask Startup

```text
python -m src.api.app
  -> set numeric thread env vars
  -> create_app()
  -> import src.api.routes
       -> instantiate MatchPredictor
       -> instantiate TournamentSimulator
            -> instantiate another MatchPredictor
  -> register blueprints
  -> app.run(host, port)
```

### Prediction Request Lifecycle

```text
POST /api/predict
  -> parse JSON
  -> validate home_team and away_team
  -> strip strings
  -> predictor.predict_match
       -> optional hot reload
       -> get latest team states
       -> build feature array
       -> scale feature array
       -> model.predict_proba
       -> for neutral: reverse teams, invert, average
       -> apply draw threshold
       -> predict scoreline via Poisson model
       -> cache and return dict
  -> jsonify response
```

### Tournament Simulation Lifecycle

```text
POST /api/simulate
  -> validate n_sims
  -> simulator.run_monte_carlo
       -> clear predictor cache
       -> disable reload
       -> multiprocessing Pool
       -> each worker creates TournamentSimulator
       -> each run resets states
       -> group stage
       -> knockout bracket
       -> champion
       -> aggregate champion counts
  -> return probabilities
```

### Shutdown Behavior

No custom shutdown hooks are implemented. Flask and Python process termination handle cleanup. Scheduler runs forever until process/container stops.

## 8. Data Flow Analysis

### Training Data Flow

```text
matches.csv + fifa_rankings.csv
  -> clean names
  -> match outcomes
  -> competitive flag
  -> ranking snapshot merge
  -> matches_clean.csv
  -> chronological Elo/form/goals
  -> rest/continent/stake/rank diffs
  -> feature_matrix.csv
  -> time split
  -> scaled X matrices
  -> model training/calibration/evaluation
  -> model pickle files + meta.json + PNGs
```

### User Input Data Flow

```text
Browser form
  -> POST /api/predict JSON
  -> route validation
  -> team latest-state lookup
  -> model feature vector
  -> probability output
  -> expected goals/scorelines
  -> JSON response
  -> browser DOM render
```

### Live Data Flow

```text
Browser /live
  -> /api/live/standings or fixtures
  -> cache check
  -> if valid: read JSON cache
  -> else: Football-Data.org or API-Football
  -> map provider shape
  -> write cache
  -> if failure: expired cache or mock
  -> flatten for frontend
```

## 9. Interview Preparation Guide

### Beginner Questions

| Question | Strong answer | Common mistake |
|---|---|---|
| What does this project do? | It predicts H/D/A football outcomes from engineered historical features and simulates World Cup tournaments with dynamic state updates. | Saying it only predicts winners and ignoring draws/probabilities/simulation. |
| What is Elo? | A rating system where expected result is a logistic function of rating difference, then ratings update by actual minus expected result. | Treating Elo as a static dataset here; this repo computes it from match history. |
| Why use `config.yaml`? | It centralizes paths and model/feature/API parameters so scripts and services share the same settings. | Hardcoding values in each script. |
| What does `pytest` test here? | Data cleaning, feature math, model selection, Poisson score grids, prediction symmetry, and simulator state update. | Claiming tests verify model accuracy. They verify code behavior. |

### Intermediate Questions

| Question | Strong answer | Follow-up |
|---|---|---|
| Why use `merge_asof` for rankings? | Rankings are snapshots, not daily records. `merge_asof` assigns the most recent prior ranking for a match/team. | What sorting does pandas require before `merge_asof`? |
| Why temporal split instead of random split? | Random split can leak future distribution/history into training evaluation. Temporal split better resembles future match prediction. | How would you validate if the tournament distribution changes? |
| Why calibration? | The app consumes probabilities in UI and simulation, so log loss/Brier and calibrated probabilities matter more than only hard-label accuracy. | Why use sigmoid for some models and isotonic for tree models? |
| How are neutral predictions made symmetric? | Predict A vs B, predict B vs A, invert reverse probabilities, average. | What is the runtime cost? |

### Advanced Questions

| Question | Strong answer | Risk to mention |
|---|---|---|
| How does the Poisson score model feed classification? | It predicts expected home/away goals, builds a scoreline probability matrix, applies Dixon-Coles low-score corrections, and sums regions for H/D/A. | Truncated max goals and rho tuning affect probabilities. |
| What are the main leakage risks? | Post-match feature leakage is mitigated by recording states before updates; ranking snapshots use backward merge. Risks remain if future raw data enters train split incorrectly or hardcoded tournament groups encode assumptions. | Need automated data versioning. |
| What would you refactor first? | Single shared feature schema module, remove unreachable fast simulator code, centralize continent/group config, stronger API validation, source metadata for artifacts. | Avoid broad rewrites before tests. |
| Why does draw recall matter? | Current holdout report shows zero draw recall. Even with good log loss, hard predictions under-serve draws. Draw-risk output helps but does not solve classification recall. | Consider threshold tuning objective beyond accuracy. |

### Senior Architecture Questions

| Question | Strong answer |
|---|---|
| How would you productionize this? | Add artifact versioning, model registry checksums, data provenance, CI, Docker build validation, locked dependency file, health/readiness endpoints, request validation, background job queue, atomic model artifact swaps, monitoring for prediction drift. |
| How would you prevent concurrent retraining issues? | Train to a new artifact directory, write metadata last, validate artifacts, then atomically swap a symlink/current pointer. Use a lock or job queue so one retrain runs at a time. |
| How would you improve modeling? | Use proper time-series backtesting, optimize calibration and draw recall separately, evaluate per tournament type, include squad/player/injury/travel venue features if sourced, benchmark against betting-market implied probabilities if legally/ethically available. |
| How would you explain trade-offs in the current design? | It is pragmatic for a portfolio ML app: CSV storage and Flask are simple and transparent, but not robust enough for high-scale or high-trust production without more infrastructure. |

## 10. Rebuild-From-Scratch Guide

1. Plan the product.
   - Define outputs: match probabilities, expected goals, tournament champion probabilities, UI dashboards.
   - Define target classes: `H`, `D`, `A`.

2. Create project scaffold.
   - Root files: `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `config.yaml`.
   - Packages: `src/data`, `src/features`, `src/models`, `src/api`, `src/utils`.
   - Folders: `data/raw`, `data/processed`, `data/features`, `models/registry`, `visualisations`, `scripts`, `tests`.

3. Implement utilities.
   - `config.py`: project root and YAML loading.
   - `logger.py`: console/file logger.

4. Add raw data contracts.
   - Put `matches.csv`, `fifa_rankings.csv`, `elo_ratings.csv` in `data/raw`.
   - Implement raw presence checks.
   - Implement schema/row-count validation.

5. Build cleaning layer.
   - Normalize team names.
   - Parse dates.
   - Drop scoreless rows.
   - Create `result` and `is_competitive`.
   - Convert rankings into dated snapshots.
   - Merge latest prior home/away rankings.
   - Save `matches_clean.csv`.

6. Build feature engineering.
   - Implement chronological Elo.
   - Implement EWMA form.
   - Implement EWMA goal stats.
   - Add rest days, continent indicators, match stake.
   - Add ranking differences.
   - Filter by configured date range and minimum matches.
   - Save `feature_matrix.csv`.

7. Build model training.
   - Define one canonical `FEATURE_COLUMNS`.
   - Split train/calibration/test by dates.
   - Fit scaler on train only.
   - Train/tune base models with time-series CV.
   - Add Poisson goal model.
   - Calibrate probabilities.
   - Select champion by log loss/Brier/accuracy tie-breaks.
   - Serialize model, scaler, score model, metadata.

8. Build evaluation.
   - Load artifacts and holdout data.
   - Generate classification report, confusion matrix, feature importance, calibration curve.
   - Update `meta.json`.

9. Build prediction runtime.
   - Load artifacts.
   - Build latest team-state lookup from feature matrix.
   - Construct feature vectors in metadata order.
   - Scale and predict probabilities.
   - Add neutral symmetric averaging.
   - Add scoreline predictions.
   - Return JSON-friendly output.

10. Build simulator.
    - Define tournament groups and hosts.
    - Simulate group round robins.
    - Rank groups by points/GD/GF.
    - Select knockout teams.
    - Simulate knockout rounds with extra time/shootouts.
    - Update team states after every match.
    - Run Monte Carlo and aggregate champions.

11. Build Flask API.
    - Add app factory.
    - Register API and page blueprints.
    - Add endpoints for health, teams, predict, simulate, team details, model metadata, plots, live cache.

12. Build UI templates.
    - Start with functional pages: prediction, simulation, insights, live.
    - Wire browser `fetch` calls to API.
    - Keep UI independent from ML internals except JSON contracts.

13. Add tests.
    - Unit test deterministic feature functions.
    - Test predictor output shape and symmetry.
    - Test model selection tie-breakers.
    - Test Poisson score grid normalization.
    - Add dummy artifact fixture for clean environments.

14. Add Docker.
    - Python base image.
    - Install system deps for LightGBM.
    - Install requirements.
    - Run tests in build.
    - Start Flask.
    - Compose web + scheduler services.

15. Production hardening.
    - Add CI.
    - Add artifact versioning and atomic swaps.
    - Add stricter request validation.
    - Add secrets management.
    - Add monitoring and model drift reports.

## 11. Key Takeaways

- The project is an end-to-end ML application: data pipeline, trained models, evaluation artifacts, API, UI, scheduler, and Docker.
- The most defensible technical choices are chronological feature engineering, temporal splits, probability-first model selection, and neutral prediction symmetry.
- The active model is a calibrated Stacking Ensemble selected by log loss, but current hard-label draw performance is poor.
- The current runtime stores state and artifacts in files rather than a database or model registry service.
- The code is interview-ready as a portfolio project if you can clearly separate what is production-grade from what is pragmatic prototype infrastructure.

## 12. Knowledge Gaps / Missing Information

Unavailable from current project context:

- Exact original raw dataset download timestamps and checksums.
- Official data licenses for the raw CSV sources.
- Personal design rationale beyond commit messages and code comments.
- Cloud deployment target, domain, hosting provider, or production environment.
- CI/CD configuration.
- Monitoring/alerting setup.
- Real API key values in `.env`.
- Notebooks with actual exploratory analysis.
- Database schema because no database exists.
- Formal API contract such as OpenAPI/Swagger.
- Model reproducibility lockfile beyond `requirements.txt`; no `pip-tools`, Poetry, Conda, or lockfile is present.

Known technical risks:

- `.dockerignore` includes `Dockerfile`.
- `_simulate_match_fast` contains unreachable optimized code.
- Feature schema is duplicated across modules.
- CORS allows all origins.
- Live mock fallback can mask provider/API failures.
- Current draw classification recall is zero in `meta.json`.
- Scheduler and web service can share mutable data/model volumes without file locking.
- Pickle/joblib artifacts require trusted local files and compatible dependency versions.

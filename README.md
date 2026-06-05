<div align="center">

# ⚽ WC 2026 Match Outcome Predictor

**A machine learning pipeline for predicting FIFA World Cup 2026 match outcomes**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

_Predict international football match outcomes (Home Win / Draw / Away Win) using historical match data, FIFA rankings, and custom-engineered features like Elo ratings, team form, and rolling goal statistics. Then simulate the entire World Cup 2026 tournament via Monte Carlo methods._

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Data Setup](#data-setup)
  - [Docker Deployment](#docker-deployment)
- [Usage](#-usage)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Running Individual Steps](#running-individual-steps)
  - [Quick Retraining](#quick-retraining)
  - [Live Match Fetch & Retrain](#live-match-fetch--retrain)
  - [Running Diagnostics](#running-diagnostics)
  - [Running Tests](#running-tests)
- [Pipeline Architecture](#-pipeline-architecture)
  - [1. Data Ingestion & Validation](#1-data-ingestion--validation)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Evaluation](#4-model-training--evaluation)
  - [5. Match Prediction](#5-match-prediction)
  - [6. Tournament Simulation](#6-tournament-simulation)
- [Web UI & REST API](#-web-ui--rest-api)
  - [Starting the Dashboard and Server](#starting-the-dashboard-and-server)
  - [Dashboard Views](#dashboard-views)
  - [REST API Endpoints](#rest-api-endpoints)
- [Model Performance](#-model-performance)
- [Configuration](#-configuration)
- [Notebooks](#-notebooks)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🌍 Overview

The **WC 2026 Predictor** is an end-to-end machine learning project that predicts the outcomes of international football matches — specifically targeting the **2026 FIFA World Cup** (hosted in USA, Canada & Mexico).

The project follows a structured ML pipeline:

```
Raw Data → Validation → Cleaning → Feature Engineering → Training → Evaluation → Prediction → Simulation
```

It processes international match history from Kaggle datasets, engineers domain-specific features rooted in football analytics (Elo ratings, rolling form, goal statistics, and FIFA rankings), trains a machine learning classifier to predict three-way outcomes (**Home Win**, **Draw**, **Away Win**), and runs **Monte Carlo simulations** of the full 48-team World Cup 2026 tournament to estimate each nation's probability of winning.

---

## ✨ Features

| Category                      | Details                                                                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Elo Rating System**         | Advanced Elo ratings computed from scratch with Home-Field Advantage (+100 Elo), logarithmic goal-margin scaling, and dynamic K-factors (K=60 for World Cup down to K=20 for friendlies) |
| **Team Form Tracking**        | Opponent-adjusted EWMA (alpha=0.3) form tracking that rewards beating higher-Elo teams                                                                                                   |
| **Goal Statistics**           | EWMA rolling goal averages (alpha=0.25) for goals scored, conceded, and goal difference                                                                                                  |
| **FIFA Rankings Integration** | Historical FIFA ranking snapshots merged via temporal join (`merge_asof`)                                                                                                                |
| **Team Name Normalisation**   | 32 team name mappings across datasets (e.g., "Korea Republic" → "South Korea")                                                                                                           |
| **Baseline Evaluation**       | Three rule-based baselines (random guessing, most-frequent class, Elo heuristic) to establish performance floors                                                                         |
| **Trained ML Model**          | Tuned & calibrated **HistGradientBoostingClassifier** (selected by lowest log loss) with ensembling (Stacking) and Platt scaling (Sigmoid) calibration                                   |
| **Symmetric Prediction**      | Neutral-venue matches use Symmetric Prediction Averaging to eliminate home/away ordering bias                                                                                            |
| **Monte Carlo Simulation**    | Full tournament simulation with dynamic Elo/form/goals updates after every simulated match                                                                                               |
| **Hot-Reloading**             | `MatchPredictor` monitors file modification times on disk and automatically reloads model artifacts without server restarts                                                              |
| **Live Data Integration**     | Cached client for RapidAPI API-Football and Football-Data.org APIs with local caching and offline mock fallbacks                                                                         |
| **Auto-Retrain Scheduler**    | Automatic background scraper to fetch recent matches, append to the database, and trigger a pipeline retrain                                                                             |
| **Interactive Dashboard**     | Complete Web UI with Predictor, Analytics, SVG radar polygon drawing, live standings, and Monte Carlo tournament runs (served via templates)                                             |
| **REST API**                  | Flask API with 10 endpoints for health checks, team listings, match predictions, tournament simulations, live scores, and model metadata                                                 |
| **Data Validation**           | Structural checks on raw files — column presence, minimum row counts, known-team assertions                                                                                              |
| **Centralised Config**        | Single `config.yaml` file as the source of truth for all parameters                                                                                                                      |
| **Structured Logging**        | Dual-output logger (console + file) with timestamped, leveled log entries                                                                                                                |
| **Diagnostic Tooling**        | Comprehensive data diagnostics script for team name overlaps, Elo coverage, and tournament type distributions                                                                            |
| **Unit Tests**                | Pytest-based test suite covering cleaning logic, feature engineering, and prediction                                                                                                     |
| **Visualisation Outputs**     | Auto-generated confusion matrix and feature importance plots                                                                                                                             |

---

## 📁 Project Structure

```
wc2026-predictor/
│
├── config.yaml              # Central configuration (paths, params, thresholds)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys, settings - git-ignored)
├── .gitignore               # Files excluded from version control
├── LICENSE                  # MIT License
├── Dockerfile               # Container setup (installs dependencies, runs tests, serves app)
├── docker-compose.yaml      # Multi-container orchestration (web server and scheduler services)
│
├── src/                     # Source code (importable Python package)
│   ├── __init__.py
│   ├── data/                # Data ingestion & cleaning
│   │   ├── fetch.py         #   → Verify raw data files exist
│   │   ├── validate.py      #   → Structural validation (schema, row counts)
│   │   └── clean.py         #   → Cleaning, normalisation, ranking merge
│   │
│   ├── features/            # Feature engineering
│   │   ├── build.py         #   → Orchestrates all feature generators
│   │   ├── elo.py           #   → Chronological Elo rating computation
│   │   ├── form.py          #   → Rolling team form (points-based)
│   │   └── goals.py         #   → Rolling goal averages (scored/conceded)
│   │
│   ├── models/              # Model training, evaluation & prediction
│   │   ├── baseline.py      #   → Rule-based baseline evaluation
│   │   ├── train.py         #   → Multi-model grid search training + serialisation
│   │   ├── evaluate.py      #   → Classification report + diagnostic plots
│   │   ├── predict.py       #   → MatchPredictor with symmetric averaging & auto-reload
│   │   └── simulate.py      #   → Monte Carlo World Cup simulation
│   │
│   ├── api/                 # Flask Web Dashboard & API
│   │   ├── app.py           #   → Flask app factory with CORS
│   │   ├── routes.py        #   → Pages blueprint and API endpoints
│   │   └── templates/       #   → Web UI dashboard pages (9 templates)
│   │
│   └── utils/               # Shared utilities
│       ├── config.py        #   → YAML config loader
│       ├── logger.py        #   → Centralised logging setup
│       └── api_football.py  #   → Live data API client with local JSON caching
│
├── scripts/                 # Standalone runnable scripts
│   ├── run_pipeline.py      #   → End-to-end pipeline (6 steps)
│   ├── retrain.py           #   → Quick retraining loop (4 steps)
│   ├── diagnostics.py       #   → Data diagnostics & coverage analysis
│   ├── fetch_recent_matches.py # → Fetches recent matches and appends to matches.csv
│   └── scheduler.py         #   → Cron-like background scheduler for Docker
│
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb         #   → Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
│
├── tests/                   # Unit tests (pytest)
│   ├── test_clean.py        #   → Tests for data cleaning logic
│   ├── test_features.py     #   → Tests for Elo, form & goals features
│   └── test_predict.py      #   → Tests for prediction logic
│
├── data/                    # Data directory (large files not committed)
│   ├── raw/                 #   → Source CSVs (manually downloaded)
│   ├── processed/           #   → Cleaned & merged datasets
│   ├── features/            #   → Final feature matrices
│   └── live_cache/          #   → Cached standings and fixtures JSON files
│
├── models/                  # Serialised models (not committed)
│   └── registry/            #   → best_model.pkl, scaler.pkl, meta.json, etc.
│
├── visualisations/          # Generated plots (confusion matrix, feature importance)
└── logs/                    # Application logs (not committed)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10** or higher
- **pip** (Python package manager)
- **Git**
- **Docker & Docker Compose** (Optional, for containerized deployment)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/veselinkalinov/wc2026-predictor.git
   cd wc2026-predictor
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

Create a `.env` file in the root of the project to configure live data scraper access. A sample format:

```env
RAPIDAPI_KEY=your_rapidapi_key_here
FOOTBALL_DATA_API_KEY=your_footballdata_api_key_here
RETRAINING_INTERVAL_HOURS=24
```

> **Warning:** Never commit your `.env` file containing active API keys to version control.

### Data Setup

The raw datasets must be manually placed in `data/raw/` for the local training pipeline to execute:

| File                | Source                                                                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `matches.csv`       | [International Football Results (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| `fifa_rankings.csv` | [FIFA World Ranking (Kaggle)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)                                          |
| `elo_ratings.csv`   | Search Kaggle for _"international football elo ratings"_                                                                            |

Ensure your `data/raw/` directory contains:

```
data/raw/
├── matches.csv
├── fifa_rankings.csv
└── elo_ratings.csv
```

Run `python -m src.data.fetch` to verify all required files are present and non-empty.

### Docker Deployment

To spin up the entire application inside Docker containers (the Flask web dashboard on port 5000 and the background retraining scheduler service):

```bash
docker-compose up --build
```

---

## 💻 Usage

### Running the Full Pipeline

The pipeline runner executes all 6 steps sequentially — from raw data validation through to model evaluation:

```bash
python -m scripts.run_pipeline
```

### Running Individual Steps

Each step can also be run independently:

```bash
# Step 1: Verify raw data files exist
python -m src.data.fetch

# Step 2: Validate raw data structure
python -m src.data.validate

# Step 3: Clean and preprocess data
python -m src.data.clean

# Step 4: Build feature matrix
python -m src.features.build

# Step 5: Evaluate baselines
python -m src.models.baseline

# Step 6: Train the model
python -m src.models.train

# Step 7: Generate evaluation report
python -m src.models.evaluate

# Step 8: Run Monte Carlo simulation (standalone)
python -m src.models.simulate
```

### Quick Retraining

To retrain the models on the processed dataset without re-running data fetching and validation checks:

```bash
python -m scripts.retrain
```

This executes: cleaning → feature building → model training → evaluation.

### Live Match Fetch & Retrain

To download completed World Cup matches from Football-Data.org, append them to `matches.csv`, and run a full retrain:

```bash
python -m scripts.fetch_recent_matches
```

### Running Diagnostics

Perform a diagnostic scan of the raw data to check names, Elo coverage, formats, and outcome ratios:

```bash
python scripts/diagnostics.py
```

### Running Tests

Execute the unit tests using `pytest`:

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_features.py -v
```

---

## 🏗️ Pipeline Architecture

### 1. Data Ingestion & Validation

- **`src/data/fetch.py`** — Verifies that the required raw CSV files exist in `data/raw/` and are non-empty.
- **`src/data/validate.py`** — Runs schema validation: checks column presence, minimum row counts, data types, and asserts the presence of historical powerhouse teams.

### 2. Data Cleaning

**`src/data/clean.py`** cleans match entries by:

- Normalising team names across different sources using a 32-entry canonical map.
- Computing match outcome targets: `H` (home win), `D` (draw), `A` (away win).
- Identifying competitive fixtures via the `is_competitive` flag.
- Merging historical FIFA rankings using pandas' `merge_asof` (temporal backward join).
- Applying date boundaries and filtering out teams with fewer than 10 total match appearances.

### 3. Feature Engineering

**`src/features/build.py`** manages three feature generators:

| Feature Module | Columns Generated                                                                        | Description                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **`elo.py`**   | `home_elo`, `away_elo`, `elo_diff`                                                       | Chronological Elo ratings computed from scratch across match history (K=32, initial=1500)            |
| **`form.py`**  | `home_form`, `away_form`, `form_diff`                                                    | Rolling points earned in the last 5 matches, normalised between 0.0 and 1.0 (cold-start default 0.5) |
| **`goals.py`** | `home_goals_scored_avg`, `home_goals_conceded_avg`, `home_goal_diff_avg` (same for away) | Rolling goal averages over the last 10 matches (cold-start default 1.2 goals)                        |

It also derives `rank_diff`, `rank_points_diff`, `is_neutral` (from match metadata), and `is_competitive`.

### 4. Model Training & Evaluation

- **`src/models/train.py`** — Evaluates multiple architectures (Logistic Regression, Random Forest, and HistGradientBoosting) using `GridSearchCV` with a `TimeSeriesSplit(n_splits=3)`. The model with the lowest test **Log Loss** is chosen as the active model.
- **`src/models/baseline.py`** — Computes three baselines (Random Guessing, Most Frequent Class, Elo Heuristic) on the test split for comparison.
- **`src/models/evaluate.py`** — Saves classification metrics, a confusion matrix heatmap (`confusion_matrix.png`), and feature importance graphs (`feature_importance.png`).

### 5. Match Prediction

- **`src/models/predict.py`** — The `MatchPredictor` loads the serialized scaler and best model. It maps the latest computed parameters (Elo, form, goals, rank) for each team.
- **Symmetric Prediction Averaging** — In neutral-venue matches, it executes predictions twice (swapping home/away designations) and averages the forward and inverted probabilities to eliminate team ordering bias.
- **Hot-Reloading** — Tracks file modifications to the model pickle file, updating prediction parameters automatically without server restarts.

### 6. Tournament Simulation

- **`src/models/simulate.py`** — The `TournamentSimulator` executes Monte Carlo simulations of the FIFA World Cup 2026.
- It hardcodes the official 12-group World Cup group draw (48 teams total) and recognizes host nations (USA, Mexico, Canada) to apply home-field advantage.
- **Dynamic State Updates** — During a tournament run, Elo, form, and goal stats are dynamically updated after each simulated match.
- **Goal Simulation** — Simulates scorelines using Poisson distributions. Penalty shootouts in knockout matches are simulated using Elo-weighted probabilities.

---

## 🌐 Web UI & REST API

### Starting the Dashboard and Server

Run the Flask application:

```bash
python -m src.api.app
```

Then visit **`http://127.0.0.1:5000/`** to view the interactive dashboard.

### Dashboard Views

- **`/` (Home)**: Displays training metrics, latency, and pipeline details.
- **`/predict` (Match Predictor)**: Interactively predict match outcomes with gauges and form cards.
- **`/analytics` (Team Analytics)**: Display team stats, last 5 matches, and an SVG-drawn radar chart.
- **`/insights` (Model Insights)**: Visualise the confusion matrix, feature importances, and hyperparameter logs.
- **`/simulate` (Tournament Simulator)**: Run Monte Carlo runs with customizable iterations and visualize champions.
- **`/live` (Live Tournament Tracker)**: Renders live World Cup standings and fixtures fetched from the API.
- **`/about`**: Overview of the methodology and project developers.
- **`/privacy` & `/terms`**: Standard legal documentation pages.

### REST API Endpoints

All endpoints return JSON responses and are prefixed with `/api`.

#### `GET /api/health`

Checks API health. Returns `{"status": "healthy", "service": "wc2026-predictor-api"}`.

#### `GET /api/teams`

Lists all known teams in alphabetical order.

#### `POST /api/predict`

Predicts outcome probabilities for a specific match.
**Body:**

```json
{
  "home_team": "Argentina",
  "away_team": "France",
  "is_neutral": 1,
  "is_competitive": 1
}
```

#### `POST /api/simulate`

Runs Monte Carlo tournament simulations.
**Body:**

```json
{ "n_sims": 1000 }
```

#### `GET /api/team-details/<team_name>`

Retrieves stats, Elo, rankings, form history, and radar stats for a team.

#### `GET /api/team-matches/<team_name>`

Gets the last 5 matches for a team.

#### `GET /api/visualisations/<filename>`

Serves generated visualization plots (`confusion_matrix.png` / `feature_importance.png`).

#### `GET /api/model-meta`

Retrieves model configuration, tuning hyperparameters, and active model metrics.

#### `GET /api/live/standings`

Fetches current live World Cup group standings.

#### `GET /api/live/fixtures`

Fetches current live World Cup fixtures and schedules.

---

## 📊 Model Performance

Performance of model architectures on the test set (matches from 2022 onwards) evaluated via hyperparameter tuning:

| Model / Baseline                      | Holdout Accuracy | Holdout Log Loss | Holdout Brier Score | Status             |
| ------------------------------------- | ---------------- | ---------------- | ------------------- | ------------------ |
| **HistGradientBoosting (Calibrated)** | 60.68%           | **0.8623**       | **0.1691**          | 🏆 **Active Best** |
| **Stacking Ensemble (Calibrated)**    | **60.78%**       | 0.8685           | **0.1699**          | Inactive           |
| **LightGBM (Calibrated)**             | 60.71%           | 0.8697           | 0.1698              | Inactive           |
| **CatBoost (Calibrated)**             | 60.71%           | 0.8726           | 0.1703              | Inactive           |
| **XGBoost (Calibrated)**              | 60.68%           | 0.8725           | 0.1704              | Inactive           |
| **Random Forest (Calibrated)**        | 60.16%           | 0.8798           | 0.1718              | Inactive           |
| **Logistic Regression (Calibrated)**  | 60.32%           | 0.8800           | 0.1723              | Inactive           |
| **Elo Heuristic Baseline**            | 59.22%           | 0.9589           | 0.1887              | Baseline Floor     |
| Uniform Random Guessing               | 33.33%           | 1.0986           | 0.2222              | Reference          |

_Note: HistGradientBoosting is selected as the best model because it minimizes Log Loss (0.8623) and Brier Score (0.1691) on the holdout test set (matches post-July 2023) after Sigmoid (Platt) calibration, providing highly calibrated probabilities for tournament simulation._

---

## ⚙️ Configuration

Central configuration parameters in [`config.yaml`](config.yaml):

```yaml
project:
  name: "wc2026-predictor"
  version: "0.1.0"

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  features: "data/features"
  models: "models/registry"
  visualisations: "visualisations"

data:
  matches_file: "matches.csv"
  rankings_file: "fifa_rankings.csv"
  date_from: "2000-01-01"
  date_to: "2026-06-10"
  min_matches: 10

features:
  form_window: 5
  goals_window: 10
  elo_k_factor: 32
  elo_initial: 1500

model:
  train_cutoff: "2022-01-01"
  random_state: 42
  test_size: 0.2
  target_column: "result"
  random_forest:
    n_estimators: 100
    max_depth: 10
  hist_gradient_boosting:
    max_iter: 100
    learning_rate: 0.1

evaluation:
  metrics:
    - "accuracy"
    - "log_loss"
    - "brier_score"

api:
  host: "0.0.0.0"
  port: 5000
  debug: false
```

---

## 📓 Notebooks

Jupyter notebooks for exploratory work are in `notebooks/`:

- `01_eda.ipynb` — Exploratory Data Analysis of match results.
- `02_feature_engineering.ipynb` — Elo and rolling window experiments.
- `03_model_evaluation.ipynb` — Model comparison metrics.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Data Science**: pandas 2.2, scikit-learn 1.5, joblib 1.4, numpy
- **Visualisation**: matplotlib 3.10, seaborn 0.13
- **Web App**: Flask 3.0, python-dotenv 1.0, requests 2.32
- **Configuration**: PyYAML 6.0
- **Testing**: pytest 8.0+
- **Containerisation**: Docker, Docker Compose

---

## 🗺️ Roadmap

- [x] Project scaffold and configuration
- [x] Data validation
- [x] Preprocessing and team normalisation
- [x] Feature building (Elo, Form, Goals)
- [x] Diagnostic scripts
- [x] Unit tests
- [x] Baselines implementation
- [x] Logistic Regression training pipeline
- [x] Model evaluation (confusion matrix, importances)
- [x] Match prediction with Symmetric Averaging
- [x] Monte Carlo simulator for WC 2026
- [x] Flask REST API endpoints
- [x] End-to-end pipeline runner script
- [x] Grid search hyperparameter tuning
- [x] Multi-model training comparison
- [x] Interactive web dashboard
- [x] Docker and Docker Compose containerization
- [ ] Cloud deployment

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ⚽ and 🐍 for the beautiful game**

</div>

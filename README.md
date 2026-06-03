<div align="center">

# ⚽ WC 2026 Match Outcome Predictor

**A machine learning pipeline for predicting FIFA World Cup 2026 match outcomes**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

*Predict international football match outcomes (Home Win / Draw / Away Win) using historical match data, FIFA rankings, and custom-engineered features like Elo ratings, team form, and rolling goal statistics. Then simulate the entire World Cup 2026 tournament via Monte Carlo methods.*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Usage](#-usage)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Running Individual Steps](#running-individual-steps)
  - [Quick Retraining](#quick-retraining)
  - [Running Diagnostics](#running-diagnostics)
  - [Running Tests](#running-tests)
- [Pipeline Architecture](#-pipeline-architecture)
  - [1. Data Ingestion & Validation](#1-data-ingestion--validation)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Evaluation](#4-model-training--evaluation)
  - [5. Match Prediction](#5-match-prediction)
  - [6. Tournament Simulation](#6-tournament-simulation)
- [REST API](#-rest-api)
  - [Starting the Server](#starting-the-server)
  - [Endpoints](#endpoints)
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

It processes **25+ years of international match history** (2000–2025) from Kaggle datasets, engineers domain-specific features rooted in football analytics, trains a multinomial Logistic Regression classifier to predict three-way outcomes (**Home Win**, **Draw**, **Away Win**), and runs **Monte Carlo simulations** of the full 48-team World Cup 2026 tournament to estimate each nation's probability of winning.

---

## ✨ Features

| Category | Details |
|---|---|
| **Elo Rating System** | Custom-built Elo ratings calculated chronologically from scratch across all international matches (K=32, initial=1500) |
| **Team Form Tracking** | Rolling form scores based on points earned in the last *N* matches (configurable window) |
| **Goal Statistics** | Rolling averages for goals scored, conceded, and goal difference per team |
| **FIFA Rankings Integration** | Historical FIFA ranking snapshots merged via temporal join (`merge_asof`) |
| **Team Name Normalisation** | 40+ team name mappings across datasets (e.g., "Korea Republic" → "South Korea") |
| **Baseline Evaluation** | Three rule-based baselines (random guessing, most-frequent class, Elo heuristic) to establish performance floors |
| **Trained ML Model** | Multinomial Logistic Regression with StandardScaler, serialised via joblib |
| **Symmetric Prediction** | Neutral-venue matches use Symmetric Prediction Averaging to eliminate home/away ordering bias |
| **Monte Carlo Simulation** | Full tournament simulation with dynamic Elo/form/goals updates after every simulated match |
| **Interactive Dashboard** | Complete Web UI with Predictor, Analytics, SVG radar polygon drawing, and Monte Carlo tournament runs (served via templates) |
| **REST API** | Flask API with endpoints for health checks, team listings, match predictions, and tournament simulations |
| **Data Validation** | Structural checks on raw files — column presence, minimum row counts, known-team assertions |
| **Centralised Config** | Single `config.yaml` file as the source of truth for all parameters |
| **Structured Logging** | Dual-output logger (console + file) with timestamped, leveled log entries |
| **Diagnostic Tooling** | Comprehensive data diagnostics script for team name overlaps, Elo coverage, and tournament type distributions |
| **Unit Tests** | Pytest-based test suite covering cleaning logic, feature engineering, and prediction |
| **Visualisation Outputs** | Auto-generated confusion matrix and feature importance plots |

---

## 📁 Project Structure

```
wc2026-predictor/
│
├── config.yaml              # Central configuration (paths, params, thresholds)
├── requirements.txt         # Python dependencies
├── .gitignore               # Files excluded from version control
├── LICENSE                  # MIT License
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
│   │   ├── train.py         #   → Logistic Regression training + serialisation
│   │   ├── evaluate.py      #   → Classification report + diagnostic plots
│   │   ├── predict.py       #   → MatchPredictor with symmetric averaging
│   │   └── simulate.py      #   → Monte Carlo World Cup simulation
│   │
│   ├── api/                 # Flask Web Dashboard & API
│   │   ├── app.py           #   → Flask app factory with CORS
│   │   ├── routes.py        #   → Pages blueprint and API endpoints
│   │   └── templates/       #   → Web UI dashboard pages (home, predict, analytics, simulate, about, insights)
│   │
│   └── utils/               # Shared utilities
│       ├── config.py        #   → YAML config loader
│       └── logger.py        #   → Centralised logging setup
│
├── scripts/                 # Standalone runnable scripts
│   ├── run_pipeline.py      #   → End-to-end pipeline (6 steps)
│   ├── retrain.py           #   → Quick retraining loop (4 steps)
│   └── diagnostics.py       #   → Data diagnostics & coverage analysis
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
├── data/                    # Data directory (raw files not committed)
│   ├── raw/                 #   → Source CSVs (manually downloaded)
│   ├── processed/           #   → Cleaned & merged datasets
│   └── features/            #   → Final feature matrices
│
├── models/                  # Serialised models (not committed)
│   └── registry/            #   → best_model.pkl, scaler.pkl, meta.json
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

### Data Setup

The raw data files are not included in the repository. You need to download them manually and place them in `data/raw/`:

| File | Source |
|---|---|
| `matches.csv` | [International Football Results (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| `fifa_rankings.csv` | [FIFA World Ranking (Kaggle)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) |
| `elo_ratings.csv` | Search Kaggle for *"international football elo ratings"* |

After downloading, your `data/raw/` directory should look like:

```
data/raw/
├── matches.csv
├── fifa_rankings.csv
└── elo_ratings.csv
```

> **Tip:** Run `python -m src.data.fetch` to verify all required files are present and non-empty.

---

## 💻 Usage

### Running the Full Pipeline

The pipeline runner executes all 6 steps sequentially — from raw data validation through to model evaluation:

```bash
python -m scripts.run_pipeline
```

This runs:
1. Raw file presence check
2. Structural data validation
3. Data cleaning and normalisation
4. Feature engineering (Elo, form, goals)
5. Logistic Regression model training
6. Evaluation report and visualisation generation

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

To retrain the model without re-running fetch and validation checks:

```bash
python -m scripts.retrain
```

This executes: cleaning → feature building → training → evaluation.

### Running Diagnostics

The diagnostics script performs a comprehensive analysis of your raw data — team name overlaps across datasets, Elo coverage, ranking date formats, tournament type distributions, and outcome class distributions:

```bash
python scripts/diagnostics.py
```

### Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_features.py -v
```

---

## 🏗️ Pipeline Architecture

### 1. Data Ingestion & Validation

- **`src/data/fetch.py`** — Verifies that all three required CSV files (`matches.csv`, `fifa_rankings.csv`, `elo_ratings.csv`) are present in `data/raw/` and are non-empty.
- **`src/data/validate.py`** — Runs structural assertions: checks column presence, enforces minimum row counts (e.g., matches ≥ 40,000 rows), validates data types, and confirms the presence of known teams (Brazil, Germany, France, etc.).

### 2. Data Cleaning

**`src/data/clean.py`** handles:

- **Date parsing** — Converts match dates and constructs ranking dates from year/semester
- **Team name normalisation** — Maps 40+ naming inconsistencies across datasets to canonical names (e.g., `"IR Iran"` → `"Iran"`, `"Côte d'Ivoire"` → `"Ivory Coast"`)
- **Outcome computation** — Derives match result labels: `H` (home win), `D` (draw), `A` (away win)
- **Competitive flag** — Classifies matches as competitive vs. friendly
- **Ranking merge** — Uses `pd.merge_asof` to attach the most recent FIFA ranking snapshot to each match
- **Fallback handling** — Assigns rank 211 and 0 points to non-FIFA teams
- **Temporal filtering** — Restricts data to the configured date range (default: 2000–2025)
- **Minimum match threshold** — Removes teams with fewer than 10 total appearances

### 3. Feature Engineering

**`src/features/build.py`** orchestrates three feature generators:

| Feature Module | Columns Generated | Description |
|---|---|---|
| **`elo.py`** | `home_elo`, `away_elo`, `elo_diff` | Chronological Elo ratings computed from scratch using a K-factor of 32 and standard expected score formula |
| **`form.py`** | `home_form`, `away_form`, `form_diff` | Rolling form score (0.0–1.0) based on points earned in the last 5 matches |
| **`goals.py`** | `home_goals_scored_avg`, `home_goals_conceded_avg`, `home_goal_diff_avg`, (same for away) | Rolling averages over the last 10 matches with cold-start default of 1.2 |

Additional derived features:
- `rank_diff` — Difference in FIFA rankings
- `rank_points_diff` — Difference in FIFA ranking points
- `is_neutral` — Whether the match is played at a neutral venue
- `is_competitive` — Whether the match is a competitive fixture

### 4. Model Training & Evaluation

**`src/models/train.py`** — Trains a multinomial Logistic Regression classifier:

- **20 input features** (Elo, form, goals, rankings, venue, competition type)
- **StandardScaler** fitted on training data, applied to both train and test sets
- **Temporal train/test split** at `2022-01-01` (no data leakage)
- **Serialisation** — Saves `best_model.pkl`, `scaler.pkl`, and `meta.json` to `models/registry/`

**`src/models/baseline.py`** — Evaluates three rule-based baselines on the test set:

| Baseline | Accuracy | Log Loss |
|---|---|---|
| Random Guessing (uniform 1/3) | ~0.33 | 1.0986 |
| Most Frequent Class (always H) | ~0.48 | 1.0046 |
| Elo Heuristic (higher Elo wins) | 0.5922 | 0.9589 |

**`src/models/evaluate.py`** — Generates a detailed evaluation report:

- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap → `visualisations/confusion_matrix.png`
- Feature importance bar chart → `visualisations/feature_importance.png`

### 5. Match Prediction

**`src/models/predict.py`** — The `MatchPredictor` class:

- Loads trained model artifacts and builds a lookup dictionary of **latest team states** (Elo, form, goal averages, FIFA rank) for all 268 teams
- Constructs feature vectors on-the-fly for any matchup
- **Symmetric Prediction Averaging** — For neutral-venue matches, predicts the match twice (swapping home/away roles), inverts the reverse probabilities, and averages. This eliminates ordering bias in predictions
- Supports **dynamic state updates** for tournament simulation (Elo, form, and goal averages updated after each simulated match)

### 6. Tournament Simulation

**`src/models/simulate.py`** — The `TournamentSimulator` class:

- Contains the **official 48-team World Cup 2026 group draw** (12 groups × 4 teams)
- Recognises host nations (USA, Mexico, Canada) and applies home advantage when a host team plays
- **Group stage**: Round-robin within each group, ranking by points → goal difference → goals for
- **Knockout stage**: Top 2 from each group + 8 best 3rd-placed teams = 32 teams in Round of 32 → Round of 16 → QF → SF → Final
- **Penalty shootouts**: Knockout draws resolved with Elo-weighted probability
- **Goal simulation**: Poisson-distributed goals aligned with the drawn result
- **Dynamic state updates**: Elo, form, and goal averages updated after every simulated match
- **Monte Carlo**: Runs *N* independent tournament simulations and returns win probabilities per team

---

## 🌐 Web UI & REST API

### Starting the Dashboard and Server

Start the Flask server locally:

```bash
python -m src.api.app
```

Once started, open **`http://127.0.0.1:5000/`** in your web browser to access the complete interactive analytics dashboard!

### Dashboard Views

- **`/` (Home)**: Portal home screen detailing training metrics, latency rates, and the ML methodology.
- **`/predict` (Match Predictor)**: Interactive head-to-head prediction engine with radial certainty gauges, ELO strength stats, and custom AI scout summaries.
- **`/analytics` (Team Analytics)**: Dynamic diagnostic panels displaying rolling form points, match histories, and a JS-rendered SVG radar chart.
- **`/insights` (Model Insights)**: Model parameters display, confusion matrices, and feature contributions charts.
- **`/simulate` (Tournament Simulator)**: Configurable Monte Carlo bracket runs displaying aggregate contenders and searchable grids of all 48 teams.

### REST API Endpoints

All REST API endpoints are prefixed with `/api`.

#### `GET /api/health`

Health check endpoint.

```json
{ "status": "healthy", "service": "wc2026-predictor-api" }
```

#### `GET /api/teams`

Returns all 268 known teams sorted alphabetically.

```json
{ "count": 268, "teams": ["Afghanistan", "Albania", "Algeria", "..."] }
```

#### `POST /api/predict`

Predict match outcome probabilities.

**Request body:**

```json
{
  "home_team": "Argentina",
  "away_team": "France",
  "is_neutral": 1,
  "is_competitive": 1
}
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `home_team` | string | ✅ | — | Name of the home team |
| `away_team` | string | ✅ | — | Name of the away team |
| `is_neutral` | int | ❌ | 1 | `1` = neutral venue, `0` = home advantage |
| `is_competitive` | int | ❌ | 1 | `1` = competitive match, `0` = friendly |

**Response:**

```json
{
  "home_team": "Argentina",
  "away_team": "France",
  "prediction": "H",
  "probabilities": {
    "home_win": 0.4498,
    "draw": 0.2696,
    "away_win": 0.2805
  }
}
```

#### `POST /api/simulate`

Run Monte Carlo tournament simulations.

**Request body:**

```json
{ "n_sims": 1000 }
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `n_sims` | int | ❌ | 100 | Number of simulations (max 5000) |

**Response:**

```json
{
  "simulations_run": 1000,
  "win_probabilities": {
    "Spain": 0.17,
    "Argentina": 0.13,
    "France": 0.10,
    "Brazil": 0.08,
    "England": 0.07,
    "..."
  }
}
```

---

## 📊 Model Performance

Comparative performance of trained model architectures on the test set (matches from 2022 onwards) after hyperparameter tuning:

| Model / Baseline | Test Accuracy | Test Log Loss | Test Brier Score | Status |
|---|---|---|---|---|
| **HistGradientBoosting (Tuned)** | **60.02%** | **0.8692** | **0.1707** | 🏆 **Active Best** |
| Logistic Regression | 59.91% | 0.8782 | 0.1722 | Inactive |
| Elo Heuristic Baseline | 59.22% | 0.9589 | 0.1887 | Baseline Floor |
| Random Forest | 58.41% | 0.8826 | 0.1735 | Inactive |

The model achieves higher accuracy and significantly better calibrated probabilities (lower log loss) compared to the best heuristic baseline.

---

## ⚙️ Configuration

All parameters are centralised in [`config.yaml`](config.yaml):

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
  date_from: "2000-01-01"
  date_to: "2025-12-31"
  min_matches: 10

features:
  form_window: 5          # Recent matches for form calculation
  goals_window: 10        # Recent matches for goal averages
  elo_k_factor: 32        # Elo rating update sensitivity
  elo_initial: 1500       # Starting Elo for all teams

model:
  train_cutoff: "2022-01-01"  # Temporal split boundary
  random_state: 42            # Reproducibility seed
  test_size: 0.2              # Fallback if not using temporal split
  target_column: "result"     # H / D / A

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

> **Important:** Never hardcode these values in source files — always import from `src.utils.config`.

---

## 📓 Notebooks

Interactive Jupyter notebooks for exploration and analysis:

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis — distributions, trends, correlations |
| `02_feature_engineering.ipynb` | Feature exploration and visualisation |
| `03_model_evaluation.ipynb` | Model comparison and evaluation metrics |

Launch Jupyter:

```bash
jupyter notebook notebooks/
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas 2.2 |
| **Machine Learning** | scikit-learn 1.5 |
| **Visualisation** | Matplotlib 3.10, Seaborn 0.13 |
| **API Framework** | Flask 3.0 |
| **Configuration** | PyYAML 6.0 |
| **Environment** | python-dotenv 1.0 |
| **Model Serialisation** | joblib 1.4 |
| **Notebooks** | Jupyter, IPyKernel |
| **Testing** | pytest |

---

## 🗺️ Roadmap

- [x] Project scaffold and configuration
- [x] Data fetching and validation
- [x] Data cleaning and preprocessing
- [x] Feature engineering (Elo, form, goals)
- [x] Diagnostic tooling
- [x] Unit test suite
- [x] Baseline model evaluation
- [x] Logistic Regression training pipeline
- [x] Model evaluation with confusion matrix and feature importance plots
- [x] Match prediction with symmetric averaging
- [x] World Cup 2026 Monte Carlo simulation
- [x] Flask REST API (health, teams, predict, simulate)
- [x] End-to-end pipeline runner
- [x] Quick retraining script
- [x] Multi-model training (Random Forest, XGBoost, HistGradientBoosting)
- [x] Hyperparameter tuning (GridSearch)
- [x] Interactive frontend dashboard
- [ ] Deployment (Docker + cloud hosting)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ⚽ and 🐍 for the beautiful game**

</div>

<div align="center">

# ⚽ WC 2026 Match Outcome Predictor

**A machine learning pipeline for predicting FIFA World Cup 2026 match outcomes**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

*Predict international football match outcomes (Home Win / Draw / Away Win) using historical match data, FIFA rankings, and custom-engineered features like Elo ratings, team form, and rolling goal statistics.*

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
  - [Running the Pipeline](#running-the-pipeline)
  - [Running Diagnostics](#running-diagnostics)
  - [Running Tests](#running-tests)
- [Pipeline Architecture](#-pipeline-architecture)
  - [1. Data Ingestion & Validation](#1-data-ingestion--validation)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Evaluation](#4-model-training--evaluation)
  - [5. Prediction API](#5-prediction-api)
- [Configuration](#-configuration)
- [Notebooks](#-notebooks)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🌍 Overview

The **WC 2026 Predictor** is an end-to-end machine learning project that aims to predict the outcomes of international football matches — specifically targeting the **2026 FIFA World Cup** (hosted in USA, Canada & Mexico).

The project follows a structured ML pipeline:

```
Raw Data → Validation → Cleaning → Feature Engineering → Training → Evaluation → Prediction API
```

It processes **25+ years of international match history** (2000–2025) from Kaggle datasets, engineers domain-specific features rooted in football analytics, and trains classification models to predict three-way outcomes: **Home Win (H)**, **Draw (D)**, or **Away Win (A)**.

---

## ✨ Features

| Category | Details |
|---|---|
| **Elo Rating System** | Custom-built Elo ratings calculated chronologically from scratch across all international matches (K=32, initial=1500) |
| **Team Form Tracking** | Rolling form scores based on points earned in the last *N* matches (configurable window) |
| **Goal Statistics** | Rolling averages for goals scored, conceded, and goal difference per team |
| **FIFA Rankings Integration** | Historical FIFA ranking snapshots merged via temporal join (`merge_asof`) |
| **Team Name Normalisation** | 40+ team name mappings across datasets (e.g., "Korea Republic" → "South Korea") |
| **Data Validation** | Structural checks on raw files — column presence, minimum row counts, known-team assertions |
| **Centralised Config** | Single `config.yaml` file as the source of truth for all parameters |
| **Structured Logging** | Dual-output logger (console + file) with timestamped, leveled log entries |
| **Unit Tests** | Pytest-based test suite covering cleaning logic, Elo calculations, form features, and goal statistics |

---

## 📁 Project Structure

```
wc2026-predictor/
│
├── config.yaml              # Central configuration (paths, params, thresholds)
├── requirements.txt         # Python dependencies
├── .gitignore               # Files excluded from version control
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
│   │   ├── baseline.py      #   → Baseline model (placeholder)
│   │   ├── train.py         #   → Model training (placeholder)
│   │   ├── evaluate.py      #   → Model evaluation (placeholder)
│   │   └── predict.py       #   → Match prediction (placeholder)
│   │
│   ├── api/                 # Flask REST API
│   │   ├── app.py           #   → Flask app factory (placeholder)
│   │   └── routes.py        #   → API endpoints (placeholder)
│   │
│   └── utils/               # Shared utilities
│       ├── config.py        #   → YAML config loader
│       └── logger.py        #   → Centralised logging setup
│
├── scripts/                 # Standalone runnable scripts
│   ├── run_pipeline.py      #   → End-to-end pipeline execution
│   ├── retrain.py           #   → Model retraining script
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
│   └── registry/            #   → Trained model artifacts
│
├── visualisations/          # Generated plots and charts
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

### Running the Pipeline

Execute the full pipeline step-by-step:

```bash
# Step 1: Verify raw data files exist
python -m src.data.fetch

# Step 2: Validate raw data structure
python -m src.data.validate

# Step 3: Clean and preprocess data
python -m src.data.clean

# Step 4: Build feature matrix
python -m src.features.build
```

### Running Diagnostics

The diagnostics script performs a comprehensive analysis of your raw data — team name overlaps across datasets, Elo coverage, ranking date formats, and tournament type distributions:

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

### 4. Model Training & Evaluation

> 🚧 **Work in Progress** — The training, evaluation, and baseline modules are scaffolded and ready for implementation.

Planned evaluation metrics (from `config.yaml`):
- Accuracy
- Log Loss
- Brier Score

The temporal train/test split is configured at `2022-01-01`.

### 5. Prediction API

> 🚧 **Work in Progress** — The Flask API is scaffolded (`src/api/app.py`, `src/api/routes.py`) and will expose prediction endpoints.

Planned configuration:
- Host: `0.0.0.0`
- Port: `5000`

---

## ⚙️ Configuration

All parameters are centralised in [`config.yaml`](config.yaml):

```yaml
project:
  name: "wc2026-predictor"
  version: "0.1.0"

features:
  form_window: 5          # Recent matches for form calculation
  goals_window: 10        # Recent matches for goal averages
  elo_k_factor: 32        # Elo rating update sensitivity
  elo_initial: 1500       # Starting Elo for all teams

model:
  train_cutoff: "2022-01-01"  # Temporal split boundary
  random_state: 42            # Reproducibility seed
  target_column: "result"     # H / D / A

data:
  date_from: "2000-01-01"
  date_to: "2025-12-31"
  min_matches: 10
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
- [ ] Baseline model implementation
- [ ] Multi-model training pipeline (Logistic Regression, Random Forest, XGBoost)
- [ ] Hyperparameter tuning
- [ ] Model evaluation dashboard
- [ ] Flask prediction API
- [ ] World Cup 2026 match simulations
- [ ] Deployment (Docker + cloud hosting)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ⚽ and 🐍 for the beautiful game**

</div>

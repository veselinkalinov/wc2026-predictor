# 00 - Orientation

## What You Will Understand After This Lesson

- What the app does.
- Why the repo is split into data, features, models, API, scripts, tests, and artifacts.
- How to explain the project in an interview without overclaiming.
- Which parts are source code and which parts are generated runtime artifacts.

## First Principles

A machine-learning web app has two broad lives:

1. Training life: collect data, clean it, build features, train models, evaluate them, and save artifacts.
2. Serving life: load saved artifacts, accept user input, build the same kind of feature vector, return predictions.

This project has both. The code is not only a notebook. It is a pipeline plus a Flask service.

## Project-Specific Walkthrough

The project predicts international football match outcomes:

- `H`: home win
- `D`: draw
- `A`: away win

It also predicts expected goals and likely scorelines with a Poisson goal model, then uses those outputs to simulate a World Cup 2026 tournament.

Current important runtime details:

- Docker serves the Flask app with Gunicorn through `wsgi.py`.
- Prediction requests can include `match_date` and optional rest-day overrides.
- When rest days are not supplied, the predictor infers them from match history in `feature_matrix.csv`.
- The predictor can hot-reload both model artifacts and feature-matrix state.
- Training now saves separate evaluation artifacts for holdout metrics and production-refit artifacts for serving.
- Live standings can be derived from finished fixtures before falling back to cache/API/mock data.

The main architecture is:

```text
data/raw/*.csv
  -> src/data/*.py
  -> data/processed/matches_clean.csv
  -> src/features/*.py
  -> data/features/feature_matrix.csv
  -> src/models/train.py
  -> models/registry/*.pkl + meta.json
  -> src/models/predict.py
  -> src/models/simulate.py
  -> src/api/app.py + src/api/routes.py
  -> src/api/templates/*.html
```

The app uses files as storage. There is no database, no auth system, and no CI/CD config in the tracked repo.

## File Groups

| Group | What it means |
|---|---|
| Root files | Project metadata, config, Docker, dependency manifest. |
| `src/data` | Raw file checks, validation, cleaning, ranking merge. |
| `src/features` | Elo, form, goals, rest days, continent, match stake. |
| `src/models` | Baselines, training, evaluation, prediction, simulation, Poisson score model. |
| `src/api` | Flask app, JSON endpoints, HTML page rendering. |
| `src/utils` | Shared config, logger, live football API/cache client, fixture-derived standings logic. |
| `scripts` | Runnable workflows for pipeline, retrain, diagnostics, scheduler, live match fetch, Elo optimization. |
| `tests` | pytest tests for deterministic behavior and runtime contracts. |
| `data`, `models`, `visualisations`, `logs` | Data snapshots, generated/local artifacts, plots, and logs used by the app. Current model artifacts remain ignored by git. |

## Interview Explanation

Use this answer:

> This is an end-to-end Python ML application for football predictions. The pipeline validates and cleans historical match data, builds chronological features like Elo, recent form, goal averages, rest days, and tournament importance, trains and calibrates several classifiers, reports holdout metrics from evaluation artifacts, refits production artifacts on all completed rows, then serves predictions through Flask and Gunicorn. At runtime, the predictor builds the same feature vector, infers missing rest days from the feature matrix, returns prediction context, and supports hot reload when artifacts change. The UI calls the Flask API, and a Monte Carlo simulator uses the prediction and scoreline models to simulate the full tournament. The design is file-based rather than database-backed, which is simple and transparent for a portfolio project but not fully production-grade.

## Common Interview Questions

| Question | Strong answer |
|---|---|
| Is this just a notebook project? | No. The notebooks are effectively empty in the current repo. The real project is a modular Python pipeline with Flask serving and tests. |
| What is the biggest design strength? | Chronological feature engineering and time-based model splitting reduce future-data leakage. |
| What is the biggest weakness? | Draw prediction is weak in the current metadata, and runtime artifacts are file-based without production-grade versioning or locking. |
| Does it use a database? | No. CSV, JSON, pickle, and log files act as storage. |
| What changed in the latest runtime/training flow? | Gunicorn now serves `wsgi:app`; predictions infer rest context; feature-matrix changes can refresh predictor state; standings can be derived from fixtures; production serving artifacts are separated from holdout evaluation artifacts. |

## Rebuild Exercise

On paper, redraw this architecture without looking:

```text
raw CSV -> cleaned CSV -> feature matrix -> trained artifacts -> predictor -> simulator/API/UI
```

Then explain what would break if `models/registry/scaler.pkl` was missing.

## Self-Check Quiz

1. What file is the central config source?
2. Which folder contains the Flask routes?
3. Which file creates the final model-ready CSV?
4. Why is chronological processing important?
5. Is a database present?

Answers:

1. `config.yaml`
2. `src/api/routes.py`
3. `src/features/build.py`
4. It prevents using future match information as pre-match features.
5. No.

## External Links

- Flask application factories: https://flask.palletsprojects.com/en/stable/patterns/appfactories/
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- Dockerfile docs: https://docs.docker.com/build/concepts/dockerfile/

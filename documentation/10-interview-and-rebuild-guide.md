# 10 - Interview and Rebuild Guide

## What You Will Understand After This Lesson

- How to explain the project at beginner, intermediate, and senior levels.
- How to answer likely interview questions.
- How to rebuild a similar app from scratch.
- How to be honest about limitations without underselling the work.

## Project Pitch

### 30-Second Version

> I built a Python ML app that predicts international football match outcomes and simulates the World Cup. It validates and cleans historical match data, engineers chronological features like Elo, recent form, goal averages, inferred rest days, rankings, and tournament importance, trains and calibrates multiple classifiers, reports holdout metrics from evaluation artifacts, then serves production-refit predictions through Flask and Gunicorn with a browser UI. A Monte Carlo simulator uses the prediction and scoreline models to estimate tournament champion probabilities.

### 2-Minute Version

> The project has two lifecycles: training and serving. The training pipeline checks raw CSVs, validates schemas, normalizes team names, creates H/D/A labels, merges FIFA ranking snapshots with backward temporal joins, builds features chronologically, trains several models, calibrates probabilities, selects the best by log loss, saves evaluation artifacts for holdout reporting, then refits production artifacts on all completed rows. The serving lifecycle loads the production scaler, model, metadata, score model, and feature matrix, builds a latest-state dictionary for teams, resolves match context, infers missing rest days from match history, constructs the same 27-feature vector at request time, predicts probabilities, symmetrizes neutral matches, and returns expected goals, scorelines, and context explaining what was used. Flask exposes JSON endpoints and HTML templates, while Docker runs the web app through Gunicorn via `wsgi:app`. The simulator runs repeated tournament simulations and updates team states after each simulated match.

### Honest Limitations

> The current project is strong as a portfolio ML application, but it is not fully production-grade. It has no database, no auth, no CI/CD, no atomic model/data artifact swap, and the current metadata shows poor draw recall. Model artifacts are intentionally ignored and local. I would improve artifact versioning, centralize feature schemas, add stricter validation, and run stronger backtesting before deploying it as a public prediction product.

## Interview Question Bank

### Beginner

| Question | Strong answer |
|---|---|
| What does the app predict? | Home win, draw, or away win, plus expected goals and likely scorelines. |
| What is the main config file? | `config.yaml`. |
| What is the training data output? | `data/features/feature_matrix.csv`. |
| What framework serves the app? | Flask. |
| What serves Flask in Docker? | Gunicorn imports `app` from `wsgi.py` and serves it as a WSGI application. |

### Intermediate

| Question | Strong answer |
|---|---|
| Why use temporal splits? | To avoid evaluating on past data after training on future data. |
| Why use calibration? | The app depends on probabilities, so predicted probabilities should match real frequencies. |
| Why merge rankings backward? | Only rankings known before or on match date should be used. |
| Why use symmetric prediction? | Neutral matches should not depend on arbitrary home/away input order. |
| How are rest days handled now? | If the request does not provide rest-day overrides, the predictor infers each team's rest from its latest known match before the prediction date and returns that context. |
| Why are there production and evaluation artifacts? | Evaluation artifacts preserve the temporal holdout used for reported metrics. Production artifacts are refit after selection on all completed rows so served predictions use the latest completed data. |

### Advanced

| Question | Strong answer |
|---|---|
| How does the Poisson model become H/D/A probabilities? | It predicts expected home/away goals, builds a scoreline matrix, then sums home-win, draw, and away-win regions. |
| What is the biggest leakage risk? | Accidentally using post-match features or future ranking snapshots. The code mitigates this with before-update features and backward joins. |
| Why choose log loss over accuracy? | Tournament simulation needs good probability estimates, not only correct hard labels. |
| Why hot-reload the feature matrix? | Latest team states, default prediction dates, and inferred rest days depend on `feature_matrix.csv`, so data-only updates should be visible without restarting the server. |
| How does production calibration work now? | `fit_production_calibrated_model` clones the selected estimator and uses `CalibratedClassifierCV` with `StratifiedKFold` out-of-fold calibration on all completed rows. |
| What would you refactor first? | Centralize feature schema, fix unreachable simulator fast path, and add artifact versioning. |

### Senior

| Question | Strong answer |
|---|---|
| How would you productionize it? | Add CI, artifact registry/versioning, atomic deploys, job queue for retraining, stricter API validation, monitoring, drift checks, and secrets management. |
| How would you improve draw prediction? | Tune threshold/objective for draw recall or macro metrics, evaluate calibration per class, rebalance or model draws separately, and compare against proper backtests. |
| How would you make retraining safe? | Train into a new versioned directory, validate artifacts, write metadata last, then atomically swap the active pointer. |
| How would you make live standings reliable? | Prefer official data with clear freshness metadata, but keep fixture-derived standings as a fallback. Store snapshots transactionally and surface stale-data status in the UI. |

## Rebuild From Scratch

1. Define the target:
   - Predict `H`, `D`, `A`.
   - Return probabilities, expected goals, and tournament simulation output.

2. Create folder structure:
   - `src/data`
   - `src/features`
   - `src/models`
   - `src/api`
   - `src/utils`
   - `scripts`
   - `tests`
   - `data`
   - `models/registry`
   - `visualisations`

3. Add config and logging:
   - Implement YAML config loader.
   - Implement centralized logger.

4. Build data pipeline:
   - Check raw files.
   - Validate schemas.
   - Normalize team names.
   - Create `result`.
   - Merge rankings with backward temporal join.

5. Build features:
   - Elo.
   - Form.
   - Goal averages.
   - Ranking differences.
   - Rest/context features.

6. Train baseline models:
   - Random baseline.
   - Most frequent class.
   - Elo heuristic.

7. Train ML models:
   - Logistic Regression.
   - Random Forest.
   - HistGradientBoosting.
   - Optional LightGBM/CatBoost/XGBoost.
   - Optional Stacking Ensemble.

8. Add calibration and evaluation:
   - Calibrated probabilities.
   - Accuracy, log loss, Brier.
   - Confusion matrix.
   - Calibration curve.
   - Separate evaluation artifacts from production-refit artifacts.

9. Build prediction runtime:
   - Load artifacts.
   - Use production artifacts for serving.
   - Build latest team states.
   - Load feature matrix state.
   - Infer missing rest days from match history.
   - Construct features in fixed order.
   - Predict and return JSON with context.

10. Build simulator:
    - Define tournament structure.
    - Simulate matches.
    - Update states.
    - Aggregate Monte Carlo results.

11. Build Flask API:
    - Health endpoint.
    - Teams endpoint.
    - Predict endpoint.
    - Simulation endpoint.
    - Metadata/visualization endpoints.

12. Build UI:
    - Prediction page.
    - Insights page.
    - Simulation page.
    - Live/cache page if API keys are available.

13. Add tests:
    - Clean functions.
    - Feature math.
    - Model selection.
    - Prediction contract.
    - Simulation state updates.

14. Add Docker:
    - Dockerfile.
    - Compose web/scheduler.
    - `.dockerignore`.
    - `wsgi.py`.
    - Gunicorn command.

## Practice Tasks

1. Explain the project without saying "AI" or "vibe coded".
2. Trace one row from `matches.csv` to model prediction.
3. Explain why `meta.json` is a runtime contract.
4. Explain why `feature_matrix.csv` is also a runtime input, not only a training artifact.
5. Explain why holdout metrics should come from `evaluation_model.pkl`, not from the production-refit artifact.
6. Name three production improvements and why they matter.
7. Rebuild a minimal version with only Logistic Regression and Flask.

## Self-Check Quiz

1. What should you avoid overclaiming?
2. What metric selects the active model by default?
3. What feature prevents arbitrary neutral venue bias?
4. What is the fastest credible rebuild path?
5. What file exists mainly for Gunicorn?

Answers:

1. Notebook analysis, production readiness, draw quality, and hidden data provenance.
2. Log loss.
3. Symmetric prediction averaging.
4. Data cleaning -> features -> Logistic Regression -> Flask `/api/predict`.
5. `wsgi.py`.

## External Links

- Kaggle Intro to Machine Learning: https://www.kaggle.com/learn/intro-to-machine-learning
- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Real Python Flask tutorials: https://realpython.com/tutorials/flask/
- Flask Gunicorn deployment: https://flask.palletsprojects.com/en/stable/deploying/gunicorn/

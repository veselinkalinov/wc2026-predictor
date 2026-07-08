# 05 - Machine Learning Models

## What You Will Understand After This Lesson

- What every classifier/regressor does in plain English.
- How the project trains, tunes, calibrates, evaluates, and selects models.
- Why accuracy alone is not enough for this project.
- What every dependency in `requirements.txt` contributes.
- Why the scoreline model needs numerical-stability guards.

## First Principles

A classifier maps input features to class probabilities or labels. In this project:

```text
27 numeric features -> probabilities for H, D, A
```

Because the app shows probabilities and runs simulations, probability quality matters. A model can have decent accuracy but poor probabilities. That is why the project tracks:

- Accuracy: how often the final class is correct.
- Log loss: how well predicted probabilities match reality.
- Brier score: mean squared error of probabilities.
- Calibration: whether predicted probabilities behave like real frequencies.

## Training Workflow

File: `src/models/train.py`

```text
feature_matrix.csv
  -> map H/D/A to 0/1/2
  -> temporal split
  -> StandardScaler
  -> GridSearchCV + TimeSeriesSplit
  -> train base models
  -> train Poisson score model
  -> optional stacking ensemble
  -> CalibratedClassifierCV
  -> draw threshold tuning
  -> holdout evaluation
  -> select winner by configured metric
  -> save holdout evaluation artifacts
  -> refit production artifacts on all completed rows
  -> serialize artifacts and metadata
```

The split is:

- Train: before `model.train_cutoff`
- Calibration: train cutoff to `model.calibration_cutoff`
- Test: after calibration cutoff

This is better than a random split for time-ordered sports data because the model should be evaluated on future-like matches.

## Holdout Evaluation vs Production Refit

The current training code has two different artifact roles:

| Role | Files | What they are for |
|---|---|---|
| Holdout evaluation | `evaluation_model.pkl`, `evaluation_scaler.pkl`, `evaluation_score_model.pkl` | Preserve the temporal split used to report honest test metrics. |
| Production serving | `best_model.pkl`, `scaler.pkl`, `score_model.pkl` | Refit on all completed feature rows so the live predictor uses the freshest completed data. |

First principle:

A model evaluated on future-like holdout data should not train on that holdout before metrics are computed. But after choosing a model family and measuring quality, a production system often refits on all available completed data to serve the strongest current artifact.

Project-specific flow:

```text
train/cal/test split
  -> tune and calibrate candidate models
  -> evaluate on holdout test set
  -> select best model by log loss
  -> save evaluation artifact for metrics
  -> refit each candidate on all completed rows
  -> save selected production artifact for runtime prediction
```

`fit_production_calibrated_model(...)` implements the production refit. It uses:

- `clone(estimator)` so the production model starts from a fresh estimator object.
- `StratifiedKFold` so each calibration fold preserves class balance as much as possible.
- `CalibratedClassifierCV(..., ensemble=False)` so calibration is learned from out-of-fold predictions instead of directly calibrating on the same in-sample predictions.
- `model.production_cv_splits` from `config.yaml`, currently `3`, capped by the smallest present class count.

Interview answer:

> The project now separates evaluation honesty from serving freshness. Holdout metrics come from `evaluation_model.pkl`, which respects the temporal split. The served `best_model.pkl` is then refit on all completed rows after selection, so predictions can use the latest completed match data without pretending those same rows were unseen test data.

## Classifiers and Regressors

### Logistic Regression

Project usage: baseline linear classifier and stacking meta-model.

Plain English:

Logistic Regression learns weights for each feature and turns the weighted score into class probabilities.

Why useful:

- Fast.
- Interpretable.
- Strong baseline.

Limit:

- Linear decision boundaries unless features already encode nonlinear patterns.

### Random Forest

Project usage: tree ensemble in model comparison.

Plain English:

A Random Forest trains many decision trees on random samples/features and averages their predictions.

Why useful:

- Handles nonlinear interactions.
- Less sensitive to scaling than linear models.

Limit:

- Can be large. Current local `random_forest.pkl` is much larger than other artifacts.

### HistGradientBoostingClassifier

Project usage: scikit-learn gradient boosting model.

Plain English:

Gradient boosting builds trees sequentially. Each new tree tries to correct errors from previous trees. The histogram version bins features for speed.

Why useful:

- Strong tabular model.
- Efficient on medium/large data.

### LightGBM

Project usage: optional external gradient boosting model if installed.

Plain English:

LightGBM is a high-performance gradient boosting library optimized for tabular data.

Why useful:

- Often very strong on structured data.
- Fast training.

Limit:

- Requires native dependencies. Docker installs `libgomp1` for OpenMP support.

### CatBoost

Project usage: optional external boosting model if installed.

Plain English:

CatBoost is a gradient boosting library known for strong defaults and categorical feature support. This project uses numeric features, but CatBoost still works as a tabular booster.

Limit:

- Can create `catboost_info/` training logs.

### XGBoost

Project usage: optional external boosting model if installed.

Plain English:

XGBoost is another high-performance gradient boosting library for tabular ML.

Project-specific detail:

`train.py` imports it in a try/except block so missing XGBoost does not break the whole project.

### Stacking Ensemble

Project usage: current active best model in `meta.json`.

Plain English:

Stacking trains several base models, then trains a final model to combine their predictions.

Why useful:

- Can blend strengths of multiple models.

Limit:

- Harder to explain and debug than a single model.

### PoissonGoalModel

Project usage: scoreline and expected-goals model.

Plain English:

Instead of directly predicting H/D/A, it predicts expected home and away goals. Then it builds a probability grid for scorelines such as 1-0, 1-1, 2-1, and sums the grid into H/D/A probabilities.

Project-specific detail:

It applies a Dixon-Coles-style low-score correction for 0-0, 0-1, 1-0, and 1-1.

Current stability detail:

`src/models/poisson_model.py` clips non-finite or extreme expected goals before building the scoreline grid. It also handles invalid probability mass by falling back to a single safe bucket before normalization.

Why this exists:

Probability vectors must not contain `NaN`, `inf`, or all-zero mass. A tournament simulator can call scoreline prediction thousands of times. One invalid matrix can break JSON responses, poison downstream probabilities, or create impossible simulation outcomes.

Interview answer:

> The Poisson model is useful because football scores are count data, but any statistical count model still needs defensive numerical handling. The project clips expected goals to a reasonable range and normalizes the scoreline matrix only after checking that the probability mass is finite and positive.

## Calibration

File: `src/models/train.py`

`CalibratedClassifierCV` wraps each trained classifier.

- Tree/boosting models use isotonic calibration.
- Linear/stacking/Poisson path uses sigmoid calibration.

Why:

If the model says "Brazil has 70% chance", then over many similar cases about 70% should happen. Calibration tries to make probability outputs more honest.

## Model Selection

Function: `select_best_model`

Default selection:

1. Lowest log loss.
2. If tied, lowest Brier score.
3. If tied, highest accuracy.

This is correct for a probability product because the UI and simulator consume probabilities.

Known issue:

Current metadata shows draw recall is `0.00`. That means the active thresholding strategy does not produce draw labels on the holdout report, even though draw probabilities are still surfaced.

Current local ignored artifact metadata from `models/registry/meta.json` reports:

| Field | Current local value |
|---|---:|
| Active model | `Stacking Ensemble` |
| Selected by | `log_loss` |
| Artifact role | `production_refit` |
| Holdout accuracy | `0.6100374064837906` |
| Holdout log loss | `0.866081836103257` |
| Holdout Brier score | `0.1693144308142882` |
| Holdout test samples | `3208` |
| Production refit samples | `15647` |
| Latest production match date | `2026-06-23` |
| Production CV splits | `3` |

These values are from local ignored artifacts, not tracked source. In an interview, distinguish between "the code trains and selects models this way" and "my current local artifact has these metrics."

## Evaluation

File: `src/models/evaluate.py`

Outputs:

- classification report
- accuracy
- log loss
- Brier score
- confusion matrix PNG
- feature importance/contribution PNG
- calibration curve PNG
- updated `meta.json`

Current detail:

`evaluate.py` reads `meta["artifacts"]["evaluation_model"]` and `meta["artifacts"]["evaluation_scaler"]` when those metadata entries exist. If those files are absent, it falls back to `best_model.pkl` and `scaler.pkl` for backward compatibility. In the current artifact contract, the evaluation path should use `evaluation_model.pkl`.

## Dependency Coverage

| Dependency | Purpose in project | Core concept | Interview angle |
|---|---|---|---|
| `pandas` | CSV loading, cleaning, merging, feature tables. | DataFrame tabular processing. | Explain `merge_asof` and chronological sorting. |
| `scipy` | Poisson PMF in scoreline model. | Scientific probability/math functions. | Explain why scorelines can be modeled with Poisson counts. |
| `scikit-learn` | Models, scaler, metrics, CV, calibration. | Standard ML toolkit. | Explain estimator API, `fit`, `predict_proba`, metrics. |
| `seaborn` | Confusion matrix heatmap. | Statistical plotting. | Explain visual model diagnostics. |
| `matplotlib` | Saves plots to PNG. | Plot rendering backend. | Explain why `Agg` is used in non-GUI environments. |
| `requests` | HTTP calls to football APIs. | Synchronous HTTP client. | Explain timeout/error handling. |
| `flask` | Web API and template serving. | Python micro web framework. | Explain routes, blueprints, app factory. |
| `pyyaml` | Loads `config.yaml`. | YAML parsing. | Explain config as data. |
| `jupyter` | Notebook tooling. | Exploratory computing. | Current notebooks are empty, so do not overclaim. |
| `notebook` | Jupyter server package. | Browser notebook UI. | Same as above. |
| `ipykernel` | Kernel for notebooks. | Python execution kernel. | Same as above. |
| `python-dotenv` | Loads `.env`. | Local environment variable loading. | Explain API keys without hardcoding secrets. |
| `joblib` | Saves/loads model artifacts. | Python object persistence for ML. | Mention trusted-artifact requirement. |
| `pytest` | Test runner. | Automated tests and fixtures. | Explain `tests/conftest.py` dummy model setup. |
| `gunicorn` | Serves the Flask app in Docker through `wsgi:app`. | WSGI HTTP server with worker processes. | Explain why the deployment runtime is different from Flask's development server. |
| `lightgbm` | Optional boosting model. | Gradient boosted trees. | Strong tabular model with native dependency. |
| `catboost` | Optional boosting model. | Gradient boosted trees. | Strong defaults, may create training logs. |
| `xgboost` | Optional boosting model. | Gradient boosted trees. | Imported dynamically for robustness. |

Versioning note:

`requirements.txt` now uses exact pins such as `pandas==2.2.2`, `scikit-learn==1.5.1`, and `gunicorn==23.0.0`. Exact pins matter because pickled ML artifacts and model behavior can change across library versions.

## Common Interview Questions

| Question | Strong answer |
|---|---|
| Why use log loss for selection? | The app needs reliable probabilities for UI and simulations, not only hard-label accuracy. |
| Why scale features? | Linear models and some optimization methods behave better when features are standardized. The same scaler must be used at prediction time. |
| Why use TimeSeriesSplit? | It keeps training folds earlier than validation folds, reducing future leakage. |
| Why calibrate? | To make predicted probabilities better match observed frequencies. |
| Why add `StratifiedKFold` for production calibration? | The production artifact uses all completed rows, so it cannot use the old temporal holdout for calibration. Stratified folds preserve class balance while producing out-of-fold calibration data. |
| Why are `evaluation_model.pkl` and `best_model.pkl` both needed? | `evaluation_model.pkl` supports honest holdout metrics. `best_model.pkl` is the production-refit artifact used by the predictor. |
| Why add numerical guards to the Poisson score model? | Scoreline probabilities are consumed by the UI and simulator. Guards prevent invalid lambdas from becoming `NaN` probabilities or broken JSON responses. |
| What would you improve in the model? | Better draw objective, more time-series backtesting, artifact versioning, and richer real-world features. |

## Rebuild Exercise

Train three models on the same feature table:

1. Logistic Regression.
2. Random Forest.
3. HistGradientBoostingClassifier.

Evaluate all three by accuracy and log loss. Pick the best by log loss and explain why.

## Self-Check Quiz

1. Which file trains the models?
2. Which file generates plots?
3. What is the active model type in current metadata?
4. Why is draw recall a known issue?
5. What does `joblib.dump` save?

Answers:

1. `src/models/train.py`
2. `src/models/evaluate.py`
3. Stacking Ensemble.
4. Current holdout classification report has zero draw recall.
5. Python model/scaler objects as serialized artifacts.

## External Links

- scikit-learn Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn calibration: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- scikit-learn Brier score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
- Kaggle Intro to Machine Learning: https://www.kaggle.com/learn/intro-to-machine-learning

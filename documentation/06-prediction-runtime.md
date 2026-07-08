# 06 - Prediction Runtime

## What You Will Understand After This Lesson

- How saved model artifacts are loaded.
- How latest team states are built.
- How a user matchup becomes a 27-feature vector.
- How neutral symmetric prediction works.
- How scoreline probabilities are added.
- How caching and hot reload work.
- How match date and rest-day context are inferred and returned to the API/UI.

## First Principles

Training and prediction must use the same feature meaning and order.

At training time:

```text
DataFrame columns -> scaler.fit_transform -> model.fit
```

At prediction time:

```text
team names -> latest team state -> same columns -> scaler transform -> model.predict_proba
```

If the order or scaling changes, predictions become invalid.

## Project-Specific Walkthrough

Prediction starts in one of two ways:

- Direct Python: `MatchPredictor().predict_match(...)`
- API: `POST /api/predict`

The core class is `src/models/predict.py::MatchPredictor`.

## Initialization Flow

When `MatchPredictor` starts:

1. Resolves model paths from `config.yaml`.
2. Loads production `scaler.pkl`.
3. Loads production `best_model.pkl` or requested model.
4. If external package loading fails, tries fallback models.
5. Loads `meta.json`.
6. Loads `score_model.pkl`.
7. Reads `feature_matrix.csv` through `_load_feature_matrix_state()`.
8. Parses feature dates and rejects invalid feature dates.
9. Sets `default_prediction_date` to the later of today and the latest feature date.
10. Builds latest team states.
11. Stores model and feature-matrix modification times for hot reload.
12. Creates an empty prediction cache.

Current artifact contract:

| Artifact | Runtime use |
|---|---|
| `best_model.pkl` | Production calibrated classifier used by `predict_match`. |
| `scaler.pkl` | Production scaler paired with `best_model.pkl`. |
| `score_model.pkl` | Production Poisson score model used by `predict_scoreline`. |
| `meta.json` | Runtime contract for feature order, class labels, thresholds, artifact roles, and evaluation metadata. |
| `evaluation_model.pkl` | Not used by normal prediction runtime; used by evaluation/reporting code. |
| `evaluation_scaler.pkl` | Paired with `evaluation_model.pkl` for holdout reporting. |

Interview answer:

> The API serves from the production-refit artifacts: `best_model.pkl`, `scaler.pkl`, and `score_model.pkl`. The evaluation artifacts exist so the reported metrics remain tied to the temporal holdout. That prevents mixing "what I serve" with "what I measured as unseen performance."

## Team State

A team state is a compact snapshot:

```text
elo
form
goals_scored_avg
goals_conceded_avg
goal_diff_avg
rank
rank_points
```

`_build_latest_team_states` finds each team's latest match in the feature matrix and extracts the correct home or away columns.

Unknown teams get defaults:

```text
elo = 1500
form = 0.5
goals = 1.2
rank = 211
rank_points = 0
```

The current implementation rebuilds this state not only when model artifacts change, but also when `data/features/feature_matrix.csv` changes. That matters because retraining or importing new matches can update team form/rest context without changing the model pickle.

## Prediction Context: Match Date and Rest Days

The API and predictor now support optional context:

```text
home_team
away_team
neutral
match_date optional
home_rest_days optional
away_rest_days optional
match_stake optional
```

First principle:

Rest days are a time-dependent feature. If the user does not manually provide rest days, the system should infer them from the latest known match before the prediction date.

Project-specific flow:

```text
predict_match(...)
  -> _resolve_prediction_date(match_date)
  -> _resolve_rest_context(...)
  -> infer_rest_days(team, prediction_date)
  -> _construct_features_numpy(...)
```

`_resolve_prediction_date(match_date)`:

- Uses the supplied `match_date` when it can be parsed.
- Falls back to `default_prediction_date` for missing/invalid values.
- Normalizes to a date so cache keys and output context are stable.

`infer_rest_days(team, match_date)`:

- Looks in `feature_matrix.csv` for the team's most recent match before the prediction date.
- Calculates the number of days between that match and the prediction date.
- Caps the result at 30 days.
- Returns 30 when no prior match is found.

`_resolve_rest_context(...)`:

- If both rest-day overrides are missing, it infers both values and marks the source as `inferred`.
- If either override is supplied, it uses the explicit override path and marks the source as `override`.

The prediction response now includes:

```json
{
  "context": {
    "match_date": "2026-06-24",
    "rest_days": {
      "home": 5,
      "away": 7,
      "source": "inferred"
    },
    "match_stake": "group"
  }
}
```

Interview answer:

> Rest days used to be a default numeric parameter. The current predictor treats rest as match context. If the user does not provide rest-day values, it infers them from historical team matches before the prediction date and returns that context so the UI can explain which values were used.

## Feature Construction

`_construct_features_numpy` creates a row in exactly the order from `meta.json`.

It computes:

- strength differences
- form differences
- goal differences
- ranking differences
- neutral/competitive context
- rest-day difference
- continent/home-continent indicators
- match stake

Then `_scale_features` manually applies:

```python
(features - self.scaler.mean_) / self.scaler.scale_
```

This matches `StandardScaler.transform`.

Common mistake:

Do not claim rest days are always hardcoded as `7`. The current default path is historical inference from the feature matrix. Explicit rest-day values are still supported for manual overrides and tests.

## Symmetric Neutral Prediction

For neutral matches, the project predicts twice:

```text
Brazil home vs Argentina away -> [P(H), P(D), P(A)]
Argentina home vs Brazil away -> [P(H), P(D), P(A)]
invert reverse -> [P(A), P(D), P(H)]
average both
```

Why:

On a neutral field, Brazil vs Argentina should not depend on which team is typed into the "home" input.

This behavior is tested in `tests/test_predict.py`.

## Scoreline Runtime

`predict_scoreline` uses `score_model.pkl` if available.

It returns:

- expected home goals
- expected away goals
- top scoreline probabilities
- internal scoreline matrix

For neutral games, it also symmetrizes the scoreline grid.

If the score model is missing, `_fallback_scoreline_payload` estimates expected goals from team scoring/conceding averages.

## Caching and Hot Reload

Prediction cache:

- Stores previous matchup results.
- Cache key rounds some state differences to reduce repeated work in simulations.

Hot reload:

- `_check_and_reload` checks model file modification time.
- Checks are throttled to once every 10 seconds.
- If the model changed, it reloads model, scaler, metadata, score model, feature matrix, and team states.
- If only `feature_matrix.csv` changed, it reloads feature matrix state and clears the prediction cache without requiring a model artifact change.

Risk:

Hot reload does not guarantee atomic artifact swaps. If a retrain writes files while the web app reads them, partial reads can still be a production concern.

Why feature-matrix reload matters:

The predictor's latest team state, default prediction date, and inferred rest days all come from `feature_matrix.csv`. Without feature reload, a finished-match import could update the CSV while the API kept serving old team states until a full process restart or model pickle change.

## Common Interview Questions

| Question | Strong answer |
|---|---|
| Why does feature order matter? | The scaler and model learned numeric columns in a fixed order. A wrong order changes the meaning of every coefficient/tree split. |
| Why average reverse predictions for neutral venues? | It removes artificial home/away input bias when no team has home advantage. |
| How are unknown teams handled? | They receive conservative default states: Elo 1500, form 0.5, goals 1.2, rank 211. |
| What does hot reload solve? | It lets the API pick up newer model files without manually restarting the server. |
| Why does hot reload also watch `feature_matrix.csv` now? | The predictor's team states and inferred rest days depend on the feature matrix, so data-only updates must refresh runtime state too. |
| What does the prediction context object add? | It makes hidden runtime assumptions visible: the match date, rest-day values, whether rest was inferred or overridden, and match stake. |
| Does the predictor use `evaluation_model.pkl`? | No. Normal predictions use production artifacts. Evaluation artifacts are for holdout reports and plots. |

## Rebuild Exercise

Write a simple predictor class that:

1. Loads a scaler and classifier.
2. Builds a feature vector from two team dictionaries.
3. Predicts forward and reverse probabilities.
4. Averages them for neutral matches.

Do not start with Flask. Start with plain Python.

## Self-Check Quiz

1. Which file defines `MatchPredictor`?
2. Which metadata field defines feature order?
3. What does `draw_risk_threshold` do?
4. Why is scoreline fallback needed?

Answers:

1. `src/models/predict.py`
2. `meta["features"]`
3. It flags high draw probability even when final prediction is not draw.
4. It keeps expected-goal output available if the Poisson score model is missing.

## External Links

- joblib persistence: https://joblib.readthedocs.io/en/latest/persistence.html
- scikit-learn StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn predict_proba concept: https://scikit-learn.org/stable/glossary.html#term-predict_proba

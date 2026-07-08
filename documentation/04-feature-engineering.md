# 04 - Feature Engineering

## What You Will Understand After This Lesson

- Why features are computed chronologically.
- How Elo, form, and goal averages work.
- How rest days, continent indicators, and match stake are added.
- How `feature_matrix.csv` becomes the model input table.
- How training-time rest-day features relate to runtime rest-day inference.

## First Principles

Raw data rarely contains the exact signals a model needs. Feature engineering converts raw rows into useful numeric inputs.

For sports prediction, good pre-match features usually describe:

- Team strength.
- Recent form.
- Attack and defense.
- Ranking/quality gap.
- Match context.
- Travel or home advantage.

The most important rule: only use information available before the match.

## Project-Specific Walkthrough

Feature construction starts from `data/processed/matches_clean.csv` and outputs `data/features/feature_matrix.csv`.

```text
matches_clean.csv
  -> compute_elo_ratings
  -> compute_form_features
  -> compute_goal_features
  -> compute_advanced_features
  -> rank diffs + neutral flag
  -> date/team filters
  -> feature_matrix.csv
```

## File-by-File Explanation

### `src/features/elo.py`

Elo estimates team strength.

Core formula:

```python
1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
```

This returns team A's expected score against team B.

K-factor logic:

- World Cup: 60
- Major continental/confederation tournaments: 50
- Qualifiers/Nations League: 40
- Friendlies/minor tournaments: 20

Goal margin:

- 0 or 1 goal margin: `1.0`
- 2 goals: `1.5`
- 3+ goals: `(11 + abs_diff) / 8`

Important implementation detail:

`compute_elo_ratings` appends `home_elo` and `away_elo` before updating ratings for the match. That means the features represent pre-match ratings, not post-match leakage.

### `src/features/form.py`

Form estimates recent performance.

`compute_ewma_form(history, alpha)`:

- Returns `0.5` if a team has no history.
- Converts match points to a 0-1 scale by dividing by 3.
- Gives more weight to recent matches.
- Clamps result to `[0, 1]`.

Opponent adjustment:

```python
home_opp_factor = max(1.0 + (away_elo - 1500.0) / 1000.0, 0.5)
```

Beating a strong opponent counts more than beating a weak opponent.

### `src/features/goals.py`

Goal features estimate attack and defense.

For each team, the project tracks:

- goals scored average
- goals conceded average
- goal difference average

Cold-start default is `1.2` goals.

Like Elo and form, these are recorded before the current match updates history.

### `src/features/build.py`

This file orchestrates all feature builders.

`compute_advanced_features` adds:

- `home_rest_days`, `away_rest_days`, `rest_days_diff`
- `home_is_home_continent`, `away_is_home_continent`, `continent_diff`
- `match_stake`

Rest days:

- Previous match date per team is tracked.
- New teams default to 30 days.
- Rest is capped at 30 days to avoid huge gaps dominating.

Important current-runtime distinction:

`src/features/build.py` computes rest-day columns for historical training rows. At prediction time, `src/models/predict.py` can infer rest days again from `feature_matrix.csv` based on the requested `match_date`. That means rest days are both:

- a training feature stored in `feature_matrix.csv`
- a serving-time context value resolved for the requested prediction

Interview answer:

> During training, rest-day features are calculated chronologically for each historical match. During serving, if the API does not receive explicit rest-day values, the predictor infers rest from the team's most recent known match before the prediction date. Both paths preserve the same feature meaning: days since the team's previous match, capped at 30.

Match stake:

- World Cup: 4
- Major tournament: 3
- Qualifier/Nations League: 2
- Other: 1

Then `build_feature_matrix` adds:

- `rank_diff`
- `rank_points_diff`
- `is_neutral`
- date range filtering
- minimum team match-count filtering
- null dropping

## Active Model Features

The active feature list in `models/registry/meta.json` has 27 columns:

```text
home_elo, away_elo, elo_diff,
home_form, away_form, form_diff,
home_goals_scored_avg, home_goals_conceded_avg, home_goal_diff_avg,
away_goals_scored_avg, away_goals_conceded_avg, away_goal_diff_avg,
home_rank, away_rank, rank_diff,
home_rank_points, away_rank_points, rank_points_diff,
is_neutral, is_competitive,
home_rest_days, away_rest_days, rest_days_diff,
home_is_home_continent, away_is_home_continent, continent_diff,
match_stake
```

The order matters. Training and prediction must use the same order.

## Common Interview Questions

| Question | Strong answer |
|---|---|
| Why compute features chronologically? | To ensure every row only uses information available before that match. |
| Why defer date filtering until `build.py`? | Older matches warm up Elo/form/goal histories before the selected training period begins. |
| What is EWMA? | Exponentially weighted moving average. Recent matches receive more weight than older matches. |
| What is a feature leakage risk here? | If post-match ratings or future ranking snapshots were used as pre-match features. The code avoids this with before-update features and backward ranking merge. |
| Why can rest days be computed in both features and prediction runtime? | Training rows need historical rest-day columns, while future predictions need context-specific rest days for the requested match date. The meaning is the same, but the execution point is different. |

## Rebuild Exercise

Use a tiny four-match dataset and manually compute:

1. Initial Elo values.
2. Pre-match Elo for each row.
3. One team's form after each match.
4. One team's goals scored/conceded average.

Then compare your manual result with the code.

## Self-Check Quiz

1. Which feature file computes `elo_diff`?
2. What is the cold-start form value?
3. What is the cold-start goals value?
4. What does `match_stake` represent?
5. Why is feature order important?

Answers:

1. `src/features/elo.py`
2. `0.5`
3. `1.2`
4. Tournament importance/context.
5. The scaler and model expect columns in the same order used during training.

## External Links

- Elo rating system overview: https://en.wikipedia.org/wiki/Elo_rating_system
- pandas time series basics: https://pandas.pydata.org/docs/user_guide/timeseries.html
- Kaggle feature engineering: https://www.kaggle.com/learn/feature-engineering

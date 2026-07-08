# 03 - Data Pipeline

## What You Will Understand After This Lesson

- How raw match and ranking data becomes `matches_clean.csv`.
- Why the pipeline validates before cleaning.
- How team-name normalization works.
- Why FIFA rankings are merged with `merge_asof`.
- What each data file contributes.
- How recent-match imports update existing rows without duplicating matches.

## First Principles

Machine-learning models are only as useful as the data they receive. A data pipeline should:

1. Check inputs exist.
2. Validate the expected schema.
3. Normalize inconsistent values.
4. Create the prediction target.
5. Join related data without leaking future information.
6. Save a clean intermediate artifact.

## Project-Specific Walkthrough

Training starts with:

- `data/raw/matches.csv`
- `data/raw/fifa_rankings.csv`
- `data/raw/elo_ratings.csv`

Current local snapshot sizes from the read-only project analysis:

| File | Rows | Columns | Date coverage |
|---|---:|---:|---|
| `data/raw/matches.csv` | 49,430 | 9 | `1872-11-30` to `2026-06-27` |
| `data/raw/fifa_rankings.csv` | 13,130 | 8 | Years `1992` to `2024` |
| `data/raw/elo_ratings.csv` | 6,678 | 4 | Historical Elo rows; date strings need parsing/normalization when used |
| `data/processed/matches_clean.csv` | 49,406 | 15 | `1872-11-30` to `2026-06-23` |
| `data/features/feature_matrix.csv` | 15,647 | 37 | `2010-01-02` to `2026-06-23` |

The current model does not directly use raw `elo_ratings.csv` as a feature table. It validates it and uses it in diagnostics, but Elo features are computed from match history in `src/features/elo.py`.

Pipeline sequence:

```text
scripts/run_pipeline.py
  -> src.data.fetch.check_raw_files()
  -> src.data.validate.run_all_validations()
  -> src.data.clean.run_cleaning()
  -> data/processed/matches_clean.csv
```

## File-by-File Explanation

### `src/data/fetch.py`

Despite the name, this file does not download data. It checks raw files.

Important behavior:

- Defines required files.
- Checks file existence.
- Checks file size is not zero.
- Logs download hints if files are missing.
- Raises `FileNotFoundError` if any required file fails.

Interview answer:

> I used this as a guard step. It fails early with a clear message instead of letting pandas fail later with a vague file error.

### `src/data/validate.py`

This file validates raw structure.

Checks:

- Minimum row counts:
  - `matches.csv`: at least 40000 rows
  - `fifa_rankings.csv`: at least 10000 rows
  - `elo_ratings.csv`: at least 5000 rows
- Required columns.
- No nulls in key match identity fields.
- Famous teams are present in match data.
- `neutral` is boolean-like.
- FIFA rank values are positive.
- Elo ratings are non-negative.

Why this matters:

Validation protects the pipeline from truncated downloads, wrong datasets, or schema drift.

### `src/data/clean.py`

This file creates `data/processed/matches_clean.csv`.

Key parts:

```python
TEAM_MAPPING = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    ...
}
```

Different data sources use different names for the same country. The mapping creates canonical names.

```python
def clean_team_name(name):
    if not isinstance(name, str):
        return name
    name = name.replace("\xa0", " ").strip()
    return TEAM_MAPPING.get(name, name)
```

This handles non-breaking spaces and known aliases.

```python
matches["result"] = "D"
matches.loc[matches["home_score"] > matches["away_score"], "result"] = "H"
matches.loc[matches["away_score"] > matches["home_score"], "result"] = "A"
```

This creates the target label for classification.

```python
matches["is_competitive"] = ~matches["tournament"].str.contains(
    "friendly", case=False, na=False
)
```

This flags non-friendly matches as competitive.

### Ranking Merge

The FIFA ranking data has snapshots, not rankings for every match date. The project converts semester to dates:

- Semester 1 -> January 1
- Semester 2 -> July 1

Then it uses `pd.merge_asof` with `direction="backward"` so every match receives the latest ranking snapshot at or before the match date.

Why backward matters:

If a match happened in March 2022, using July 2022 rankings would leak future information. Backward merge avoids that.

### Recent Match Upsert Flow

File: `scripts/fetch_recent_matches.py`

This script updates `data/raw/matches.csv` with finished World Cup matches from the football-data API.

Older behavior was conceptually "append new finished rows." The current behavior is safer:

```text
load existing rows
  -> fetch finished World Cup matches
  -> normalize team names and dates
  -> find an existing exact or near-date match row
  -> if existing row has missing scores, fill the scores
  -> if row is already scored, skip it
  -> if no matching row exists, append it
  -> rewrite CSV only if something changed
  -> retrain only if the CSV changed
```

Important helper responsibilities:

| Helper | What it teaches |
|---|---|
| `load_existing_matches()` | Reads CSV rows as dictionaries so the script can preserve columns while deciding what changed. |
| `_has_missing_score()` | Detects rows where a fixture exists but final score is blank. |
| `_same_world_cup_match()` | Compares teams/tournament/date and allows a one-day tolerance to handle UTC/local date mismatch. |
| `_find_existing_match_index()` | Locates the row that should be updated instead of appended. |
| `upsert_matches()` | Applies update/append/skip decisions and returns whether retraining is needed. |

Interview answer:

> The import is an upsert instead of a blind append. That matters because future fixtures may already exist with blank scores. When the final result arrives, the script should complete the existing row, not duplicate the match. It also skips retraining if no data changed, which avoids unnecessary model artifact churn.

Latest observed refresh:

Commit `d6821e5` completed four June 23, 2026 World Cup rows that were previously blank:

| Date | Match | Final score |
|---|---|---|
| `2026-06-23` | Portugal vs Uzbekistan | `5-0` |
| `2026-06-23` | Colombia vs DR Congo | `1-0` |
| `2026-06-23` | England vs Ghana | `0-0` |
| `2026-06-23` | Panama vs Croatia | `0-1` |

Those four completed matches explain the processed-data and feature-matrix row-count increase from the previous curriculum snapshot.

## Data Files

| File | Role |
|---|---|
| `data/raw/matches.csv` | Historical matches and scores. |
| `data/raw/fifa_rankings.csv` | Ranking snapshots and points. |
| `data/raw/elo_ratings.csv` | Validated/diagnostic source, not direct feature source in current model. |
| `data/processed/matches_clean.csv` | Cleaned matches with target labels and rankings. |
| `data/features/feature_matrix.csv` | Model-ready features consumed by training and prediction runtime. Covered deeply in lesson 04. |

## Common Mistakes

- Assuming `fetch.py` downloads data. It only checks files.
- Using exact-date ranking joins. Most matches would miss rankings.
- Filtering dates too early. That would remove historical warm-up for Elo/form/goals.
- Claiming raw Elo data powers model features. Current features compute Elo from matches.
- Blindly appending imported matches. The current script upserts to avoid duplicate World Cup rows.
- Retraining on every scheduler tick. The current script only retrains when the raw match CSV changed.

## Interview Questions

| Question | Strong answer |
|---|---|
| Why normalize team names? | Model features require consistent team identities across match and ranking sources. |
| Why use `merge_asof`? | Rankings are temporal snapshots. The latest prior snapshot is the correct pre-match value. |
| What target does the model predict? | `result`, with classes `H`, `D`, `A`. |
| Why fill missing rankings with rank 211 and points 0? | It prevents NaNs for teams missing FIFA ranking coverage while representing them as low-ranked/default teams. |
| Why does `fetch_recent_matches.py` use an upsert? | Finished scores may arrive for fixtures that already exist as blank-score rows. Updating avoids duplicate matches and keeps retraining tied to real data changes. |

## Rebuild Exercise

Write a small script that:

1. Loads a CSV with `home_score` and `away_score`.
2. Creates `result`.
3. Creates `is_competitive`.
4. Saves a cleaned CSV.

Then add one mapping such as `USA -> United States`.

## Self-Check Quiz

1. What creates `matches_clean.csv`?
2. What does `direction="backward"` prevent?
3. Which column becomes the ML target?
4. What happens to matches with null scores?

Answers:

1. `src/data/clean.py`
2. Future ranking leakage.
3. `result`
4. They are dropped.

## External Links

- pandas `merge_asof`: https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
- pandas `read_csv`: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
- Kaggle Learn pandas: https://www.kaggle.com/learn/pandas

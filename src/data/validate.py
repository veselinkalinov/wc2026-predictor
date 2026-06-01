"""
validate.py

Responsibility: assert that raw data files meet structural expectations
before the cleaning pipeline runs.

Checks column presence, minimum row counts, and known value constraints.
Raises ValueError with a descriptive message on any failed assertion.
"""

import pandas as pd
from pathlib import Path
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum row counts — deliberately conservative to catch truncated downloads
MIN_ROWS = {
    "matches.csv":       40000,
    "fifa_rankings.csv": 10000,
    "elo_ratings.csv":    5000,
}

# Required columns per file
REQUIRED_COLUMNS = {
    "matches.csv": [
        "date", "home_team", "away_team",
        "home_score", "away_score", "tournament", "neutral"
    ],
    "fifa_rankings.csv": [
        "date", "rank", "team", "total.points"
    ],
    "elo_ratings.csv": [
        "date", "team", "rating"
    ],
}

# Known teams that must be present in matches.csv
KNOWN_TEAMS = [
    "Brazil", "Germany", "France", "Argentina",
    "Spain", "England", "Italy", "Netherlands"
]


def _load(filename: str) -> pd.DataFrame:
    raw_dir = Path(config["paths"]["raw_data"])
    return pd.read_csv(raw_dir / filename)


def validate_matches() -> None:
    logger.info("Validating matches.csv ...")
    df = _load("matches.csv")

    # Row count
    if len(df) < MIN_ROWS["matches.csv"]:
        raise ValueError(
            f"matches.csv has {len(df)} rows, expected at least {MIN_ROWS['matches.csv']}. "
            f"File may be truncated or incorrect."
        )

    # Required columns
    for col in REQUIRED_COLUMNS["matches.csv"]:
        if col not in df.columns:
            raise ValueError(
                f"matches.csv is missing required column: '{col}'")

    # No nulls in key identity columns
    for col in ["date", "home_team", "away_team", "tournament"]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(
                f"matches.csv has {null_count} null values in '{col}'. "
                f"This column must be fully populated."
            )

    # Known teams present
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    for team in KNOWN_TEAMS:
        if team not in all_teams:
            raise ValueError(
                f"Expected team '{team}' not found in matches.csv. "
                f"Team name normalisation may be required."
            )

    # Neutral column is boolean-like
    neutral_values = set(df["neutral"].unique())
    if not neutral_values.issubset({True, False}):
        raise ValueError(
            f"matches.csv 'neutral' column contains unexpected values: {neutral_values}"
        )

    logger.info(f"matches.csv passed validation. Shape: {df.shape}")


def validate_rankings() -> None:
    logger.info("Validating fifa_rankings.csv ...")
    df = _load("fifa_rankings.csv")

    # Row count
    if len(df) < MIN_ROWS["fifa_rankings.csv"]:
        raise ValueError(
            f"fifa_rankings.csv has {len(df)} rows, expected at least {MIN_ROWS['fifa_rankings.csv']}. "
            f"File may be truncated or incorrect."
        )

    # Required columns
    for col in REQUIRED_COLUMNS["fifa_rankings.csv"]:
        if col not in df.columns:
            raise ValueError(
                f"fifa_rankings.csv is missing required column: '{col}'")

    # Rank column must be positive integers
    if (df["rank"] < 1).any():
        raise ValueError("fifa_rankings.csv contains rank values less than 1.")

    logger.info(f"fifa_rankings.csv passed validation. Shape: {df.shape}")


def validate_elo() -> None:
    logger.info("Validating elo_ratings.csv ...")
    df = _load("elo_ratings.csv")

    # Row count
    if len(df) < MIN_ROWS["elo_ratings.csv"]:
        raise ValueError(
            f"elo_ratings.csv has {len(df)} rows, expected at least {MIN_ROWS['elo_ratings.csv']}. "
            f"File may be truncated or incorrect."
        )

    # Required columns
    for col in REQUIRED_COLUMNS["elo_ratings.csv"]:
        if col not in df.columns:
            raise ValueError(
                f"elo_ratings.csv is missing required column: '{col}'")

    # Ratings must be positive
    valid_ratings = df["rating"].dropna()
    if (valid_ratings < 0).any():
        raise ValueError("elo_ratings.csv contains negative rating values.")

    logger.info(f"elo_ratings.csv passed validation. Shape: {df.shape}")


def run_all_validations() -> None:
    """
    Run all validation checks. Raises on the first failure encountered.
    """
    validate_matches()
    validate_rankings()
    validate_elo()
    logger.info("All validations passed.")


if __name__ == "__main__":
    run_all_validations()

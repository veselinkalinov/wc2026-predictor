"""
elo.py

Responsibility: Calculate chronological Elo ratings from scratch
for all international teams based on match history.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score probability for team A against team B.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process matches chronologically to calculate running Elo ratings.

    Returns a DataFrame with matches and new columns:
      - home_elo: Elo of home team before match
      - away_elo: Elo of away team before match
      - elo_diff: home_elo - away_elo
    """
    logger.info("Computing Elo ratings from scratch...")

    # Read parameters from config
    k_factor = config["features"]["elo_k_factor"]
    initial_elo = config["features"]["elo_initial"]

    # Sort matches chronologically to process rating updates in order
    df = matches_df.sort_values("date").copy()

    # Track current Elo rating for every team
    current_elos = {}

    home_elos = []
    away_elos = []

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        result = row["result"]

        # Initialise teams if they haven't appeared before
        if home_team not in current_elos:
            current_elos[home_team] = initial_elo
        if away_team not in current_elos:
            current_elos[away_team] = initial_elo

        # Get ratings BEFORE the match
        h_elo = current_elos[home_team]
        a_elo = current_elos[away_team]

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Convert result (H/D/A) to actual points for Home Team
        # Win = 1.0, Draw = 0.5, Loss = 0.0
        if result == "H":
            actual_home = 1.0
        elif result == "D":
            actual_home = 0.5
        else:
            actual_home = 0.0

        actual_away = 1.0 - actual_home

        # Calculate expected scores
        expected_home = calculate_expected_score(h_elo, a_elo)
        expected_away = 1.0 - expected_home

        # Update running ratings
        current_elos[home_team] = h_elo + \
            k_factor * (actual_home - expected_home)
        current_elos[away_team] = a_elo + \
            k_factor * (actual_away - expected_away)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    logger.info(
        f"Finished Elo computation. Unique teams tracked: {len(current_elos)}")
    return df

"""
elo.py
Responsibility: Calculate chronological Elo ratings from scratch
for all international teams based on match history.

Upgrades:
1. Goal-margin logarithmic weighting.
2. Home-field advantage adjustment.
3. Dynamic K-factors based on tournament tier.
"""

import math
from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score probability for team A against team B.
    Using the standard logistic Elo curve: 1 / (1 + 10^((R_b - R_a) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def get_k_factor(tournament: str) -> int:
    """
    Return K-factor based on tournament importance tier.
    Matches are tier-weighted to prevent friendlies from swings
    while ensuring World Cups carry high significance.
    """
    t = str(tournament).lower()
    if t == "fifa world cup":
        return 60
    elif any(comp in t for comp in ["uefa euro", "copa américa", "african cup of nations",
                                    "afc asian cup", "concacaf gold cup",
                                    "confederations cup"]):
        return 50
    elif "qualification" in t or "nations league" in t:
        return 40
    else:
        return 20  # Friendlies and minor tournaments


def goal_margin_multiplier(goal_diff: int) -> float:
    """
    Logarithmic multiplier based on goal margin.
    A 5-0 win should reward more points than a 1-0 win, but with diminishing returns.
    """
    abs_diff = abs(goal_diff)
    if abs_diff <= 1:
        return 1.0
    elif abs_diff == 2:
        return 1.5
    else:
        # Diminishing returns: (11 + diff) / 8
        return (11.0 + abs_diff) / 8.0


def compute_elo_ratings(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Process matches chronologically to calculate running Elo ratings.

    Returns:
      - df: DataFrame with new columns: home_elo, away_elo, elo_diff
      - current_elos: A dict of final ratings {team_name: final_elo}
    """
    logger.info("Computing advanced Elo ratings from scratch...")

    initial_elo = config["features"]["elo_initial"]
    hfa_bonus = config["features"].get("elo_home_advantage", 100)

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
        home_score = row["home_score"]
        away_score = row["away_score"]
        neutral = row["neutral"]
        tournament = row["tournament"]

        # Initialise teams if they haven't appeared before
        if home_team not in current_elos:
            current_elos[home_team] = float(initial_elo)
        if away_team not in current_elos:
            current_elos[away_team] = float(initial_elo)

        # Get ratings BEFORE the match
        h_elo = current_elos[home_team]
        a_elo = current_elos[away_team]

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # 1. Apply Home-Field Advantage (HFA) adjustment for expected score calculation
        # If neutral is False, home team gets virtual rating bump
        h_elo_adjusted = h_elo
        if not neutral:
            h_elo_adjusted += hfa_bonus

        # Calculate expected scores based on adjusted ratings
        expected_home = calculate_expected_score(h_elo_adjusted, a_elo)
        expected_away = 1.0 - expected_home

        # Convert result (H/D/A) to actual points for Home Team
        if result == "H":
            actual_home = 1.0
        elif result == "D":
            actual_home = 0.5
        else:
            actual_home = 0.0
        actual_away = 1.0 - actual_home

        # 2. Determine K-factor dynamically by tournament tier
        k_factor = get_k_factor(tournament)

        # 3. Determine Goal Margin Multiplier
        goal_diff = int(home_score - away_score)
        multiplier = goal_margin_multiplier(goal_diff)

        # 4. Update running ratings
        rating_change_home = k_factor * \
            multiplier * (actual_home - expected_home)
        rating_change_away = k_factor * \
            multiplier * (actual_away - expected_away)

        current_elos[home_team] = h_elo + rating_change_home
        current_elos[away_team] = a_elo + rating_change_away

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    logger.info(
        f"Finished Elo computation. Unique teams tracked: {len(current_elos)}"
    )
    return df, current_elos

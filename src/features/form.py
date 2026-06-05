"""
form.py
Responsibility: Compute rolling team form features based on points won
in recent matches, enhanced with:
1. Exponentially Weighted Moving Average (EWMA) decay.
2. Opponent-strength Elo adjustments.
"""
from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_ewma_form(history: list, alpha: float) -> float:
    """
    Compute exponentially weighted moving average of form points.
    history: list of opponent-adjusted points earned.
    alpha: exponential decay factor.
    Returns: A form score scaled between 0.0 and 1.0.
    """
    if not history:
        return 0.5  # Cold-start default

    # Weights: (1-alpha)^age, where age is 0 for the most recent match
    n = len(history)
    weights = [(1.0 - alpha) ** i for i in range(n-1, -1, -1)]

    total_weight = sum(weights)

    # Standardize points by dividing by 3 (so a standard win = 1.0)
    weighted_sum = sum(w * (pts / 3.0) for w, pts in zip(weights, history))

    form_score = weighted_sum / total_weight

    # Clamp to [0, 1] range to avoid floating-point drift out of bounds
    return min(max(form_score, 0.0), 1.0)


def compute_form_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process matches chronologically to calculate rolling form.
    Returns a DataFrame with matches and new columns:
      - home_form: Form score of home team before match [0.0 - 1.0]
      - away_form: Form score of away team before match [0.0 - 1.0]
      - form_diff: home_form - away_form
    """

    logger.info("Computing advanced opponent-adjusted team form features...")

    # Read parameters from config
    window = config["features"]["form_window"]
    alpha = config["features"].get("form_alpha", 0.3)

    # Sort matches chronologically to process form updates in order
    df = matches_df.sort_values("date").copy()

    # Track list of adjusted match points earned for each team: {team: [adj_pts1, adj_pts2, ...]}
    team_history = {}
    home_forms = []
    away_forms = []

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        result = row["result"]
        home_elo = row["home_elo"]
        away_elo = row["away_elo"]

        # Initialise teams if they haven't appeared before
        if home_team not in team_history:
            team_history[home_team] = []
        if away_team not in team_history:
            team_history[away_team] = []

        # 1. Calculate form BEFORE the match (historical form) using EWMA on recent window
        h_history = team_history[home_team][-window:]
        a_history = team_history[away_team][-window:]

        home_forms.append(compute_ewma_form(h_history, alpha))
        away_forms.append(compute_ewma_form(a_history, alpha))

        # 2. Compute points earned in the current match
        if result == "H":
            home_pts, away_pts = 3.0, 0.0
        elif result == "D":
            home_pts, away_pts = 1.0, 1.0
        else:
            home_pts, away_pts = 0.0, 3.0

        # 3. Apply Opponent-Strength Elo adjustment
        # opponent_factor = 1.0 + (opponent_elo - 1500) / 1000
        # Clamped at min 0.5 to prevent negative points or division issues against very low teams
        home_opp_factor = max(1.0 + (away_elo - 1500.0) / 1000.0, 0.5)
        away_opp_factor = max(1.0 + (home_elo - 1500.0) / 1000.0, 0.5)

        home_pts_adj = home_pts * home_opp_factor
        away_pts_adj = away_pts * away_opp_factor

        # 4. Append current adjusted points to history for future matches
        team_history[home_team].append(home_pts_adj)
        team_history[away_team].append(away_pts_adj)

    df["home_form"] = home_forms
    df["away_form"] = away_forms
    df["form_diff"] = df["home_form"] - df["away_form"]

    logger.info("Finished team form computation.")
    return df

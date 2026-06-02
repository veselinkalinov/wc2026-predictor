"""
form.py

Responsibility: Compute rolling team form features based on points won
in recent matches.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_form_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process matches chronologically to calculate rolling form.

    Returns a DataFrame with matches and new columns:
      - home_form: Form score of home team before match [0.0 - 1.0]
      - away_form: Form score of away team before match [0.0 - 1.0]
      - form_diff: home_form - away_form
    """
    logger.info("Computing team form features...")

    # Read parameters from config
    window = config["features"]["form_window"]

    # Sort matches chronologically to process form updates in order
    df = matches_df.sort_values("date").copy()

    # Track list of match points earned for each team: {team_name: [points1, points2, ...]}
    team_history = {}

    home_forms = []
    away_forms = []

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        result = row["result"]

        # Initialise teams if they haven't appeared before
        if home_team not in team_history:
            team_history[home_team] = []
        if away_team not in team_history:
            team_history[away_team] = []

        # 1. Calculate form BEFORE the match (historical form)
        for team, form_list in [(home_team, home_forms), (away_team, away_forms)]:
            history = team_history[team]
            if len(history) == 0:
                # Default to neutral form (0.5) for the team's first match
                form_list.append(0.5)
            else:
                # Take last N matches or all matches if less than N
                recent_history = history[-window:]
                actual_points = sum(recent_history)
                max_possible = 3 * len(recent_history)
                form_score = actual_points / max_possible
                form_list.append(form_score)

        # 2. Compute points earned in the current match
        if result == "H":
            home_pts, away_pts = 3, 0
        elif result == "D":
            home_pts, away_pts = 1, 1
        else:
            home_pts, away_pts = 0, 3

        # 3. Append current points to history for future matches
        team_history[home_team].append(home_pts)
        team_history[away_team].append(away_pts)

    df["home_form"] = home_forms
    df["away_form"] = away_forms
    df["form_diff"] = df["home_form"] - df["away_form"]

    logger.info("Finished team form computation.")
    return df

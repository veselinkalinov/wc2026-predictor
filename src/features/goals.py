"""
goals.py
Responsibility: Compute rolling goal-based statistics (scored, conceded, differences)
to represent team offensive and defensive capabilities.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_goal_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process matches chronologically to calculate rolling goal statistics.

    Returns a DataFrame with matches and new columns:
      - home_goals_scored_avg
      - home_goals_conceded_avg
      - home_goal_diff_avg
      - away_goals_scored_avg
      - away_goals_conceded_avg
      - away_goal_diff_avg
    """

    logger.info("Compute rolling goal features...")

    # Read params from config
    window = config["features"]["goals_window"]

    # Sort matches chronologically to process goal stats in order
    df = matches_df.sort_values("date").copy()

    # Track goals scored and conceded history for each team: {team_name: [goals1, goals2, ...]}
    goals_scored_history = {}
    goals_conceded_history = {}
    home_scored_avgs = []
    home_conceded_avgs = []
    away_scored_avgs = []
    away_conceded_avgs = []

    # Default value for new teams
    DEFAULT_GOALS = 1.2

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]

        # Initialise history lists if they haven't appeared before
        for team in [home_team, away_team]:
            if team not in goals_scored_history:
                goals_scored_history[team] = []
            if team not in goals_conceded_history:
                goals_conceded_history[team] = []

                # 1. Calculate averages BEFORE the match (historical averages)
        for team, scored_list, conceded_list in [(home_team, home_scored_avgs, home_conceded_avgs), (away_team, away_scored_avgs, away_conceded_avgs)]:
            scored_hist = goals_scored_history[team]
            conceded_hist = goals_conceded_history[team]

            if len(scored_hist) == 0:
                # Cold-start defaults
                scored_list.append(DEFAULT_GOALS)
                conceded_list.append(DEFAULT_GOALS)
            else:
                # Take last N matches
                recent_scored = scored_hist[-window:]
                recent_conceded = conceded_hist[-window:]

                scored_list.append(sum(recent_scored) / len(recent_scored))
                conceded_list.append(
                    sum(recent_conceded) / len(recent_conceded))

        # 2. Append current match details to history for future matches
        # For Home Team: scored = home_score, conceded = away_score
        goals_scored_history[home_team].append(home_score)
        goals_conceded_history[home_team].append(away_score)

        # For Away Team: scored = away_score, conceded = home_score
        goals_scored_history[away_team].append(away_score)
        goals_conceded_history[away_team].append(home_score)

    # 3. Add features to DataFrame and calculate differences
    df["home_goals_scored_avg"] = home_scored_avgs
    df["home_goals_conceded_avg"] = home_conceded_avgs
    df["home_goal_diff_avg"] = df["home_goals_scored_avg"] - \
        df["home_goals_conceded_avg"]

    df["away_goals_scored_avg"] = away_scored_avgs
    df["away_goals_conceded_avg"] = away_conceded_avgs
    df["away_goal_diff_avg"] = df["away_goals_scored_avg"] - \
        df["away_goals_conceded_avg"]

    logger.info("Finished rolling goal features computation.")
    return df

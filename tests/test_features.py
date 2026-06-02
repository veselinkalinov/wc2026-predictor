"""
test_features.py
Unit tests for feature engineering logic in src/features/
"""
import pandas as pd
from src.features.elo import calculate_expected_score, compute_elo_ratings
from src.features.form import compute_form_features
from src.features.goals import compute_goal_features


def test_expected_score():
    # If ratings are equal, both should have a 50% chance of winning
    assert calculate_expected_score(1500, 1500) == 0.5
    # A stronger team should have a higher expected score
    assert calculate_expected_score(1600, 1400) > 0.5
    # A weaker team should have a lower expected score
    assert calculate_expected_score(1400, 1600) < 0.5


def test_compute_elo_ratings():
    # Setup test matches
    data = {
        "date": ["2024-06-01", "2024-06-02"],
        "home_team": ["Germany", "Brazil"],
        "away_team": ["Spain", "Argentina"],
        "home_score": [2, 1],
        "away_score": [0, 2],
        "result": ["H", "A"],  # Germany wins, Argentina wins
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    # Run Elo calculation
    result_df = compute_elo_ratings(df)
    # Germany and Spain start at 1500
    assert result_df.loc[0, "home_elo"] == 1500
    assert result_df.loc[0, "away_elo"] == 1500
    assert result_df.loc[0, "elo_diff"] == 0
    # Brazil starts at 1500, Argentina starts at 1500
    assert result_df.loc[1, "home_elo"] == 1500
    assert result_df.loc[1, "away_elo"] == 1500


def test_compute_form_features():
    # Setup test matches: team A plays 2 matches
    data = {
        "date": ["2024-06-01", "2024-06-02"],
        "home_team": ["Germany", "Spain"],
        "away_team": ["Spain", "Germany"],
        # Germany wins first (3 pts), Germany wins second as away (3 pts)
        "result": ["H", "A"],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    result_df = compute_form_features(df)
    # First match: both teams have no history -> neutral form (0.5)
    assert result_df.loc[0, "home_form"] == 0.5
    assert result_df.loc[0, "away_form"] == 0.5
    # Second match: Germany has 3 points from 1 match -> form is 3/3 = 1.0
    # Spain has 0 points from 1 match -> form is 0/3 = 0.0
    # Germany is the away team in match 2
    assert result_df.loc[1, "away_form"] == 1.0  # Germany
    assert result_df.loc[1, "home_form"] == 0.0  # Spain


def test_compute_goal_features():
    data = {
        "date": ["2024-06-01", "2024-06-02"],
        "home_team": ["Germany", "Spain"],
        "away_team": ["Spain", "Germany"],
        "home_score": [3, 0],
        "away_score": [1, 2],  # Germany 3-1 Spain, Spain 0-2 Germany
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    result_df = compute_goal_features(df)
    # First match: no history -> default values (1.2)
    assert result_df.loc[0, "home_goals_scored_avg"] == 1.2
    assert result_df.loc[0, "home_goals_conceded_avg"] == 1.2
    # Second match: Germany has scored 3, conceded 1. Spain has scored 1, conceded 3.
    # Germany is away, Spain is home.
    assert result_df.loc[1, "away_goals_scored_avg"] == 3.0  # Germany scored 3
    # Germany conceded 1
    assert result_df.loc[1, "away_goals_conceded_avg"] == 1.0
    assert result_df.loc[1, "home_goals_scored_avg"] == 1.0  # Spain scored 1
    # Spain conceded 3
    assert result_df.loc[1, "home_goals_conceded_avg"] == 3.0

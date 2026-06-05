"""
test_features.py
Unit tests for updated feature engineering logic in src/features/
"""
import pandas as pd
import numpy as np
import pytest
from src.features.elo import (
    calculate_expected_score,
    compute_elo_ratings,
    get_k_factor,
    goal_margin_multiplier
)
from src.features.form import compute_form_features, compute_ewma_form
from src.features.goals import compute_goal_features, compute_ewma_goals

# Set up some configuration mock constants so tests can run without reading config.yaml
# We can mock config if needed, but the modules import config directly.
# Since we already updated config.yaml, it will read the correct new parameters!


def test_expected_score():
    assert calculate_expected_score(1500, 1500) == 0.5
    assert calculate_expected_score(1600, 1400) > 0.5
    assert calculate_expected_score(1400, 1600) < 0.5


def test_get_k_factor():
    # World Cup Finals should return K=60
    assert get_k_factor("FIFA World Cup") == 60

    # Continental finals should return K=50
    assert get_k_factor("UEFA Euro") == 50
    assert get_k_factor("Copa América") == 50

    # Qualifiers & Nations League should return K=40
    assert get_k_factor("FIFA World Cup qualification") == 40
    assert get_k_factor("UEFA Nations League") == 40

    # Friendlies and other matches should return K=20
    assert get_k_factor("Friendly") == 20
    assert get_k_factor("King's Cup") == 20


def test_goal_margin_multiplier():
    # Margin of 0 or 1 should return 1.0
    assert goal_margin_multiplier(0) == 1.0
    assert goal_margin_multiplier(1) == 1.0
    assert goal_margin_multiplier(-1) == 1.0

    # Margin of 2 should return 1.5
    assert goal_margin_multiplier(2) == 1.5
    assert goal_margin_multiplier(-2) == 1.5

    # Margin of 3 or more: (11 + diff) / 8
    # Margin of 3 -> (11 + 3) / 8 = 1.75
    assert goal_margin_multiplier(3) == 1.75
    assert goal_margin_multiplier(-4) == 1.875


def test_compute_ewma_form():
    # If history is empty, return cold-start 0.5
    assert compute_ewma_form([], alpha=0.3) == 0.5

    # Standard win (3 points -> 3/3 = 1.0)
    assert compute_ewma_form([3.0], alpha=0.3) == 1.0

    # EWMA weights: most recent elements get higher weights
    # [loss (0), win (3)] -> Win is most recent
    # Win (1.0) should have weight 1.0, Loss (0.0) should have weight 0.7
    # Weighted avg = (1.0 * 1.0 + 0.7 * 0.0) / (1.0 + 0.7) = 1.0 / 1.7 = 0.588
    assert compute_ewma_form([0.0, 3.0], alpha=0.3) == pytest.approx(1.0 / 1.7)


def test_compute_ewma_goals():
    # Empty history returns default value
    assert compute_ewma_goals([], alpha=0.25, default=1.2) == 1.2

    # Single match goal score
    assert compute_ewma_goals([3.0], alpha=0.25, default=1.2) == 3.0

    # Recent elements carry more weight
    # [1.0 goal, 3.0 goals] -> 3.0 is most recent
    # weight of 3.0 = 1.0, weight of 1.0 = (1 - 0.25) = 0.75
    # Weighted avg = (1.0 * 3.0 + 0.75 * 1.0) / (1.0 + 0.75) = 3.75 / 1.75 = 2.1428
    assert compute_ewma_goals([1.0, 3.0], alpha=0.25,
                              default=1.2) == pytest.approx(3.75 / 1.75)


def test_compute_elo_ratings():
    data = {
        "date": ["2024-06-01", "2024-06-02"],
        "home_team": ["Germany", "Brazil"],
        "away_team": ["Spain", "Argentina"],
        "home_score": [2, 1],
        "away_score": [0, 2],
        "result": ["H", "A"],
        "neutral": [False, True],
        "tournament": ["Friendly", "FIFA World Cup"]
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    result_df, elos_dict = compute_elo_ratings(df)

    assert "home_elo" in result_df.columns
    assert "away_elo" in result_df.columns
    assert "elo_diff" in result_df.columns

    # First match: both start at initial (1500)
    assert result_df.loc[0, "home_elo"] == 1500.0
    assert result_df.loc[0, "away_elo"] == 1500.0

    # Final Elo ratings should be returned in dict
    assert "Germany" in elos_dict
    assert "Spain" in elos_dict

    # Germany won a home friendly match (neutral=False) -> should increase Elo
    assert elos_dict["Germany"] > 1500.0
    assert elos_dict["Spain"] < 1500.0


def test_compute_form_features():
    data = {
        "date": ["2024-06-01", "2024-06-02"],
        "home_team": ["Germany", "Spain"],
        "away_team": ["Spain", "Germany"],
        "result": ["H", "A"],
        "home_elo": [1500.0, 1490.0],
        "away_elo": [1500.0, 1510.0]
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    result_df = compute_form_features(df)

    assert "home_form" in result_df.columns
    assert "away_form" in result_df.columns
    assert "form_diff" in result_df.columns

    # First match: no history, starts at cold-start 0.5
    assert result_df.loc[0, "home_form"] == 0.5
    assert result_df.loc[0, "away_form"] == 0.5


def test_compute_goal_features():
    data = {
        "date": ["2024-06-01", "2024-06-02"],
        "home_team": ["Germany", "Spain"],
        "away_team": ["Spain", "Germany"],
        "home_score": [3, 0],
        "away_score": [1, 2],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    result_df = compute_goal_features(df)

    assert "home_goals_scored_avg" in result_df.columns
    assert "home_goals_conceded_avg" in result_df.columns
    assert "home_goal_diff_avg" in result_df.columns

    # First match: cold-start defaults 1.2
    assert result_df.loc[0, "home_goals_scored_avg"] == 1.2
    assert result_df.loc[0, "home_goals_conceded_avg"] == 1.2

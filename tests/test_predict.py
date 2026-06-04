"""
test_predict.py

Unit tests for the prediction pipeline in src/models/predict.py.
"""

import numpy as np
from src.models.predict import MatchPredictor


def test_predictor_initialisation():
    predictor = MatchPredictor()

    # Assert model artifacts are loaded
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert len(predictor.features) == 20
    assert list(predictor.classes) == ["H", "D", "A"]

    # Assert team states were parsed
    assert len(predictor.team_states) > 0
    assert "Brazil" in predictor.team_states


def test_prediction_output_and_probabilities():
    predictor = MatchPredictor()
    result = predictor.predict_match("Brazil", "Argentina", is_neutral=1)

    # Verify return dictionary shape and keys
    assert "home_team" in result
    assert "away_team" in result
    assert "probabilities" in result
    assert "prediction" in result

    probs = result["probabilities"]
    assert "home_win" in probs
    assert "draw" in probs
    assert "away_win" in probs

    # Sum of probabilities must equal 1.0 (with a margin of floating point error)
    prob_sum = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert np.isclose(prob_sum, 1.0, atol=1e-3)

    # Target label must be one of H, D, A
    assert result["prediction"] in ["H", "D", "A"]


def test_neutral_symmetric_prediction():
    predictor = MatchPredictor()

    # Predict forward matchup
    forward = predictor.predict_match("Germany", "France", is_neutral=1)

    # Predict reverse matchup
    reverse = predictor.predict_match("France", "Germany", is_neutral=1)

    # For neutral matches, home_win in forward must equal away_win in reverse, and vice versa.
    # Draws must be identical.
    assert np.isclose(forward["probabilities"]["home_win"],
                      reverse["probabilities"]["away_win"])
    assert np.isclose(forward["probabilities"]["away_win"],
                      reverse["probabilities"]["home_win"])
    assert np.isclose(forward["probabilities"]["draw"],
                      reverse["probabilities"]["draw"])


def test_non_neutral_asymmetric_prediction():
    predictor = MatchPredictor()

    # Predict non-neutral (Germany is home team, playing in Germany)
    forward = predictor.predict_match("Germany", "France", is_neutral=0)

    # Predict non-neutral (France is home team, playing in France)
    reverse = predictor.predict_match("France", "Germany", is_neutral=0)

    # For non-neutral matches, the home field advantage makes the prediction asymmetric.
    # Therefore, swapping team order should not produce exact mirror probabilities.
    assert not np.isclose(
        forward["probabilities"]["home_win"], reverse["probabilities"]["away_win"])

"""
routes.py

Responsibility: Define Flask API routes for health checks, listing teams,
match prediction, and tournament simulation.
"""

from flask import Blueprint, request, jsonify
from src.models.predict import MatchPredictor
from src.models.simulate import TournamentSimulator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create the API blueprint
api_bp = Blueprint("api", __name__)

# Global instances loaded once on startup for speed
logger.info("Loading ML predictor and simulator models into API memory...")
predictor = MatchPredictor()
simulator = TournamentSimulator()
logger.info("API memory loaded.")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "healthy", "service": "wc2026-predictor-api"})


@api_bp.route("/teams", methods=["GET"])
def get_teams():
    """
    Get a list of all unique team names sorted alphabetically.
    """
    teams = sorted(list(predictor.team_states.keys()))
    return jsonify({"teams": teams, "count": len(teams)})


@api_bp.route("/predict", methods=["POST"])
def predict_matchup():
    """
    Predict outcome probabilities for a match.
    JSON input:
      - home_team (str, required)
      - away_team (str, required)
      - is_neutral (int, optional, default: 1)
      - is_competitive (int, optional, default: 1)
    """
    data = request.get_json() or {}

    home_team = data.get("home_team")
    away_team = data.get("away_team")
    is_neutral = data.get("is_neutral", 1)
    is_competitive = data.get("is_competitive", 1)

    # Validation
    if not home_team or not away_team:
        return jsonify({"error": "Missing required fields: 'home_team' and 'away_team'"}), 400

    # Clean strings in case of slight space variations
    home_team = home_team.strip()
    away_team = away_team.strip()

    try:
        prediction = predictor.predict_match(
            home_team=home_team,
            away_team=away_team,
            is_neutral=is_neutral,
            is_competitive=is_competitive
        )
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"API prediction failed: {str(e)}")
        return jsonify({"error": f"Failed to predict match: {str(e)}"}), 500


@api_bp.route("/simulate", methods=["POST"])
def simulate_tournament():
    """
    Run Monte Carlo simulations of the World Cup tournament.
    JSON input:
      - n_sims (int, optional, default: 100)
    """
    data = request.get_json() or {}
    # Defaulting to 100 in API for quick responses
    n_sims = data.get("n_sims", 100)

    if not isinstance(n_sims, int) or n_sims <= 0:
        return jsonify({"error": "Parameter 'n_sims' must be a positive integer"}), 400

    # Restrict simulation runs in API to prevent server timeout
    if n_sims > 5000:
        return jsonify({"error": "Parameter 'n_sims' cannot exceed 5000 in a single API call"}), 400

    try:
        champs = simulator.run_monte_carlo(n_sims=n_sims)
        # Convert pandas Series output to JSON dictionary mapping
        results_dict = champs.to_dict()
        return jsonify({
            "simulations_run": n_sims,
            "win_probabilities": results_dict
        })
    except Exception as e:
        logger.error(f"API simulation failed: {str(e)}")
        return jsonify({"error": f"Failed to run simulation: {str(e)}"}), 500

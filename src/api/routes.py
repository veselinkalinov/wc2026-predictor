"""
routes.py

Responsibility: Define Flask API routes for health checks, listing teams,
match prediction, and tournament simulation, as well as page rendering.
"""

import json
from pathlib import Path
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, send_from_directory
from src.models.predict import MatchPredictor
from src.models.simulate import TournamentSimulator
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create the blueprints
api_bp = Blueprint("api", __name__)
pages_bp = Blueprint("pages", __name__, template_folder="templates")

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


@api_bp.route("/team-details/<team_name>", methods=["GET"])
def get_team_details(team_name):
    """
    Get ELO, FIFA rank, form, and average goals for a specific team.
    """
    team_name = team_name.strip()
    state = predictor.get_team_state(team_name)
    if not state or (state["elo"] == 1500.0 and state["rank"] == 211.0):
        if team_name not in predictor.team_states:
            return jsonify({"error": f"Team '{team_name}' not found"}), 404
            
    goals_scored = state.get("goals_scored_avg", 1.2)
    goals_conceded = state.get("goals_conceded_avg", 1.2)
    elo = state.get("elo", 1500.0)
    form = state.get("form", 0.5)
    
    # Dynamic calculations for radar charts
    attack = min(99.0, max(30.0, float(goals_scored * 20.0 + (211 - state["rank"]) * 0.15 + (elo - 1000) * 0.02)))
    defense = min(99.0, max(30.0, float(100.0 - goals_conceded * 25.0 + (211 - state["rank"]) * 0.10 + (elo - 1000) * 0.01)))
    tactics = min(99.0, max(30.0, float(elo * 0.045)))
    fitness = min(99.0, max(30.0, float(form * 100.0)))
    
    # Calculate win, draw, loss ratios from feature matrix for additional stats
    df = predictor.feature_matrix
    team_df = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)]
    total_matches = len(team_df)
    
    wins = 0
    if total_matches > 0:
        for _, row in team_df.iterrows():
            is_home = row["home_team"] == team_name
            if row["result"] == "H" and is_home:
                wins += 1
            elif row["result"] == "A" and not is_home:
                wins += 1
        win_ratio = round((wins / total_matches) * 100)
    else:
        win_ratio = 50

    return jsonify({
        "team": team_name,
        "state": state,
        "win_ratio": win_ratio,
        "radar": {
            "attack": round(attack, 1),
            "defense": round(defense, 1),
            "tactics": round(tactics, 1),
            "fitness": round(fitness, 1)
        }
    })


@api_bp.route("/team-matches/<team_name>", methods=["GET"])
def get_team_matches(team_name):
    """
    Get recent matches for a specific team.
    """
    team_name = team_name.strip()
    df = predictor.feature_matrix
    team_df = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)]
    
    recent = team_df.sort_values("date", ascending=False).head(5)
    
    matches_list = []
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team_name
        result = row["result"]
        
        if result == "H":
            outcome = "W" if is_home else "L"
        elif result == "A":
            outcome = "L" if is_home else "W"
        else:
            outcome = "D"
            
        # Try to parse scores, fallback if missing
        h_score = int(row["home_score"]) if "home_score" in row and not pd.isna(row["home_score"]) else 0
        a_score = int(row["away_score"]) if "away_score" in row and not pd.isna(row["away_score"]) else 0
            
        matches_list.append({
            "date": pd.to_datetime(row["date"]).strftime("%b %d, %Y"),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "opponent": row["away_team"] if is_home else row["home_team"],
            "tournament": row["tournament"] if "tournament" in row else "International Match",
            "is_home": is_home,
            "outcome": outcome,
            "score_text": f"{h_score} - {a_score}"
        })
        
    return jsonify({"matches": matches_list})


@api_bp.route("/visualisations/<filename>", methods=["GET"])
def serve_visualisation(filename):
    """
    Serve generated plots directly to the frontend.
    """
    vis_dir = Path(config["paths"]["visualisations"])
    return send_from_directory(vis_dir, filename)


@api_bp.route("/model-meta", methods=["GET"])
def get_model_meta():
    """
    Get training metadata and comparison metrics.
    """
    meta_path = Path(config["paths"]["models"]) / "meta.json"
    if not meta_path.exists():
        return jsonify({"error": "Model metadata not found"}), 404
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return jsonify(meta)


# Page Rendering Handlers
@pages_bp.route("/")
def render_home_page():
    return render_template("home.html")


@pages_bp.route("/predict")
def render_predict_page():
    return render_template("predict.html")


@pages_bp.route("/analytics")
def render_analytics_page():
    return render_template("analytics.html")


@pages_bp.route("/insights")
def render_insights_page():
    return render_template("insights.html")


@pages_bp.route("/about")
def render_about_page():
    return render_template("about.html")


@pages_bp.route("/simulate")
def render_simulate_page():
    return render_template("simulate.html")

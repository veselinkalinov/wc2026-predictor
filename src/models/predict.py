"""
predict.py

Responsibility: Load the trained model artifacts, construct feature vectors for
live matchups, and perform symmetric predictions for neutral venues. Includes
auto-reload on disk file modification for zero-restart model updates.
"""

from pathlib import Path
import time
import json
import pandas as pd
import numpy as np
import joblib
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Geographic continent mapping for top national teams
CONTINENT_MAP = {
    # Europe (UEFA)
    "Germany": "Europe", "France": "Europe", "England": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Netherlands": "Europe", "Portugal": "Europe", "Belgium": "Europe", "Croatia": "Europe", "Denmark": "Europe",
    "Sweden": "Europe", "Switzerland": "Europe", "Poland": "Europe", "Austria": "Europe", "Ukraine": "Europe",
    "Turkey": "Europe", "Russia": "Europe", "Wales": "Europe", "Scotland": "Europe", "Republic of Ireland": "Europe",
    # South America (CONMEBOL)
    "Brazil": "South America", "Argentina": "South America", "Uruguay": "South America", "Colombia": "South America",
    "Chile": "South America", "Peru": "South America", "Ecuador": "South America", "Paraguay": "South America",
    "Venezuela": "South America", "Bolivia": "South America",
    # North/Central America (CONCACAF)
    "United States": "North America", "Mexico": "North America", "Canada": "North America", "Costa Rica": "North America",
    "Jamaica": "North America", "Honduras": "North America", "Panama": "North America", "El Salvador": "North America",
    # Africa (CAF)
    "Senegal": "Africa", "Morocco": "Africa", "Algeria": "Africa", "Nigeria": "Africa", "Egypt": "Africa",
    "Cameroon": "Africa", "Ghana": "Africa", "Ivory Coast": "Africa", "Tunisia": "Africa", "Mali": "Africa",
    # Asia (AFC)
    "Japan": "Asia", "South Korea": "Asia", "Iran": "Asia", "Australia": "Asia", "Saudi Arabia": "Asia",
    "Qatar": "Asia", "Iraq": "Asia", "United Arab Emirates": "Asia", "China PR": "Asia",
    # Oceania (OFC)
    "New Zealand": "Oceania"
}


class MatchPredictor:
    def __init__(self, model_filename="best_model.pkl"):
        models_dir = Path(config["paths"]["models"])
        features_dir = Path(config["paths"]["features"])

        self.model_filename = model_filename
        self.model_path = models_dir / model_filename
        if not self.model_path.exists():
            logger.warning(f"Requested model file {model_filename} not found. Falling back to best_model.pkl.")
            self.model_path = models_dir / "best_model.pkl"
            self.model_filename = "best_model.pkl"

        self.scaler_path = models_dir / "scaler.pkl"
        self.meta_path = models_dir / "meta.json"
        self.feature_matrix_path = features_dir / "feature_matrix.csv"

        # Load artifacts
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.features = meta["features"]
        self.classes = meta["classes"]
        
        # Load draw threshold for this specific model if it exists in the meta comparison
        model_display_name = {
            "logistic_regression.pkl": "Logistic Regression",
            "random_forest.pkl": "Random Forest",
            "histgradientboosting.pkl": "HistGradientBoosting",
            "lightgbm.pkl": "LightGBM",
            "catboost.pkl": "CatBoost",
            "xgboost.pkl": "XGBoost",
            "poisson_goal_model.pkl": "Poisson Goal Model",
            "stacking_ensemble.pkl": "Stacking Ensemble"
        }.get(self.model_filename)
        
        if model_display_name and "comparison" in meta and model_display_name in meta["comparison"]:
            self.draw_threshold = meta["comparison"][model_display_name].get("draw_threshold", 1.0)
        else:
            self.draw_threshold = meta.get("draw_threshold", 1.0)

        # Load the latest matches to get recent stats for teams
        self.feature_matrix = pd.read_csv(self.feature_matrix_path)
        self.feature_matrix["date"] = pd.to_datetime(
            self.feature_matrix["date"])

        # Build a lookup dictionary of the latest state for each team
        self.team_states = self._build_latest_team_states()

        # Track file modification time for auto-reloads
        self.last_loaded_time = self.model_path.stat(
            ).st_mtime if self.model_path.exists() else 0
        self.last_check_time = 0.0
        self.prediction_cache = {}

    def clear_prediction_cache(self) -> None:
        """
        Clear the prediction cache. Useful before starting a fresh Monte Carlo run.
        """
        self.prediction_cache = {}

    def _check_and_reload(self) -> bool:
        """
        Check if the model file on disk has been updated, and reload if necessary.
        """
        if getattr(self, "disable_reload", False):
            return False
        current_time = time.time()
        # Throttle file system checks to once every 10 seconds to optimize loop speed
        if current_time - self.last_check_time < 10.0:
            return False
        self.last_check_time = current_time

        if not self.model_path.exists():
            return False
        mtime = self.model_path.stat().st_mtime
        if mtime > self.last_loaded_time:
            logger.info(
                f"Model file update detected on disk for {self.model_filename}. Reloading model artifacts and team states...")
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                self.features = meta["features"]
                self.classes = meta["classes"]
                
                model_display_name = {
                    "logistic_regression.pkl": "Logistic Regression",
                    "random_forest.pkl": "Random Forest",
                    "histgradientboosting.pkl": "HistGradientBoosting",
                    "lightgbm.pkl": "LightGBM",
                    "catboost.pkl": "CatBoost",
                    "xgboost.pkl": "XGBoost",
                    "poisson_goal_model.pkl": "Poisson Goal Model",
                    "stacking_ensemble.pkl": "Stacking Ensemble"
                }.get(self.model_filename)
                
                if model_display_name and "comparison" in meta and model_display_name in meta["comparison"]:
                    self.draw_threshold = meta["comparison"][model_display_name].get("draw_threshold", 1.0)
                else:
                    self.draw_threshold = meta.get("draw_threshold", 1.0)

                # Reload feature matrix and team states
                self.feature_matrix = pd.read_csv(self.feature_matrix_path)
                self.feature_matrix["date"] = pd.to_datetime(
                    self.feature_matrix["date"])
                self.team_states = self._build_latest_team_states()

                self.last_loaded_time = mtime
                logger.info("Reload completed successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to reload model artifacts: {str(e)}")
        return False

    def _build_team_state(self, team_name: str, team_matches: pd.DataFrame) -> dict:
        """
        Extract the latest metrics for a single team.
        """
        # Sort matches to get the absolute latest snapshot
        latest_match = team_matches.sort_values("date").iloc[-1]

        # Determine if the team was home or away in their latest match
        is_home = latest_match["home_team"] == team_name

        if is_home:
            return {
                "elo": float(latest_match["home_elo"]),
                "form": float(latest_match["home_form"]),
                "goals_scored_avg": float(latest_match["home_goals_scored_avg"]),
                "goals_conceded_avg": float(latest_match["home_goals_conceded_avg"]),
                "goal_diff_avg": float(latest_match["home_goal_diff_avg"]),
                "rank": float(latest_match["home_rank"]),
                "rank_points": float(latest_match["home_rank_points"]),
            }
        else:
            return {
                "elo": float(latest_match["away_elo"]),
                "form": float(latest_match["away_form"]),
                "goals_scored_avg": float(latest_match["away_goals_scored_avg"]),
                "goals_conceded_avg": float(latest_match["away_goals_conceded_avg"]),
                "goal_diff_avg": float(latest_match["away_goal_diff_avg"]),
                "rank": float(latest_match["away_rank"]),
                "rank_points": float(latest_match["away_rank_points"]),
            }

    def _build_latest_team_states(self) -> dict:
        """
        Examine the entire feature matrix and capture the latest state of each team.
        """
        logger.info("Building latest team states dictionary...")
        states = {}
        all_teams = set(self.feature_matrix["home_team"].unique()) | set(
            self.feature_matrix["away_team"].unique())

        for team in all_teams:
            # Get all matches involving this team
            team_df = self.feature_matrix[(self.feature_matrix["home_team"] == team) | (
                self.feature_matrix["away_team"] == team)]
            if len(team_df) > 0:
                states[team] = self._build_team_state(team, team_df)

        logger.info(f"Loaded states for {len(states)} teams.")
        return states

    def update_team_state(self, team: str, state: dict) -> None:
        """
        Updates the running state of a team.
        """
        self.team_states[team] = state

    def get_team_state(self, team: str) -> dict:
        """
        Return the team state, falling back to a default state if not found.
        """
        if team in self.team_states:
            return self.team_states[team]

        return {
            "elo": 1500.0,
            "form": 0.5,
            "goals_scored_avg": 1.2,
            "goals_conceded_avg": 1.2,
            "goal_diff_avg": 0.0,
            "rank": 211.0,
            "rank_points": 0.0,
        }

    def _construct_features_numpy(
        self, home_team: str, away_team: str, is_neutral: int, is_competitive: int
    ) -> np.ndarray:
        """
        Build the feature row in the exact order the model expects using numpy (optimized).
        """
        h_state = self.get_team_state(home_team)
        a_state = self.get_team_state(away_team)

        # Compute travel continent features
        h_cont = CONTINENT_MAP.get(home_team)
        a_cont = CONTINENT_MAP.get(away_team)
        if is_neutral == 0:
            h_is_home_cont = 1
            a_is_home_cont = 1 if h_cont == a_cont else 0
        else:
            host_continent = "North America"
            h_is_home_cont = 1 if h_cont == host_continent else 0
            a_is_home_cont = 1 if a_cont == host_continent else 0

        row_map = {
            "home_elo": h_state["elo"],
            "away_elo": a_state["elo"],
            "elo_diff": h_state["elo"] - a_state["elo"],
            "home_form": h_state["form"],
            "away_form": a_state["form"],
            "form_diff": h_state["form"] - a_state["form"],
            "home_goals_scored_avg": h_state["goals_scored_avg"],
            "home_goals_conceded_avg": h_state["goals_conceded_avg"],
            "home_goal_diff_avg": h_state["goal_diff_avg"],
            "away_goals_scored_avg": a_state["goals_scored_avg"],
            "away_goals_conceded_avg": a_state["goals_conceded_avg"],
            "away_goal_diff_avg": a_state["goal_diff_avg"],
            "home_rank": h_state["rank"],
            "away_rank": a_state["rank"],
            "rank_diff": h_state["rank"] - a_state["rank"],
            "home_rank_points": h_state["rank_points"],
            "away_rank_points": a_state["rank_points"],
            "rank_points_diff": h_state["rank_points"] - a_state["rank_points"],
            "is_neutral": is_neutral,
            "is_competitive": is_competitive,
            "home_rest_days": 30.0,
            "away_rest_days": 30.0,
            "rest_days_diff": 0.0,
            "home_is_home_continent": float(h_is_home_cont),
            "away_is_home_continent": float(a_is_home_cont),
            "continent_diff": float(h_is_home_cont - a_is_home_cont),
            "match_stake": 4.0 if is_competitive == 1 else 1.0
        }

        # Build list in the exact order of self.features
        vals = [row_map[feat] for feat in self.features]
        return np.array([vals])

    def _construct_features(
        self, home_team: str, away_team: str, is_neutral: int, is_competitive: int
    ) -> pd.DataFrame:
        """
        Build the feature row in the exact order the model expects.
        """
        h_state = self.get_team_state(home_team)
        a_state = self.get_team_state(away_team)

        h_cont = CONTINENT_MAP.get(home_team)
        a_cont = CONTINENT_MAP.get(away_team)
        if is_neutral == 0:
            h_is_home_cont = 1
            a_is_home_cont = 1 if h_cont == a_cont else 0
        else:
            host_continent = "North America"
            h_is_home_cont = 1 if h_cont == host_continent else 0
            a_is_home_cont = 1 if a_cont == host_continent else 0

        row = {
            "home_elo": h_state["elo"],
            "away_elo": a_state["elo"],
            "elo_diff": h_state["elo"] - a_state["elo"],
            "home_form": h_state["form"],
            "away_form": a_state["form"],
            "form_diff": h_state["form"] - a_state["form"],
            "home_goals_scored_avg": h_state["goals_scored_avg"],
            "home_goals_conceded_avg": h_state["goals_conceded_avg"],
            "home_goal_diff_avg": h_state["goal_diff_avg"],
            "away_goals_scored_avg": a_state["goals_scored_avg"],
            "away_goals_conceded_avg": a_state["goals_conceded_avg"],
            "away_goal_diff_avg": a_state["goal_diff_avg"],
            "home_rank": h_state["rank"],
            "away_rank": a_state["rank"],
            "rank_diff": h_state["rank"] - a_state["rank"],
            "home_rank_points": h_state["rank_points"],
            "away_rank_points": a_state["rank_points"],
            "rank_points_diff": h_state["rank_points"] - a_state["rank_points"],
            "is_neutral": is_neutral,
            "is_competitive": is_competitive,
            "home_rest_days": 30.0,
            "away_rest_days": 30.0,
            "rest_days_diff": 0.0,
            "home_is_home_continent": float(h_is_home_cont),
            "away_is_home_continent": float(a_is_home_cont),
            "continent_diff": float(h_is_home_cont - a_is_home_cont),
            "match_stake": 4.0 if is_competitive == 1 else 1.0
        }

        return pd.DataFrame([row])[self.features]

    def predict_match(
        self, home_team: str, away_team: str, is_neutral: int = 1, is_competitive: int = 1
    ) -> dict:
        """
        Predict outcomes using Symmetric Prediction Averaging.
        """
        self._check_and_reload()

        h_state = self.get_team_state(home_team)
        a_state = self.get_team_state(away_team)

        cache_key = (
            home_team,
            away_team,
            round((h_state["elo"] - a_state["elo"]) / 15.0) * 15.0,
            round((h_state["form"] - a_state["form"]) / 0.05) * 0.05,
            round((h_state["goals_scored_avg"] -
                  a_state["goals_conceded_avg"]) / 0.2) * 0.2,
            round((a_state["goals_scored_avg"] -
                  h_state["goals_conceded_avg"]) / 0.2) * 0.2,
            is_neutral,
            is_competitive
        )

        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # 1. Forward direction: Team A as Home, Team B as Away
        feat_forward = self._construct_features_numpy(
            home_team, away_team, is_neutral, is_competitive)

        feat_forward_scaled = (
            feat_forward - self.scaler.mean_) / self.scaler.scale_

        probs_forward = self.model.predict_proba(feat_forward_scaled)[0]

        if is_neutral == 1:
            # 2. Reverse direction: Team B as Home, Team A as Away
            feat_reverse = self._construct_features_numpy(
                away_team, home_team, is_neutral, is_competitive)

            feat_reverse_scaled = (
                feat_reverse - self.scaler.mean_) / self.scaler.scale_

            probs_reverse = self.model.predict_proba(feat_reverse_scaled)[0]

            probs_reverse_inverted = np.array(
                [probs_reverse[2], probs_reverse[1], probs_reverse[0]])

            # 3. Average forward and inverted reverse predictions
            probs_final = (probs_forward + probs_reverse_inverted) / 2.0
        else:
            probs_final = probs_forward

        # Extract prediction class with threshold
        p_home, p_draw, p_away = probs_final[0], probs_final[1], probs_final[2]
        if p_draw >= self.draw_threshold and self.draw_threshold < 1.0:
            prediction = self.classes[1]  # "D"
        else:
            prediction = self.classes[0] if p_home >= p_away else self.classes[2]  # "H" or "A"

        res_dict = {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": {
                "home_win": round(float(probs_final[0]), 4),
                "draw": round(float(probs_final[1]), 4),
                "away_win": round(float(probs_final[2]), 4),
            },
            "prediction": prediction,
        }
        self.prediction_cache[cache_key] = res_dict
        return res_dict

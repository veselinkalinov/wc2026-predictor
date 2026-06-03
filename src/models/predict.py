"""
predict.py

Responsibility: Load the trained model artifacts, construct feature vectors for
live matchups, and perform symmetric predictions for neutral venues.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MatchPredictor:
    def __init__(self):
        models_dir = Path(config["paths"]["models"])

        # Load artifacts
        self.model = joblib.load(models_dir / "best_model.pkl")
        self.scaler = joblib.load(models_dir / "scaler.pkl")

        with open(models_dir / "meta.json", "r") as f:
            meta = json.load(f)

        self.features = meta["features"]
        self.classes = meta["classes"]

        # Load the latest matches to get recent stats for teams
        features_dir = Path(config["paths"]["features"])
        self.feature_matrix = pd.read_csv(features_dir / "feature_matrix.csv")
        self.feature_matrix["date"] = pd.to_datetime(
            self.feature_matrix["date"])

        # Build a lookup dictionary of the latest state for each team
        self.team_states = self._build_latest_team_states()

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
        Updates the running state of a team. Used during tournament simulation
        to update Elo/form after a game is simulated.
        """
        self.team_states[team] = state

    def get_team_state(self, team: str) -> dict:
        """
        Return the team state, falling back to a default state if not found.
        """
        if team in self.team_states:
            return self.team_states[team]

        # Default state for unknown teams
        return {
            "elo": 1500.0,
            "form": 0.5,
            "goals_scored_avg": 1.2,
            "goals_conceded_avg": 1.2,
            "goal_diff_avg": 0.0,
            "rank": 211.0,
            "rank_points": 0.0,
        }

    def _construct_features(
        self, home_team: str, away_team: str, is_neutral: int, is_competitive: int
    ) -> pd.DataFrame:
        """
        Build the feature row in the exact order the model expects.
        """
        h_state = self.get_team_state(home_team)
        a_state = self.get_team_state(away_team)

        # Assemble row dictionary matching the feature list order in train.py
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
        }

        # Convert to DataFrame to ensure feature ordering is matched via indexing
        return pd.DataFrame([row])[self.features]

    def predict_match(
        self, home_team: str, away_team: str, is_neutral: int = 1, is_competitive: int = 1
    ) -> dict:
        """
        Predict outcomes using Symmetric Prediction Averaging.
        """
        # 1. Forward direction: Team A as Home, Team B as Away
        feat_forward = self._construct_features(
            home_team, away_team, is_neutral, is_competitive)
        feat_forward_scaled = self.scaler.transform(feat_forward.values)
        probs_forward = self.model.predict_proba(feat_forward_scaled)[
            0]  # [p_H, p_D, p_A]

        if is_neutral == 1:
            # 2. Reverse direction: Team B as Home, Team A as Away
            feat_reverse = self._construct_features(
                away_team, home_team, is_neutral, is_competitive)
            feat_reverse_scaled = self.scaler.transform(feat_reverse.values)
            probs_reverse = self.model.predict_proba(feat_reverse_scaled)[0]

            # Invert the reverse probabilities: swap index 0 (H) and index 2 (A)
            probs_reverse_inverted = np.array(
                [probs_reverse[2], probs_reverse[1], probs_reverse[0]])

            # 3. Average the forward and inverted reverse predictions
            probs_final = (probs_forward + probs_reverse_inverted) / 2.0
        else:
            # Keep asymmetric predictions if a host team actually has home field advantage
            probs_final = probs_forward

        # Extract prediction class
        pred_idx = np.argmax(probs_final)
        prediction = self.classes[pred_idx]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": {
                "home_win": round(float(probs_final[0]), 4),
                "draw": round(float(probs_final[1]), 4),
                "away_win": round(float(probs_final[2]), 4),
            },
            "prediction": prediction,
        }

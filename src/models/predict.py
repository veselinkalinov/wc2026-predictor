"""
predict.py

Responsibility: Load the trained model artifacts, construct feature vectors for
live matchups, and perform symmetric predictions for neutral venues. Includes
auto-reload on disk file modification for zero-restart model updates.
"""

from pathlib import Path
import time
import json
import os
import pandas as pd
import numpy as np
import joblib
from src.utils.config import config
from src.utils.logger import get_logger

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

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
        self.score_model_path = models_dir / "score_model.pkl"
        self.feature_matrix_path = features_dir / "feature_matrix.csv"

        # Load scaler
        self.scaler = joblib.load(self.scaler_path)

        # Load model with fallback safety for missing packages (e.g. xgboost in containers)
        try:
            self.model = joblib.load(self.model_path)
        except (ModuleNotFoundError, ImportError) as e:
            logger.warning(f"Failed to load model {self.model_filename} due to missing package: {e}. Attempting fallback to standard models...")
            # Try to fall back to scikit-learn models that don't need external packages
            fallback_options = ["histgradientboosting.pkl", "logistic_regression.pkl", "best_model.pkl"]
            loaded = False
            for fallback in fallback_options:
                fallback_path = models_dir / fallback
                if fallback_path.exists() and fallback != self.model_filename:
                    try:
                        self.model = joblib.load(fallback_path)
                        self.model_path = fallback_path
                        self.model_filename = fallback
                        logger.info(f"Successfully fell back to {fallback}")
                        loaded = True
                        break
                    except (ModuleNotFoundError, ImportError):
                        continue
            if not loaded:
                raise e

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.meta = meta
        self.features = meta["features"]
        self.classes = meta["classes"]
        self.score_model = self._load_score_model()
        
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

        # Track file modification time for auto-reloads
        self.last_loaded_time = self.model_path.stat(
            ).st_mtime if self.model_path.exists() else 0
        self.last_feature_matrix_loaded_time = 0
        self.last_check_time = 0.0
        self.prediction_cache = {}
        self._load_feature_matrix_state()

    def clear_prediction_cache(self) -> None:
        """
        Clear the prediction cache. Useful before starting a fresh Monte Carlo run.
        """
        self.prediction_cache = {}

    def _feature_matrix_mtime(self) -> float:
        if not self.feature_matrix_path.exists():
            return 0.0
        return self.feature_matrix_path.stat().st_mtime

    def _load_feature_matrix_state(self) -> None:
        """
        Reload the feature matrix and rebuild derived team analytics state.
        """
        self.feature_matrix = pd.read_csv(self.feature_matrix_path)
        parsed_dates = pd.to_datetime(
            self.feature_matrix["date"], errors="coerce", format="mixed")
        if parsed_dates.isna().any():
            bad_dates = self.feature_matrix.loc[parsed_dates.isna(), "date"].head(3).tolist()
            raise ValueError(f"Invalid feature matrix date values: {bad_dates}")
        self.feature_matrix["date"] = parsed_dates
        self.default_prediction_date = max(
            self.feature_matrix["date"].max().normalize(),
            pd.Timestamp.today().normalize(),
        )
        self.team_states = self._build_latest_team_states()
        self.last_feature_matrix_loaded_time = self._feature_matrix_mtime()
        self.clear_prediction_cache()

    def _check_and_reload(self) -> bool:
        """
        Check if model artifacts or the feature matrix changed, and reload if necessary.
        """
        if getattr(self, "disable_reload", False):
            return False
        current_time = time.time()
        # Throttle file system checks to once every 10 seconds to optimize loop speed
        if current_time - self.last_check_time < 10.0:
            return False
        self.last_check_time = current_time

        model_mtime = self.model_path.stat().st_mtime if self.model_path.exists() else 0
        feature_matrix_mtime = self._feature_matrix_mtime()
        model_updated = model_mtime > self.last_loaded_time
        feature_matrix_updated = feature_matrix_mtime > self.last_feature_matrix_loaded_time
        reloaded = False

        if model_updated:
            logger.info(
                f"Model file update detected on disk for {self.model_filename}. Reloading model artifacts and team states...")
            try:
                # Load new model with package presence checks
                try:
                    new_model = joblib.load(self.model_path)
                except (ModuleNotFoundError, ImportError) as e:
                    logger.warning(f"Failed to hot-reload updated model file due to missing package: {e}. Keeping current model.")
                else:
                    self.model = new_model
                    self.scaler = joblib.load(self.scaler_path)
                    with open(self.meta_path, "r") as f:
                        meta = json.load(f)
                    self.meta = meta
                    self.features = meta["features"]
                    self.classes = meta["classes"]
                    self.score_model = self._load_score_model()

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

                    self.last_loaded_time = model_mtime
                    reloaded = True
            except Exception as e:
                logger.error(f"Failed to reload model artifacts: {str(e)}")

        if feature_matrix_updated or reloaded:
            try:
                logger.info("Feature matrix update detected on disk. Reloading team states...")
                self._load_feature_matrix_state()
                reloaded = True
            except Exception as e:
                logger.error(f"Failed to reload feature matrix state: {str(e)}")

        if reloaded:
            logger.info("Reload completed successfully.")
        return reloaded

    def _load_score_model(self):
        """
        Load the dedicated scoreline model for expected goals and score grids.
        """
        if self.score_model_path.exists():
            try:
                return joblib.load(self.score_model_path)
            except Exception as e:
                logger.warning(f"Failed to load score model: {e}")
        return None

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

    def _resolve_prediction_date(self, match_date: str | pd.Timestamp | None = None) -> pd.Timestamp:
        if match_date is None:
            return self.default_prediction_date
        date = pd.to_datetime(match_date, errors="coerce")
        if pd.isna(date):
            return self.default_prediction_date
        return date.normalize()

    def infer_rest_days(self, team: str, match_date: str | pd.Timestamp | None = None) -> float:
        """
        Infer rest days from the latest known match before the prediction date.
        The value is capped at 30 to match training-time feature engineering.
        """
        prediction_date = self._resolve_prediction_date(match_date)
        team_matches = self.feature_matrix[
            ((self.feature_matrix["home_team"] == team) |
             (self.feature_matrix["away_team"] == team)) &
            (self.feature_matrix["date"] < prediction_date)
        ]
        if team_matches.empty:
            return 30.0

        last_match_date = team_matches["date"].max().normalize()
        rest_days = max((prediction_date - last_match_date).days, 0)
        return float(min(rest_days, 30))

    def _resolve_rest_context(
        self,
        home_team: str,
        away_team: str,
        match_date: str | pd.Timestamp | None = None,
        home_rest_days: float | None = None,
        away_rest_days: float | None = None,
    ) -> dict:
        prediction_date = self._resolve_prediction_date(match_date)
        inferred_home = self.infer_rest_days(home_team, prediction_date)
        inferred_away = self.infer_rest_days(away_team, prediction_date)

        home_value = inferred_home if home_rest_days is None else float(home_rest_days)
        away_value = inferred_away if away_rest_days is None else float(away_rest_days)

        return {
            "prediction_date": prediction_date,
            "home_rest_days": home_value,
            "away_rest_days": away_value,
            "source": "inferred" if home_rest_days is None and away_rest_days is None else "override",
        }

    def _construct_features_numpy(
        self,
        home_team: str,
        away_team: str,
        is_neutral: int,
        is_competitive: int,
        match_stake: float | None = None,
        home_rest_days: float | None = None,
        away_rest_days: float | None = None,
        match_date: str | pd.Timestamp | None = None,
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

        if match_stake is None:
            match_stake = 4.0 if is_competitive == 1 else 1.0
        rest_context = self._resolve_rest_context(
            home_team, away_team, match_date, home_rest_days, away_rest_days)
        home_rest_days = rest_context["home_rest_days"]
        away_rest_days = rest_context["away_rest_days"]

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
            "home_rest_days": float(home_rest_days),
            "away_rest_days": float(away_rest_days),
            "rest_days_diff": float(home_rest_days) - float(away_rest_days),
            "home_is_home_continent": float(h_is_home_cont),
            "away_is_home_continent": float(a_is_home_cont),
            "continent_diff": float(h_is_home_cont - a_is_home_cont),
            "match_stake": float(match_stake)
        }

        # Build list in the exact order of self.features
        vals = [row_map[feat] for feat in self.features]
        return np.array([vals])

    def _construct_features(
        self,
        home_team: str,
        away_team: str,
        is_neutral: int,
        is_competitive: int,
        match_stake: float | None = None,
        home_rest_days: float | None = None,
        away_rest_days: float | None = None,
        match_date: str | pd.Timestamp | None = None,
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

        if match_stake is None:
            match_stake = 4.0 if is_competitive == 1 else 1.0
        rest_context = self._resolve_rest_context(
            home_team, away_team, match_date, home_rest_days, away_rest_days)
        home_rest_days = rest_context["home_rest_days"]
        away_rest_days = rest_context["away_rest_days"]

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
            "home_rest_days": float(home_rest_days),
            "away_rest_days": float(away_rest_days),
            "rest_days_diff": float(home_rest_days) - float(away_rest_days),
            "home_is_home_continent": float(h_is_home_cont),
            "away_is_home_continent": float(a_is_home_cont),
            "continent_diff": float(h_is_home_cont - a_is_home_cont),
            "match_stake": float(match_stake)
        }

        return pd.DataFrame([row])[self.features]

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        return (features - self.scaler.mean_) / self.scaler.scale_

    def _scoreline_payload_from_grid(self, grid: np.ndarray, top_n: int = 5) -> dict:
        home_goals_axis = np.arange(grid.shape[0])[:, None]
        away_goals_axis = np.arange(grid.shape[1])[None, :]
        expected_home = float(np.sum(grid * home_goals_axis))
        expected_away = float(np.sum(grid * away_goals_axis))

        flat_order = np.argsort(grid.ravel())[::-1][:top_n]
        top_scorelines = []
        for flat_idx in flat_order:
            h_goals, a_goals = np.unravel_index(flat_idx, grid.shape)
            top_scorelines.append({
                "home_goals": int(h_goals),
                "away_goals": int(a_goals),
                "probability": round(float(grid[h_goals, a_goals]), 4),
            })

        return {
            "expected_goals": {
                "home": round(expected_home, 3),
                "away": round(expected_away, 3),
            },
            "scoreline_probabilities": top_scorelines,
            "scoreline_matrix": grid,
        }

    def _fallback_scoreline_payload(self, home_team: str, away_team: str) -> dict:
        h_state = self.get_team_state(home_team)
        a_state = self.get_team_state(away_team)
        expected_home = max(
            0.5, (h_state["goals_scored_avg"] + a_state["goals_conceded_avg"]) / 2.0)
        expected_away = max(
            0.5, (a_state["goals_scored_avg"] + h_state["goals_conceded_avg"]) / 2.0)
        return {
            "expected_goals": {
                "home": round(float(expected_home), 3),
                "away": round(float(expected_away), 3),
            },
            "scoreline_probabilities": [],
            "scoreline_matrix": None,
        }

    def predict_scoreline(
        self,
        home_team: str,
        away_team: str,
        is_neutral: int = 1,
        is_competitive: int = 1,
        match_stake: float | None = None,
        home_rest_days: float | None = None,
        away_rest_days: float | None = None,
        match_date: str | pd.Timestamp | None = None,
        top_n: int = 5,
    ) -> dict:
        """
        Predict expected goals and a scoreline probability grid for a matchup.
        Neutral matches are symmetrized by averaging the forward score grid with
        the transposed reverse-order grid.
        """
        if self.score_model is None:
            return self._fallback_scoreline_payload(home_team, away_team)

        feat_forward = self._construct_features_numpy(
            home_team, away_team, is_neutral, is_competitive,
            match_stake=match_stake,
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            match_date=match_date,
        )
        feat_forward_scaled = self._scale_features(feat_forward)
        grid_forward = self.score_model.predict_scoreline_matrices(
            feat_forward_scaled)[0]

        if is_neutral == 1:
            feat_reverse = self._construct_features_numpy(
                away_team, home_team, is_neutral, is_competitive,
                match_stake=match_stake,
                home_rest_days=away_rest_days,
                away_rest_days=home_rest_days,
                match_date=match_date,
            )
            feat_reverse_scaled = self._scale_features(feat_reverse)
            grid_reverse = self.score_model.predict_scoreline_matrices(
                feat_reverse_scaled)[0].T
            grid = (grid_forward + grid_reverse) / 2.0
            grid = grid / grid.sum()
        else:
            grid = grid_forward

        return self._scoreline_payload_from_grid(grid, top_n=top_n)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        is_neutral: int = 1,
        is_competitive: int = 1,
        match_stake: float | None = None,
        home_rest_days: float | None = None,
        away_rest_days: float | None = None,
        match_date: str | pd.Timestamp | None = None,
    ) -> dict:
        """
        Predict outcomes using Symmetric Prediction Averaging.
        """
        self._check_and_reload()

        h_state = self.get_team_state(home_team)
        a_state = self.get_team_state(away_team)

        if match_stake is None:
            match_stake = 4.0 if is_competitive == 1 else 1.0
        rest_context = self._resolve_rest_context(
            home_team, away_team, match_date, home_rest_days, away_rest_days)
        resolved_match_date = rest_context["prediction_date"]
        resolved_home_rest_days = rest_context["home_rest_days"]
        resolved_away_rest_days = rest_context["away_rest_days"]

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
            is_competitive,
            float(match_stake),
            resolved_match_date.strftime("%Y-%m-%d"),
            round(float(resolved_home_rest_days), 1),
            round(float(resolved_away_rest_days), 1),
            rest_context["source"],
        )

        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # 1. Forward direction: Team A as Home, Team B as Away
        feat_forward = self._construct_features_numpy(
            home_team, away_team, is_neutral, is_competitive,
            match_stake=match_stake,
            home_rest_days=resolved_home_rest_days,
            away_rest_days=resolved_away_rest_days,
            match_date=resolved_match_date)

        feat_forward_scaled = self._scale_features(feat_forward)

        probs_forward = self.model.predict_proba(feat_forward_scaled)[0]

        if is_neutral == 1:
            # 2. Reverse direction: Team B as Home, Team A as Away
            feat_reverse = self._construct_features_numpy(
                away_team, home_team, is_neutral, is_competitive,
                match_stake=match_stake,
                home_rest_days=resolved_away_rest_days,
                away_rest_days=resolved_home_rest_days,
                match_date=resolved_match_date)

            feat_reverse_scaled = self._scale_features(feat_reverse)

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

        argmax_prediction = self.classes[int(np.argmax(probs_final))]
        confidence = float(np.max(probs_final))
        draw_risk_threshold = float(self.meta.get("draw_risk_threshold", 0.30))
        draw_risk = bool(p_draw >= draw_risk_threshold)
        scoreline_payload = self.predict_scoreline(
            home_team=home_team,
            away_team=away_team,
            is_neutral=is_neutral,
            is_competitive=is_competitive,
            match_stake=match_stake,
            home_rest_days=resolved_home_rest_days,
            away_rest_days=resolved_away_rest_days,
            match_date=resolved_match_date,
            top_n=5,
        )

        res_dict = {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": {
                "home_win": round(float(probs_final[0]), 4),
                "draw": round(float(probs_final[1]), 4),
                "away_win": round(float(probs_final[2]), 4),
            },
            "prediction": prediction,
            "expected_goals": scoreline_payload["expected_goals"],
            "scoreline_probabilities": scoreline_payload["scoreline_probabilities"],
            "decision": {
                "argmax": argmax_prediction,
                "balanced": prediction,
                "confidence": round(confidence, 4),
                "draw_risk": draw_risk,
            },
            "model_info": {
                "type": self.meta.get("model_type", self.model_filename),
                "selected_by": self.meta.get("selected_by", "accuracy"),
                "log_loss": self.meta.get("test_metrics", {}).get("log_loss"),
                "brier_score": self.meta.get("test_metrics", {}).get("brier_score"),
            },
            "context": {
                "match_date": resolved_match_date.strftime("%Y-%m-%d"),
                "rest_days": {
                    "home": round(float(resolved_home_rest_days), 1),
                    "away": round(float(resolved_away_rest_days), 1),
                    "source": rest_context["source"],
                },
                "match_stake": float(match_stake),
            },
        }
        self.prediction_cache[cache_key] = res_dict
        return res_dict

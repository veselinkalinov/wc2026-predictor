"""
conftest.py

Responsibility: Automatically prepare dummy models for unit tests
if they are missing (e.g. during a clean Docker container build).
This isolates build environments from git-ignored models.
"""

import pytest
from pathlib import Path
import numpy as np
import json
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from src.utils.config import config


@pytest.fixture(scope="session", autouse=True)
def ensure_dummy_models():
    """
    Autouse fixture that runs at session start. Generates toy model assets
    under models/registry if the directory is missing files, ensuring
    clean test compliance on a new checkout.
    """
    models_dir = Path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    meta_path = models_dir / "meta.json"

    # If any core model artifact is missing, generate dummy replacements
    if not best_model_path.exists() or not scaler_path.exists() or not meta_path.exists():
        print("\n[conftest.py] Core model files missing. Generating toy models for test execution...")

        # 1. Toy data: 100 samples with 27 features (to allow tree classifier splits)
        np.random.seed(42)
        X = np.random.randn(100, 27)
        y = np.random.choice([0, 1, 2], size=100)  # Classes: H (0), D (1), A (2)

        # 2. Scaler fit and dump
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)

        # 3. Base model fit and dump
        base_hgb = HistGradientBoostingClassifier(max_iter=5, random_state=42)
        base_hgb.fit(X_scaled, y)
        joblib.dump(base_hgb, models_dir / "best_model_uncalibrated.pkl")
        joblib.dump(base_hgb, models_dir / "histgradientboosting.pkl")

        # 4. Calibration fit and dump
        calibrated_hgb = CalibratedClassifierCV(estimator=base_hgb, method="sigmoid", cv="prefit")
        calibrated_hgb.fit(X_scaled, y)
        joblib.dump(calibrated_hgb, best_model_path)

        # 5. Logistic Regression for simulator fast fallback compatibility
        base_lr = LogisticRegression(random_state=42)
        base_lr.fit(X_scaled, y)
        calibrated_lr = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv="prefit")
        calibrated_lr.fit(X_scaled, y)
        joblib.dump(calibrated_lr, models_dir / "logistic_regression.pkl")

        # 6. Meta.json definition
        meta = {
            "model_type": "HistGradientBoosting",
            "features": [
                "home_elo", "away_elo", "elo_diff",
                "home_form", "away_form", "form_diff",
                "home_goals_scored_avg", "home_goals_conceded_avg", "home_goal_diff_avg",
                "away_goals_scored_avg", "away_goals_conceded_avg", "away_goal_diff_avg",
                "home_rank", "away_rank", "rank_diff",
                "home_rank_points", "away_rank_points", "rank_points_diff",
                "is_neutral", "is_competitive",
                "home_rest_days", "away_rest_days", "rest_days_diff",
                "home_is_home_continent", "away_is_home_continent", "continent_diff",
                "match_stake"
            ],
            "best_params": {
                "learning_rate": 0.1,
                "max_depth": 3,
                "max_iter": 5
            },
            "test_metrics": {
                "accuracy": 0.60,
                "log_loss": 0.86,
                "brier_score": 0.17
            },
            "classes": ["H", "D", "A"]
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        print("[conftest.py] Toy model files successfully written to models/registry/.")

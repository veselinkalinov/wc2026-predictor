"""
train.py
Responsibility: Train a multinomial Logistic Regression model using a temporal split,
standardise features, and serialise the model artifacts to the model registry.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# List of numeric features to train on (explicitly defined)
FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form", "away_form", "form_diff",
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_goal_diff_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_goal_diff_avg",
    "home_rank", "away_rank", "rank_diff",
    "home_rank_points", "away_rank_points", "rank_points_diff",
    "is_neutral", "is_competitive"
]


def train_model() -> None:
    features_dir = Path(config["paths"]["features"])
    models_dir = Path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = features_dir / "feature_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {matrix_path}. Run build.py first."
        )

    # 1. Load data
    logger.info("Loading feature matrix...")
    df = pd.read_csv(matrix_path)
    df["date"] = pd.to_datetime(df["date"])

    # Convert targets to class indices: H -> 0, D -> 1, A -> 2
    class_map = {"H": 0, "D": 1, "A": 2}
    df["target"] = df[config["model"]["target_column"]].map(class_map)

    # 2. Temporal train/test split
    cutoff = pd.to_datetime(config["model"]["train_cutoff"])

    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()

    logger.info(f"Split data at cutoff {cutoff.date()}:")
    logger.info(f"  Train matches: {len(train_df)}")
    logger.info(f"  Test matches:  {len(test_df)}")

    # Extract features and targets
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    # 3. Standardize features
    logger.info("Fitting StandardScalar on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test set using train scaling parameters!
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Multinomial Logistic Regression model
    logger.info("Training multinomial Logistic Regression...")
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=config["model"]["random_state"],
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate on test set
    y_pred_probs = model.predict_proba(X_test_scaled)
    y_preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_preds)
    loss = log_loss(y_test, y_pred_probs)

    logger.info(f"Training complete. Evaluation on test set:")
    logger.info(f"  Accuracy: {acc:.4f} (Elo Baseline: 0.5922)")
    logger.info(f"  Log Loss: {loss:.4f} (Elo Baseline: 0.9589)")

    # 6. Save model, scaler, and metadata
    logger.info("Serialising artifacts to registry...")
    joblib.dump(model, models_dir / "best_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    # Save meta json with feature listing and scores
    meta = {
        "model_type": "Logistic Regression",
        "features": FEATURE_COLUMNS,
        "test_metrics": {
            "accuracy": float(acc),
            "log_loss": float(loss),
        },
        "classes": ["H", "D", "A"]
    }

    with open(models_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    logger.info("Saved all model artifacts successfully")


if __name__ == "__main__":
    train_model()

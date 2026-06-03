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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.utils.config import config
from src.utils.logger import get_logger

import importlib
try:
    importlib.import_module("xgboost")
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

    # 4. Define base models and hyperparameter grids for tuning
    base_models = {
        "Logistic Regression": LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=config["model"]["random_state"],
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=config["model"]["random_state"],
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=config["model"]["random_state"],
        )
    }

    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15]
        },
        "HistGradientBoosting": {
            "max_iter": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 8]
        }
    }

    if XGBOOST_AVAILABLE:
        logger.info("XGBoost is available. Adding to comparison...")
        xgb_module = importlib.import_module("xgboost")
        base_models["XGBoost"] = xgb_module.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=config["model"]["random_state"],
            eval_metric="mlogloss"
        )
        grids["XGBoost"] = {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    else:
        logger.info("XGBoost package not installed. Skipping.")

    # 5. Train and tune all models using TimeSeriesSplit cross-validation
    comparison_results = {}
    trained_models = {}
    best_params_dict = {}

    # Pre-build one-hot encoding of targets for Brier score calculation
    n_samples = len(y_test)
    y_test_one_hot = np.zeros((n_samples, 3))
    for i, idx in enumerate(y_test):
        y_test_one_hot[i, idx] = 1.0

    # Create 3-fold temporal cross-validation split on training data
    cv = TimeSeriesSplit(n_splits=3)

    logger.info("Starting hyperparameter tuning and model evaluation...")
    for model_name, clf in base_models.items():
        logger.info(f"Tuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=grids[model_name],
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)

        best_clf = grid_search.best_estimator_
        trained_models[model_name] = best_clf
        best_params_dict[model_name] = grid_search.best_params_
        logger.info(
            f"  Best params for {model_name}: {grid_search.best_params_}")

        # Predict and evaluate on test set
        y_pred_probs = best_clf.predict_proba(X_test_scaled)
        y_preds = best_clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_preds)
        loss = log_loss(y_test, y_pred_probs)

        # Calculate Brier score
        brier_scores = [brier_score_loss(
            y_test_one_hot[:, c], y_pred_probs[:, c]) for c in range(3)]
        avg_brier = np.mean(brier_scores)

        comparison_results[model_name] = {
            "accuracy": acc,
            "log_loss": loss,
            "brier_score": avg_brier
        }
        logger.info(
            f"  {model_name} (Tuned) -> Accuracy: {acc:.4f}, Log Loss: {loss:.4f}, Brier Score: {avg_brier:.4f}")

    # Log Comparison Table
    logger.info("\n" + "=" * 60 +
                "\nTUNED MODEL COMPARISON TABLE\n" + "=" * 60)
    for model_name, res in comparison_results.items():
        logger.info(
            f"{model_name:<25} | Accuracy: {res['accuracy']:.4f} | Log Loss: {res['log_loss']:.4f} | Brier: {res['brier_score']:.4f}")
    logger.info("=" * 60)

    # 6. Select the best model (minimising Log Loss)
    best_model_name = min(comparison_results,
                          key=lambda k: comparison_results[k]["log_loss"])
    best_model = trained_models[best_model_name]
    logger.info(
        f"Winning Model: {best_model_name} (Lowest Test Log Loss: {comparison_results[best_model_name]['log_loss']:.4f})")

    # 7. Serialise artifacts to registry
    logger.info("Serialising winning model to best_model.pkl...")
    joblib.dump(best_model, models_dir / "best_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    # Save all individual tuned models
    for model_name, clf in trained_models.items():
        clean_name = model_name.lower().replace(" ", "_")
        joblib.dump(clf, models_dir / f"{clean_name}.pkl")

    # Save meta json with feature listing and comparison details
    meta = {
        "model_type": best_model_name,
        "features": FEATURE_COLUMNS,
        "best_params": best_params_dict[best_model_name],
        "test_metrics": {
            "accuracy": float(comparison_results[best_model_name]["accuracy"]),
            "log_loss": float(comparison_results[best_model_name]["log_loss"]),
            "brier_score": float(comparison_results[best_model_name]["brier_score"])
        },
        "comparison": {
            name: {
                "accuracy": float(res["accuracy"]),
                "log_loss": float(res["log_loss"]),
                "brier_score": float(res["brier_score"])
            } for name, res in comparison_results.items()
        },
        "all_best_params": best_params_dict,
        "classes": ["H", "D", "A"]
    }

    with open(models_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    logger.info("Saved all model artifacts successfully")


if __name__ == "__main__":
    train_model()

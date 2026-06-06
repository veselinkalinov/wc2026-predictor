"""
train.py
Responsibility: Train, tune, ensemble, and calibrate classification models.
Using a 3-way temporal split:
- Train: pre-2022 (to learn patterns)
- Calibration: Jan 2022 - Jun 2023 (to calibrate probabilities)
- Test: Jul 2023 - Jun 2026 (to evaluate performance)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from src.utils.config import config
from src.utils.logger import get_logger
from src.models.poisson_model import PoissonGoalModel

# Import boosters with fallback checks
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = get_logger(__name__)

# Feature columns used by the model (including the new features)
FEATURE_COLUMNS = [
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
]


def find_optimal_draw_threshold(y_probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Search for a draw threshold on the calibration set that maximizes accuracy.
    If P(Draw) >= threshold, predict Draw. Otherwise predict argmax of Home vs Away.
    """
    best_acc = 0.0
    best_theta = 1.0  # Default to 1.0 (equivalent to standard argmax)
    
    # Baseline argmax accuracy
    baseline_preds = np.argmax(y_probs, axis=1)
    baseline_acc = accuracy_score(y_true, baseline_preds)
    best_acc = baseline_acc
    
    # Search over possible thresholds
    for theta in np.arange(0.15, 0.45, 0.01):
        preds = []
        for prob in y_probs:
            p_home, p_draw, p_away = prob[0], prob[1], prob[2]
            if p_draw >= theta:
                preds.append(1)  # Draw
            else:
                preds.append(0 if p_home >= p_away else 2)
        
        acc = accuracy_score(y_true, np.array(preds))
        if acc > best_acc:
            best_acc = acc
            best_theta = theta
            
    return float(best_theta)


def predict_classes_with_threshold(y_probs: np.ndarray, theta: float) -> np.ndarray:
    """
    Predict classes using the optimal draw threshold.
    """
    preds = []
    for prob in y_probs:
        p_home, p_draw, p_away = prob[0], prob[1], prob[2]
        if p_draw >= theta and theta < 1.0:
            preds.append(1)  # Draw
        else:
            preds.append(0 if p_home >= p_away else 2)
    return np.array(preds)


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

    # Convert target labels to indices: H -> 0, D -> 1, A -> 2
    class_map = {"H": 0, "D": 1, "A": 2}
    df["target"] = df[config["model"]["target_column"]].map(class_map)

    # 2. Three-Way Temporal Split
    cutoff_train = pd.to_datetime(config["model"]["train_cutoff"])
    cutoff_cal = pd.to_datetime(config["model"]["calibration_cutoff"])

    train_df = df[df["date"] < cutoff_train].copy()
    cal_df = df[(df["date"] >= cutoff_train) &
                (df["date"] < cutoff_cal)].copy()
    test_df = df[df["date"] >= cutoff_cal].copy()

    logger.info(f"Split data:")
    logger.info(
        f"  Train matches:       {len(train_df)} (pre-{cutoff_train.date()})")
    logger.info(
        f"  Calibration matches: {len(cal_df)} ({cutoff_train.date()} to {cutoff_cal.date()})")
    logger.info(
        f"  Test matches:        {len(test_df)} (post-{cutoff_cal.date()})")

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values

    X_cal = cal_df[FEATURE_COLUMNS].values
    y_cal = cal_df["target"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    # 3. Standardize features
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
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
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, 20]
        },
        "HistGradientBoosting": {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    }

    # Conditionally add LightGBM
    if LIGHTGBM_AVAILABLE:
        logger.info("LightGBM is available. Adding to comparison...")
        base_models["LightGBM"] = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=config["model"]["random_state"],
            verbose=-1
        )
        grids["LightGBM"] = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }

    # Conditionally add CatBoost
    if CATBOOST_AVAILABLE:
        logger.info("CatBoost is available. Adding to comparison...")
        base_models["CatBoost"] = CatBoostClassifier(
            loss_function="MultiClass",
            random_state=config["model"]["random_state"],
            verbose=0
        )
        grids["CatBoost"] = {
            "iterations": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [3, 5, 7]
        }

    # Conditionally add XGBoost
    if XGBOOST_AVAILABLE:
        logger.info("XGBoost is available. Adding to comparison...")
        base_models["XGBoost"] = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=config["model"]["random_state"],
            eval_metric="mlogloss"
        )
        grids["XGBoost"] = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }

    # 5. Tune and fit base models on training data
    cv = TimeSeriesSplit(n_splits=config["model"].get("cv_splits", 5))
    trained_base_models = {}
    best_params_dict = {}

    logger.info("Starting hyperparameter tuning...")
    for model_name, clf in base_models.items():
        logger.info(f"Tuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=grids[model_name],
            cv=cv,
            scoring="accuracy",  # Target accuracy directly in tuning
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)

        best_base = grid_search.best_estimator_
        trained_base_models[model_name] = best_base
        best_params_dict[model_name] = grid_search.best_params_
        logger.info(f"  Best params: {grid_search.best_params_}")

    # Train Poisson Goal Model
    logger.info("Training Poisson Goal Model...")
    poisson_model = PoissonGoalModel(alpha=1.0)
    y_train_goals = np.column_stack([train_df["home_score"].values, train_df["away_score"].values])
    poisson_model.fit(X_train_scaled, y_train_goals)
    trained_base_models["Poisson Goal Model"] = poisson_model
    best_params_dict["Poisson Goal Model"] = {"alpha": 1.0}

    # 6. Add Stacking Classifier
    stack_estimators = []
    for name in ["HistGradientBoosting", "LightGBM", "CatBoost", "XGBoost"]:
        if name in trained_base_models:
            stack_estimators.append((name.lower(), trained_base_models[name]))

    if len(stack_estimators) >= 2:
        logger.info("Building Stacking Classifier Meta-Ensemble...")
        stacking_clf = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                random_state=config["model"]["random_state"]
            ),
            cv=3,
            n_jobs=-1
        )
        logger.info("Fitting Stacking Classifier on training set...")
        stacking_clf.fit(X_train_scaled, y_train)
        trained_base_models["Stacking Ensemble"] = stacking_clf
        best_params_dict["Stacking Ensemble"] = {}

    # 7. Apply Probability Calibration and Draw Threshold Tuning
    comparison_results = {}
    calibrated_models = {}
    draw_thresholds = {}

    n_samples = len(y_test)
    y_test_one_hot = np.zeros((n_samples, 3))
    for i, idx in enumerate(y_test):
        y_test_one_hot[i, idx] = 1.0

    logger.info("Calibrating and evaluating models on Holdout Test Set...")
    for model_name, clf in trained_base_models.items():
        # Tree models use isotonic calibration, linear/stacking use sigmoid
        method = "isotonic" if model_name in ["Random Forest", "HistGradientBoosting", "LightGBM", "CatBoost", "XGBoost"] else "sigmoid"

        calibrated_clf = CalibratedClassifierCV(
            estimator=clf,
            method=method,
            cv="prefit"
        )

        # Fit calibration mapping on calibration set
        calibrated_clf.fit(X_cal_scaled, y_cal)
        calibrated_models[model_name] = calibrated_clf

        # Tune draw threshold on calibration set probabilities
        y_cal_probs = calibrated_clf.predict_proba(X_cal_scaled)
        opt_draw_threshold = find_optimal_draw_threshold(y_cal_probs, y_cal)
        draw_thresholds[model_name] = opt_draw_threshold

        # Evaluate calibrated model with threshold on holdout test set
        y_pred_probs = calibrated_clf.predict_proba(X_test_scaled)
        y_preds = predict_classes_with_threshold(y_pred_probs, opt_draw_threshold)

        acc = accuracy_score(y_test, y_preds)
        loss = log_loss(y_test, y_pred_probs)

        # Calculate Brier Score
        brier_scores = [brier_score_loss(
            y_test_one_hot[:, c], y_pred_probs[:, c]) for c in range(3)]
        avg_brier = np.mean(brier_scores)

        comparison_results[model_name] = {
            "accuracy": acc,
            "log_loss": loss,
            "brier_score": avg_brier,
            "draw_threshold": opt_draw_threshold
        }
        logger.info(
            f"  {model_name} (Calibrated) -> Accuracy: {acc:.4f}, Log Loss: {loss:.4f}, Brier: {avg_brier:.4f}, Draw Threshold: {opt_draw_threshold:.2f}"
        )

    # Log Comparison Table
    logger.info("\n" + "=" * 80 +
                "\nTUNED & CALIBRATED MODEL HOLDOUT COMPARISON (ACCURACY CRITERION)\n" + "=" * 80)
    for model_name, res in comparison_results.items():
        logger.info(
            f"{model_name:<25} | Accuracy: {res['accuracy']:.4f} | Log Loss: {res['log_loss']:.4f} | Brier: {res['brier_score']:.4f} | Draw Thresh: {res['draw_threshold']:.2f}"
        )
    logger.info("=" * 80)

    # 8. Select the overall best model (maximizing Holdout Accuracy)
    best_model_name = max(comparison_results,
                          key=lambda k: comparison_results[k]["accuracy"])

    logger.info(
        f"Winning Model: {best_model_name} (Highest Holdout Accuracy: {comparison_results[best_model_name]['accuracy']:.4f})")

    winning_calibrated_model = calibrated_models[best_model_name]
    winning_uncalibrated_base = trained_base_models[best_model_name]
    winning_threshold = draw_thresholds[best_model_name]

    # 9. Serialise artifacts to models registry
    logger.info("Serialising winning models to registry...")
    joblib.dump(winning_calibrated_model, models_dir / "best_model.pkl")
    joblib.dump(winning_uncalibrated_base, models_dir / "best_model_uncalibrated.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    # Save all individual calibrated models
    for name, clf in calibrated_models.items():
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(clf, models_dir / f"{clean_name}.pkl")

    # Save classification report for the winner
    y_pred_probs = winning_calibrated_model.predict_proba(X_test_scaled)
    y_preds = predict_classes_with_threshold(y_pred_probs, winning_threshold)
    class_names = ["H (Home)", "D (Draw)", "A (Away)"]
    report = classification_report(y_test, y_preds, target_names=class_names)

    # Save meta.json
    meta = {
        "model_type": best_model_name,
        "features": FEATURE_COLUMNS,
        "best_params": best_params_dict[best_model_name] if best_model_name in best_params_dict else {},
        "draw_threshold": winning_threshold,
        "test_metrics": {
            "accuracy": float(comparison_results[best_model_name]["accuracy"]),
            "log_loss": float(comparison_results[best_model_name]["log_loss"]),
            "brier_score": float(comparison_results[best_model_name]["brier_score"])
        },
        "comparison": {
            name: {
                "accuracy": float(res["accuracy"]),
                "log_loss": float(res["log_loss"]),
                "brier_score": float(res["brier_score"]),
                "draw_threshold": float(res["draw_threshold"])
            } for name, res in comparison_results.items()
        },
        "all_best_params": best_params_dict,
        "classes": ["H", "D", "A"],
        "evaluation": {
            "test_accuracy": float(comparison_results[best_model_name]["accuracy"]),
            "test_log_loss": float(comparison_results[best_model_name]["log_loss"]),
            "test_brier_score": float(comparison_results[best_model_name]["brier_score"]),
            "test_samples": int(len(y_test)),
            "classification_report": report
        }
    }

    with open(models_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    logger.info("Saved all model artifacts successfully.")


if __name__ == "__main__":
    train_model()

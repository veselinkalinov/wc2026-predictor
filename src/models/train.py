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

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.models.poisson_model import PoissonGoalModel
from src.utils.config import config
from src.utils.logger import get_logger

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
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form",
    "away_form",
    "form_diff",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_goal_diff_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_goal_diff_avg",
    "home_rank",
    "away_rank",
    "rank_diff",
    "home_rank_points",
    "away_rank_points",
    "rank_points_diff",
    "is_neutral",
    "is_competitive",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_is_home_continent",
    "away_is_home_continent",
    "continent_diff",
    "match_stake",
]


def select_best_model(
    comparison_results: dict, selection_metric: str = "log_loss"
) -> str:
    """
    Select the champion model using probability-first ranking.

    Lower log loss is preferred. Ties are resolved by lower Brier score, then
    higher accuracy. Accuracy can still be selected explicitly for experiments.
    """
    if selection_metric == "accuracy":
        return max(
            comparison_results,
            key=lambda k: (
                comparison_results[k]["accuracy"],
                -comparison_results[k]["log_loss"],
                -comparison_results[k]["brier_score"],
            ),
        )

    return min(
        comparison_results,
        key=lambda k: (
            comparison_results[k]["log_loss"],
            comparison_results[k]["brier_score"],
            -comparison_results[k]["accuracy"],
        ),
    )


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


def calibration_method_for_model(model_name: str) -> str:
    """
    Choose the probability calibration method used for each model family.
    """
    if model_name in [
        "Random Forest",
        "HistGradientBoosting",
        "LightGBM",
        "CatBoost",
        "XGBoost",
    ]:
        return "isotonic"
    return "sigmoid"


def fit_production_calibrated_model(
    model_name: str,
    estimator,
    X_all_scaled: np.ndarray,
    y_all: np.ndarray,
) -> tuple[CalibratedClassifierCV, object, int]:
    """
    Refit a calibrated production artifact on all completed rows.

    The final base estimator is trained on all rows. Calibration is learned from
    out-of-fold predictions, so production probabilities do not come from a
    calibrator fitted directly on the same in-sample predictions.
    """
    class_counts = np.bincount(y_all, minlength=3)
    min_class_count = int(class_counts[class_counts > 0].min())
    requested_splits = int(config["model"].get("production_cv_splits", 3))
    n_splits = min(requested_splits, min_class_count)
    if n_splits < 2:
        raise ValueError(
            "At least two samples from each present class are required for production calibration."
        )

    production_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config["model"]["random_state"],
    )
    production_model = CalibratedClassifierCV(
        estimator=clone(estimator),
        method=calibration_method_for_model(model_name),
        cv=production_cv,
        n_jobs=-1,
        ensemble=False,
    )
    production_model.fit(X_all_scaled, y_all)

    production_base = production_model.calibrated_classifiers_[0].estimator
    return production_model, production_base, n_splits


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
    cal_df = df[(df["date"] >= cutoff_train) & (df["date"] < cutoff_cal)].copy()
    test_df = df[df["date"] >= cutoff_cal].copy()

    logger.info("Split data:")
    logger.info(f"  Train matches:       {len(train_df)} (pre-{cutoff_train.date()})")
    logger.info(
        f"  Calibration matches: {len(cal_df)} ({cutoff_train.date()} to {cutoff_cal.date()})"
    )
    logger.info(f"  Test matches:        {len(test_df)} (post-{cutoff_cal.date()})")

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
        ),
    }

    grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1.0, 10.0]},
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, 20],
        },
        "HistGradientBoosting": {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
    }

    # Conditionally add LightGBM
    if LIGHTGBM_AVAILABLE:
        logger.info("LightGBM is available. Adding to comparison...")
        base_models["LightGBM"] = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=config["model"]["random_state"],
            verbose=-1,
        )
        grids["LightGBM"] = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        }

    # Conditionally add CatBoost
    if CATBOOST_AVAILABLE:
        logger.info("CatBoost is available. Adding to comparison...")
        base_models["CatBoost"] = CatBoostClassifier(
            loss_function="MultiClass",
            random_state=config["model"]["random_state"],
            verbose=0,
        )
        grids["CatBoost"] = {
            "iterations": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [3, 5, 7],
        }

    # Conditionally add XGBoost
    if XGBOOST_AVAILABLE:
        logger.info("XGBoost is available. Adding to comparison...")
        base_models["XGBoost"] = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=config["model"]["random_state"],
            eval_metric="mlogloss",
        )
        grids["XGBoost"] = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        }

    selection_metric = config["model"].get("selection_metric", "log_loss")
    grid_scoring = "accuracy" if selection_metric == "accuracy" else "neg_log_loss"
    logger.info(f"Model selection metric: {selection_metric}")
    logger.info(f"Grid-search scoring: {grid_scoring}")

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
            scoring=grid_scoring,
            n_jobs=-1,
        )
        grid_search.fit(X_train_scaled, y_train)

        best_base = grid_search.best_estimator_
        trained_base_models[model_name] = best_base
        best_params_dict[model_name] = grid_search.best_params_
        logger.info(f"  Best params: {grid_search.best_params_}")

    # Train Poisson Goal Model
    logger.info("Training Poisson Goal Model...")
    score_model_config = config.get("score_model", {})
    poisson_model = PoissonGoalModel(
        alpha=score_model_config.get("alpha", 1.0),
        rho=score_model_config.get("rho", 0.0),
        max_goals=score_model_config.get("max_goals", 10),
    )
    y_train_goals = np.column_stack(
        [train_df["home_score"].values, train_df["away_score"].values]
    )
    poisson_model.fit(X_train_scaled, y_train_goals)
    y_cal_goals = np.column_stack(
        [cal_df["home_score"].values, cal_df["away_score"].values]
    )
    rho_grid = score_model_config.get("rho_grid")
    tuned_rho, rho_cal_nll = poisson_model.tune_rho(
        X_cal_scaled, y_cal_goals, rho_grid=rho_grid
    )
    logger.info(
        f"  Poisson Goal Model Dixon-Coles rho: {tuned_rho:.3f} | Calibration score NLL: {rho_cal_nll:.4f}"
    )
    trained_base_models["Poisson Goal Model"] = poisson_model
    best_params_dict["Poisson Goal Model"] = {
        "alpha": poisson_model.alpha,
        "rho": poisson_model.rho,
        "max_goals": poisson_model.max_goals,
    }

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
                random_state=config["model"]["random_state"],
            ),
            cv=3,
            n_jobs=-1,
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
        method = calibration_method_for_model(model_name)

        calibrated_clf = CalibratedClassifierCV(
            estimator=clf, method=method, cv="prefit"
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
        brier_scores = [
            brier_score_loss(y_test_one_hot[:, c], y_pred_probs[:, c]) for c in range(3)
        ]
        avg_brier = np.mean(brier_scores)

        comparison_results[model_name] = {
            "accuracy": acc,
            "log_loss": loss,
            "brier_score": avg_brier,
            "draw_threshold": opt_draw_threshold,
        }
        logger.info(
            f"  {model_name} (Calibrated) -> Accuracy: {acc:.4f}, Log Loss: {loss:.4f}, Brier: {avg_brier:.4f}, Draw Threshold: {opt_draw_threshold:.2f}"
        )

    # Log Comparison Table
    logger.info(
        "\n"
        + "=" * 80
        + f"\nTUNED & CALIBRATED MODEL HOLDOUT COMPARISON ({selection_metric.upper()} CRITERION)\n"
        + "=" * 80
    )
    for model_name, res in comparison_results.items():
        logger.info(
            f"{model_name:<25} | Accuracy: {res['accuracy']:.4f} | Log Loss: {res['log_loss']:.4f} | Brier: {res['brier_score']:.4f} | Draw Thresh: {res['draw_threshold']:.2f}"
        )
    logger.info("=" * 80)

    # 8. Select the overall best model using the configured probability-first metric
    best_model_name = select_best_model(comparison_results, selection_metric)

    logger.info(
        f"Winning Model: {best_model_name} (selected by {selection_metric}: "
        f"accuracy={comparison_results[best_model_name]['accuracy']:.4f}, "
        f"log_loss={comparison_results[best_model_name]['log_loss']:.4f}, "
        f"brier={comparison_results[best_model_name]['brier_score']:.4f})"
    )

    winning_evaluation_model = calibrated_models[best_model_name]
    winning_evaluation_base = trained_base_models[best_model_name]
    winning_threshold = draw_thresholds[best_model_name]

    # 9. Refit production artifacts on all completed rows.
    logger.info("Refitting selected production model on all completed feature rows...")
    X_all = df[FEATURE_COLUMNS].values
    y_all = df["target"].values
    production_scaler = StandardScaler()
    X_all_scaled = production_scaler.fit_transform(X_all)

    production_models = {}
    production_bases = {}
    production_cv_splits = {}
    for model_name, base_model in trained_base_models.items():
        logger.info(f"Refitting production artifact for {model_name}...")
        production_model, production_base, n_splits = fit_production_calibrated_model(
            model_name,
            base_model,
            X_all_scaled,
            y_all,
        )
        production_models[model_name] = production_model
        production_bases[model_name] = production_base
        production_cv_splits[model_name] = n_splits

    production_score_model = PoissonGoalModel(
        alpha=poisson_model.alpha,
        rho=poisson_model.rho,
        max_goals=poisson_model.max_goals,
    )
    y_all_goals = np.column_stack([df["home_score"].values, df["away_score"].values])
    production_score_model.fit(X_all_scaled, y_all_goals)

    winning_production_model = production_models[best_model_name]
    winning_production_base = production_bases[best_model_name]

    # 10. Serialise artifacts to models registry
    logger.info(
        "Serialising production and holdout evaluation artifacts to registry..."
    )
    joblib.dump(winning_production_model, models_dir / "best_model.pkl")
    joblib.dump(winning_production_base, models_dir / "best_model_uncalibrated.pkl")
    joblib.dump(production_score_model, models_dir / "score_model.pkl")
    joblib.dump(production_scaler, models_dir / "scaler.pkl")

    joblib.dump(winning_evaluation_model, models_dir / "evaluation_model.pkl")
    joblib.dump(
        winning_evaluation_base, models_dir / "evaluation_model_uncalibrated.pkl"
    )
    joblib.dump(scaler, models_dir / "evaluation_scaler.pkl")
    joblib.dump(poisson_model, models_dir / "evaluation_score_model.pkl")

    # Save all individual production calibrated models.
    for name, clf in production_models.items():
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(clf, models_dir / f"{clean_name}.pkl")

    # Save classification report for the holdout winner
    y_pred_probs = winning_evaluation_model.predict_proba(X_test_scaled)
    y_preds = predict_classes_with_threshold(y_pred_probs, winning_threshold)
    class_names = ["H (Home)", "D (Draw)", "A (Away)"]
    report = classification_report(y_test, y_preds, target_names=class_names)

    # Save meta.json
    meta = {
        "model_type": best_model_name,
        "selected_by": selection_metric,
        "artifact_role": "production_refit",
        "features": FEATURE_COLUMNS,
        "best_params": best_params_dict[best_model_name]
        if best_model_name in best_params_dict
        else {},
        "draw_threshold": winning_threshold,
        "draw_risk_threshold": float(config["model"].get("draw_risk_threshold", 0.30)),
        "score_model": {
            "model_type": "Dixon-Coles Poisson Goal Model",
            "artifact": "score_model.pkl",
            "alpha": float(poisson_model.alpha),
            "rho": float(poisson_model.rho),
            "max_goals": int(poisson_model.max_goals),
            "calibration_score_nll": float(rho_cal_nll),
        },
        "test_metrics": {
            "accuracy": float(comparison_results[best_model_name]["accuracy"]),
            "log_loss": float(comparison_results[best_model_name]["log_loss"]),
            "brier_score": float(comparison_results[best_model_name]["brier_score"]),
        },
        "comparison": {
            name: {
                "accuracy": float(res["accuracy"]),
                "log_loss": float(res["log_loss"]),
                "brier_score": float(res["brier_score"]),
                "draw_threshold": float(res["draw_threshold"]),
            }
            for name, res in comparison_results.items()
        },
        "all_best_params": best_params_dict,
        "production_refit": {
            "enabled": True,
            "samples": int(len(df)),
            "latest_match_date": df["date"].max().strftime("%Y-%m-%d"),
            "cv_splits": int(production_cv_splits[best_model_name]),
            "calibration": "CalibratedClassifierCV ensemble=False with StratifiedKFold out-of-fold calibration",
            "note": "best_model.pkl and scaler.pkl are refit on all completed rows; holdout metrics come from evaluation_model.pkl.",
        },
        "artifacts": {
            "production_model": "best_model.pkl",
            "production_uncalibrated_model": "best_model_uncalibrated.pkl",
            "production_scaler": "scaler.pkl",
            "production_score_model": "score_model.pkl",
            "evaluation_model": "evaluation_model.pkl",
            "evaluation_uncalibrated_model": "evaluation_model_uncalibrated.pkl",
            "evaluation_scaler": "evaluation_scaler.pkl",
            "evaluation_score_model": "evaluation_score_model.pkl",
        },
        "temporal_split": {
            "train_before": cutoff_train.strftime("%Y-%m-%d"),
            "calibration_from": cutoff_train.strftime("%Y-%m-%d"),
            "calibration_before": cutoff_cal.strftime("%Y-%m-%d"),
            "test_from": cutoff_cal.strftime("%Y-%m-%d"),
            "train_samples": int(len(train_df)),
            "calibration_samples": int(len(cal_df)),
            "test_samples": int(len(test_df)),
        },
        "classes": ["H", "D", "A"],
        "evaluation": {
            "test_accuracy": float(comparison_results[best_model_name]["accuracy"]),
            "test_log_loss": float(comparison_results[best_model_name]["log_loss"]),
            "test_brier_score": float(
                comparison_results[best_model_name]["brier_score"]
            ),
            "test_samples": int(len(y_test)),
            "draw_threshold": float(winning_threshold),
            "classification_report": report,
        },
    }

    with open(models_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    logger.info("Saved all model artifacts successfully.")


if __name__ == "__main__":
    train_model()

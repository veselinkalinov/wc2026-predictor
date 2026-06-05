"""
evaluate.py
Responsibility: Evaluate the trained model, generate classification report,
confusion matrix heatmap, feature importance charts, and reliability (calibration) curves.
"""

from src.utils.logger import get_logger
from src.utils.config import config
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form", "away_form", "form_diff",
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_goal_diff_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_goal_diff_avg",
    "home_rank", "away_rank", "rank_diff",
    "home_rank_points", "away_rank_points", "rank_points_diff",
    "is_neutral", "is_competitive"
]


def generate_evaluation_report() -> None:
    models_dir = Path(config["paths"]["models"])
    features_dir = Path(config["paths"]["features"])
    vis_dir = Path(config["paths"]["visualisations"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load artifacts
    logger.info("Loading model artifacts for evaluation...")
    # Calibrated winning model
    model = joblib.load(models_dir / "best_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")

    with open(models_dir / "meta.json", "r") as f:
        meta = json.load(f)

    # 2. Load feature matrix and split
    df = pd.read_csv(features_dir / "feature_matrix.csv")
    df["date"] = pd.to_datetime(df["date"])

    class_map = {"H": 0, "D": 1, "A": 2}
    df["target"] = df[config["model"]["target_column"]].map(class_map)

    # Test set begins at calibration_cutoff (July 2023 onwards)
    cutoff = pd.to_datetime(config["model"].get(
        "calibration_cutoff", config["model"]["train_cutoff"]))
    test_df = df[df["date"] >= cutoff].copy()

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    X_test_scaled = scaler.transform(X_test)

    # 3. Predictions
    y_preds = model.predict(X_test_scaled)
    y_pred_probs = model.predict_proba(X_test_scaled)

    # 4. Classification Report
    class_names = ["H (Home)", "D (Draw)", "A (Away)"]
    report = classification_report(y_test, y_preds, target_names=class_names)

    logger.info(f"\nClassification Report (Holdout Test Set):\n{report}")

    # 5. Metrics
    acc = accuracy_score(y_test, y_preds)
    loss = log_loss(y_test, y_pred_probs)

    n_samples = len(y_test)
    y_test_one_hot = np.zeros((n_samples, 3))
    for i, idx in enumerate(y_test):
        y_test_one_hot[i, idx] = 1.0

    brier_scores = [brier_score_loss(
        y_test_one_hot[:, c], y_pred_probs[:, c]) for c in range(3)]
    avg_brier = np.mean(brier_scores)

    logger.info(f"Test Accuracy:    {acc:.4f}")
    logger.info(f"Test Log Loss:    {loss:.4f}")
    logger.info(f"Test Brier Score: {avg_brier:.4f}")

    # 6. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {meta.get('model_type', 'Model')}")
    plt.tight_layout()

    cm_path = vis_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {cm_path}")

    # 7. Feature Importance Plot
    # Access the underlying model inside CalibratedClassifierCV
    base_est = model.estimator if hasattr(model, "estimator") else model

    fig2, ax2 = plt.subplots(figsize=(10, 8))

    if hasattr(base_est, "feature_importances_"):
        importances = base_est.feature_importances_
        indices = np.argsort(importances)[::-1]

        ax2.barh(
            [FEATURE_COLUMNS[i] for i in indices],
            importances[indices],
            color="steelblue"
        )
        ax2.set_xlabel("Importance")
        ax2.set_title(
            f"Feature Importances — {meta.get('model_type', 'Model')}")
        ax2.invert_yaxis()
    elif hasattr(base_est, "coef_"):
        # For Logistic Regression
        avg_coef = np.mean(np.abs(base_est.coef_), axis=0)
        indices = np.argsort(avg_coef)[::-1]

        ax2.barh(
            [FEATURE_COLUMNS[i] for i in indices],
            avg_coef[indices],
            color="coral"
        )
        ax2.set_xlabel("|Coefficient| (averaged across classes)")
        ax2.set_title(
            f"Feature Coefficients — {meta.get('model_type', 'Model')}")
        ax2.invert_yaxis()
    elif hasattr(base_est, "estimators_") and hasattr(base_est, "final_estimator_"):
        # For Stacking Classifier: display coefficients of the final meta-learner
        meta_coef = base_est.final_estimator_.coef_  # Shape (3, 3 * len(estimators))
        base_names = [name for name, _ in base_est.estimators]
        n_est = len(base_names)
        
        # Reshape and average across classes (axis 0) and predictions (axis 2)
        # to get a single weight for each base model
        coef_reshaped = np.abs(meta_coef).reshape(3, n_est, 3)
        model_weights = np.mean(coef_reshaped, axis=(0, 2))
        
        ax2.barh(
            [f"{name} prediction scale" for name in base_names],
            model_weights,
            color="purple"
        )
        ax2.set_xlabel("Meta-learner Coefficient Weight (Averaged)")
        ax2.set_title(f"Stacking Meta-Learner Model Contributions")
        ax2.invert_yaxis()
    else:
        # Fallback to Permutation Feature Importance for HistGradientBoosting (Finding #8)
        logger.info("Computing permutation feature importance on holdout test set...")
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                model, X_test_scaled, y_test, n_repeats=5, random_state=config["model"]["random_state"], n_jobs=-1
            )
            importances = result.importances_mean
            indices = np.argsort(importances)[::-1]

            ax2.barh(
                [FEATURE_COLUMNS[i] for i in indices],
                importances[indices],
                color="teal"
            )
            ax2.set_xlabel("Mean Decrease in Holdout Accuracy")
            ax2.set_title(
                f"Feature Importance (Permutation) — {meta.get('model_type', 'Model')}")
            ax2.invert_yaxis()
        except Exception as pe:
            logger.error(f"Failed to calculate permutation importance: {pe}")
            ax2.text(0.5, 0.5, "Feature importance not available for this model type.",
                     ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    fi_path = vis_dir / "feature_importance.png"
    fig2.savefig(fi_path, dpi=150)
    plt.close(fig2)
    logger.info(f"Saved feature importance plot to {fi_path}")

    # 8. Calibration Curve (Reliability Diagram) Plot
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for c_idx, (class_name, color) in enumerate(zip(["Home Win", "Draw", "Away Win"], colors)):
        y_true_binary = y_test_one_hot[:, c_idx]
        y_prob = y_pred_probs[:, c_idx]

        # Calculate reliability curve
        prob_true, prob_pred = calibration_curve(
            y_true_binary, y_prob, n_bins=10, strategy="uniform")

        ax3.plot(
            prob_pred, prob_true, "s-", color=color,
            label=f"{class_name} (Brier: {brier_scores[c_idx]:.4f})"
        )

    # Add perfectly calibrated identity line
    ax3.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("True Probability in Bin")
    ax3.set_title(
        f"Calibration Curves (Reliability Diagram) — {meta.get('model_type', 'Model')}")
    ax3.legend(loc="lower right")
    ax3.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    cal_curve_path = vis_dir / "calibration_curve.png"
    fig3.savefig(cal_curve_path, dpi=150)
    plt.close(fig3)
    logger.info(f"Saved calibration curve to {cal_curve_path}")

    # 9. Save evaluation summary to meta
    eval_summary = {
        "test_accuracy": float(acc),
        "test_log_loss": float(loss),
        "test_brier_score": float(avg_brier),
        "test_samples": int(len(y_test)),
        "classification_report": report
    }

    meta["evaluation"] = eval_summary
    with open(models_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    logger.info(
        "Evaluation complete. Updated meta.json with calibrated holdout results.")


if __name__ == "__main__":
    generate_evaluation_report()

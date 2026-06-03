"""
evaluate.py

Responsibility: Load the trained model and scaler, evaluate them on the test set,
print detailed classification reports, and save diagnostic plots.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, log_loss, accuracy_score
from sklearn.inspection import permutation_importance
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_evaluation_report() -> None:
    models_dir = Path(config["paths"]["models"])
    features_dir = Path(config["paths"]["features"])
    vis_dir = Path(config["paths"]["visualisations"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify files exist
    model_path = models_dir / "best_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    meta_path = models_dir / "meta.json"
    matrix_path = features_dir / "feature_matrix.csv"

    for p in [model_path, scaler_path, meta_path, matrix_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required evaluation file not found: {p}")

    # 2. Load model, scaler, metadata and data
    logger.info("Loading artifacts and data for evaluation...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["features"]
    df = pd.read_csv(matrix_path)
    df["date"] = pd.to_datetime(df["date"])

    # 3. Create test set split (temporal)
    cutoff = pd.to_datetime(config["model"]["train_cutoff"])
    test_df = df[df["date"] >= cutoff].copy()

    class_map = {"H": 0, "D": 1, "A": 2}
    y_test = test_df[config["model"]["target_column"]].map(class_map).values
    X_test = test_df[feature_cols].values

    # 4. Scale and Predict
    X_test_scaled = scaler.transform(X_test)
    y_pred_probs = model.predict_proba(X_test_scaled)
    y_preds = model.predict(X_test_scaled)

    # 5. Calculate Metrics
    acc = accuracy_score(y_test, y_preds)
    loss = log_loss(y_test, y_pred_probs)

    # Calculate multi-class Brier score
    # Convert y_test to one-hot encoding
    n_samples = len(y_test)
    y_test_one_hot = np.zeros((n_samples, 3))
    for i, idx in enumerate(y_test):
        y_test_one_hot[i, idx] = 1.0

    brier_scores = [brier_score_loss(
        y_test_one_hot[:, i], y_pred_probs[:, i]) for i in range(3)]
    avg_brier = np.mean(brier_scores)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"Accuracy:    {acc:.4f}  (Elo Heuristic Baseline: 0.5922)")
    print(f"Log Loss:    {loss:.4f}  (Elo Heuristic Baseline: 0.9589)")
    print(f"Brier Score: {avg_brier:.4f}  (Elo Heuristic Baseline: 0.1887)")
    print("-" * 60)

    # Classification Report
    print("Classification Report:")
    target_names = ["Home Win (H)", "Draw (D)", "Away Win (A)"]
    print(classification_report(y_test, y_preds, target_names=target_names))
    print("-" * 60)

    # 6. Plot and Save Confusion Matrix Heatmap
    logger.info("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f"Confusion Matrix (Accuracy: {acc:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = vis_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

    # 7. Plot and Save Feature Importance (Model-Agnostic / Model-Specific)
    logger.info("Generating Feature Importance plot...")
    model_type = meta.get("model_type", "Unknown Model")
    
    if hasattr(model, "coef_"):
        # Linear models like Logistic Regression
        logger.info(f"Extracting coefficients for {model_type}...")
        importances = np.mean(np.abs(model.coef_), axis=0)
        title = f"Feature Importance ({model_type} Absolute Coefficients)"
        xlabel = "Average Absolute Coefficient Weight"
    elif hasattr(model, "feature_importances_"):
        # Tree models like Random Forest, XGBoost
        logger.info(f"Extracting Gini/Gain feature importances for {model_type}...")
        importances = model.feature_importances_
        title = f"Feature Importance ({model_type} Gini Importance)"
        xlabel = "Relative Importance"
    else:
        # Falling back to Permutation Importance (e.g. HistGradientBoosting)
        logger.info(f"Computing permutation importance on the test set for {model_type}...")
        result = permutation_importance(
            model,
            X_test_scaled,
            y_test,
            n_repeats=10,
            random_state=config["model"]["random_state"]
        )
        importances = result.importances_mean
        title = f"Feature Importance ({model_type} Permutation Importance)"
        xlabel = "Mean Accuracy Decrease"

    feat_imp = pd.Series(
        importances, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind="barh", color="skyblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Features")
    plt.tight_layout()

    fi_path = vis_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=150)
    plt.close()
    logger.info(f"Saved feature importance plot to {fi_path}")


if __name__ == "__main__":
    generate_evaluation_report()

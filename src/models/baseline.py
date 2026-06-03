"""
baseline.py

Responsibility: Evaluate simple rule-based baselines on the test set
to establish performance floors for ML models.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_baselines() -> None:
    features_dir = Path(config["paths"]["features"])
    # Fixed name: feature_matrix (singular)
    matrix_path = features_dir / "feature_matrix.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {matrix_path}. Run build.py first."
        )

    # 1. Load feature matrix
    df = pd.read_csv(matrix_path)
    df["date"] = pd.to_datetime(df["date"])

    # 2. Temporal train/test split (based on config train_cutoff)
    cutoff = pd.to_datetime(config["model"]["train_cutoff"])
    test_df = df[df["date"] >= cutoff].copy()

    logger.info(
        f"Loaded test set for baseline evaluation. Cutoff: {cutoff.date()}")
    logger.info(f"Test set size: {len(test_df)} matches.")

    # Target classes: H, D, A
    y_test = test_df[config["model"]["target_column"]].values
    n_samples = len(y_test)

    # Map target strings to class indices: H -> 0, D -> 1, A -> 2
    class_map = {"H": 0, "D": 1, "A": 2}
    # Added brackets to fix numpy array syntax
    y_test_idx = np.array([class_map[val] for val in y_test])

    # Convert y_test_idx to a one-hot representation for Brier score and Log Loss
    y_test_one_hot = np.zeros((n_samples, 3))
    for i, idx in enumerate(y_test_idx):
        y_test_one_hot[i, idx] = 1.0

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION ON TEST SET")
    print("=" * 60)

    # ----------------------------------------------------
    # Baseline 1: Random Guessing (Uniform Probabilities)
    # ----------------------------------------------------
    # Assigns 1/3 probability to H, D, and A
    probs_random = np.full((n_samples, 3), 1.0 / 3.0)
    preds_random = np.random.choice(["H", "D", "A"], size=n_samples)

    acc_random = accuracy_score(y_test, preds_random)
    loss_random = log_loss(y_test_idx, probs_random, labels=[0, 1, 2])
    brier_random = np.mean(
        [brier_score_loss(y_test_one_hot[:, i], probs_random[:, i]) for i in range(3)])

    print(f"1. Random Guessing:")
    print(f"   Accuracy:   {acc_random:.4f}")
    print(f"   Log Loss:   {loss_random:.4f}")
    print(f"   Brier Score: {brier_random:.4f}")
    print("-" * 50)

    # ----------------------------------------------------
    # Baseline 2: Most Frequent Class (Always Home Win)
    # ----------------------------------------------------
    # Predict H for all matches.
    # To compute log loss, we use the historical training set base rates as probabilities
    # (e.g. H=0.481, D=0.233, A=0.286) to avoid infinite loss on wrong predictions.
    train_df = df[df["date"] < cutoff]
    class_counts = train_df[config["model"]
                            ["target_column"]].value_counts(normalize=True)

    p_h = class_counts.get("H", 0.48)
    p_d = class_counts.get("D", 0.23)
    p_a = class_counts.get("A", 0.29)
    probs_mfc = np.tile([p_h, p_d, p_a], (n_samples, 1))
    preds_mfc = np.full(n_samples, "H")

    acc_mfc = accuracy_score(y_test, preds_mfc)
    loss_mfc = log_loss(y_test_idx, probs_mfc, labels=[0, 1, 2])
    brier_mfc = np.mean(
        [brier_score_loss(y_test_one_hot[:, i], probs_mfc[:, i]) for i in range(3)])

    print(f"2. Most Frequent Class (Always Home Win):")
    print(f"   Accuracy:   {acc_mfc:.4f}")
    print(f"   Log Loss:   {loss_mfc:.4f}")
    print(f"   Brier Score: {brier_mfc:.4f}")
    print("-" * 50)

    # ----------------------------------------------------
    # Baseline 3: Heuristic (Higher Elo Wins)
    # ----------------------------------------------------
    # Predict H if home_elo >= away_elo, else A (Draws default to H)
    preds_elo = np.where(test_df["home_elo"] >= test_df["away_elo"], "H", "A")

    # To calculate a log loss for the Elo baseline, we will generate simple
    # pseudo-probabilities. If home_elo >= away_elo: [0.6, 0.2, 0.2], else [0.2, 0.2, 0.6]
    probs_elo = np.zeros((n_samples, 3))
    for i, row in enumerate(test_df.itertuples()):
        if row.home_elo >= row.away_elo:
            probs_elo[i] = [0.60, 0.20, 0.20]
        else:
            probs_elo[i] = [0.20, 0.20, 0.60]

    acc_elo = accuracy_score(y_test, preds_elo)
    loss_elo = log_loss(y_test_idx, probs_elo, labels=[0, 1, 2])
    brier_elo = np.mean(
        [brier_score_loss(y_test_one_hot[:, i], probs_elo[:, i]) for i in range(3)])

    print(f"3. Elo Heuristic (Higher Elo Wins):")
    print(f"   Accuracy:   {acc_elo:.4f}")
    print(f"   Log Loss:   {loss_elo:.4f}")
    print(f"   Brier Score: {brier_elo:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_baselines()

"""
build.py

Responsibility: Run all feature generators and assemble the final feature matrix.
Saves the output to data/features/feature_matrix.csv.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger
from src.features.elo import compute_elo_ratings
from src.features.form import compute_form_features
from src.features.goals import compute_goal_features

logger = get_logger(__name__)


def build_feature_matrix() -> None:
    processed_dir = Path(config["paths"]["processed_data"])
    features_dir = Path(config["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "matches_clean.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned matches file not found at {input_path}. Run clean.py first."
        )

    # 1. Load cleaned matches
    logger.info(f"Loading cleaned matches from {input_path}...")
    df = pd.read_csv(input_path)

    # Convert date to datetime for correct chronological sorting inside feature modules
    df["date"] = pd.to_datetime(df["date"])

    # 2. Chain feature calculators
    # Since each function returns the DataFrame with new columns, we chain them
    df = compute_elo_ratings(df)
    df = compute_form_features(df)
    df = compute_goal_features(df)

    # 3. Add ranking differences
    logger.info("Computing ranking difference features...")
    df["rank_diff"] = df["home_rank"] - df["away_rank"]
    df["rank_points_diff"] = df["home_rank_points"] - df["away_rank_points"]

    # 4. Standardize boolean features
    df["is_neutral"] = df["neutral"].astype(int)

    # Check for NaNs
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if len(columns_with_nulls) > 0:
        logger.warning(
            f"NaN values found in compiled DataFrame: \n{columns_with_nulls}")
        # Drop rows with null features if any slipped through
        df = df.dropna().copy()
        logger.info(f"Dropped rows with NaNs. Remaining matches: {len(df)}")

    # 5. Save the final feature matrix
    out_path = features_dir / "feature_matrix.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved feature matrix to {out_path}. Shape: {df.shape}")


if __name__ == "__main__":
    build_feature_matrix()

"""
retrain.py

Responsibility: Run a quick retraining loop:
cleaning -> feature building -> model training -> evaluation.
Skips fetch and validation checks for speed.
"""

import sys
from src.utils.logger import get_logger
from src.data.clean import run_cleaning
from src.features.build import build_feature_matrix
from src.models.train import train_model
from src.models.evaluate import generate_evaluation_report

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 70)
    logger.info("STARTING QUICK RETRAINING LOOP")
    logger.info("=" * 70)

    try:
        # Step 1: Preprocessing and cleaning
        logger.info("[STEP 1/4] Running data cleaning...")
        run_cleaning()

        # Step 2: Feature engineering
        logger.info("[STEP 2/4] Building feature matrix...")
        build_feature_matrix()

        # Step 3: Model training
        logger.info("[STEP 3/4] Training model...")
        train_model()

        # Step 4: Model evaluation
        logger.info("[STEP 4/4] Evaluating model...")
        generate_evaluation_report()

        logger.info("=" * 70)
        logger.info("QUICK RETRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

    except Exception as e:
        logger.critical(f"Retraining loop failed! Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

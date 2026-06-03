"""
run_pipeline.py

Responsibility: Run the entire end-to-end ML pipeline sequentially:
raw file check -> structural validation -> cleaning -> feature engineering -> training -> evaluation.
"""

import sys
from src.utils.logger import get_logger
from src.data.fetch import check_raw_files
from src.data.validate import run_all_validations
from src.data.clean import run_cleaning
from src.features.build import build_feature_matrix
from src.models.train import train_model
from src.models.evaluate import generate_evaluation_report

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 70)
    logger.info("STARTING END-TO-END MACHINE LEARNING PIPELINE")
    logger.info("=" * 70)

    try:
        # Step 1: Raw data presence check
        logger.info("[STEP 1/6] Checking raw file presence...")
        check_raw_files()

        # Step 2: Data validation checks
        logger.info("[STEP 2/6] Running structural data validations...")
        run_all_validations()

        # Step 3: Preprocessing and cleaning
        logger.info("[STEP 3/6] Running data cleaning and normalization...")
        run_cleaning()

        # Step 4: Feature engineering
        logger.info("[STEP 4/6] Engineering ELO, form, and goal features...")
        build_feature_matrix()

        # Step 5: Model training
        logger.info("[STEP 5/6] Training Logistic Regression model...")
        train_model()

        # Step 6: Model evaluation
        logger.info("[STEP 6/6] Generating evaluation reports and plots...")
        generate_evaluation_report()

        logger.info("=" * 70)
        logger.info("ML PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

    except Exception as e:
        logger.critical(f"Pipeline failed at step! Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

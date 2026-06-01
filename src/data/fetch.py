"""
fetch.py

Responsibility: verify that raw data files are present in data/raw/.

For this project the Kaggle source files (matches.csv, fifa_rankings.csv)
are downloaded manually and placed in data/raw/. This module confirms they
exist and are non-empty before the pipeline proceeds.

The elo_ratings.csv file is also manually sourced and placed in data/raw/.

If any file is missing, a clear error is raised pointing to the resolution.
"""

from pathlib import Path
from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def check_raw_files() -> None:
    """
    Verify that all required raw data files exist and are non-empty.
    Raises FileNotFoundError with a descriptive message if any file is missing.
    """
    raw_dir = Path(config["paths"]["raw_data"])

    required_files = {
        "matches.csv": (
            "Download from: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017"
        ),
        "fifa_rankings.csv": (
            "Download from: https://www.kaggle.com/datasets/cashncarry/fifaworldranking"
        ),
        "elo_ratings.csv": (
            "Download from Kaggle: search 'international football elo ratings'"
        ),
    }

    all_present = True

    for filename, instructions in required_files.items():
        filepath = raw_dir / filename

        if not filepath.exists():
            log.error(f"Missing file: {filepath}")
            log.error(f"Resolution: {instructions}")
            all_present = False
            continue

        size_bytes = filepath.stat().st_size
        if size_bytes == 0:
            log.error(f"Empty file: {filepath}")
            all_present = False
            continue

        log.info(f"Found: {filename} ({size_bytes / 1024:.1f} KB)")

    if not all_present:
        raise FileNotFoundError(
            "One or more required raw data files are missing or empty. "
            "See log output above for details."
        )

    log.info("All required raw data files are present.")


if __name__ == "__main__":
    check_raw_files()

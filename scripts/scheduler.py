"""
scheduler.py

Responsibility: Periodic cron-like scheduler running as a separate service
in Docker compose to trigger scripts/fetch_recent_matches.py at set intervals.
"""

import os
import time
from src.utils.logger import get_logger
from scripts.fetch_recent_matches import main as fetch_and_retrain

logger = get_logger(__name__)

INTERVAL_HOURS = float(os.getenv("RETRAINING_INTERVAL_HOURS", "24"))
INTERVAL_SECONDS = int(INTERVAL_HOURS * 3600)

def main():
    logger.info("=" * 60)
    logger.info(f"SCHEDULER SERVICE STARTING (Interval: {INTERVAL_HOURS}h / {INTERVAL_SECONDS}s)")
    logger.info("=" * 60)
    
    # Warm-up delay on container startup to let Flask initialize first
    logger.info("Scheduler warming up for 15 seconds...")
    time.sleep(15)
    
    while True:
        try:
            logger.info("Scheduler: Beginning scheduled execution cycle...")
            fetch_and_retrain()
            logger.info("Scheduler: Execution cycle completed successfully.")
        except Exception as e:
            logger.error(f"Scheduler Encountered Error: {str(e)}")
            
        logger.info(f"Scheduler: Sleeping for {INTERVAL_HOURS} hours before next cycle...")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()

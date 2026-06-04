"""
fetch_recent_matches.py

Responsibility: Fetches finished World Cup matches from Football-Data.org,
cleans and normalises them, appends new records to data/raw/matches.csv,
and triggers the model retraining pipeline.
"""

import os
import csv
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.utils.config import config, PROJECT_ROOT
from src.data.clean import clean_team_name, run_cleaning
from src.features.build import build_feature_matrix
from src.models.train import train_model
from src.models.evaluate import generate_evaluation_report

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
MATCHES_CSV_PATH = PROJECT_ROOT / config["paths"]["raw_data"] / config["data"]["matches_file"]

def load_existing_matches():
    """
    Load existing matches into a set of (date, home_team, away_team) for fast deduplication.
    """
    existing = set()
    if not MATCHES_CSV_PATH.exists():
        logger.warning(f"matches.csv not found at {MATCHES_CSV_PATH}. A new file will be created.")
        return existing

    with open(MATCHES_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add((row["date"], row["home_team"], row["away_team"]))
    return existing

def query_world_cup_matches():
    """
    Fetch all matches for the World Cup (competition WC) from Football-Data.org.
    """
    if not API_KEY:
        logger.warning("FOOTBALL_DATA_API_KEY not found in environment variables.")
        return []

    url = "https://api.football-data.org/v4/competitions/WC/matches"
    headers = {
        "X-Auth-Token": API_KEY
    }

    try:
        logger.info("Fetching World Cup matches from Football-Data.org...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("matches", [])
    except Exception as e:
        logger.error(f"Failed to fetch World Cup matches: {str(e)}")
        if 'response' in locals() and hasattr(response, 'text'):
            logger.error(f"Response details: {response.text}")
        return []

def parse_and_clean_matches(matches):
    """
    Parse Football-Data.org matches list into matches.csv expected formats.
    """
    parsed = []
    for m in matches:
        try:
            status = m.get("status")
            if status != "FINISHED":
                continue # Only process finished matches

            # Extract date (YYYY-MM-DD)
            utc_date = m.get("utcDate", "")
            if not utc_date:
                continue
            date_str = utc_date[:10]

            home = clean_team_name(m.get("homeTeam", {}).get("name"))
            away = clean_team_name(m.get("awayTeam", {}).get("name"))
            
            score = m.get("score", {})
            full_time = score.get("fullTime", {})
            home_score = full_time.get("home")
            away_score = full_time.get("away")

            if home_score is None or away_score is None:
                continue

            city = m.get("venue", "Unknown")
            area_name = m.get("area", {}).get("name", "USA/Canada/Mexico")
            
            neutral = "TRUE" if area_name != home else "FALSE"

            parsed.append({
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "tournament": "FIFA World Cup",
                "city": city,
                "country": area_name,
                "neutral": neutral
            })
        except Exception as e:
            logger.warning(f"Error parsing match: {str(e)}")
            continue
    return parsed

def main():
    logger.info("=" * 70)
    logger.info("STARTING WORLD CUP MATCHES FETCH & RETRAINING LOOP")
    logger.info("=" * 70)

    existing_keys = load_existing_matches()
    logger.info(f"Loaded {len(existing_keys)} existing matches from matches.csv")

    fetched_matches = []

    if API_KEY:
        matches = query_world_cup_matches()
        if matches:
            fetched_matches = parse_and_clean_matches(matches)
            logger.info(f"Fetched {len(fetched_matches)} finished World Cup matches from Football-Data.org.")
    else:
        logger.warning("FOOTBALL_DATA_API_KEY is not configured in .env. Skipping fetch.")

    # Deduplicate and append
    new_matches = []
    for m in fetched_matches:
        key = (m["date"], m["home_team"], m["away_team"])
        if key not in existing_keys:
            new_matches.append(m)
            existing_keys.add(key)

    if not new_matches:
        logger.info("No new finished World Cup matches found. matches.csv is fully up to date.")
    else:
        logger.info(f"Appending {len(new_matches)} new matches to matches.csv...")
        
        file_exists = MATCHES_CSV_PATH.exists()
        with open(MATCHES_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"
            ])
            if not file_exists:
                writer.writeheader()
            for m in new_matches:
                writer.writerow(m)
                
        logger.info("Successfully updated matches.csv")

    # Trigger the retraining pipeline steps
    logger.info("Triggering retraining pipeline steps...")
    
    logger.info("[STEP 1/4] Running data cleaning and ranking merges...")
    run_cleaning()
    
    logger.info("[STEP 2/4] Engineering Elo, form, and rolling goal averages...")
    build_feature_matrix()
    
    logger.info("[STEP 3/4] Tuning and training HistGradientBoosting classifier...")
    train_model()
    
    logger.info("[STEP 4/4] Writing test confusion matrix and feature contribution heatmaps...")
    generate_evaluation_report()

    logger.info("=" * 70)
    logger.info("MATCHES REFRESH AND MODEL RETRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

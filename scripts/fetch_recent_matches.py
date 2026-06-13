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
MATCH_FIELDNAMES = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral"
]

def load_existing_matches():
    """
    Load existing matches as rows so finished API results can update
    pre-seeded fixture rows that currently have blank scores.
    """
    if not MATCHES_CSV_PATH.exists():
        logger.warning(f"matches.csv not found at {MATCHES_CSV_PATH}. A new file will be created.")
        return []

    with open(MATCHES_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _has_missing_score(row):
    return row.get("home_score") in ("", None) or row.get("away_score") in ("", None)


def _parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _same_world_cup_match(row, match):
    return (
        row.get("home_team") == match["home_team"]
        and row.get("away_team") == match["away_team"]
        and row.get("tournament") == match["tournament"]
    )


def _find_existing_match_index(rows, match):
    """
    Find an existing match row.

    Football-Data.org dates are UTC, while the seeded World Cup schedule uses
    local match dates. For late CONCACAF kickoffs this can be one day apart, so
    fall back to the same teams/tournament within one day.
    """
    match_date = _parse_date(match["date"])

    exact_matches = [
        idx for idx, row in enumerate(rows)
        if _same_world_cup_match(row, match) and row.get("date") == match["date"]
    ]
    if exact_matches:
        return exact_matches[0]

    nearby_matches = []
    for idx, row in enumerate(rows):
        row_date = _parse_date(row.get("date"))
        if not row_date or not match_date or not _same_world_cup_match(row, match):
            continue
        if abs((row_date - match_date).days) <= 1:
            nearby_matches.append(idx)

    missing_nearby = [idx for idx in nearby_matches if _has_missing_score(rows[idx])]
    if missing_nearby:
        return missing_nearby[0]
    if nearby_matches:
        return nearby_matches[0]
    return None


def upsert_matches(matches):
    """
    Update blank fixture rows with final scores, append genuinely new matches,
    and skip rows that already contain a score.
    """
    rows = load_existing_matches()
    updated_count = 0
    appended_count = 0
    skipped_count = 0

    for match in matches:
        existing_idx = _find_existing_match_index(rows, match)

        if existing_idx is None:
            rows.append(match)
            appended_count += 1
            continue

        existing_row = rows[existing_idx]
        if not _has_missing_score(existing_row):
            skipped_count += 1
            continue

        existing_row["home_score"] = str(match["home_score"])
        existing_row["away_score"] = str(match["away_score"])
        existing_row["tournament"] = existing_row.get("tournament") or match["tournament"]
        for field in ("city", "country", "neutral"):
            if existing_row.get(field) in ("", None):
                existing_row[field] = match[field]
        updated_count += 1

    if updated_count == 0 and appended_count == 0:
        logger.info(
            "No new or blank World Cup match rows to update. "
            f"Skipped {skipped_count} already-scored matches."
        )
        return False

    MATCHES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MATCHES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Successfully updated matches.csv "
        f"(updated={updated_count}, appended={appended_count}, skipped={skipped_count})."
    )
    return True

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

    existing_count = len(load_existing_matches())
    logger.info(f"Loaded {existing_count} existing matches from matches.csv")

    fetched_matches = []

    if API_KEY:
        matches = query_world_cup_matches()
        if matches:
            fetched_matches = parse_and_clean_matches(matches)
            logger.info(f"Fetched {len(fetched_matches)} finished World Cup matches from Football-Data.org.")
    else:
        logger.warning("FOOTBALL_DATA_API_KEY is not configured in .env. Skipping fetch.")

    match_data_changed = upsert_matches(fetched_matches)
    if not match_data_changed:
        logger.info("Skipping retraining because matches.csv did not change.")
        return

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

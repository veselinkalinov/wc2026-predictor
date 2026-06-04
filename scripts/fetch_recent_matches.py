"""
fetch_recent_matches.py

Responsibility: Fetches recent international matches (from 2024 to June 2026)
from API-Football, cleans and normalizes them, appends new records to data/raw/matches.csv,
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

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
MATCHES_CSV_PATH = PROJECT_ROOT / \
    config["paths"]["raw_data"] / config["data"]["matches_file"]

# Target leagues for fetching:
# 10 = Friendlies
# 5 = CONMEBOL Qualifiers
# 4 = UEFA Qualifiers
# 32 = CONCACAF Qualifiers
# 31 = CAF Qualifiers
# 30 = AFC Qualifiers
TARGET_LEAGUES = {
    10: "Friendly",
    5: "FIFA World Cup qualification",
    4: "FIFA World Cup qualification",
    32: "FIFA World Cup qualification",
    31: "FIFA World Cup qualification",
    30: "FIFA World Cup qualification"
}
SEASONS = [2024, 2025, 2026]

# Mock data fallback for testing without API keys (adds simulated friendly results in June 2026)
MOCK_RECENT_MATCHES = [
    {"date": "2026-06-01", "home_team": "Germany", "away_team": "France", "home_score": 2, "away_score": 1,
        "tournament": "Friendly", "city": "Berlin", "country": "Germany", "neutral": "FALSE"},
    {"date": "2026-06-02", "home_team": "Brazil", "away_team": "Spain", "home_score": 2, "away_score": 2,
        "tournament": "Friendly", "city": "Madrid", "country": "Spain", "neutral": "TRUE"},
    {"date": "2026-06-03", "home_team": "United States", "away_team": "Ecuador", "home_score": 1, "away_score": 0,
        "tournament": "Friendly", "city": "New York", "country": "United States", "neutral": "FALSE"},
    {"date": "2026-06-04", "home_team": "Mexico", "away_team": "Colombia", "home_score": 1, "away_score": 2,
        "tournament": "Friendly", "city": "Mexico City", "country": "Mexico", "neutral": "FALSE"},
]


def load_existing_matches():
    """
    Load existing matches into a set of (date, home_team, away_team) for fast deduplication.
    """
    existing = set()
    if not MATCHES_CSV_PATH.exists():
        logger.warning(
            f"matches.csv not found at {MATCHES_CSV_PATH}. A new file will be created.")
        return existing

    with open(MATCHES_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add((row["date"], row["home_team"], row["away_team"]))
    return existing


def query_fixtures(league_id, season):
    """
    Fetch finished fixtures for a specific league and season from API-Football.
    """
    if not RAPIDAPI_KEY:
        return []

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    params = {
        "league": league_id,
        "season": season,
        "status": "FT"  # Finished matches only
    }

    try:
        logger.info(
            f"Fetching fixtures for league {league_id}, season {season}...")
        response = requests.get(url, headers=headers,
                                params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        errors = data.get("errors", [])
        if errors:
            logger.error(f"API-Football error: {errors}")
            return []

        return data.get("response", [])
    except Exception as e:
        logger.error(
            f"Failed to fetch league {league_id} season {season}: {str(e)}")
        return []


def parse_and_clean_fixtures(fixtures, tournament_fallback):
    """
    Parse API-Football fixtures list into matches.csv expected formats.
    """
    parsed = []
    for f in fixtures:
        try:
            fixture_data = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            league = f.get("league", {})

            date_str = fixture_data.get("date", "")[:10]  # Get YYYY-MM-DD
            home = clean_team_name(teams.get("home", {}).get("name"))
            away = clean_team_name(teams.get("away", {}).get("name"))
            home_score = goals.get("home")
            away_score = goals.get("away")

            if home_score is None or away_score is None:
                continue  # Skip unplayed

            tournament = league.get("name", tournament_fallback)
            city = fixture_data.get("venue", {}).get("city", "Unknown")
            country = league.get("country", "Unknown")

            # If match is played in a country that doesn't match home team's name
            neutral = "TRUE" if country != home else "FALSE"

            parsed.append({
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "tournament": tournament,
                "city": city,
                "country": country,
                "neutral": neutral
            })
        except Exception as e:
            logger.warning(f"Error parsing fixture: {str(e)}")
            continue
    return parsed


def main():
    logger.info("=" * 70)
    logger.info("STARTING RECENT MATCHES FETCH & RETRAINING LOOP")
    logger.info("=" * 70)

    existing_keys = load_existing_matches()
    logger.info(
        f"Loaded {len(existing_keys)} existing matches from matches.csv")

    fetched_matches = []

    if RAPIDAPI_KEY:
        logger.info(
            "RAPIDAPI_KEY found. Fetching live matches from API-Football...")
        for league_id, tourney_name in TARGET_LEAGUES.items():
            for season in SEASONS:
                fixtures = query_fixtures(league_id, season)
                if fixtures:
                    parsed = parse_and_clean_fixtures(fixtures, tourney_name)
                    fetched_matches.extend(parsed)
        logger.info(
            f"Fetched {len(fetched_matches)} total finished fixtures from API-Football.")
    else:
        logger.warning("RAPIDAPI_KEY not found in environment variables.")
        logger.info(
            "Falling back to pre-defined mock friendly matches for demonstration...")
        fetched_matches = MOCK_RECENT_MATCHES

    # Deduplicate and append
    new_matches = []
    for m in fetched_matches:
        key = (m["date"], m["home_team"], m["away_team"])
        if key not in existing_keys:
            new_matches.append(m)
            existing_keys.add(key)

    if not new_matches:
        logger.info("No new matches found. matches.csv is fully up to date.")
    else:
        logger.info(
            f"Appending {len(new_matches)} new matches to matches.csv...")

        file_exists = MATCHES_CSV_PATH.exists()
        with open(MATCHES_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                                    "date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"])
            if not file_exists:
                writer.writeheader()
            for m in new_matches:
                writer.writerow(m)

        logger.info("Successfully updated matches.csv")

    # Trigger the retraining pipeline steps
    logger.info("Triggering retraining pipeline steps...")

    logger.info("[STEP 1/4] Running data cleaning and ranking merges...")
    run_cleaning()

    logger.info(
        "[STEP 2/4] Engineering Elo, form, and rolling goal averages...")
    build_feature_matrix()

    logger.info(
        "[STEP 3/4] Tuning and training HistGradientBoosting classifier...")
    train_model()

    logger.info(
        "[STEP 4/4] Writing test confusion matrix and feature contribution heatmaps...")
    generate_evaluation_report()

    logger.info("=" * 70)
    logger.info("MATCHES REFRESH AND MODEL RETRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

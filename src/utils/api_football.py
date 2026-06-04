"""
api_football.py

Responsibility: Handles all API calls to API-Football (via RapidAPI),
caching response JSONs locally and providing mock fallbacks if offline,
no credentials are set, or request limits are reached.
"""

import os
import json
from pathlib import Path
import requests
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT

logger = get_logger(__name__)

# Load local environment variables
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
CACHE_DIR = PROJECT_ROOT / "data" / "live_cache"
FIXTURES_CACHE = CACHE_DIR / "fixtures.json"
STANDINGS_CACHE = CACHE_DIR / "standings.json"

# Official Groups from simulate.py
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}


def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _generate_mock_standings():
    """
    Generate starting standings for the 12 groups in API-Football format.
    """
    response_data = []
    league_obj = {
        "id": 1,
        "name": "World Cup",
        "country": "World",
        "logo": "https://media.api-sports.io/football/leagues/1.png",
        "flag": None,
        "season": 2026,
        "standings": []
    }

    # API-Football groups standings is an array of arrays (each representing a group)
    for group_name, teams in GROUPS.items():
        group_standings = []
        for i, team in enumerate(teams):
            group_standings.append({
                "rank": i + 1,
                "team": {
                    "id": 1000 + hash(team) % 1000,
                    "name": team,
                    "logo": f"https://media.api-sports.io/football/teams/{1000 + hash(team) % 1000}.png"
                },
                "points": 0,
                "goalsDiff": 0,
                "group": f"Group {group_name}",
                "form": "",
                "status": "same",
                "description": "Possible Qualification",
                "all": {
                    "played": 0,
                    "win": 0,
                    "draw": 0,
                    "lose": 0,
                    "goals": {
                        "for": 0,
                        "against": 0
                    }
                },
                "update": "2026-06-04T00:00:00+00:00"
            })
        league_obj["standings"].append(group_standings)

    response_data.append({"league": league_obj})
    return {"response": response_data, "errors": [], "results": len(response_data)}


def _generate_mock_fixtures():
    """
    Generate round-robin fixtures schedule for group stages of World Cup 2026.
    """
    response_fixtures = []
    match_id = 200000

    # Let's start the tournament matches on June 11, 2026
    start_day = 11

    # Loop over groups and generate 6 round-robin matches for each
    for group_idx, (group_name, teams) in enumerate(GROUPS.items()):
        # Schedule pairings
        pairings = [
            (teams[0], teams[1], start_day + group_idx % 4),        # Matchday 1
            (teams[2], teams[3], start_day + group_idx % 4),
            (teams[0], teams[2], start_day + 4 + group_idx % 4),    # Matchday 2
            (teams[3], teams[1], start_day + 4 + group_idx % 4),
            (teams[3], teams[0], start_day + 8 + group_idx % 4),    # Matchday 3
            (teams[1], teams[2], start_day + 8 + group_idx % 4)
        ]

        for home, away, day in pairings:
            fixture_date = f"2026-06-{day:02d}T18:00:00+00:00"
            
            # Since current local time is June 4, 2026, matches are in the future
            status_long = "Not Started"
            status_short = "NS"
            elapsed = 0
            goals_home = None
            goals_away = None
            
            response_fixtures.append({
                "fixture": {
                    "id": match_id,
                    "referee": "TBD",
                    "timezone": "UTC",
                    "date": fixture_date,
                    "timestamp": 1781197200 + (day - 11) * 86400,
                    "periods": {
                        "first": None,
                        "second": None
                    },
                    "venue": {
                        "id": 100 + match_id % 10,
                        "name": "MetLife Stadium" if day % 2 == 0 else "Azteca Stadium",
                        "city": "East Rutherford" if day % 2 == 0 else "Mexico City"
                    },
                    "status": {
                        "long": status_long,
                        "short": status_short,
                        "elapsed": elapsed
                    }
                },
                "league": {
                    "id": 1,
                    "name": "World Cup",
                    "country": "World",
                    "logo": "https://media.api-sports.io/football/leagues/1.png",
                    "flag": None,
                    "season": 2026,
                    "round": f"Group Stage - Group {group_name}"
                },
                "teams": {
                    "home": {
                        "id": 1000 + hash(home) % 1000,
                        "name": home,
                        "logo": f"https://media.api-sports.io/football/teams/{1000 + hash(home) % 1000}.png",
                        "winner": None
                    },
                    "away": {
                        "id": 1000 + hash(away) % 1000,
                        "name": away,
                        "logo": f"https://media.api-sports.io/football/teams/{1000 + hash(away) % 1000}.png",
                        "winner": None
                    }
                },
                "goals": {
                    "home": goals_home,
                    "away": goals_away
                },
                "score": {
                    "halftime": {
                        "home": None,
                        "away": None
                    },
                    "fulltime": {
                        "home": goals_home,
                        "away": goals_away
                    },
                    "extratime": {
                        "home": None,
                        "away": None
                    },
                    "penalty": {
                        "home": None,
                        "away": None
                    }
                }
            })
            match_id += 1

    return {"response": response_fixtures, "errors": [], "results": len(response_fixtures)}


def fetch_from_api(endpoint: str, params: dict) -> dict:
    """
    Low-level query to RapidAPI API-Football endpoints.
    """
    if not RAPIDAPI_KEY:
        logger.warning("RAPIDAPI_KEY is not set in environment. Skipping API request.")
        return {}

    url = f"https://api-football-v1.p.rapidapi.com/v3/{endpoint}"
    headers = {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY
    }

    try:
        logger.info(f"Querying API-Football: {url} with params {params}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        errors = data.get("errors", [])
        if errors:
            logger.error(f"API-Football returned errors: {errors}")
            return {}
            
        return data
    except Exception as e:
        logger.error(f"API-Football query failed for {endpoint}: {str(e)}")
        return {}


def get_standings(bypass_cache: bool = False) -> dict:
    """
    Fetch standings for World Cup 2026 (League=1, Season=2026).
    Checks cache first, then API. Generates mock data if both fail.
    """
    _ensure_cache_dir()

    if not bypass_cache and STANDINGS_CACHE.exists():
        try:
            logger.info("Loading standings from local cache.")
            with open(STANDINGS_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read standings cache: {str(e)}")

    # Fetch from API
    api_data = fetch_from_api("standings", {"league": 1, "season": 2026})
    
    if api_data and api_data.get("response"):
        try:
            with open(STANDINGS_CACHE, "w", encoding="utf-8") as f:
                json.dump(api_data, f, indent=4)
            logger.info("Saved standings from API to cache.")
            return api_data
        except Exception as e:
            logger.error(f"Failed to write standings cache: {str(e)}")
            return api_data

    # Fallback to Mock
    logger.info("API standings fetch failed. Utilizing default mock standings.")
    mock_standings = _generate_mock_standings()
    
    if not STANDINGS_CACHE.exists():
        try:
            with open(STANDINGS_CACHE, "w", encoding="utf-8") as f:
                json.dump(mock_standings, f, indent=4)
        except Exception:
            pass

    return mock_standings


def get_fixtures(bypass_cache: bool = False) -> dict:
    """
    Fetch fixtures for World Cup 2026 (League=1, Season=2026).
    Checks cache first, then API. Generates mock data if both fail.
    """
    _ensure_cache_dir()

    if not bypass_cache and FIXTURES_CACHE.exists():
        try:
            logger.info("Loading fixtures from local cache.")
            with open(FIXTURES_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read fixtures cache: {str(e)}")

    # Fetch from API
    api_data = fetch_from_api("fixtures", {"league": 1, "season": 2026})
    
    if api_data and api_data.get("response"):
        try:
            with open(FIXTURES_CACHE, "w", encoding="utf-8") as f:
                json.dump(api_data, f, indent=4)
            logger.info("Saved fixtures from API to cache.")
            return api_data
        except Exception as e:
            logger.error(f"Failed to write fixtures cache: {str(e)}")
            return api_data

    # Fallback to Mock
    logger.info("API fixtures fetch failed. Utilizing default mock fixtures.")
    mock_fixtures = _generate_mock_fixtures()
    
    if not FIXTURES_CACHE.exists():
        try:
            with open(FIXTURES_CACHE, "w", encoding="utf-8") as f:
                json.dump(mock_fixtures, f, indent=4)
        except Exception:
            pass

    return mock_fixtures

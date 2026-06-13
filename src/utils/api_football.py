"""
api_football.py

Responsibility: Handles all API calls to API-Football (via RapidAPI) or
Football-Data.org, caching response JSONs locally and providing mock fallbacks
if offline, no credentials are set, or request limits are reached.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT
from src.data.clean import clean_team_name

logger = get_logger(__name__)

# Load local environment variables
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
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


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if the cache file exists and has not expired according to the configured TTL."""
    if not cache_path.exists():
        return False
    from src.utils.config import config
    ttl_hours = float(config.get("api", {}).get("cache_ttl_hours", 2.0))
    mtime = cache_path.stat().st_mtime
    return (time.time() - mtime) <= (ttl_hours * 3600)


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


def _stable_team_id(team_name: str) -> int:
    return 1000 + (sum(ord(ch) for ch in team_name) % 1000)


def _group_from_fixture(item: dict, home: str, away: str) -> str:
    round_str = item.get("league", {}).get("round", "")
    if "Group " in round_str:
        return "Group " + round_str.split("Group ")[-1].strip()

    for group_name, teams in GROUPS.items():
        if home in teams and away in teams:
            return f"Group {group_name}"
    return "Group Stage"


def _empty_standing_row(team: str, group_name: str) -> dict:
    return {
        "rank": 1,
        "team": {
            "id": _stable_team_id(team),
            "name": team,
            "logo": f"https://media.api-sports.io/football/teams/{_stable_team_id(team)}.png"
        },
        "points": 0,
        "goalsDiff": 0,
        "group": group_name,
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
        "update": datetime.now().isoformat()
    }


def _apply_result(row: dict, goals_for: int, goals_against: int) -> None:
    row["all"]["played"] += 1
    row["all"]["goals"]["for"] += goals_for
    row["all"]["goals"]["against"] += goals_against
    row["goalsDiff"] = row["all"]["goals"]["for"] - row["all"]["goals"]["against"]

    if goals_for > goals_against:
        row["all"]["win"] += 1
        row["points"] += 3
        row["form"] += "W"
    elif goals_for < goals_against:
        row["all"]["lose"] += 1
        row["form"] += "L"
    else:
        row["all"]["draw"] += 1
        row["points"] += 1
        row["form"] += "D"


def _generate_standings_from_fixtures(fixtures_data: dict) -> dict:
    """
    Build group standings from finished fixture results when the dedicated
    standings API is unavailable, stale, or not populated yet.
    """
    response = fixtures_data.get("response", []) if fixtures_data else []
    if not response:
        return {}

    standings_by_group = {}
    for group_name, teams in GROUPS.items():
        display_group = f"Group {group_name}"
        standings_by_group[display_group] = {
            team: _empty_standing_row(team, display_group)
            for team in teams
        }

    finished_count = 0
    for item in response:
        fixture = item.get("fixture", {})
        status_short = fixture.get("status", {}).get("short", "").upper()
        if status_short not in {"FT", "AET", "PEN"}:
            continue

        teams = item.get("teams", {})
        home = clean_team_name(teams.get("home", {}).get("name"))
        away = clean_team_name(teams.get("away", {}).get("name"))
        goals = item.get("goals", {})
        home_goals = goals.get("home")
        away_goals = goals.get("away")
        if not home or not away or home_goals is None or away_goals is None:
            continue

        group_name = _group_from_fixture(item, home, away)
        standings_by_group.setdefault(group_name, {})
        standings_by_group[group_name].setdefault(home, _empty_standing_row(home, group_name))
        standings_by_group[group_name].setdefault(away, _empty_standing_row(away, group_name))

        _apply_result(standings_by_group[group_name][home], int(home_goals), int(away_goals))
        _apply_result(standings_by_group[group_name][away], int(away_goals), int(home_goals))
        finished_count += 1

    if finished_count == 0:
        return {}

    league_obj = {
        "id": 1,
        "name": "World Cup",
        "country": "World",
        "logo": "https://media.api-sports.io/football/leagues/1.png",
        "flag": None,
        "season": 2026,
        "standings": []
    }

    for group_name in sorted(standings_by_group):
        group_rows = list(standings_by_group[group_name].values())
        group_rows.sort(
            key=lambda row: (
                -row["points"],
                -row["goalsDiff"],
                -row["all"]["goals"]["for"],
                row["team"]["name"],
            )
        )
        for rank, row in enumerate(group_rows, start=1):
            row["rank"] = rank
        league_obj["standings"].append(group_rows)

    return {"response": [{"league": league_obj}], "errors": [], "results": 1}


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


def fetch_from_football_data(endpoint: str) -> dict:
    """
    Low-level query to Football-Data.org API.
    """
    if not FOOTBALL_DATA_API_KEY:
        return {}

    url = f"https://api.football-data.org/v4/{endpoint}"
    headers = {
        "X-Auth-Token": FOOTBALL_DATA_API_KEY
    }

    try:
        logger.info(f"Querying Football-Data.org: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Football-Data.org query failed for {endpoint}: {str(e)}")
        return {}


def map_football_data_standings(fd_data: dict) -> dict:
    """
    Map Football-Data.org standings response into API-Football format.
    Normalises team names to the model's canonical mapping.
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

    fd_standings = fd_data.get("standings", [])
    for group_data in fd_standings:
        # Standard standings filters
        if group_data.get("type") != "TOTAL" or group_data.get("stage") != "GROUP_STAGE":
            continue

        group_raw = group_data.get("group", "")
        group_name = "Group " + group_raw.split("GROUP_")[-1] if "GROUP_" in group_raw else group_raw

        group_standings = []
        for row in group_data.get("table", []):
            team_info = row.get("team", {})
            raw_team_name = team_info.get("name", "Unknown")
            clean_name = clean_team_name(raw_team_name)

            group_standings.append({
                "rank": row.get("position", 1),
                "team": {
                    "id": team_info.get("id", 0),
                    "name": clean_name,
                    "logo": team_info.get("crest", "")
                },
                "points": row.get("points", 0),
                "goalsDiff": row.get("goalDifference", 0),
                "group": group_name,
                "form": "",
                "status": "same",
                "description": "Possible Qualification",
                "all": {
                    "played": row.get("playedGames", 0),
                    "win": row.get("won", 0),
                    "draw": row.get("draw", 0),
                    "lose": row.get("lost", 0),
                    "goals": {
                        "for": row.get("goalsFor", 0),
                        "against": row.get("goalsAgainst", 0)
                    }
                },
                "update": datetime.now().isoformat()
            })
        league_obj["standings"].append(group_standings)

    response_data.append({"league": league_obj})
    return {"response": response_data, "errors": [], "results": len(response_data)}


def map_football_data_fixtures(fd_data: dict) -> dict:
    """
    Map Football-Data.org matches response into API-Football fixtures format.
    Normalises team names to the model's canonical mapping.
    """
    response_fixtures = []
    
    fd_matches = fd_data.get("matches", [])
    for m in fd_matches:
        fixture_date = m.get("utcDate", "")
        status_raw = m.get("status", "SCHEDULED").upper()
        
        status_short = "NS"
        status_long = "Not Started"
        if status_raw == "FINISHED":
            status_short = "FT"
            status_long = "Finished"
        elif status_raw in ["IN_PLAY", "PAUSED", "LIVE"]:
            status_short = "LIVE"
            status_long = "In Play"

        group_raw = m.get("group", "")
        group_name = "Group Stage"
        if group_raw:
            group_name = "Group Stage - Group " + (group_raw.split("GROUP_")[-1] if "GROUP_" in group_raw else group_raw)

        home_team = m.get("homeTeam", {})
        away_team = m.get("awayTeam", {})
        clean_home = clean_team_name(home_team.get("name", "Unknown"))
        clean_away = clean_team_name(away_team.get("name", "Unknown"))
        
        score = m.get("score", {})
        full_time = score.get("fullTime", {})

        # Compute timestamp safely
        timestamp = 0
        if fixture_date:
            try:
                dt_str = fixture_date.replace("Z", "+00:00")
                timestamp = int(datetime.fromisoformat(dt_str).timestamp())
            except Exception:
                pass

        response_fixtures.append({
            "fixture": {
                "id": m.get("id", 0),
                "referee": "TBD",
                "timezone": "UTC",
                "date": fixture_date,
                "timestamp": timestamp,
                "periods": {
                    "first": None,
                    "second": None
                },
                "venue": {
                    "id": None,
                    "name": "TBD",
                    "city": "TBD"
                },
                "status": {
                    "long": status_long,
                    "short": status_short,
                    "elapsed": 0
                }
            },
            "league": {
                "id": 1,
                "name": "World Cup",
                "country": "World",
                "logo": "https://media.api-sports.io/football/leagues/1.png",
                "flag": None,
                "season": 2026,
                "round": group_name
            },
            "teams": {
                "home": {
                    "id": home_team.get("id", 0),
                    "name": clean_home,
                    "logo": home_team.get("crest", ""),
                    "winner": None
                },
                "away": {
                    "id": away_team.get("id", 0),
                    "name": clean_away,
                    "logo": away_team.get("crest", ""),
                    "winner": None
                }
            },
            "goals": {
                "home": full_time.get("home"),
                "away": full_time.get("away")
            },
            "score": {
                "fulltime": {
                    "home": full_time.get("home"),
                    "away": full_time.get("away")
                }
            }
        })

    return {"response": response_fixtures, "errors": [], "results": len(response_fixtures)}


def get_standings(bypass_cache: bool = False) -> dict:
    """
    Fetch standings for World Cup 2026.
    Checks cache first, then Football-Data.org (if key exists) or API-Football.
    Generates mock data if all credentials or calls fail or return empty.
    """
    _ensure_cache_dir()

    fixture_standings = _generate_standings_from_fixtures(get_fixtures())
    if fixture_standings:
        try:
            with open(STANDINGS_CACHE, "w", encoding="utf-8") as f:
                json.dump(fixture_standings, f, indent=4)
            logger.info("Saved standings derived from finished fixtures to cache.")
        except Exception as e:
            logger.error(f"Failed to write derived standings cache: {str(e)}")
        return fixture_standings

    if not bypass_cache and _is_cache_valid(STANDINGS_CACHE):
        try:
            logger.info("Loading standings from valid local cache.")
            with open(STANDINGS_CACHE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                response = cached_data.get("response", [])
                if response:
                    standings = response[0].get("league", {}).get("standings", [])
                    if standings and any(len(group) > 0 for group in standings):
                        return cached_data
        except Exception as e:
            logger.error(f"Failed to read standings cache: {str(e)}")

    # Try Football-Data.org API if key configured
    if FOOTBALL_DATA_API_KEY:
        fd_data = fetch_from_football_data("competitions/WC/standings")
        if fd_data and fd_data.get("standings"):
            mapped_data = map_football_data_standings(fd_data)
            response = mapped_data.get("response", [])
            if response:
                standings = response[0].get("league", {}).get("standings", [])
                if standings and any(len(group) > 0 for group in standings):
                    try:
                        with open(STANDINGS_CACHE, "w", encoding="utf-8") as f:
                            json.dump(mapped_data, f, indent=4)
                        logger.info("Saved standings from Football-Data.org to cache.")
                        return mapped_data
                    except Exception as e:
                        logger.error(f"Failed to write standings cache: {str(e)}")
                        return mapped_data

    # Fetch from API-Football
    api_data = fetch_from_api("standings", {"league": 1, "season": 2026})
    
    if api_data and api_data.get("response"):
        response = api_data.get("response", [])
        if response:
            standings = response[0].get("league", {}).get("standings", [])
            if standings and any(len(group) > 0 for group in standings):
                try:
                    with open(STANDINGS_CACHE, "w", encoding="utf-8") as f:
                        json.dump(api_data, f, indent=4)
                    logger.info("Saved standings from API to cache.")
                    return api_data
                except Exception as e:
                    logger.error(f"Failed to write standings cache: {str(e)}")
                    return api_data

    # Fallback to expired cache first if it exists
    if STANDINGS_CACHE.exists():
        try:
            logger.warning("API standings fetch failed. Utilizing expired standings cache.")
            with open(STANDINGS_CACHE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                response = cached_data.get("response", [])
                if response:
                    standings = response[0].get("league", {}).get("standings", [])
                    if standings and any(len(group) > 0 for group in standings):
                        return cached_data
        except Exception as e:
            logger.error(f"Failed to load expired standings cache: {str(e)}")

    # Fallback to Mock
    logger.info("API standings fetch failed and no cache available. Utilizing default mock standings.")
    mock_standings = _generate_mock_standings()
    
    try:
        with open(STANDINGS_CACHE, "w", encoding="utf-8") as f:
            json.dump(mock_standings, f, indent=4)
    except Exception:
        pass

    return mock_standings




def get_fixtures(bypass_cache: bool = False) -> dict:
    """
    Fetch fixtures for World Cup 2026.
    Checks cache first, then Football-Data.org (if key exists) or API-Football.
    Generates mock data if all credentials or calls fail.
    """
    _ensure_cache_dir()

    if not bypass_cache and _is_cache_valid(FIXTURES_CACHE):
        try:
            logger.info("Loading fixtures from valid local cache.")
            with open(FIXTURES_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read fixtures cache: {str(e)}")

    # Try Football-Data.org API if key configured
    if FOOTBALL_DATA_API_KEY:
        fd_data = fetch_from_football_data("competitions/WC/matches")
        if fd_data and fd_data.get("matches"):
            mapped_data = map_football_data_fixtures(fd_data)
            try:
                with open(FIXTURES_CACHE, "w", encoding="utf-8") as f:
                    json.dump(mapped_data, f, indent=4)
                logger.info("Saved fixtures from Football-Data.org to cache.")
                return mapped_data
            except Exception as e:
                logger.error(f"Failed to write fixtures cache: {str(e)}")
                return mapped_data

    # Fetch from API-Football
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

    # Fallback to expired cache first if it exists
    if FIXTURES_CACHE.exists():
        try:
            logger.warning("API fixtures fetch failed. Utilizing expired fixtures cache.")
            with open(FIXTURES_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load expired fixtures cache: {str(e)}")

    # Fallback to Mock
    logger.info("API fixtures fetch failed and no cache available. Utilizing default mock fixtures.")
    mock_fixtures = _generate_mock_fixtures()
    
    try:
        with open(FIXTURES_CACHE, "w", encoding="utf-8") as f:
            json.dump(mock_fixtures, f, indent=4)
    except Exception:
        pass

    return mock_fixtures

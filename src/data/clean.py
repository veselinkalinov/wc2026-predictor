"""
clean.py

Responsibility: Clean and preprocess raw data files.
Creates data/processed/matches_clean.csv containing cleaned matches
merged with FIFA ranking snapshots.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping dictionary to normalise team names across matches.csv and fifa_rankings.csv
TEAM_MAPPING = {
    "USA": "United States",
    "US Virgin Islands": "U.S. Virgin Islands",
    "United States Virgin Islands": "U.S. Virgin Islands",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Zaire": "DR Congo",
    "Chinese Taipei": "Taiwan",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St. Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Brunei Darussalam": "Brunei",
    "Swaziland": "Eswatini",
    "The Gambia": "Gambia",
    "Ireland": "Republic of Ireland",
    "East Timor": "Timor-Leste",
    "Curacao": "Curaçao",
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def clean_team_name(name):
    if not isinstance(name, str):
        return name
    # Clean whitespace and non-breaking spaces
    name = name.replace("\xa0", " ").strip()
    # Map to canonical name if it exists in the dictionary
    return TEAM_MAPPING.get(name, name)


def run_cleaning() -> None:
    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading raw datasets...")
    matches = pd.read_csv(raw_dir / config["data"]["matches_file"])
    rankings = pd.read_csv(raw_dir / config["data"]["rankings_file"])

    # 2. Parse dates
    logger.info("Parsing dates...")
    matches["date"] = pd.to_datetime(matches["date"])

    # Construct rankings date: Sem 1 -> Jan 1, Sem 2 -> Jul 1
    rankings["month"] = rankings["semester"].apply(
        lambda s: 1 if s == 1 else 7)
    rankings["date"] = pd.to_datetime(
        rankings["date"].astype(str) + "-" +
        rankings["month"].astype(str) + "-01"
    )
    rankings = rankings.drop(columns=["month", "semester"])

    # 3. Clean and Normalise Team Names
    logger.info("Normalising team names...")
    matches["home_team"] = matches["home_team"].apply(clean_team_name)
    matches["away_team"] = matches["away_team"].apply(clean_team_name)
    rankings["team"] = rankings["team"].apply(clean_team_name)

    # Remove rows with null scores in matches
    initial_matches = len(matches)
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    dropped_scores = initial_matches - len(matches)
    if dropped_scores > 0:
        logger.info(f"Dropped {dropped_scores} matches with null scores.")

    # 4. Compute target result column
    # H = Home Win, A = Away Win, D = Draw
    logger.info("Computing match outcomes...")
    matches["result"] = "D"
    matches.loc[matches["home_score"] > matches["away_score"], "result"] = "H"
    matches.loc[matches["away_score"] > matches["home_score"], "result"] = "A"

    # Add binary feature is_competitive
    # Competitive matches include tournament games/qualifiers, excluding friendlies
    matches["is_competitive"] = ~matches["tournament"].str.contains(
        "friendly", case=False, na=False
    )
    matches["is_competitive"] = matches["is_competitive"].astype(int)

    # 5. Merge rankings snapshot using merge_asof
    # merge_asof requires both dataframes to be sorted by date
    logger.info("Merging FIFA rankings snapshot...")
    matches = matches.sort_values("date")
    rankings = rankings.sort_values("date")

    # Rename ranking columns to avoid conflicts
    rankings_home = rankings.rename(
        columns={"team": "home_team", "rank": "home_rank",
                 "total.points": "home_rank_points"}
    )[["date", "home_team", "home_rank", "home_rank_points"]]

    rankings_away = rankings.rename(
        columns={"team": "away_team", "rank": "away_rank",
                 "total.points": "away_rank_points"}
    )[["date", "away_team", "away_rank", "away_rank_points"]]

    # Merge home rankings
    matches = pd.merge_asof(
        matches,
        rankings_home,
        on="date",
        by="home_team",
        direction="backward",  # match the most recent ranking before or on the match date
    )

    # Merge away rankings
    matches = pd.merge_asof(
        matches,
        rankings_away,
        on="date",
        by="away_team",
        direction="backward",
    )

    # 6. Handle missing rankings for non-FIFA teams
    # Fallback to lowest FIFA rank (211) and 0 points to prevent NaNs
    logger.info("Handling missing rankings with fallbacks...")
    matches["home_rank"] = matches["home_rank"].fillna(211.0)
    matches["home_rank_points"] = matches["home_rank_points"].fillna(0.0)
    matches["away_rank"] = matches["away_rank"].fillna(211.0)
    matches["away_rank_points"] = matches["away_rank_points"].fillna(0.0)

    # 7. Apply configuration filters
    # Filter by date range (from config: 2000-01-01 to 2025-12-31)
    date_from = pd.to_datetime(config["data"]["date_from"])
    date_to = pd.to_datetime(config["data"]["date_to"])

    matches = matches[(matches["date"] >= date_from) &
                      (matches["date"] <= date_to)].copy()
    logger.info(
        f"Filtered date range ({date_from.date()} to {date_to.date()}). Matches: {len(matches)}")

    # Filter teams with minimum matches (config: min_matches = 10)
    min_matches = config["data"]["min_matches"]
    team_counts = pd.concat(
        [matches["home_team"], matches["away_team"]]).value_counts()
    valid_teams = team_counts[team_counts >= min_matches].index

    matches = matches[
        matches["home_team"].isin(
            valid_teams) & matches["away_team"].isin(valid_teams)
    ].copy()
    logger.info(
        f"Filtered teams with < {min_matches} matches. Remaining: {len(matches)}")

    # 8. Save cleaned matches
    out_path = processed_dir / "matches_clean.csv"
    matches.to_csv(out_path, index=False)
    logger.info(f"Saved cleaned matches to {out_path}. Shape: {matches.shape}")


if __name__ == "__main__":
    run_cleaning()

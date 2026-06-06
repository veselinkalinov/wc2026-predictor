"""
build.py

Responsibility: Run all feature generators and assemble the final feature matrix.
Saves the output to data/features/feature_matrix.csv.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger
from src.features.elo import compute_elo_ratings
from src.features.form import compute_form_features
from src.features.goals import compute_goal_features

logger = get_logger(__name__)


def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing rest days, travel continent, and match stake features...")
    # Sort chronologically
    df = df.sort_values("date").copy()
    
    # 1. Rest days calculation
    last_match_date = {}
    home_rest_days = []
    away_rest_days = []
    
    for idx, row in df.iterrows():
        date = row["date"]
        h_team = row["home_team"]
        a_team = row["away_team"]
        
        # Home team rest
        if h_team in last_match_date:
            h_rest = (date - last_match_date[h_team]).days
        else:
            h_rest = 30 # Default
        
        # Away team rest
        if a_team in last_match_date:
            a_rest = (date - last_match_date[a_team]).days
        else:
            a_rest = 30 # Default
            
        # Clamp to max 30 days
        h_rest = min(h_rest, 30)
        a_rest = min(a_rest, 30)
        
        home_rest_days.append(h_rest)
        away_rest_days.append(a_rest)
        
        # Update last match dates
        last_match_date[h_team] = date
        last_match_date[a_team] = date
        
    df["home_rest_days"] = home_rest_days
    df["away_rest_days"] = away_rest_days
    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
    
    # 2. Travel Fatigue / Continent
    CONTINENT_MAP = {
        # Europe (UEFA)
        "Germany": "Europe", "France": "Europe", "England": "Europe", "Italy": "Europe", "Spain": "Europe",
        "Netherlands": "Europe", "Portugal": "Europe", "Belgium": "Europe", "Croatia": "Europe", "Denmark": "Europe",
        "Sweden": "Europe", "Switzerland": "Europe", "Poland": "Europe", "Austria": "Europe", "Ukraine": "Europe",
        "Turkey": "Europe", "Russia": "Europe", "Wales": "Europe", "Scotland": "Europe", "Republic of Ireland": "Europe",
        # South America (CONMEBOL)
        "Brazil": "South America", "Argentina": "South America", "Uruguay": "South America", "Colombia": "South America",
        "Chile": "South America", "Peru": "South America", "Ecuador": "South America", "Paraguay": "South America",
        "Venezuela": "South America", "Bolivia": "South America",
        # North/Central America (CONCACAF)
        "United States": "North America", "Mexico": "North America", "Canada": "North America", "Costa Rica": "North America",
        "Jamaica": "North America", "Honduras": "North America", "Panama": "North America", "El Salvador": "North America",
        # Africa (CAF)
        "Senegal": "Africa", "Morocco": "Africa", "Algeria": "Africa", "Nigeria": "Africa", "Egypt": "Africa",
        "Cameroon": "Africa", "Ghana": "Africa", "Ivory Coast": "Africa", "Tunisia": "Africa", "Mali": "Africa",
        # Asia (AFC)
        "Japan": "Asia", "South Korea": "Asia", "Iran": "Asia", "Australia": "Asia", "Saudi Arabia": "Asia",
        "Qatar": "Asia", "Iraq": "Asia", "United Arab Emirates": "Asia", "China PR": "Asia",
        # Oceania (OFC)
        "New Zealand": "Oceania"
    }
    
    home_is_home_continent = []
    away_is_home_continent = []
    
    for idx, row in df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        country = row.get("country", "")
        
        h_cont = CONTINENT_MAP.get(h_team)
        a_cont = CONTINENT_MAP.get(a_team)
        m_cont = CONTINENT_MAP.get(country) if isinstance(country, str) else None
        
        if h_cont is not None and m_cont is not None and h_cont == m_cont:
            home_is_home_continent.append(1)
        else:
            home_is_home_continent.append(0)
            
        if a_cont is not None and m_cont is not None and a_cont == m_cont:
            away_is_home_continent.append(1)
        else:
            away_is_home_continent.append(0)
            
    df["home_is_home_continent"] = home_is_home_continent
    df["away_is_home_continent"] = away_is_home_continent
    df["continent_diff"] = df["home_is_home_continent"] - df["away_is_home_continent"]
    
    # 3. Match Stake / Tier
    def get_match_stake(tourn):
        t = str(tourn).lower()
        if t == "fifa world cup":
            return 4
        elif any(comp in t for comp in ["uefa euro", "copa américa", "african cup of nations",
                                        "afc asian cup", "concacaf gold cup",
                                        "confederations cup"]):
            return 3
        elif "qualification" in t or "nations league" in t:
            return 2
        else:
            return 1
            
    df["match_stake"] = df["tournament"].apply(get_match_stake)
    
    return df


def build_feature_matrix() -> None:
    processed_dir = Path(config["paths"]["processed_data"])
    features_dir = Path(config["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "matches_clean.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned matches file not found at {input_path}. Run clean.py first."
        )

    # 1. Load cleaned matches
    logger.info(f"Loading cleaned matches from {input_path}...")
    df = pd.read_csv(input_path)

    # Convert date to datetime for correct chronological sorting inside feature modules
    df["date"] = pd.to_datetime(df["date"])

    # 2. Chain feature calculators
    df, _ = compute_elo_ratings(df)
    df = compute_form_features(df)
    df = compute_goal_features(df)
    df = compute_advanced_features(df)

    # 3. Add ranking differences
    logger.info("Computing ranking difference features...")
    df["rank_diff"] = df["home_rank"] - df["away_rank"]
    df["rank_points_diff"] = df["home_rank_points"] - df["away_rank_points"]

    # 4. Standardize boolean features
    df["is_neutral"] = df["neutral"].astype(int)

    # 4b. Apply configuration filters (deferred from clean.py to avoid Elo cold-start)
    # Filter by date range
    date_from = pd.to_datetime(config["data"]["date_from"])
    date_to = pd.to_datetime(config["data"]["date_to"])
    df = df[(df["date"] >= date_from) & (df["date"] <= date_to)].copy()
    logger.info(f"Filtered date range ({date_from.date()} to {date_to.date()}). Matches: {len(df)}")

    # Filter teams with minimum matches
    min_matches = config["data"]["min_matches"]
    team_counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    valid_teams = team_counts[team_counts >= min_matches].index
    df = df[df["home_team"].isin(valid_teams) & df["away_team"].isin(valid_teams)].copy()
    logger.info(f"Filtered teams with < {min_matches} matches. Remaining: {len(df)}")

    # Check for NaNs
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if len(columns_with_nulls) > 0:
        logger.warning(
            f"NaN values found in compiled DataFrame: \n{columns_with_nulls}")
        # Drop rows with null features if any slipped through
        df = df.dropna().copy()
        logger.info(f"Dropped rows with NaNs. Remaining matches: {len(df)}")

    # 5. Save the final feature matrix
    out_path = features_dir / "feature_matrix.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved feature matrix to {out_path}. Shape: {df.shape}")


if __name__ == "__main__":
    build_feature_matrix()

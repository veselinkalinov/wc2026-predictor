"""
test_clean.py

Unit tests for data cleaning functions in src/data/clean.py.
"""

import pandas as pd
from src.data.clean import clean_team_name, TEAM_MAPPING


def test_clean_team_name():
    # Test normalization of known mapping values
    assert clean_team_name("USA") == "United States"
    assert clean_team_name("Korea Republic") == "South Korea"
    assert clean_team_name("Côte d'Ivoire") == "Ivory Coast"

    # Test handling of non-breaking space (Unicode \xa0)
    assert clean_team_name("Brazil\xa0") == "Brazil"
    assert clean_team_name("\xa0Argentina\xa0") == "Argentina"

    # Test team that is not in the mapping dictionary (should remain unchanged)
    assert clean_team_name("Germany") == "Germany"


def test_match_outcome_and_competitive_flag():
    # We will simulate a small matches dataframe to test outcomes and competitive flags
    data = {
        "date": ["2024-06-01", "2024-06-02", "2024-06-03"],
        "home_team": ["Germany", "Brazil", "France"],
        "away_team": ["Spain", "Argentina", "England"],
        "home_score": [2, 1, 1],
        "away_score": [1, 1, 3],
        "tournament": ["UEFA Euro", "Friendly", "FIFA World Cup"],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Deriving outcomes manually to verify logic
    df["result"] = "D"
    df.loc[df["home_score"] > df["away_score"], "result"] = "H"
    df.loc[df["away_score"] > df["home_score"], "result"] = "A"

    df["is_competitive"] = ~df["tournament"].str.contains(
        "friendly", case=False, na=False
    )
    df["is_competitive"] = df["is_competitive"].astype(int)

    # Outcomes checks
    assert df.loc[0, "result"] == "H"  # Germany 2 - 1 Spain
    assert df.loc[1, "result"] == "D"  # Brazil 1 - 1 Argentina
    assert df.loc[2, "result"] == "A"  # France 1 - 3 England

    # Competitive flag checks
    assert df.loc[0, "is_competitive"] == 1  # UEFA Euro is competitive
    assert df.loc[1, "is_competitive"] == 0  # Friendly is not competitive
    assert df.loc[2, "is_competitive"] == 1  # FIFA World Cup is competitive

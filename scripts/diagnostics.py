"""
Diagnostic script: team name overlap + Elo coverage analysis.
Runs the checks that were left incomplete in the Claude conversation.
"""
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

print("=" * 70)
print("DIAGNOSTIC 1: Dataset shapes and basic info")
print("=" * 70)

matches = pd.read_csv(RAW / "matches.csv")
rankings = pd.read_csv(RAW / "fifa_rankings.csv")
elo = pd.read_csv(RAW / "elo_ratings.csv")

print(f"matches.csv:       {matches.shape[0]:>6} rows x {matches.shape[1]} cols")
print(f"fifa_rankings.csv:  {rankings.shape[0]:>6} rows x {rankings.shape[1]} cols")
print(f"elo_ratings.csv:    {elo.shape[0]:>6} rows x {elo.shape[1]} cols")

print(f"\nmatches columns: {list(matches.columns)}")
print(f"rankings columns: {list(rankings.columns)}")
print(f"elo columns: {list(elo.columns)}")

# Post-2000 matches
matches["date"] = pd.to_datetime(matches["date"])
post2000 = matches[matches["date"] >= "2000-01-01"]
print(f"\nPost-2000 matches: {len(post2000)}")

# Outcome distribution
if "home_score" in matches.columns and "away_score" in matches.columns:
    valid = post2000.dropna(subset=["home_score", "away_score"])
    results = []
    for _, r in valid.iterrows():
        if r["home_score"] > r["away_score"]:
            results.append("H")
        elif r["away_score"] > r["home_score"]:
            results.append("A")
        else:
            results.append("D")
    from collections import Counter
    dist = Counter(results)
    total = len(results)
    print(f"Outcome distribution (post-2000, {total} matches with scores):")
    for k in ["H", "D", "A"]:
        print(f"  {k}: {dist[k]:>5} ({100*dist[k]/total:.1f}%)")

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Team name overlap analysis")
print("=" * 70)

# Get unique teams from each dataset
match_teams = set(matches["home_team"].unique()) | set(matches["away_team"].unique())
rank_teams = set(rankings["team"].unique()) if "team" in rankings.columns else set()

# Clean elo team names (strip non-breaking spaces)
elo["team"] = elo["team"].str.replace("\xa0", " ").str.strip()
elo_teams = set(elo["team"].unique())

print(f"Unique teams in matches:  {len(match_teams)}")
print(f"Unique teams in rankings: {len(rank_teams)}")
print(f"Unique teams in elo:      {len(elo_teams)}")

# Overlaps
in_matches_not_rankings = match_teams - rank_teams
in_matches_not_elo = match_teams - elo_teams
in_rankings_not_matches = rank_teams - match_teams
in_elo_not_matches = elo_teams - match_teams

print(f"\nTeams in matches but NOT in rankings: {len(in_matches_not_rankings)}")
print(f"Teams in matches but NOT in elo:      {len(in_matches_not_elo)}")
print(f"Teams in rankings but NOT in matches:  {len(in_rankings_not_matches)}")
print(f"Teams in elo but NOT in matches:        {len(in_elo_not_matches)}")

# Post-2000 teams only
post2000_teams = set(post2000["home_team"].unique()) | set(post2000["away_team"].unique())
post2000_not_rankings = post2000_teams - rank_teams
post2000_not_elo = post2000_teams - elo_teams

print(f"\nPost-2000 teams: {len(post2000_teams)}")
print(f"Post-2000 teams NOT in rankings: {len(post2000_not_rankings)}")
print(f"Post-2000 teams NOT in elo:      {len(post2000_not_elo)}")

# Show the missing teams (sorted)
print(f"\n--- Post-2000 teams missing from rankings ({len(post2000_not_rankings)}) ---")
for t in sorted(post2000_not_rankings):
    print(f"  {repr(t)}")

print(f"\n--- Post-2000 teams missing from elo ({len(post2000_not_elo)}) ---")
for t in sorted(post2000_not_elo):
    print(f"  {repr(t)}")

# Show ranking teams not in matches (potential name variants)
print(f"\n--- Ranking teams NOT in matches ({len(in_rankings_not_matches)}) ---")
for t in sorted(in_rankings_not_matches):
    print(f"  {repr(t)}")

print(f"\n--- Elo teams NOT in matches ({len(in_elo_not_matches)}) ---")
for t in sorted(in_elo_not_matches):
    print(f"  {repr(t)}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Elo coverage analysis")
print("=" * 70)

elo["date"] = pd.to_datetime(elo["date"], dayfirst=True, errors="coerce")
print(f"Elo date range: {elo['date'].min()} to {elo['date'].max()}")
print(f"Elo null dates: {elo['date'].isna().sum()}")
print(f"Elo null ratings: {elo['rating'].isna().sum()}")
print(f"Elo unique teams: {elo['team'].nunique()}")

# How many post-2000 match-team pairs could be covered by elo?
elo_post2000 = elo[elo["date"] >= "2000-01-01"]
print(f"Elo rows post-2000: {len(elo_post2000)}")
print(f"Elo unique teams post-2000: {elo_post2000['team'].nunique()}")

# Coverage: for each post-2000 match, can we find elo for BOTH teams?
# (Approximate: check if team exists in elo at all)
elo_available_teams = set(elo_post2000["team"].unique())
both_covered = 0
home_only = 0
away_only = 0
neither = 0
for _, row in post2000.iterrows():
    h = row["home_team"] in elo_available_teams
    a = row["away_team"] in elo_available_teams
    if h and a:
        both_covered += 1
    elif h:
        home_only += 1
    elif a:
        away_only += 1
    else:
        neither += 1

print(f"\nPost-2000 match Elo coverage (team exists in elo, ignoring date proximity):")
print(f"  Both teams covered:   {both_covered:>5} ({100*both_covered/len(post2000):.1f}%)")
print(f"  Home only:            {home_only:>5} ({100*home_only/len(post2000):.1f}%)")
print(f"  Away only:            {away_only:>5} ({100*away_only/len(post2000):.1f}%)")
print(f"  Neither:              {neither:>5} ({100*neither/len(post2000):.1f}%)")

print("\n" + "=" * 70)
print("DIAGNOSTIC 4: FIFA Rankings date format check")
print("=" * 70)
print(f"Rankings 'date' sample values: {rankings['date'].head(10).tolist()}")
print(f"Rankings 'date' dtype: {rankings['date'].dtype}")
if "semester" in rankings.columns:
    print(f"Rankings 'semester' unique values: {sorted(rankings['semester'].unique())}")
print(f"Rankings date range: {rankings['date'].min()} to {rankings['date'].max()}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 5: Tournament types in matches")
print("=" * 70)
tourney_counts = post2000["tournament"].value_counts()
print(f"Tournament types (post-2000):")
for t, c in tourney_counts.items():
    is_friendly = "friendly" in t.lower()
    print(f"  {'[F]' if is_friendly else '   '} {t}: {c}")

friendly_count = post2000[post2000["tournament"].str.contains("friendly", case=False, na=False)].shape[0]
competitive_count = len(post2000) - friendly_count
print(f"\nFriendlies: {friendly_count} ({100*friendly_count/len(post2000):.1f}%)")
print(f"Competitive: {competitive_count} ({100*competitive_count/len(post2000):.1f}%)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

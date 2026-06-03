"""
simulate.py

Responsibility: Run Monte Carlo simulations of the FIFA World Cup 2026,
updating team states (Elo, form, goals) dynamically after every match.
"""

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from src.models.predict import MatchPredictor
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Official World Cup 2026 groups and team mappings
# Standardised to matches.csv team names!
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

HOSTS = {"United States", "Mexico", "Canada"}


class TournamentSimulator:
    def __init__(self):
        # We initialise the MatchPredictor to get model access and latest real-world stats
        self.predictor = MatchPredictor()

        # Save a backup of the real-world baseline team states
        self.baseline_states = json.loads(
            json.dumps(self.predictor.team_states))

    def _reset_states(self) -> None:
        """
        Reset predictor's team states to baseline before starting a new simulation run.
        """
        self.predictor.team_states = json.loads(
            json.dumps(self.baseline_states))

    def _update_stats_after_match(
        self, home_team: str, away_team: str, home_goals: int, away_goals: int, result: str
    ) -> None:
        """
        Dynamic State Update: Updates Elo, form, and goal averages for both teams
        in the predictor's state registry immediately after a simulated game.
        """
        k_factor = config["features"]["elo_k_factor"]
        form_window = config["features"]["form_window"]
        goals_window = config["features"]["goals_window"]

        # Get current states
        h_state = self.predictor.get_team_state(home_team).copy()
        a_state = self.predictor.get_team_state(away_team).copy()

        # --- 1. Update Elo ---
        expected_home = 1.0 / \
            (1.0 + 10.0 ** ((a_state["elo"] - h_state["elo"]) / 400.0))
        actual_home = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)

        h_state["elo"] = h_state["elo"] + \
            k_factor * (actual_home - expected_home)
        a_state["elo"] = a_state["elo"] + k_factor * \
            ((1.0 - actual_home) - (1.0 - expected_home))
        h_state["elo_diff"] = h_state["elo"] - a_state["elo"]
        a_state["elo_diff"] = a_state["elo"] - h_state["elo"]

        # --- 2. Update Form (approximate update via rolling factor) ---
        # Instead of storing full list histories in memory during MC, we update form
        # using a simple exponential smoothing factor equivalent to a rolling window.
        alpha_form = 1.0 / form_window
        h_pts = 3 if result == "H" else (1 if result == "D" else 0)
        a_pts = 3 if result == "A" else (1 if result == "D" else 0)

        h_state["form"] = (1.0 - alpha_form) * \
            h_state["form"] + alpha_form * (h_pts / 3.0)
        a_state["form"] = (1.0 - alpha_form) * \
            a_state["form"] + alpha_form * (a_pts / 3.0)

        # --- 3. Update Goals (approximate update) ---
        alpha_goals = 1.0 / goals_window
        h_state["goals_scored_avg"] = (
            1.0 - alpha_goals) * h_state["goals_scored_avg"] + alpha_goals * home_goals
        h_state["goals_conceded_avg"] = (
            1.0 - alpha_goals) * h_state["goals_conceded_avg"] + alpha_goals * away_goals
        h_state["goal_diff_avg"] = h_state["goals_scored_avg"] - \
            h_state["goals_conceded_avg"]

        a_state["goals_scored_avg"] = (
            1.0 - alpha_goals) * a_state["goals_scored_avg"] + alpha_goals * away_goals
        a_state["goals_conceded_avg"] = (
            1.0 - alpha_goals) * a_state["goals_conceded_avg"] + alpha_goals * home_goals
        a_state["goal_diff_avg"] = a_state["goals_scored_avg"] - \
            a_state["goals_conceded_avg"]

        # Save back into predictor
        self.predictor.update_team_state(home_team, h_state)
        self.predictor.update_team_state(away_team, a_state)

    def _simulate_match(self, team_a: str, team_b: str, is_knockout: bool = False) -> tuple:
        """
        Simulate a match between team A and team B.
        Returns: (home_goals, away_goals, winner)
        """
        # Determine host crowd advantage
        # If A is a host playing at home -> is_neutral = 0. If B is host -> swap them to Home and set is_neutral = 0.
        if team_a in HOSTS and team_b not in HOSTS:
            home_team, away_team = team_a, team_b
            is_neutral = 0
        elif team_b in HOSTS and team_a not in HOSTS:
            home_team, away_team = team_b, team_a
            is_neutral = 0
        else:
            home_team, away_team = team_a, team_b
            is_neutral = 1

        # Predict outcome probabilities
        pred = self.predictor.predict_match(
            home_team, away_team, is_neutral=is_neutral, is_competitive=1)
        probs = pred["probabilities"]

        # Draw outcome from prediction probabilities
        outcomes = ["H", "D", "A"]
        weights = [probs["home_win"], probs["draw"], probs["away_win"]]
        result = random.choices(outcomes, weights=weights)[0]

        # Draw goals using Poisson distributions based on team strengths
        h_state = self.predictor.get_team_state(home_team)
        a_state = self.predictor.get_team_state(away_team)

        # Expected goals lambda
        lambda_h = max(
            0.5, (h_state["goals_scored_avg"] + a_state["goals_conceded_avg"]) / 2.0)
        lambda_a = max(
            0.5, (a_state["goals_scored_avg"] + h_state["goals_conceded_avg"]) / 2.0)

        # Draw goal counts
        home_goals = np.random.poisson(lambda_h)
        away_goals = np.random.poisson(lambda_a)

        # Align goal counts with drawn result to avoid contradictions (e.g. result H but goals 1-2)
        if result == "H" and home_goals <= away_goals:
            home_goals = away_goals + random.randint(1, 2)
        elif result == "A" and away_goals <= home_goals:
            away_goals = home_goals + random.randint(1, 2)
        elif result == "D":
            home_goals = away_goals  # Equalise scores for draw

        # Update stats in predictor's temporary state
        self._update_stats_after_match(
            home_team, away_team, home_goals, away_goals, result)

        # Map back to original team names in case of host swap
        if result == "H":
            winner = home_team
        elif result == "A":
            winner = away_team
        else:
            winner = None

        if is_knockout and result == "D":
            # If draw in knockout, simulate penalty shootout (50/50 weight slightly adjusted by Elo)
            elo_a = self.predictor.get_team_state(team_a)["elo"]
            elo_b = self.predictor.get_team_state(team_b)["elo"]
            # Slightly favors stronger team
            prob_a_shootout = 0.5 + (elo_a - elo_b) / 2000.0
            prob_a_shootout = max(0.2, min(0.8, prob_a_shootout))

            winner = team_a if random.random() < prob_a_shootout else team_b

        return home_goals, away_goals, winner

    def simulate_group_stage(self) -> dict:
        """
        Simulate the group stage for all 12 groups.
        Returns: standings: {group_letter: pd.DataFrame of standings}
        """
        standings = {}

        for group_letter, teams in GROUPS.items():
            # Initialise group table
            table = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams}

            # Round robin matches (6 games per group)
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a, team_b = teams[i], teams[j]
                    gf_a, gf_b, winner = self._simulate_match(
                        team_a, team_b, is_knockout=False)

                    # Update table
                    table[team_a]["gf"] += gf_a
                    table[team_b]["gf"] += gf_b
                    table[team_a]["gd"] += (gf_a - gf_b)
                    table[team_b]["gd"] += (gf_b - gf_a)

                    if winner == team_a:
                        table[team_a]["pts"] += 3
                    elif winner == team_b:
                        table[team_b]["pts"] += 3
                    else:
                        table[team_a]["pts"] += 1
                        table[team_b]["pts"] += 1

            # Convert to DataFrame and sort by points -> GD -> GF
            group_df = pd.DataFrame.from_dict(
                table, orient="index").reset_index()
            group_df.columns = ["team", "pts", "gd", "gf"]
            group_df = group_df.sort_values(
                by=["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
            standings[group_letter] = group_df

        return standings

    def _determine_knockout_teams(self, standings: dict) -> list:
        """
        Collect group winners, runners-up, and 8 best 3rd-placed teams.
        """
        knockout_teams = []
        third_place_teams = []

        for group_letter, df in standings.items():
            # Top 2 advance directly
            knockout_teams.append(df.loc[0, "team"])  # Winner
            knockout_teams.append(df.loc[1, "team"])  # Runner-up

            # Record 3rd place details for comparison
            third_df = df.loc[2:2].copy()
            third_df["group"] = group_letter
            third_place_teams.append(third_df)

        # Concatenate 3rd placed teams and rank them
        third_df_all = pd.concat(third_place_teams).sort_values(
            by=["pts", "gd", "gf"], ascending=False).reset_index(drop=True)

        # Take the top 8
        best_third_teams = list(third_df_all.loc[:7, "team"].values)
        knockout_teams.extend(best_third_teams)

        return knockout_teams

    def simulate_tournament(self) -> str:
        """
        Simulate a single end-to-end tournament and return the champion.
        """
        self._reset_states()

        # 1. Simulate Group Stage
        standings = self.simulate_group_stage()

        # 2. Extract 32 Knockout teams
        ko_teams = self._determine_knockout_teams(standings)
        # Randomise pairings to represent tournament bracket (simplified representation)
        random.shuffle(ko_teams)

        # 3. Knockout Rounds
        # Round of 32
        r32_winners = []
        for i in range(0, 32, 2):
            _, _, winner = self._simulate_match(
                ko_teams[i], ko_teams[i+1], is_knockout=True)
            r32_winners.append(winner)

        # Round of 16
        r16_winners = []
        for i in range(0, 16, 2):
            _, _, winner = self._simulate_match(
                r32_winners[i], r32_winners[i+1], is_knockout=True)
            r16_winners.append(winner)

        # Quarter-finals
        qf_winners = []
        for i in range(0, 8, 2):
            _, _, winner = self._simulate_match(
                r16_winners[i], r16_winners[i+1], is_knockout=True)
            qf_winners.append(winner)

        # Semi-finals
        sf_winners = []
        for i in range(0, 4, 2):
            _, _, winner = self._simulate_match(
                qf_winners[i], qf_winners[i+1], is_knockout=True)
            sf_winners.append(winner)

        # Final
        _, _, champion = self._simulate_match(
            sf_winners[0], sf_winners[1], is_knockout=True)
        return champion

    def run_monte_carlo(self, n_sims: int = 1000) -> pd.Series:
        """
        Run Monte Carlo simulations of the entire World Cup.
        Returns a sorted Series of team win probabilities.
        """
        logger.info(
            f"Starting Monte Carlo simulation of {n_sims} tournament runs...")
        champions = []

        for i in range(n_sims):
            if (i + 1) % 250 == 0:
                logger.info(f"Simulated {i + 1}/{n_sims} tournaments...")
            champions.append(self.simulate_tournament())

        counts = pd.Series(champions).value_counts()
        probabilities = counts / n_sims
        return probabilities


if __name__ == "__main__":
    # Let's run a test simulation of 100 runs
    sim = TournamentSimulator()
    champs = sim.run_monte_carlo(n_sims=100)
    print("\n--- TEST MONTE CARLO RESULTS (100 runs) ---")
    print(champs.head(10))

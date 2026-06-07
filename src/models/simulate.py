from multiprocessing import Pool
from src.utils.logger import get_logger
from src.utils.config import config
from src.features.elo import get_k_factor, goal_margin_multiplier
from src.models.predict import MatchPredictor, CONTINENT_MAP
import pandas as pd
import numpy as np
import random
import json
from pathlib import Path
import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING_BACKEND"] = "threading"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

"""
simulate.py
Responsibility: Run Monte Carlo simulations of the FIFA World Cup 2026,
updating team states (Elo, form, goals) dynamically after every match.

Upgrades:
1. Dynamic K-factors (K=60 for World Cup) and goal margin multipliers.
2. Home-field advantage Elo adjustment for hosting nations.
3. Configurable EWMA decay factors.
"""


logger = get_logger(__name__)

# Official World Cup 2026 groups and team mappings
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

        # Fast copy cache for Monte Carlo runs
        self.fast_baseline_states = {k: v.copy()
                                     for k, v in self.baseline_states.items()}

    def _load_fast_model_params(self) -> None:
        try:
            import joblib
            models_dir = Path(config["paths"]["models"])
            lr_path = models_dir / "logistic_regression.pkl"
            scaler_path = models_dir / "scaler.pkl"
            if lr_path.exists() and scaler_path.exists():
                model = joblib.load(lr_path)
                scaler = joblib.load(scaler_path)
                clf = model.calibrated_classifiers_[0]

                self.fast_W = clf.estimator.coef_  # (3, 27) now 27 features
                self.fast_intercept = clf.estimator.intercept_  # (3,)
                self.fast_mean = scaler.mean_  # (27,)
                self.fast_scale = scaler.scale_  # (27,)
                self.fast_a = np.array([c.a_ for c in clf.calibrators])
                self.fast_b = np.array([c.b_ for c in clf.calibrators])

                self.fast_W_scaled = self.fast_W / self.fast_scale
                self.fast_intercept_scaled = self.fast_intercept - \
                    np.sum(self.fast_W * self.fast_mean /
                           self.fast_scale, axis=1)
                self.fast_model_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load fast model parameters: {e}")

    def _reset_states(self) -> None:
        """
        Reset predictor's team states to baseline before starting a new simulation run.
        """
        # If the predictor reloaded new states from disk, update baseline_states first!
        if self.predictor._check_and_reload():
            self.baseline_states = json.loads(
                json.dumps(self.predictor.team_states))
            self.fast_baseline_states = {k: v.copy()
                                         for k, v in self.baseline_states.items()}

        # Fast dictionary copy instead of slow JSON serialization
        self.predictor.team_states = {
            k: v.copy() for k, v in self.fast_baseline_states.items()
        }

    def _reset_states_fast(self) -> None:
        """
        Extremely fast copy of team states for Monte Carlo runs.
        """
        self.predictor.team_states = {
            k: v.copy() for k, v in self.fast_baseline_states.items()}

    def _update_stats_after_match(
        self, home_team: str, away_team: str, home_goals: int, away_goals: int, result: str, is_neutral: int
    ) -> None:
        """
        Dynamic State Update: Updates Elo, form, and goal averages for both teams
        in the predictor's state registry immediately after a simulated game.
        """
        # Overridden dynamically: World Cup matches use Tier 1 K-factor (60)
        k_factor = get_k_factor("FIFA World Cup")

        # Read form and goal parameters from config
        form_window = config["features"]["form_window"]
        goals_window = config["features"]["goals_window"]
        alpha_form = config["features"].get("form_alpha", 0.3)
        alpha_goals = config["features"].get("goals_alpha", 0.25)
        hfa_bonus = config["features"].get("elo_home_advantage", 100)

        # Get current states
        h_state = self.predictor.get_team_state(home_team).copy()
        a_state = self.predictor.get_team_state(away_team).copy()

        # --- 1. Update Elo with HFA and Goal Margin scaling ---
        h_elo_adjusted = h_state["elo"]
        if is_neutral == 0:
            h_elo_adjusted += hfa_bonus

        expected_home = 1.0 / \
            (1.0 + 10.0 ** ((a_state["elo"] - h_elo_adjusted) / 400.0))
        actual_home = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)

        # Calculate goal difference and margin multiplier
        goal_diff = int(home_goals - away_goals)
        multiplier = goal_margin_multiplier(goal_diff)

        h_state["elo"] = h_state["elo"] + k_factor * \
            multiplier * (actual_home - expected_home)
        a_state["elo"] = a_state["elo"] + k_factor * multiplier * \
            ((1.0 - actual_home) - (1.0 - expected_home))

        h_state["elo_diff"] = h_state["elo"] - a_state["elo"]
        a_state["elo_diff"] = a_state["elo"] - h_state["elo"]

        # --- 2. Update Form via EWMA smoothing ---
        h_pts = 3.0 if result == "H" else (1.0 if result == "D" else 0.0)
        a_pts = 3.0 if result == "A" else (1.0 if result == "D" else 0.0)

        # Apply opponent-adjusted points formula (just like in form.py)
        home_opp_factor = max(1.0 + (a_state["elo"] - 1500.0) / 1000.0, 0.5)
        away_opp_factor = max(1.0 + (h_state["elo"] - 1500.0) / 1000.0, 0.5)

        h_pts_adj = h_pts * home_opp_factor
        a_pts_adj = a_pts * away_opp_factor

        h_state["form"] = (1.0 - alpha_form) * \
            h_state["form"] + alpha_form * (h_pts_adj / 3.0)
        a_state["form"] = (1.0 - alpha_form) * \
            a_state["form"] + alpha_form * (a_pts_adj / 3.0)

        # Clamp form to [0.0, 1.0]
        h_state["form"] = min(max(h_state["form"], 0.0), 1.0)
        a_state["form"] = min(max(a_state["form"], 0.0), 1.0)

        # --- 3. Update Goals via EWMA smoothing ---
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
        Simulate a match between team A and team B using the best model.
        """
        if team_a in HOSTS and team_b not in HOSTS:
            home_team, away_team = team_a, team_b
            is_neutral = 0
        elif team_b in HOSTS and team_a not in HOSTS:
            home_team, away_team = team_b, team_a
            is_neutral = 0
        else:
            home_team, away_team = team_a, team_b
            is_neutral = 1

        scoreline = self.predictor.predict_scoreline(
            home_team, away_team, is_neutral=is_neutral, is_competitive=1, match_stake=4.0)
        grid = scoreline.get("scoreline_matrix")
        if grid is None:
            expected_goals = scoreline["expected_goals"]
            lambda_h = expected_goals["home"]
            lambda_a = expected_goals["away"]
            home_goals = np.random.poisson(lambda_h)
            away_goals = np.random.poisson(lambda_a)
        else:
            flat_idx = np.random.choice(grid.size, p=grid.ravel())
            home_goals, away_goals = np.unravel_index(flat_idx, grid.shape)
            home_goals = int(home_goals)
            away_goals = int(away_goals)

        expected_goals = scoreline["expected_goals"]
        lambda_h = max(0.1, float(expected_goals["home"]))
        lambda_a = max(0.1, float(expected_goals["away"]))

        if home_goals > away_goals:
            result = "H"
        elif away_goals > home_goals:
            result = "A"
        else:
            result = "D"

        if result == "H":
            winner = home_team
        elif result == "A":
            winner = away_team
        else:
            winner = None

        if is_knockout and result == "D":
            lambda_h_et = lambda_h / 3.0
            lambda_a_et = lambda_a / 3.0
            et_home_goals = np.random.poisson(lambda_h_et)
            et_away_goals = np.random.poisson(lambda_a_et)

            home_goals += et_home_goals
            away_goals += et_away_goals

            if home_goals > away_goals:
                result = "H"
                winner = home_team
            elif away_goals > home_goals:
                result = "A"
                winner = away_team
            else:
                elo_a = self.predictor.get_team_state(team_a)["elo"]
                elo_b = self.predictor.get_team_state(team_b)["elo"]
                prob_a_shootout = 0.5 + (elo_a - elo_b) / 2000.0
                prob_a_shootout = max(0.2, min(0.8, prob_a_shootout))
                winner = team_a if random.random() < prob_a_shootout else team_b

        self._update_stats_after_match(
            home_team, away_team, home_goals, away_goals, result, is_neutral)

        if home_team == team_b:
            return away_goals, home_goals, winner
        else:
            return home_goals, away_goals, winner

    def _simulate_match_fast(self, team_a: str, team_b: str, is_knockout: bool = False) -> tuple:
        """
        Extremely fast simulation of a match using pure NumPy Logistic Regression.
        """
        return self._simulate_match(team_a, team_b, is_knockout=is_knockout)

        if team_a in HOSTS and team_b not in HOSTS:
            home_team, away_team = team_a, team_b
            is_neutral = 0
        elif team_b in HOSTS and team_a not in HOSTS:
            home_team, away_team = team_b, team_a
            is_neutral = 0
        else:
            home_team, away_team = team_a, team_b
            is_neutral = 1

        h_state = self.predictor.get_team_state(home_team)
        a_state = self.predictor.get_team_state(away_team)

        # Compute travel continent features
        h_cont = CONTINENT_MAP.get(home_team)
        a_cont = CONTINENT_MAP.get(away_team)
        if is_neutral == 0:
            h_is_home_cont = 1
            a_is_home_cont = 1 if h_cont == a_cont else 0
        else:
            host_continent = "North America"
            h_is_home_cont = 1 if h_cont == host_continent else 0
            a_is_home_cont = 1 if a_cont == host_continent else 0

        # Feature array vectorised (27 features now)
        feat = np.array([[
            h_state["elo"],
            a_state["elo"],
            h_state["elo"] - a_state["elo"],
            h_state["form"],
            a_state["form"],
            h_state["form"] - a_state["form"],
            h_state["goals_scored_avg"],
            h_state["goals_conceded_avg"],
            h_state["goal_diff_avg"],
            a_state["goals_scored_avg"],
            a_state["goals_conceded_avg"],
            a_state["goal_diff_avg"],
            h_state["rank"],
            a_state["rank"],
            h_state["rank"] - a_state["rank"],
            h_state["rank_points"],
            a_state["rank_points"],
            h_state["rank_points"] - a_state["rank_points"],
            is_neutral,
            1,  # is_competitive defaults to 1
            30.0,  # home_rest_days
            30.0,  # away_rest_days
            0.0,   # rest_days_diff
            float(h_is_home_cont),
            float(a_is_home_cont),
            float(h_is_home_cont - a_is_home_cont),
            4.0  # match_stake (WC matches are tier 4)
        ]])

        f = np.dot(feat, self.fast_W_scaled.T) + self.fast_intercept_scaled
        p = 1.0 / (1.0 + np.exp(self.fast_a * f + self.fast_b))
        probs_final = p[0] / np.sum(p[0])

        outcomes = ["H", "D", "A"]
        result = random.choices(outcomes, weights=probs_final)[0]

        lambda_h = max(
            0.5, (h_state["goals_scored_avg"] + a_state["goals_conceded_avg"]) / 2.0)
        lambda_a = max(
            0.5, (a_state["goals_scored_avg"] + h_state["goals_conceded_avg"]) / 2.0)

        home_goals = np.random.poisson(lambda_h)
        away_goals = np.random.poisson(lambda_a)

        if result == "H" and home_goals <= away_goals:
            home_goals = away_goals + random.randint(1, 2)
        elif result == "A" and away_goals <= home_goals:
            away_goals = home_goals + random.randint(1, 2)
        elif result == "D":
            home_goals = away_goals

        if result == "H":
            winner = home_team
        elif result == "A":
            winner = away_team
        else:
            winner = None

        if is_knockout and result == "D":
            lambda_h_et = lambda_h / 3.0
            lambda_a_et = lambda_a / 3.0
            et_home_goals = np.random.poisson(lambda_h_et)
            et_away_goals = np.random.poisson(lambda_a_et)

            home_goals += et_home_goals
            away_goals += et_away_goals

            if home_goals > away_goals:
                result = "H"
                winner = home_team
            elif away_goals > home_goals:
                result = "A"
                winner = away_team
            else:
                elo_a = h_state["elo"]
                elo_b = a_state["elo"]
                prob_a_shootout = 0.5 + (elo_a - elo_b) / 2000.0
                prob_a_shootout = max(0.2, min(0.8, prob_a_shootout))
                winner = team_a if random.random() < prob_a_shootout else team_b

        self._update_stats_after_match(
            home_team, away_team, home_goals, away_goals, result, is_neutral)

        if home_team == team_b:
            return away_goals, home_goals, winner
        else:
            return home_goals, away_goals, winner

    def simulate_group_stage(self) -> dict:
        """
        Simulate the group stage for all 12 groups.
        """
        standings = {}

        for group_letter, teams in GROUPS.items():
            table = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams}

            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a, team_b = teams[i], teams[j]
                    gf_a, gf_b, winner = self._simulate_match(
                        team_a, team_b, is_knockout=False)

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

            group_df = pd.DataFrame.from_dict(
                table, orient="index").reset_index()
            group_df.columns = ["team", "pts", "gd", "gf"]
            group_df = group_df.sort_values(
                by=["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
            standings[group_letter] = group_df

        return standings

    def fast_simulate_group_stage(self) -> dict:
        standings = {}
        for group_letter, teams in GROUPS.items():
            table = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams}
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a, team_b = teams[i], teams[j]
                    gf_a, gf_b, winner = self._simulate_match_fast(
                        team_a, team_b, is_knockout=False)
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

            sorted_teams = sorted(
                table.items(),
                key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
                reverse=True
            )
            standings[group_letter] = [
                {"team": k, "pts": v["pts"], "gd": v["gd"], "gf": v["gf"]}
                for k, v in sorted_teams
            ]
        return standings

    def _determine_knockout_teams(self, standings: dict) -> list:
        knockout_teams = []
        third_place_teams = []

        for group_letter, df in standings.items():
            knockout_teams.append(df.loc[0, "team"])  # Winner
            knockout_teams.append(df.loc[1, "team"])  # Runner-up

            third_df = df.loc[2:2].copy()
            third_df["group"] = group_letter
            third_place_teams.append(third_df)

        third_df_all = pd.concat(third_place_teams).sort_values(
            by=["pts", "gd", "gf"], ascending=False).reset_index(drop=True)

        best_third_teams = list(third_df_all.loc[:7, "team"].values)
        knockout_teams.extend(best_third_teams)

        return knockout_teams

    def fast_get_knockout_bracket_teams(self, standings: dict) -> list:
        third_place_teams = []
        for group_letter, group_list in standings.items():
            third_info = group_list[2]
            third_place_teams.append({
                "team": third_info["team"],
                "group": group_letter,
                "pts": third_info["pts"],
                "gd": third_info["gd"],
                "gf": third_info["gf"]
            })

        third_place_teams.sort(key=lambda x: (
            x["pts"], x["gd"], x["gf"]), reverse=True)
        best_thirds = third_place_teams[:8]

        available_thirds = list(best_thirds)
        match_requirements = {
            3: {"A", "B", "C", "D", "F"},
            6: {"C", "D", "F", "G", "H"},
            7: {"C", "E", "F", "H", "I"},
            8: {"E", "H", "I", "J", "K"},
            9: {"A", "E", "H", "I", "J"},
            10: {"B", "E", "F", "I", "J"},
            13: {"E", "F", "G", "I", "J"},
            16: {"D", "E", "I", "J", "L"}
        }

        assigned_thirds = {}
        for match_num, allowed_groups in match_requirements.items():
            matched_team = None
            for t in available_thirds:
                if t["group"] in allowed_groups:
                    matched_team = t
                    break
            if matched_team:
                assigned_thirds[match_num] = matched_team["team"]
                available_thirds.remove(matched_team)
            else:
                if available_thirds:
                    matched_team = available_thirds[0]
                    assigned_thirds[match_num] = matched_team["team"]
                    available_thirds.remove(matched_team)
                else:
                    assigned_thirds[match_num] = "Unknown"

        ko_teams = []
        ko_teams.extend([standings["A"][1]["team"], standings["B"][1]["team"]])
        ko_teams.extend([standings["C"][0]["team"], standings["F"][1]["team"]])
        ko_teams.extend([standings["E"][0]["team"], assigned_thirds[3]])
        ko_teams.extend([standings["F"][0]["team"], standings["C"][1]["team"]])
        ko_teams.extend([standings["E"][1]["team"], standings["I"][1]["team"]])
        ko_teams.extend([standings["I"][0]["team"], assigned_thirds[6]])
        ko_teams.extend([standings["A"][0]["team"], assigned_thirds[7]])
        ko_teams.extend([standings["L"][0]["team"], assigned_thirds[8]])
        ko_teams.extend([standings["G"][0]["team"], assigned_thirds[9]])
        ko_teams.extend([standings["D"][0]["team"], assigned_thirds[10]])
        ko_teams.extend([standings["H"][0]["team"], standings["J"][1]["team"]])
        ko_teams.extend([standings["K"][1]["team"], standings["L"][1]["team"]])
        ko_teams.extend([standings["B"][0]["team"], assigned_thirds[13]])
        ko_teams.extend([standings["D"][1]["team"], standings["G"][1]["team"]])
        ko_teams.extend([standings["J"][0]["team"], standings["H"][1]["team"]])
        ko_teams.extend([standings["K"][0]["team"], assigned_thirds[16]])

        return ko_teams

    def simulate_tournament(self) -> str:
        """
        Simulate a single end-to-end tournament and return the champion.
        """
        self._reset_states()
        standings = self.simulate_group_stage()
        ko_teams = self._get_knockout_bracket_teams(standings)

        # Knockout Rounds
        r32_winners = []
        for i in range(0, 32, 2):
            _, _, winner = self._simulate_match(
                ko_teams[i], ko_teams[i+1], is_knockout=True)
            r32_winners.append(winner)

        r16_winners = []
        for i in range(0, 16, 2):
            _, _, winner = self._simulate_match(
                r32_winners[i], r32_winners[i+1], is_knockout=True)
            r16_winners.append(winner)

        qf_winners = []
        for i in range(0, 8, 2):
            _, _, winner = self._simulate_match(
                r16_winners[i], r16_winners[i+1], is_knockout=True)
            qf_winners.append(winner)

        sf_winners = []
        for i in range(0, 4, 2):
            _, _, winner = self._simulate_match(
                qf_winners[i], qf_winners[i+1], is_knockout=True)
            sf_winners.append(winner)

        _, _, champion = self._simulate_match(
            sf_winners[0], sf_winners[1], is_knockout=True)
        return champion

    def simulate_tournament_fast(self) -> str:
        self._reset_states_fast()
        standings = self.fast_simulate_group_stage()
        ko_teams = self.fast_get_knockout_bracket_teams(standings)

        r32_winners = []
        for i in range(0, 32, 2):
            _, _, winner = self._simulate_match_fast(
                ko_teams[i], ko_teams[i+1], is_knockout=True)
            r32_winners.append(winner)

        r16_winners = []
        for i in range(0, 16, 2):
            _, _, winner = self._simulate_match_fast(
                r32_winners[i], r32_winners[i+1], is_knockout=True)
            r16_winners.append(winner)

        qf_winners = []
        for i in range(0, 8, 2):
            _, _, winner = self._simulate_match_fast(
                r16_winners[i], r16_winners[i+1], is_knockout=True)
            qf_winners.append(winner)

        sf_winners = []
        for i in range(0, 4, 2):
            _, _, winner = self._simulate_match_fast(
                qf_winners[i], qf_winners[i+1], is_knockout=True)
            sf_winners.append(winner)

        _, _, champion = self._simulate_match_fast(
            sf_winners[0], sf_winners[1], is_knockout=True)
        return champion

    def _get_knockout_bracket_teams(self, standings: dict) -> list:
        third_place_teams = []
        for group_letter, df in standings.items():
            third_team = df.loc[2, "team"]
            pts = df.loc[2, "pts"]
            gd = df.loc[2, "gd"]
            gf = df.loc[2, "gf"]
            third_place_teams.append({
                "team": third_team,
                "group": group_letter,
                "pts": pts,
                "gd": gd,
                "gf": gf
            })

        third_place_teams.sort(key=lambda x: (
            x["pts"], x["gd"], x["gf"]), reverse=True)
        best_thirds = third_place_teams[:8]

        available_thirds = list(best_thirds)
        match_requirements = {
            3: {"A", "B", "C", "D", "F"},
            6: {"C", "D", "F", "G", "H"},
            7: {"C", "E", "F", "H", "I"},
            8: {"E", "H", "I", "J", "K"},
            9: {"A", "E", "H", "I", "J"},
            10: {"B", "E", "F", "I", "J"},
            13: {"E", "F", "G", "I", "J"},
            16: {"D", "E", "I", "J", "L"}
        }

        assigned_thirds = {}
        for match_num, allowed_groups in match_requirements.items():
            matched_team = None
            for t in available_thirds:
                if t["group"] in allowed_groups:
                    matched_team = t
                    break

            if matched_team:
                assigned_thirds[match_num] = matched_team["team"]
                available_thirds.remove(matched_team)
            else:
                if available_thirds:
                    matched_team = available_thirds[0]
                    assigned_thirds[match_num] = matched_team["team"]
                    available_thirds.remove(matched_team)
                else:
                    assigned_thirds[match_num] = "Unknown"

        ko_teams = []
        ko_teams.extend([standings["A"].loc[1, "team"],
                        standings["B"].loc[1, "team"]])
        ko_teams.extend([standings["C"].loc[0, "team"],
                        standings["F"].loc[1, "team"]])
        ko_teams.extend([standings["E"].loc[0, "team"], assigned_thirds[3]])
        ko_teams.extend([standings["F"].loc[0, "team"],
                        standings["C"].loc[1, "team"]])
        ko_teams.extend([standings["E"].loc[1, "team"],
                        standings["I"].loc[1, "team"]])
        ko_teams.extend([standings["I"].loc[0, "team"], assigned_thirds[6]])
        ko_teams.extend([standings["A"].loc[0, "team"], assigned_thirds[7]])
        ko_teams.extend([standings["L"].loc[0, "team"], assigned_thirds[8]])
        ko_teams.extend([standings["G"].loc[0, "team"], assigned_thirds[9]])
        ko_teams.extend([standings["D"].loc[0, "team"], assigned_thirds[10]])
        ko_teams.extend([standings["H"].loc[0, "team"],
                        standings["J"].loc[1, "team"]])
        ko_teams.extend([standings["K"].loc[1, "team"],
                        standings["L"].loc[1, "team"]])
        ko_teams.extend([standings["B"].loc[0, "team"], assigned_thirds[13]])
        ko_teams.extend([standings["D"].loc[1, "team"],
                        standings["G"].loc[1, "team"]])
        ko_teams.extend([standings["J"].loc[0, "team"],
                        standings["H"].loc[1, "team"]])
        ko_teams.extend([standings["K"].loc[0, "team"], assigned_thirds[16]])

        return ko_teams

    def simulate_detailed_tournament(self) -> dict:
        """
        Simulate a single end-to-end tournament and return full details.
        """
        self.predictor.clear_prediction_cache()
        self._reset_states()

        group_matches = []
        group_standings_data = {}

        # 1. Simulate Group Stage and capture details
        for group_letter, teams in GROUPS.items():
            table = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams}

            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a, team_b = teams[i], teams[j]
                    gf_a, gf_b, winner = self._simulate_match(
                        team_a, team_b, is_knockout=False)

                    # Log match details
                    group_matches.append({
                        "group": group_letter,
                        "home_team": team_a,
                        "away_team": team_b,
                        "home_goals": int(gf_a),
                        "away_goals": int(gf_b),
                        "winner": winner
                    })

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

            group_standings_data[group_letter] = group_df.to_dict(
                orient="records")

        standings_dfs = {}
        for group_letter in GROUPS.keys():
            standings_dfs[group_letter] = pd.DataFrame(
                group_standings_data[group_letter])

        ko_teams = self._get_knockout_bracket_teams(standings_dfs)

        # 3. Simulate Knockout Rounds
        knockout_rounds = {
            "r32": [],
            "r16": [],
            "qf": [],
            "sf": [],
            "final": []
        }

        def run_knockout_stage(teams_list, round_key):
            next_round_teams = []
            for i in range(0, len(teams_list), 2):
                team_a = teams_list[i]
                team_b = teams_list[i+1]

                # Run match
                gf_a, gf_b, winner = self._simulate_match(
                    team_a, team_b, is_knockout=True)

                shootout_winner = None
                if gf_a == gf_b:
                    shootout_winner = winner

                knockout_rounds[round_key].append({
                    "home_team": team_a,
                    "away_team": team_b,
                    "home_goals": int(gf_a),
                    "away_goals": int(gf_b),
                    "winner": winner,
                    "shootout_winner": shootout_winner
                })
                next_round_teams.append(winner)
            return next_round_teams

        r32_winners = run_knockout_stage(ko_teams, "r32")
        r16_winners = run_knockout_stage(r32_winners, "r16")
        qf_winners = run_knockout_stage(r16_winners, "qf")
        sf_winners = run_knockout_stage(qf_winners, "sf")
        champion = run_knockout_stage(sf_winners, "final")[0]

        return {
            "group_matches": group_matches,
            "group_standings": group_standings_data,
            "knockout_rounds": knockout_rounds,
            "champion": champion
        }

    def run_monte_carlo(self, n_sims: int = 1000) -> pd.Series:
        """
        Run Monte Carlo simulations of the entire World Cup.
        """
        self.predictor.clear_prediction_cache()
        self.predictor.disable_reload = True
        logger.info(
            f"Starting Monte Carlo simulation of {n_sims} tournament runs...")

        # Determine number of worker processes
        num_workers = max(1, os.cpu_count() - 1)
        logger.info(f"Spawning {num_workers} parallel worker processes...")

        # Run simulations concurrently across all cores
        with Pool(processes=num_workers, initializer=_init_worker) as pool:
            champions = pool.map(_run_single_sim, range(n_sims))

        self.predictor.disable_reload = False
        counts = pd.Series(champions).value_counts()
        probabilities = counts / n_sims
        return probabilities


# --- Multiprocessing Workers ---
_worker_simulator = None


def _init_worker():
    global _worker_simulator
    # Initialize a single simulator per process (loads the model once)
    _worker_simulator = TournamentSimulator()
    # Disable auto-reload inside workers to maximize loop speed
    _worker_simulator.predictor.disable_reload = True


def _run_single_sim(_):
    # Runs a single tournament simulation and returns the champion
    return _worker_simulator.simulate_tournament()


if __name__ == "__main__":
    # Let's run a test simulation of 100 runs
    sim = TournamentSimulator()
    champs = sim.run_monte_carlo(n_sims=100)
    print("\n--- TEST MONTE CARLO RESULTS (100 runs) ---")
    print(champs.head(10))

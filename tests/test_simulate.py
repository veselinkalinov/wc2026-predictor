from src.models.simulate import TournamentSimulator


def test_simulated_away_goal_diff_uses_away_conceded_average():
    simulator = TournamentSimulator()
    simulator.predictor.team_states["Home Test"] = {
        "elo": 1500.0,
        "form": 0.5,
        "goals_scored_avg": 1.0,
        "goals_conceded_avg": 1.0,
        "goal_diff_avg": 0.0,
        "rank": 100.0,
        "rank_points": 1200.0,
    }
    simulator.predictor.team_states["Away Test"] = {
        "elo": 1500.0,
        "form": 0.5,
        "goals_scored_avg": 1.0,
        "goals_conceded_avg": 1.0,
        "goal_diff_avg": 0.0,
        "rank": 101.0,
        "rank_points": 1190.0,
    }

    simulator._update_stats_after_match(
        "Home Test", "Away Test", home_goals=3, away_goals=1, result="H", is_neutral=1)

    away_state = simulator.predictor.get_team_state("Away Test")
    assert away_state["goal_diff_avg"] == (
        away_state["goals_scored_avg"] - away_state["goals_conceded_avg"])

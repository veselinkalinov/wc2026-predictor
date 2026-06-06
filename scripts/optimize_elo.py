import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import config
from src.features.elo import compute_elo_ratings
from src.features.form import compute_form_features
from src.features.goals import compute_goal_features

def main():
    print("=" * 70)
    print("OPTIMIZING ELO AND EWMA DECAY PARAMETERS")
    print("=" * 70)

    processed_dir = Path(config["paths"]["processed_data"])
    input_path = processed_dir / "matches_clean.csv"
    if not input_path.exists():
        print(f"Cleaned matches file not found at {input_path}. Please run run_pipeline.py first.")
        sys.exit(1)

    # Load matches
    matches = pd.read_csv(input_path)
    matches["date"] = pd.to_datetime(matches["date"])

    # Define grids to search
    elo_home_advantages = [50, 75, 100, 125, 150]
    form_alphas = [0.15, 0.25, 0.35, 0.45]
    goals_alphas = [0.15, 0.25, 0.35, 0.45]

    best_loss = float("inf")
    best_params = {}

    # Target column map
    class_map = {"H": 0, "D": 1, "A": 2}
    matches["target"] = matches["result"].map(class_map)

    # Set up static cutoffs matching config
    cutoff_train = pd.to_datetime(config["model"]["train_cutoff"])
    cutoff_cal = pd.to_datetime(config["model"]["calibration_cutoff"])

    # We only train and validate on competitive matches after a warm-up period
    date_from = pd.to_datetime(config["data"]["date_from"])
    
    total_iterations = len(elo_home_advantages) * len(form_alphas) * len(goals_alphas)
    current_iteration = 0

    print(f"Running grid search over {total_iterations} combinations...")

    for hfa in elo_home_advantages:
        for f_alpha in form_alphas:
            for g_alpha in goals_alphas:
                current_iteration += 1
                
                # Apply parameters to global config dictionary dynamically
                config["features"]["elo_home_advantage"] = hfa
                config["features"]["form_alpha"] = f_alpha
                config["features"]["goals_alpha"] = g_alpha

                # Copy matches to avoid modifying the original dataframe
                df = matches.copy()

                # Recompute ratings/features
                df, _ = compute_elo_ratings(df)
                df = compute_form_features(df)
                df = compute_goal_features(df)

                # Add differences
                df["rank_diff"] = df["home_rank"] - df["away_rank"]
                df["rank_points_diff"] = df["home_rank_points"] - df["away_rank_points"]
                df["is_neutral"] = df["neutral"].astype(int)

                # Filter dataset down to date_from for model evaluation
                df_filtered = df[(df["date"] >= date_from) & (df["date"] <= cutoff_cal)].copy()
                df_filtered = df_filtered.dropna(subset=["home_elo", "away_elo", "home_form", "away_form", "target"])

                # Splits
                train_df = df_filtered[df_filtered["date"] < cutoff_train]
                val_df = df_filtered[df_filtered["date"] >= cutoff_train]

                if len(train_df) == 0 or len(val_df) == 0:
                    continue

                # Feature subset for baseline evaluation
                features = [
                    "elo_diff", "form_diff", "home_goal_diff_avg", "away_goal_diff_avg", 
                    "rank_diff", "rank_points_diff", "is_neutral"
                ]

                X_train = train_df[features].values
                y_train = train_df["target"].values
                X_val = val_df[features].values
                y_val = val_df["target"].values

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Fit a simple logistic regression
                clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)
                clf.fit(X_train_scaled, y_train)

                val_probs = clf.predict_proba(X_val_scaled)
                loss = log_loss(y_val, val_probs)

                if loss < best_loss:
                    best_loss = loss
                    best_params = {
                        "elo_home_advantage": hfa,
                        "form_alpha": f_alpha,
                        "goals_alpha": g_alpha
                    }
                    print(f"[{current_iteration}/{total_iterations}] New best: HFA={hfa}, FormAlpha={f_alpha}, GoalAlpha={g_alpha} -> Log Loss: {loss:.5f}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Log Loss: {best_loss:.5f}")
    print(f"Best Parameters: {best_params}")
    print("=" * 70)

    # Update config.yaml with best parameters
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # Merge optimized parameters
    full_config["features"]["elo_home_advantage"] = best_params["elo_home_advantage"]
    full_config["features"]["form_alpha"] = best_params["form_alpha"]
    full_config["features"]["goals_alpha"] = best_params["goals_alpha"]

    with open(config_path, "w") as f:
        yaml.safe_dump(full_config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {config_path} with optimized features parameters.")

if __name__ == "__main__":
    main()

import numpy as np
from scipy.stats import poisson
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import PoissonRegressor


class PoissonGoalModel(BaseEstimator, ClassifierMixin):
    """
    Goal-count model that converts expected goals into H/D/A probabilities.

    The model keeps two Poisson regressors for expected home and away goals,
    then applies a Dixon-Coles style low-score correction to the scoreline
    matrix. It remains a scikit-learn compatible classifier so it can be
    calibrated and compared beside the other 1X2 models.
    """

    def __init__(self, alpha=1.0, rho=0.0, max_goals=10):
        self.alpha = alpha
        self.rho = rho
        self.max_goals = max_goals
        self.classes_ = np.array([0, 1, 2])  # H=0, D=1, A=2
        self.home_regressor = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        self.away_regressor = PoissonRegressor(alpha=self.alpha, max_iter=1000)

    def fit(self, X, y):
        self.home_regressor = PoissonRegressor(
            alpha=self.alpha, max_iter=1000)
        self.away_regressor = PoissonRegressor(
            alpha=self.alpha, max_iter=1000)

        y_arr = np.asarray(y)
        if len(y_arr.shape) == 2:
            self.home_regressor.fit(X, y_arr[:, 0])
            self.away_regressor.fit(X, y_arr[:, 1])
        else:
            dummy_home_goals = np.where(y_arr == 0, 2, np.where(y_arr == 1, 1, 0))
            dummy_away_goals = np.where(y_arr == 2, 2, np.where(y_arr == 1, 1, 0))
            self.home_regressor.fit(X, dummy_home_goals)
            self.away_regressor.fit(X, dummy_away_goals)
        return self

    def predict_expected_goals(self, X) -> np.ndarray:
        lambda_h = self.home_regressor.predict(X)
        lambda_a = self.away_regressor.predict(X)
        max_lambda = max(float(getattr(self, "max_goals", 10)), 1.0)
        lambda_h = np.nan_to_num(lambda_h, nan=1.0, posinf=max_lambda, neginf=0.1)
        lambda_a = np.nan_to_num(lambda_a, nan=1.0, posinf=max_lambda, neginf=0.1)
        lambda_h = np.clip(lambda_h, 0.1, max_lambda)
        lambda_a = np.clip(lambda_a, 0.1, max_lambda)
        return np.column_stack([lambda_h, lambda_a])

    def _dixon_coles_tau(self, home_goals: int, away_goals: int, lambda_h: float, lambda_a: float) -> float:
        rho = getattr(self, "rho", 0.0)
        if home_goals == 0 and away_goals == 0:
            return 1.0 - (lambda_h * lambda_a * rho)
        if home_goals == 0 and away_goals == 1:
            return 1.0 + (lambda_h * rho)
        if home_goals == 1 and away_goals == 0:
            return 1.0 + (lambda_a * rho)
        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        return 1.0

    def scoreline_matrix_for_lambdas(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        max_goals = int(getattr(self, "max_goals", 10))
        goals = np.arange(max_goals + 1)
        p_h = poisson.pmf(goals, lambda_h)
        p_a = poisson.pmf(goals, lambda_a)
        p_h_sum = p_h.sum()
        p_a_sum = p_a.sum()
        if not np.isfinite(p_h_sum) or p_h_sum <= 0:
            p_h = np.zeros_like(goals, dtype=float)
            p_h[min(int(round(lambda_h)), max_goals)] = 1.0
        else:
            p_h = p_h / p_h_sum
        if not np.isfinite(p_a_sum) or p_a_sum <= 0:
            p_a = np.zeros_like(goals, dtype=float)
            p_a[min(int(round(lambda_a)), max_goals)] = 1.0
        else:
            p_a = p_a / p_a_sum

        grid = np.outer(p_h, p_a)
        for h in (0, 1):
            for a in (0, 1):
                grid[h, a] *= self._dixon_coles_tau(h, a, lambda_h, lambda_a)

        grid = np.maximum(grid, 0.0)
        total = grid.sum()
        if total <= 0:
            return np.outer(p_h, p_a)
        return grid / total

    def predict_scoreline_matrices(self, X) -> list[np.ndarray]:
        expected_goals = self.predict_expected_goals(X)
        return [
            self.scoreline_matrix_for_lambdas(lambda_h, lambda_a)
            for lambda_h, lambda_a in expected_goals
        ]

    def predict_proba(self, X):
        matrices = self.predict_scoreline_matrices(X)
        probs = []
        for grid in matrices:
            p_home_win = float(np.sum(np.tril(grid, -1)))
            p_draw = float(np.sum(np.diag(grid)))
            p_away_win = float(np.sum(np.triu(grid, 1)))
            probs.append([p_home_win, p_draw, p_away_win])
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def tune_rho(self, X_cal, y_goals, rho_grid=None):
        if rho_grid is None:
            rho_grid = np.round(np.arange(-0.20, 0.205, 0.01), 3)

        y_arr = np.asarray(y_goals)
        expected_goals = self.predict_expected_goals(X_cal)
        best_rho = float(getattr(self, "rho", 0.0))
        best_nll = float("inf")
        max_goals = int(getattr(self, "max_goals", 10))

        for rho in rho_grid:
            self.rho = float(rho)
            nll = 0.0
            for (lambda_h, lambda_a), (home_goals, away_goals) in zip(expected_goals, y_arr):
                grid = self.scoreline_matrix_for_lambdas(lambda_h, lambda_a)
                h = int(min(max(home_goals, 0), max_goals))
                a = int(min(max(away_goals, 0), max_goals))
                nll -= np.log(max(grid[h, a], 1e-12))
            if nll < best_nll:
                best_nll = nll
                best_rho = float(rho)

        self.rho = best_rho
        return best_rho, best_nll / max(len(y_arr), 1)

    def scoreline_dict(self, X, top_n=5) -> list[dict]:
        expected_goals = self.predict_expected_goals(X)
        matrices = self.predict_scoreline_matrices(X)
        payloads = []
        for (lambda_h, lambda_a), grid in zip(expected_goals, matrices):
            flat_order = np.argsort(grid.ravel())[::-1][:top_n]
            top_scorelines = []
            for flat_idx in flat_order:
                h, a = np.unravel_index(flat_idx, grid.shape)
                top_scorelines.append({
                    "home_goals": int(h),
                    "away_goals": int(a),
                    "probability": round(float(grid[h, a]), 4),
                })
            payloads.append({
                "expected_goals": {
                    "home": round(float(lambda_h), 3),
                    "away": round(float(lambda_a), 3),
                },
                "top_scorelines": top_scorelines,
            })
        return payloads

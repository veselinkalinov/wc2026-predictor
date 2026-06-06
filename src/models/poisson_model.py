import numpy as np
from scipy.stats import poisson
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import PoissonRegressor

class PoissonGoalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = np.array([0, 1, 2])  # H=0, D=1, A=2
        self.home_regressor = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        self.away_regressor = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        
    def fit(self, X, y):
        # We assume y is the 1D target (H/D/A), but we need the actual goals scored
        # To avoid breaking sklearn pipelines, we require y_goals to be passed or we
        # retrieve it. If y is 2D (scores), we use it. Otherwise, we expect y_goals to be passed.
        # However, to be compatible with GridSearchCV, we can pass y_goals as y during fit!
        if len(y.shape) == 2:
            self.home_regressor.fit(X, y[:, 0])
            self.away_regressor.fit(X, y[:, 1])
        else:
            # Fallback if y is 1D target (we shouldn't call this directly without goals, but we add a fallback)
            # This fallback assumes standard default goals if not available
            dummy_home_goals = np.where(y == 0, 2, np.where(y == 1, 1, 0))
            dummy_away_goals = np.where(y == 2, 2, np.where(y == 1, 1, 0))
            self.home_regressor.fit(X, dummy_home_goals)
            self.away_regressor.fit(X, dummy_away_goals)
        return self
        
    def predict_proba(self, X):
        lambda_h = self.home_regressor.predict(X)
        lambda_a = self.away_regressor.predict(X)
        
        # Ensure lambdas are positive and reasonable
        lambda_h = np.maximum(lambda_h, 0.1)
        lambda_a = np.maximum(lambda_a, 0.1)
        
        probs = []
        for lh, la in zip(lambda_h, lambda_a):
            # Construct score grid up to 10 goals
            max_goals = 10
            x_range = np.arange(max_goals + 1)
            p_h = poisson.pmf(x_range, lh)
            p_a = poisson.pmf(x_range, la)
            
            # Normalize to sum to 1.0
            p_h /= p_h.sum()
            p_a /= p_a.sum()
            
            grid = np.outer(p_h, p_a)  # grid[x, y] is P(H=x, A=y)
            
            p_home_win = np.sum(np.tril(grid, -1))
            p_draw = np.sum(np.diag(grid))
            p_away_win = np.sum(np.triu(grid, 1))
            
            probs.append([p_home_win, p_draw, p_away_win])
            
        return np.array(probs)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

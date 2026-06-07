import numpy as np

from src.models.poisson_model import PoissonGoalModel


def test_dixon_coles_scoreline_probabilities_sum_to_one():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 4))
    y_goals = rng.poisson(lam=1.3, size=(40, 2))

    model = PoissonGoalModel(alpha=0.1, rho=0.05, max_goals=8)
    model.fit(X, y_goals)

    matrices = model.predict_scoreline_matrices(X[:3])
    for matrix in matrices:
        assert matrix.shape == (9, 9)
        assert np.isclose(matrix.sum(), 1.0)
        assert np.all(matrix >= 0.0)

    probs = model.predict_proba(X[:3])
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_dixon_coles_rho_tuning_returns_grid_value():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 3))
    y_goals = rng.poisson(lam=1.1, size=(30, 2))
    rho_grid = [-0.05, 0.0, 0.05]

    model = PoissonGoalModel(alpha=0.1, max_goals=6)
    model.fit(X, y_goals)
    rho, nll = model.tune_rho(X, y_goals, rho_grid=rho_grid)

    assert rho in rho_grid
    assert nll > 0

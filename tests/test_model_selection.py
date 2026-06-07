from src.models.train import select_best_model


def test_select_best_model_prefers_lowest_log_loss():
    comparison = {
        "Accuracy Winner": {
            "accuracy": 0.62,
            "log_loss": 0.95,
            "brier_score": 0.18,
        },
        "Probability Winner": {
            "accuracy": 0.60,
            "log_loss": 0.86,
            "brier_score": 0.17,
        },
    }

    assert select_best_model(comparison, "log_loss") == "Probability Winner"


def test_select_best_model_uses_brier_then_accuracy_tiebreakers():
    comparison = {
        "A": {
            "accuracy": 0.61,
            "log_loss": 0.90,
            "brier_score": 0.18,
        },
        "B": {
            "accuracy": 0.60,
            "log_loss": 0.90,
            "brier_score": 0.17,
        },
        "C": {
            "accuracy": 0.62,
            "log_loss": 0.90,
            "brier_score": 0.17,
        },
    }

    assert select_best_model(comparison, "log_loss") == "C"

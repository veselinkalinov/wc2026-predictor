# WC2026 Predictor Curriculum

This curriculum turns the project reference documentation into a study path. Use it when you want to understand the code well enough to explain it in an interview or rebuild a similar project from scratch.

The original reference file remains here:

- `wc2026-files/MASTER_TECHNICAL_DOCUMENTATION.md`

## Current Code Covered

This curriculum has been updated against the project state ending at commit `d6821e5` (`Refit production model on completed matches`) plus the current working tree read-only analysis. The important changes since the first curriculum draft are:

- Docker now starts the Flask app through Gunicorn and `wsgi.py`, not the Flask development server.
- `requirements.txt` uses exact package pins and includes `gunicorn==23.0.0`.
- `data/raw`, `data/processed`, and `data/features` CSV snapshots are now tracked project inputs/artifacts, while `models/` and `wc2026-files/` are ignored.
- `MatchPredictor` now infers rest days from match history, accepts optional `match_date`, returns prediction context, and hot-reloads when `feature_matrix.csv` changes.
- The live standings utility can derive group standings from fixture results before falling back to external APIs or mock/cache data.
- `scripts/fetch_recent_matches.py` now upserts World Cup match rows and skips retraining when no data changed.
- The Poisson score model now guards against non-finite expected goals and scoreline matrices.
- Tests now cover rest-day inference, feature-matrix reload, and Poisson numerical stability.
- Training now separates holdout evaluation artifacts from production-refit artifacts: `evaluation_model.pkl`/`evaluation_scaler.pkl` are used for reported metrics, while `best_model.pkl`/`scaler.pkl` are refit on all completed rows for serving.
- The latest data refresh completed four June 23, 2026 World Cup fixtures and increased the feature matrix to 15,647 rows through `2026-06-23`.

## How To Study This

Use the lessons in order. Each lesson follows the same structure:

- What you will understand after the lesson
- First-principles explanation
- Project-specific walkthrough
- File-by-file or code-block explanations
- Interview questions and strong answers
- Rebuild exercise
- Self-check quiz
- External links

Do not try to memorize the whole project. Learn the flow:

```text
raw data
  -> validation
  -> cleaning
  -> feature engineering
  -> model training
  -> evaluation
  -> prediction runtime
  -> tournament simulation
  -> Flask API and UI
```

## 10-Day Study Schedule

| Day | Study target | Output you should produce |
|---|---|---|
| 1 | `00-orientation.md`, `01-folder-and-file-map.md` | Explain the whole repo in 2 minutes. |
| 2 | `02-python-foundations.md` | Explain imports, modules, config, logging, scripts, Docker basics. |
| 3 | `03-data-pipeline.md` | Trace `matches.csv` into `matches_clean.csv`. |
| 4 | `04-feature-engineering.md` | Explain Elo, form, goals, rest, continent, and match stake features. |
| 5 | First half of `05-machine-learning-models.md` | Explain train/cal/test split, scaling, metrics, baselines. |
| 6 | Second half of `05-machine-learning-models.md` | Explain every classifier/regressor and calibration. |
| 7 | `06-prediction-runtime.md` | Trace one `/api/predict` request through the model. |
| 8 | `07-tournament-simulation.md` | Explain one Monte Carlo tournament run. |
| 9 | `08-flask-api-and-ui.md`, `09-testing-deployment-and-risks.md` | Explain the web app, tests, Docker, and known risks. |
| 10 | `10-interview-and-rebuild-guide.md`, `11-current-changes.md` | Practice answers, outline a rebuild from scratch, and explain the latest changes. |

If you only have one week, combine days 5 and 6, and skim the UI templates in day 9.

## Curriculum Files

1. [Orientation](00-orientation.md)
2. [Folder and File Map](01-folder-and-file-map.md)
3. [Python Foundations](02-python-foundations.md)
4. [Data Pipeline](03-data-pipeline.md)
5. [Feature Engineering](04-feature-engineering.md)
6. [Machine Learning Models](05-machine-learning-models.md)
7. [Prediction Runtime](06-prediction-runtime.md)
8. [Tournament Simulation](07-tournament-simulation.md)
9. [Flask API and UI](08-flask-api-and-ui.md)
10. [Testing, Deployment, and Risks](09-testing-deployment-and-risks.md)
11. [Interview and Rebuild Guide](10-interview-and-rebuild-guide.md)
12. [Current Changes Deep Dive](11-current-changes.md)
13. [Resources](appendix-resources.md)
14. [Glossary](appendix-glossary.md)

## Fast Path For The New Changes

If you already studied the first curriculum draft, review these in order:

1. [Current Changes Deep Dive](11-current-changes.md)
2. [Prediction Runtime](06-prediction-runtime.md)
3. [Flask API and UI](08-flask-api-and-ui.md)
4. [Testing, Deployment, and Risks](09-testing-deployment-and-risks.md)
5. [Folder and File Map](01-folder-and-file-map.md)

## What This Curriculum Does Not Claim

- It does not claim the app is production-grade.
- It does not claim the trained model is optimal.
- It does not invent missing history. If the repo does not contain a fact, the lessons say so.
- It does not document secret values from `.env`.

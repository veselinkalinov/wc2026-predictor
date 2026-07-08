# Appendix - Glossary

## Data Terms

| Term | Meaning in this project |
|---|---|
| Raw data | Original CSV files in `data/raw`. |
| Processed data | Cleaned match table in `data/processed/matches_clean.csv`. |
| Feature matrix | Final model-ready table in `data/features/feature_matrix.csv`. |
| Feature-matrix reload | Runtime behavior where the predictor reloads team state when `feature_matrix.csv` changes. |
| Target | The value the classifier learns to predict: `result`. |
| Schema | Expected columns and types in a data file. |
| Temporal join | Joining data by time, such as ranking snapshot before a match date. |
| Leakage | Accidentally using future or post-outcome information during training/prediction. |

## Feature Terms

| Term | Meaning |
|---|---|
| Elo | Rating system estimating team strength from match results. |
| K-factor | Controls how much Elo changes after a match. |
| Home-field advantage | Virtual rating/context boost for non-neutral home teams. |
| EWMA | Exponentially weighted moving average, with more weight on recent events. |
| Form | Recent opponent-adjusted point performance. |
| Goal average | Rolling goals scored/conceded estimate. |
| Match stake | Numeric tournament importance level. |
| Rest-day inference | Runtime calculation that estimates a team's rest days from its latest known match before the prediction date. |
| Match context | Prediction metadata such as match date, rest days, rest source, and match stake. |

## ML Terms

| Term | Meaning |
|---|---|
| Classifier | Model that predicts categories such as H/D/A. |
| Regressor | Model that predicts numeric values such as expected goals. |
| Logistic Regression | Linear classifier that outputs probabilities. |
| Random Forest | Ensemble of decision trees trained on random subsets. |
| Gradient boosting | Sequential tree ensemble that corrects previous errors. |
| Stacking | Ensemble where a final model combines base model predictions. |
| Calibration | Adjusting model probabilities to better match observed frequencies. |
| Log loss | Probability-based loss; lower is better. |
| Brier score | Mean squared error of predicted probabilities; lower is better. |
| Accuracy | Fraction of correct hard-label predictions. |
| Holdout test set | Data kept out of training/calibration for final evaluation. |
| TimeSeriesSplit | Cross-validation that respects chronological order. |
| StratifiedKFold | Cross-validation that tries to preserve class balance in each fold. Used for production calibration after refitting on all completed rows. |
| Out-of-fold calibration | Calibration learned from predictions made on folds not used to fit that fold's estimator, reducing direct in-sample calibration bias. |
| Production refit | Final serving artifact trained/calibrated on all completed rows after model selection and holdout evaluation are done. |
| Evaluation artifact | Saved model/scaler used to report holdout metrics without mixing them with production-refit artifacts. |

## Runtime Terms

| Term | Meaning |
|---|---|
| Artifact | Saved output such as model pickle, scaler, metadata, or plot. |
| `meta.json` | Runtime contract describing model type, features, classes, metrics, thresholds. |
| Artifact role | Metadata field explaining whether a saved artifact is for production serving or holdout evaluation. |
| Scaler | Object that standardizes features using training mean and scale. |
| Hot reload | Predictor checks model and feature-matrix file modification times and reloads runtime state when needed. |
| Prediction cache | In-memory dictionary of recent matchup predictions. |
| Symmetric prediction | Averaging forward and reverse predictions for neutral matches. |
| Scoreline matrix | Grid of probabilities for home goals vs away goals. |

## Web Terms

| Term | Meaning |
|---|---|
| Flask | Python web framework used by the app. |
| App factory | Function that creates and configures a Flask app. |
| Blueprint | Flask route grouping mechanism. |
| WSGI | Python standard interface between a web server and a Python web application. |
| Gunicorn | WSGI HTTP server used by Docker to serve the Flask app through `wsgi:app`. |
| Route | URL handler function. |
| JSON endpoint | API route that returns JSON. |
| Template | HTML file rendered by Flask. |
| CORS | Browser policy controlling cross-origin API access. |
| `fetch` | Browser API for HTTP requests from JavaScript. |

## Deployment Terms

| Term | Meaning |
|---|---|
| Dockerfile | Instructions for building a container image. |
| Docker Compose | Tool for running multiple services together. |
| Service | A container role, such as `web` or `scheduler`. |
| Volume mount | Mapping local files into a container. |
| Environment variable | Runtime setting supplied outside code, often secrets/config. |
| WSGI entry point | A module-level app object that a WSGI server imports, implemented here as `wsgi.py`. |

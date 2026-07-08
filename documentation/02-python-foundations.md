# 02 - Python Foundations

## What You Will Understand After This Lesson

- How imports connect the project.
- How `config.yaml` becomes a Python dictionary.
- How logging works across scripts and Flask routes.
- Why scripts use `if __name__ == "__main__"`.
- How the virtual environment, requirements file, Dockerfile, Compose file, WSGI entry point, and Gunicorn runtime fit together.

## First Principles

Python projects usually separate reusable code from executable entrypoints.

- Reusable code lives in packages and modules.
- Entry points import reusable code and call one main workflow.
- Configuration should be read in one place.
- Logging should be centralized so every module emits consistent messages.

## Project-Specific Walkthrough

The reusable package is `src`. Scripts and Flask import from it:

```python
from src.utils.logger import get_logger
from src.data.clean import run_cleaning
from src.features.build import build_feature_matrix
```

The project depends on `PYTHONPATH` or running from the project root so `src.*` imports resolve. Docker sets `PYTHONPATH=/app`.

## Code-Block Explanations

### Config Loading

File: `src/utils/config.py`

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

This starts from `src/utils/config.py`, resolves the absolute path, then goes two levels up to the project root.

```python
def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
```

This reads YAML safely into a dictionary. `safe_load` is used instead of unsafe YAML loading.

```python
config = load_config()
```

This executes at import time. Any module that imports `config` gets the same loaded dictionary.

### Logging

File: `src/utils/logger.py`

```python
if logger.handlers:
    return logger
```

This prevents duplicate handlers when multiple modules call `get_logger(__name__)`.

```python
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(log_dir / "app.log")
```

The same log message can appear in the terminal and in `logs/app.log`. Console level is INFO, file level is DEBUG.

### Script Entry Points

Files: `scripts/run_pipeline.py`, `scripts/retrain.py`, `scripts/scheduler.py`, `scripts/fetch_recent_matches.py`, `scripts/optimize_elo.py`

```python
if __name__ == "__main__":
    main()
```

This means the script runs when executed directly, but importing it does not immediately run the workflow.

### WSGI and Gunicorn Runtime

First principle:

Flask is a web framework. It knows how to route requests once Python receives them, but the development server is not the normal production entry point. A WSGI server is the process that receives HTTP requests, calls the Flask application object, and sends the response back.

File: `wsgi.py`

```python
from src.api.app import create_app

app = create_app()
```

This file exists because Gunicorn expects a module and variable name. The command `gunicorn wsgi:app` means:

```text
import module named wsgi
find variable named app
call that object as the WSGI application
```

The project uses the Flask app factory pattern in `src/api/app.py`, but Gunicorn needs a concrete application object at runtime. `wsgi.py` bridges those two ideas.

Common mistake:

Do not say Gunicorn "runs `src/api/app.py`." It imports `wsgi.py`, and `wsgi.py` creates the Flask app by calling `create_app()`.

### Docker Runtime

File: `Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 -r requirements.txt
COPY . .
RUN python -m pytest tests/ -v
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]
```

The container installs dependencies, copies source, runs tests during build, and starts Gunicorn. The `--access-logfile -` and `--error-logfile -` settings send logs to container stdout/stderr so Docker can collect them.

File: `docker-compose.yaml`

The `web` service starts the Flask app. The `scheduler` service starts `scripts.scheduler`. Both mount local folders so data/model changes are visible inside containers.

## File-by-File Explanation

| File | What to understand |
|---|---|
| `config.yaml` | Runtime parameters: paths, data range, feature params, model cutoffs, API host/port. |
| `requirements.txt` | Exact dependency pins used by local venv and Docker, including `gunicorn==23.0.0` for container web serving. |
| `src/utils/config.py` | Loads config once at import time. |
| `src/utils/logger.py` | Creates consistent console/file loggers. |
| `Dockerfile` | Builds a test-checked Flask image and runs Gunicorn against `wsgi:app`. |
| `docker-compose.yaml` | Runs web and scheduler services together. |
| `wsgi.py` | Creates the module-level Flask app object expected by Gunicorn. |
| `.gitignore` | Keeps generated models/logs/local docs out of git while allowing current data snapshots. |
| `.dockerignore` | Keeps local-only files out of Docker build context. |

## Common Interview Questions

| Question | Strong answer |
|---|---|
| Why use `config.yaml` instead of constants everywhere? | It centralizes parameters and paths so pipeline behavior can change without editing many modules. |
| What is the risk of `config = load_config()` at import time? | Config changes after startup are not automatically reloaded. Long-running services need restart or explicit reload logic. |
| Why avoid duplicate logger handlers? | Otherwise repeated imports can cause the same log message to be written multiple times. |
| Why does Docker run tests during build? | It catches missing dependencies and broken runtime contracts before the image is used. |
| Why add `wsgi.py` if the app already has `create_app()`? | The factory is flexible for tests/configuration, while `wsgi.py` exposes the concrete app object that Gunicorn imports. |
| Why use Gunicorn instead of `python -m src.api.app` in Docker? | Gunicorn is a WSGI server designed for serving Python web apps with workers, timeouts, and production-style logging. Flask's built-in server is mainly for development. |

## Rebuild Exercise

Create a minimal project with:

- `config.yaml`
- `src/utils/config.py`
- `src/utils/logger.py`
- `scripts/run_pipeline.py`

Make the script read one config value and log it.

## Self-Check Quiz

1. What does `PROJECT_ROOT` solve?
2. Why does Docker set `PYTHONPATH=/app`?
3. What does `if __name__ == "__main__"` prevent?
4. Why is `.env` ignored?

Answers:

1. It lets modules build paths relative to the repo root.
2. So `src.*` imports resolve inside the container.
3. It prevents workflows from running just because the module was imported.
4. It may contain API keys/secrets.

## External Links

- Python modules: https://docs.python.org/3/tutorial/modules.html
- Python logging: https://docs.python.org/3/library/logging.html
- PyYAML docs: https://pyyaml.org/wiki/PyYAMLDocumentation
- Dockerfile docs: https://docs.docker.com/build/concepts/dockerfile/
- Docker Compose concepts: https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-docker-compose/
- Flask Gunicorn deployment: https://flask.palletsprojects.com/en/stable/deploying/gunicorn/
- Gunicorn official site: https://gunicorn.org/

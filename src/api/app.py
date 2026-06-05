import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING_BACKEND"] = "threading"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

"""
app.py

Responsibility: Flask application factory. Configures extensions, CORS,
and registers blueprints for routes.
"""

from flask import Flask
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    # Load configuration parameters
    app.config["DEBUG"] = config["api"]["debug"]
    app.config["HOST"] = config["api"]["host"]
    app.config["PORT"] = config["api"]["port"]

    # Basic CORS support (CORS is needed to connect to frontend dashboards)
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    # Register blueprints (routes will be defined in routes.py)
    from src.api.routes import api_bp, pages_bp
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(pages_bp, url_prefix="")

    logger.info("Flask application created and configured.")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=app.config["HOST"], port=app.config["PORT"])

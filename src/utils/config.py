import yaml
from pathlib import Path

# Absolute path to the project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Single shared instance — import this directly in other modules
config = load_config()

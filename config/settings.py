"""
config/settings.py
==================
Loads the project configuration ONCE and gives the rest of the code a single,
clean way to ask questions like:

    from config.settings import settings
    settings.paths.parquet        # -> absolute Path to data/parquet
    settings.db_url               # -> SQLAlchemy connection string
    settings.config["ifrs9"]["sicr_dpd_threshold"]   # -> 30

Why this exists:
- Paths are resolved to ABSOLUTE paths from the project root, so the code works
  no matter which folder you run it from.
- Secrets (DB password, API keys) come from a .env file and are NEVER hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv

# The project root is the folder that contains this 'config' package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from a .env file at the project root (if present).
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml() -> dict:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings:
    """A small wrapper around the YAML config + environment secrets."""

    def __init__(self) -> None:
        self.config: dict = _load_yaml()

        # Turn the 'paths' section into absolute Paths for convenience.
        raw_paths = self.config.get("paths", {})
        self.paths = SimpleNamespace(
            **{key: (PROJECT_ROOT / value) for key, value in raw_paths.items()}
        )
        self.project_root = PROJECT_ROOT

        # Make sure the working folders actually exist.
        for p in self.paths.__dict__.values():
            p.mkdir(parents=True, exist_ok=True)

    # ---- Database -----------------------------------------------------
    @property
    def db_url(self) -> str:
        """SQLAlchemy/psycopg2 connection string built from config + .env."""
        db = self.config["database"]
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", db.get("host", "localhost"))
        port = os.getenv("DB_PORT", db.get("port", 5432))
        name = os.getenv("DB_NAME", db.get("name", "ifrs9"))
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"

    @property
    def random_seed(self) -> int:
        return int(self.config["project"]["random_seed"])


# A single shared instance the whole project imports.
settings = Settings()

"""
tests/test_smoke.py
===================
A tiny "does the building stand up?" test. It checks that:
  1. The config loads,
  2. The expected folders exist,
  3. The logger can be created.

Run it with:  pytest
If this passes, your Stage 0 foundation is healthy.
"""

from config.settings import settings
from src.utils.logger import get_logger


def test_config_loads():
    assert settings.config["project"]["name"] == "ifrs9_ecl"
    assert settings.random_seed == 42


def test_paths_exist():
    assert settings.paths.raw_data.exists()
    assert settings.paths.parquet.exists()
    assert settings.paths.logs.exists()


def test_db_url_builds():
    url = settings.db_url
    assert url.startswith("postgresql+psycopg2://")
    assert "/ifrs9" in url


def test_logger_works():
    log = get_logger("test_logger")
    log.info("smoke test logging works")
    assert log.name == "test_logger"


def test_ifrs9_thresholds_present():
    cfg = settings.config["ifrs9"]
    assert cfg["sicr_dpd_threshold"] == 30
    assert cfg["default_dpd_threshold"] == 90

"""
conftest.py (project root)
==========================
Putting this file at the root makes pytest add the project root to the import
path, so tests can do `from config.settings import settings` and
`from src.utils.logger import get_logger` without any extra setup.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

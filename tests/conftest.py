"""Shared pytest configuration for the toric-code test suite.

The tests are intended to be run from the repository root with:

    pytest

This file ensures that the local package is importable even if the package has
not been installed in editable mode.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

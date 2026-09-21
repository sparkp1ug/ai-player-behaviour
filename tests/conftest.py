"""Shared fixtures.

src/ modules import each other flatly (`from schema import ...`), so src/ has
to be on the path before any test imports them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def games():
    """A generated catalogue. Built here rather than read from data/games.csv
    so the suite works on a clean clone — data/ is gitignored."""
    from generate_synthetic_data import generate_games

    return generate_games(15, np.random.default_rng(42))


@pytest.fixture(scope="session")
def machine(games):
    from slot_machine import SlotMachine

    return SlotMachine(games, seed=7)

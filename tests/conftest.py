"""Gemeinsame Fixtures: kleine Modelle und von Hand gebaute Ergebnisse, an denen jede Einheit nachgerechnet wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eli_pool as P  # noqa: E402


def small(**kw) -> P.Model:
    """Kleines Modell (8 Spieler, 4 Saisons zu 3 Runden), für Einheiten- und Orakeltests."""
    base = {"n": 8, "seasons": 4, "games": 3, "churn": 0.25, "leave": "zufällig", "k": 20.0}
    base.update(kw)
    return P.Model(**base)


def hand_pool(rating=(0.0,), true=(0.0,), anchor=None, base=0.0, model=None, **kw) -> P.Pool:
    """Ergebnis von Hand: ein Lauf, eine Saison je Eintrag von `rating`/`true`; alle übrigen Felder null."""
    s = len(rating)
    z = np.zeros((1, s))
    nan = np.full((1, s), np.nan)
    fields = dict(mean_rating=np.array([rating], dtype=float), mean_true=np.array([true], dtype=float), base_true=np.array([base], dtype=float),
                  anchor=nan if anchor is None else np.array([anchor], dtype=float), total_start=z.copy(), total_end=z.copy(), created=z.copy(), floor_lift=z.copy(),
                  churn_delta=z.copy(), anchor_in=z.copy(), anchor_out=z.copy(), entry_mean=nan.copy(), entry_msq=nan.copy(), final_rating=np.zeros((1, 3)),
                  final_true=np.zeros((1, 3)), final_age=np.zeros((1, 3)))
    fields.update({k: np.array(v, dtype=float) for k, v in kw.items()})
    return P.Pool(model or small(), (0,), **fields)


@pytest.fixture
def model():
    return small()

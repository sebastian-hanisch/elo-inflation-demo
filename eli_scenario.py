"""Szenario: Reglerwerte (Settings) und das daraus erzeugte Modell des offenen Feldes."""

from dataclasses import dataclass

import eli_constants as C
import eli_pool as P


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seasons: int = C.DEFAULT_SEASONS
    games: int = C.DEFAULT_GAMES
    churn: float = C.DEFAULT_CHURN
    leave: str = C.DEFAULT_LEAVE
    gap: float = C.DEFAULT_GAP
    trend: float = C.DEFAULT_TREND
    k: float = C.DEFAULT_K
    use_k0: bool = False
    k0: float = C.DEFAULT_K0                  # nur mit use_k0
    use_floor: bool = False
    floor: float = C.DEFAULT_FLOOR            # nur mit use_floor
    k_steps: bool = False
    start: str = C.START[0]
    prior: float = C.DEFAULT_PRIOR            # nur bei Platzierungspartien
    use_anchor: bool = True
    join: int = C.DEFAULT_JOIN                # nur mit Anker
    seed: int = C.DEFAULT_SEED

    @property
    def placement(self) -> bool:
        return self.start == C.START[1]

    @property
    def anchor_join(self):
        """Saisons, die vor dem Beitritt des Ankers gespielt sind (None = kein Anker); der Anker spielt mindestens eine Saison."""
        if not self.use_anchor:
            return None
        return min(max(self.join, C.JOIN_MIN), self.seasons - 2)


def join_max(seasons: int) -> int:
    """Spätester Beitritt des Ankers: danach bleiben mindestens zwei Saisons (plus eine, damit der Regler eine Spanne hat)."""
    return seasons - 2


def make_model(s: Settings) -> P.Model:
    return P.Model(n=s.n, seasons=s.seasons, games=s.games, churn=s.churn, leave=s.leave, gap=s.gap, trend=s.trend, k=s.k, k0=s.k0 if s.use_k0 else None,
                   floor=s.floor if s.use_floor else None, k_steps=s.k_steps, placement=s.placement, prior=s.prior, anchor_join=s.anchor_join)

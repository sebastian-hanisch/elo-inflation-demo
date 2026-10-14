"""Offenes Feld: Saisons, Abgang und Zugang, Lernen und Altern, Referenzspieler (lokal; der gemeinsame Generator liefert nur die Gründerstärken und den Zufallsstrom).

Aufbau: n Spieler mit gleichabständigen Gründerstärken (Streuung POP_SD, Mittel 0 = 1500) und ein Platz n für den Referenzspieler („Anker“, konstante Stärke 0). Je Saison
`games` Runden; in jeder Runde spielen alle aktiven Plätze in zufälliger Paarung (bei ungerader Zahl setzt einer aus). Nach der Saison ändern sich die Stärken (Trend aller,
Aufholen der Lernenden), die Kennzahlen werden gemessen, dann gehen n_out Spieler und ebenso viele Neue kommen. Zufall nur über SplitMix64; der Verbrauch je Lauf hat
eine feste Reihenfolge und Größe, die nicht von der Platzierung oder der Abgangsregel abhängt (gepaarte Vergleiche teilen ihre Zufallszahlen)."""

import math
from dataclasses import dataclass

import numpy as np

import eli_constants as C
import eli_rating as R
import turnier_generator as T

_SALT = 0x6E10


@dataclass(frozen=True)
class Model:
    n: int = C.DEFAULT_N
    seasons: int = C.DEFAULT_SEASONS
    games: int = C.DEFAULT_GAMES
    churn: float = C.DEFAULT_CHURN
    leave: str = C.DEFAULT_LEAVE
    gap: float = C.DEFAULT_GAP                 # Neue sind so viel schwächer als ihre spätere Stärke und holen GROWTH je Saison auf
    trend: float = C.DEFAULT_TREND             # Änderung der Stärke aller Spieler je Saison; Neue kommen auf dem Niveau des Feldes
    k: float = C.DEFAULT_K
    k0: object = None                          # Anlauf-K in Punkten oder None
    floor: object = None                       # Untergrenze der Wertung in Punkten oder None
    k_steps: bool = False
    placement: bool = False                    # Neue starten bei der Schätzung aus Platzierungspartien statt bei 1500
    prior: float = C.DEFAULT_PRIOR
    anchor_join: object = None                 # Saisons, die vor dem Beitritt des Ankers gespielt sind, oder None
    anchor_stay: int = 0                       # Saisons, die der Anker bleibt (0 = bis zum Ende)

    @property
    def n_out(self) -> int:
        return int(round(self.churn * self.n))

    @property
    def slots(self) -> int:
        return self.n + 1


@dataclass(frozen=True)
class Pool:
    """Ergebnis für B Läufe, Wertungen in θ-Einheiten (0 = 1500)."""
    model: Model
    seeds: tuple
    mean_rating: np.ndarray      # (B, S) Mittel der Wertungen der n Feldspieler nach der Saison
    mean_true: np.ndarray        # (B, S) Mittel ihrer wahren Stärken nach der Saison
    base_true: np.ndarray        # (B,) Mittel der Gründerstärken
    anchor: np.ndarray           # (B, S) Wertung des Ankers nach der Saison, NaN außerhalb seiner Zeit
    total_start: np.ndarray      # (B, S) Summe der Wertungen aller aktiven Spieler zu Beginn der Saison (nach dem Beitritt des Ankers)
    total_end: np.ndarray        # (B, S) Summe nach der Saison, vor Abgang und Zugang
    created: np.ndarray          # (B, S) Punkte, die die Partien der Saison erzeugt haben (ungleiches K)
    floor_lift: np.ndarray       # (B, S) Punkte, die die Untergrenze der Saison erzeugt hat
    churn_delta: np.ndarray      # (B, S) Startwertungen der Neuen minus Wertungen der Gehenden, am Ende der Saison
    anchor_in: np.ndarray        # (B, S) Startwertung des Ankers, wenn er zu Beginn der Saison beitritt, sonst 0
    anchor_out: np.ndarray       # (B, S) minus Wertung des Ankers, wenn er am Ende der Saison geht, sonst 0
    entry_mean: np.ndarray       # (B, S) Mittel (Startwertung − wahre Stärke) der Neuen am Ende der Saison, NaN ohne Zugang
    entry_msq: np.ndarray        # (B, S) Mittel des Quadrats
    final_rating: np.ndarray     # (B, n + 1)
    final_true: np.ndarray       # (B, n + 1)
    final_age: np.ndarray        # (B, n + 1) Saisons im Feld (Anker: Saisons seit dem Beitritt)

    @property
    def runs(self) -> int:
        return len(self.seeds)


def normals(rng: T.SplitMix64, size: int) -> np.ndarray:
    """Standardnormalverteilte Zahlen per Box-Muller aus der Gleichverteilten des Generators."""
    u1 = 1.0 - rng.uniform(size)
    u2 = rng.uniform(size)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def draws(seed: int, seasons: int, games: int, slots: int) -> dict:
    """Alle Zufallszahlen eines Laufs in fester Reihenfolge: Potenziale der Neuen, Paarungsschlüssel, Ausgänge, Abgangsschlüssel, Gegner und Ausgänge der Platzierung."""
    rng = T.SplitMix64(int(seed) ^ _SALT)
    S, G, P = seasons, games, slots
    return {
        "z": normals(rng, S * P).reshape(S, P),
        "pair": rng.uniform(S * G * P).reshape(S, G, P),
        "win": rng.uniform(S * G * (P // 2)).reshape(S, G, P // 2),
        "leave": rng.uniform(S * P).reshape(S, P),
        "opp": rng.uniform(S * P * C.PLACE_GAMES).reshape(S, P, C.PLACE_GAMES),
        "pwin": rng.uniform(S * P * C.PLACE_GAMES).reshape(S, P, C.PLACE_GAMES),
    }


def founder_strengths(n: int) -> np.ndarray:
    """Gleichabständige Gründerstärken des Generators, Mittel 0, Standardabweichung POP_SD."""
    return T.make_strengths(n, C.POP_SD * math.sqrt(12.0 / (n * n - 1)))


def leavers(model: Model, rating, age, key) -> np.ndarray:
    """Plätze (B, n_out) der Spieler, die gehen. `key` (B, n) sind Gleichverteilte für die Auswahl bzw. den Gleichstand."""
    if model.leave == C.LEAVE[0]:
        order = key
    elif model.leave == C.LEAVE[1]:
        order = rating + 1e-9 * key                       # niedrigste Wertung zuerst, Gleichstand zufällig
    elif model.leave == C.LEAVE[2]:
        order = -age + 0.5 * key                          # längste Zeit im Feld zuerst, Gleichstand zufällig
    else:
        raise ValueError(f"unbekannte Abgangsregel: {model.leave}")
    return np.argsort(order, axis=1, kind="stable")[:, :model.n_out]


def simulate(model: Model, seeds) -> Pool:
    n, S, G, P = model.n, model.seasons, model.games, model.slots
    seeds = tuple(int(s) for s in seeds)
    B = len(seeds)
    if n < 2 or S < 1 or G < 1:
        raise ValueError("n, seasons und games müssen mindestens 2, 1 und 1 sein")
    if model.n_out > n - 1:
        raise ValueError("Zu viele Abgänge")
    if model.anchor_join is not None and not 0 <= model.anchor_join < S:
        raise ValueError("Der Anker muss innerhalb der Saisons beitreten")
    dr = [draws(s, S, G, P) for s in seeds]
    z, upair, uwin, uleave, uopp, upw = (np.stack([d[key] for d in dr]) for key in ("z", "pair", "win", "leave", "opp", "pwin"))
    rows = np.arange(B)[:, None]
    rating = np.zeros((B, P))
    true = np.zeros((B, P))
    true[:, :n] = founder_strengths(n)[None, :]
    potential = true.copy()
    played = np.zeros((B, P))
    age = np.zeros((B, P))
    active = np.zeros(P, dtype=bool)
    active[:n] = True
    base = true[:, :n].mean(axis=1)
    level = 0.0
    out = {key: np.zeros((B, S)) for key in ("mean_rating", "mean_true", "total_start", "total_end", "created", "floor_lift", "churn_delta", "anchor_in", "anchor_out")}
    out["anchor"] = np.full((B, S), np.nan)
    out["entry_mean"] = np.full((B, S), np.nan)
    out["entry_msq"] = np.full((B, S), np.nan)
    floor_on = model.floor is not None
    field = np.zeros((B, P), dtype=bool)
    field[:, :n] = True
    for s in range(S):
        if model.anchor_join is not None and s == model.anchor_join:
            active[n] = True
            true[:, n] = 0.0
            potential[:, n] = 0.0
            played[:, n] = 0.0
            age[:, n] = 0.0
            start = np.zeros(B)
            if model.placement:
                start = R.placement_estimates(true[:, n:n + 1], rating, true, field, uopp[:, s, n:n + 1], upw[:, s, n:n + 1], model.prior)[:, 0]
            rating[:, n] = start
            out["anchor_in"][:, s] = start
        out["total_start"][:, s] = rating[:, active].sum(axis=1)
        m = int(active.sum()) // 2
        for g in range(G):
            keys = np.where(active[None, :], upair[:, s, g, :], 2.0)
            order = np.argsort(keys, axis=1, kind="stable")
            out["created"][:, s] += R.play_round(rating, true, played, order[:, 0:2 * m:2], order[:, 1:2 * m:2], uwin[:, s, g, :m], model.k, model.k0, model.k_steps)
            if floor_on:
                out["floor_lift"][:, s] += R.apply_floor(rating, active, model.floor)
        out["total_end"][:, s] = rating[:, active].sum(axis=1)
        if model.trend:
            true[:, :n] += model.trend
            potential[:, :n] += model.trend
        level += model.trend
        true[:, :n] += (potential[:, :n] - true[:, :n]) * C.GROWTH
        age[:, :n] += 1
        out["mean_rating"][:, s] = rating[:, :n].mean(axis=1)
        out["mean_true"][:, s] = true[:, :n].mean(axis=1)
        if active[n]:
            out["anchor"][:, s] = rating[:, n]
            age[:, n] += 1
            if model.anchor_stay and s == model.anchor_join + model.anchor_stay - 1:
                out["anchor_out"][:, s] = -rating[:, n]
                active[n] = False
        n_out = model.n_out
        if n_out and s < S - 1:
            idx = leavers(model, rating[:, :n], age[:, :n], uleave[:, s, :n])
            pot = level + C.POP_SD * z[rows, s, idx]
            new_true = pot - model.gap
            stay = field.copy()
            stay[rows, idx] = False
            start = np.zeros((B, n_out))
            if model.placement:
                start = R.placement_estimates(new_true, rating, true, stay, uopp[rows, s, idx], upw[rows, s, idx], model.prior)
            out["churn_delta"][:, s] = start.sum(axis=1) - rating[rows, idx].sum(axis=1)
            err = start - new_true
            out["entry_mean"][:, s] = err.mean(axis=1)
            out["entry_msq"][:, s] = (err ** 2).mean(axis=1)
            rating[rows, idx] = start
            true[rows, idx] = new_true
            potential[rows, idx] = pot
            played[rows, idx] = 0.0
            age[rows, idx] = 0.0
    return Pool(model, seeds, out["mean_rating"], out["mean_true"], base, out["anchor"], out["total_start"], out["total_end"], out["created"], out["floor_lift"],
                out["churn_delta"], out["anchor_in"], out["anchor_out"], out["entry_mean"], out["entry_msq"], rating.copy(), true.copy(), age.copy())

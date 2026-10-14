"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit θ (400 Punkte = Faktor 10 der Gewinnchancen), 173,72
BASE = 1500.0                               # Mitte der Elo-Skala in der Anzeige und Startwertung
POP_SD = 0.8                                # Streuung der wahren Stärken im Feld (θ-Einheiten, ≈ 139 Punkte): Gründer und Zugänge
GROWTH = 0.25                               # Anteil des Abstands zur eigenen Spitze, den ein lernender Neuer je Saison aufholt
G0 = 10.0                                   # Anlauf-K: nach G0 Partien ist der Aufschlag halbiert
STEP_FROM, STEP_K = 1600.0, 10.0            # K-Stufe: ab dieser Wertung gilt K = 10
PLACE_GAMES = 10                            # Platzierungspartien eines Neuen
PLACE_ITERS = 25                            # Newton-Schritte der Platzierungsschätzung (der Prior hält das Problem strikt konkav)

N_MIN, N_MAX, N_STEP, DEFAULT_N = 10, 100, 2, 40
SEASONS_MIN, SEASONS_MAX, DEFAULT_SEASONS = 5, 60, 40
GAMES_MIN, GAMES_MAX, DEFAULT_GAMES = 5, 40, 20                              # Runden je Saison (jeder spielt je Runde einmal)
CHURN_MIN, CHURN_MAX, CHURN_STEP, DEFAULT_CHURN = 0.0, 0.4, 0.05, 0.15       # Anteil, der je Saison geht (und ebenso viele kommen)
LEAVE = ("zufällig", "die Schwächsten", "die Veteranen")
DEFAULT_LEAVE = LEAVE[1]
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 0.0, 2.0, 0.25, 0.0                # Abstand eines Neuen unter seiner späteren Stärke (θ-Einheiten)
TREND_MIN, TREND_MAX, TREND_STEP, DEFAULT_TREND = -0.04, 0.04, 0.01, 0.0     # Änderung der Stärke aller je Saison (θ-Einheiten)
K_MIN, K_MAX, DEFAULT_K = 5, 60, 20                                          # Punkte
K0_MIN, K0_MAX, DEFAULT_K0 = 20, 300, 120                                    # Anlauf-K in Punkten
FLOOR_MIN, FLOOR_MAX, FLOOR_STEP, DEFAULT_FLOOR = 1000, 1450, 50, 1350       # Untergrenze der Wertung in Punkten
START = ("Startwertung 1500", "Platzierungspartien")
PRIOR_MIN, PRIOR_MAX, PRIOR_STEP, DEFAULT_PRIOR = 0.3, 1.5, 0.1, 1.0         # Prior der Platzierung (Standardabweichung, θ-Einheiten); darüber wächst die Drift unbegrenzt (Experiment 6)
JOIN_MIN, DEFAULT_JOIN = 0, 2                                                # Saisons, die vor dem Beitritt des Ankers gespielt sind (Obergrenze: Saisons − 2)
SEED_MAX, DEFAULT_SEED = 999999, 17                                         # ein typischer Lauf: Drift und Anker nahe dem Mittel der 200 Läufe 0 bis 199 (nicht der auffälligste)

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_SEASONS, EXP_GAMES, EXP_CHURN, EXP_K = 40, 40, 20, 0.15, 20
EXP_MARKS = (10, 20, 40)                                                     # Saisons, nach denen die Drift berichtet wird
ANCHOR_ERAS = (2, 15, 30)                                                    # Saisons, die bis zum Beitritt des Ankers gespielt sind
ANCHOR_STAY = 4                                                              # Saisons, die der Anker im Experiment spielt (80 Partien)
LEARN_STEP = 0.02                                                            # Lernen (+) bzw. Altern (−) des ganzen Feldes je Saison im Experiment 2
EXP_K0 = 120                                                                 # Anlauf-K in den Experimenten
EXP_FLOOR = 1350
PRIOR_GRID = (0.5, 1.0, 2.0)

COLOR_RATING = "#2F6B65"
COLOR_TRUTH = "#e67e22"
COLOR_ANCHOR = "#8e6bb0"
COLOR_CREATED = "#c0392b"
PALETTE = ("#2F6B65", "#8e6bb0", "#d98c1f", "#c0392b", "#3d7ab5", "#7a8f3c", "#b0577e", "#5a5a5a")

_BASE = {"n": DEFAULT_N, "seasons": DEFAULT_SEASONS, "games": DEFAULT_GAMES, "churn": DEFAULT_CHURN, "leave": DEFAULT_LEAVE, "gap": DEFAULT_GAP, "trend": DEFAULT_TREND,
         "k": DEFAULT_K, "use_k0": False, "k0": DEFAULT_K0, "use_floor": False, "floor": DEFAULT_FLOOR, "k_steps": False, "start": START[0], "prior": DEFAULT_PRIOR,
         "use_anchor": True, "join": DEFAULT_JOIN, "seed": DEFAULT_SEED}
PRESETS = {
    "Geschlossenes Feld": {**_BASE, "churn": 0.0},
    "Die Schwächsten gehen": dict(_BASE),
    "Das Feld lernt": {**_BASE, "leave": LEAVE[0], "trend": 0.02},
    "Neue unter ihrer Startwertung": {**_BASE, "leave": LEAVE[0], "gap": 1.0},
}
PRESET_HELP = {
    "Geschlossenes Feld": "Niemand geht, niemand kommt: Die Summe der Wertungen bleibt 40 · 1500, das Mittel steht still. Die Wertungen sind relativ, aber hier bewegt sich nichts.",
    "Die Schwächsten gehen": "Jede Saison gehen 15 % mit der niedrigsten Wertung, Neue kommen bei 1500: Die Wertungen im Feld steigen langsamer als die Fähigkeit (Deflation).",
    "Das Feld lernt": "Alle werden je Saison um 0,02 Einheiten besser, Neue kommen auf dem Niveau des Feldes: Die Wertungen bleiben bei 1500, der Anker verliert Punkte.",
    "Neue unter ihrer Startwertung": "Neue kommen bei 1500, sind aber 1 Einheit schwächer als ihre spätere Stärke und lernen: Die Neuen tragen Punkte ins Feld (Inflation), solange sie lernen.",
}

"""Literatur-Referenzwerte für Elo-Inflation (ORACLE-PLAYBOOK §5.1). Nur Werte, die in einer Quelle eingesehen wurden (Stand der Einsicht: 2026-10-17).

Umrechnung: Die Demo rechnet in θ-Einheiten (1 Einheit = SCALE = 400/ln 10 ≈ 173,72 Elo-Punkte, 0 = 1500) und mit k = K/SCALE. Die Tests rechnen die Quellwerte mit
θ = (R − 1500)/SCALE in die Demo-Einheiten (`eli_rating.sig`, `eli_rating.play_round`) und zurück (R = 1500 + SCALE·θ).

Werte
-----
1. Erwartung 0,64 bei 100 Punkten Unterschied und 0,76 bei 200 Punkten (Gleichstand 0,50).
   Quelle: FIDE Handbook, B.02 FIDE Rating Regulations (effective from 1 January 2022 till 29 February 2024), Abschnitt 8.1, Tabelle 8.1.2 („Rtg Dif / H / L“): Zeilen
   „0-3 .50 .50“, „99-106 .64 .36“, „198-206 .76 .24“. URL: https://handbook.fide.com/chapter/B022022 (per WebFetch eingesehen am 2026-10-17; die Zeilen kamen aus der
   Zusammenfassung des Abrufwerkzeugs, der Wortlaut der ganzen Tabelle ließ sich nicht abrufen). Bestätigt durch https://en.wikipedia.org/wiki/Elo_rating_system
   (am 2026-10-17): „A player whose rating is 100 points greater than their opponent's is expected to score 64%“, bei 200 Punkten 76%. Nicht getestet: Bereichsränder
   und 400 Punkte (Tafel .92, logistisch 0,909): Die FIDE-Tafel stammt aus einer Normalverteilung, die Demo rechnet logistisch.
2. Änderung einer Wertung nach FIDE-Regel 8.3.2: ΔR = Punkte − PD (PD aus Tabelle 8.1.2), Wertungsänderung = K · ΔR. Dieselbe Quelle und Einsicht wie 1. Mit K = 20, 100 Punkte
   Vorsprung und Sieg des Höherbewerteten ist die Änderung 20·(1 − 0,64) = 7,2 Punkte (mit PD aus der gerundeten Tafel; die Demo rechnet 20·(1 − 0,6401) = 7,20).
3. Erwartungen des Beispielturniers der Wikipedia: Spieler A mit 1613 gegen 1609, 1477, 1388, 1586, 1720: 0,51 / 0,69 / 0,79 / 0,54 / 0,35 (gerundet auf zwei Stellen).
   Quelle: https://en.wikipedia.org/wiki/Elo_rating_system, Abschnitt zum Rechenbeispiel, am 2026-10-17 per WebFetch eingesehen. Der Remis-Ausgang dieses Beispiels wird
   hier nicht verwendet (die Simulation kennt nur Sieg und Niederlage), nur die Erwartungen.

Nicht eingesehen (siehe Bericht „offen“): Elo (1978); der Verlauf des Anlauf-K (K + (K0 − K)/(1 + g/10)) und die K-Stufe bei 1600 Punkten sind Annahmen der Demo ohne Quelle (die
FIDE-Regeln nennen K = 40 für Neue bis 30 Partien, K = 20 unter 2400, K = 10 ab 2400, aber keinen solchen Verlauf); quantitative Literaturwerte zu Inflation und Deflation der Skala."""

import numpy as np
import pytest

import eli_constants as C
import eli_rating as R

S = C.SCALE


def _theta(points: float) -> float:
    return (points - C.BASE) / S


def test_fide_table_8_1_2_expected_scores():
    assert round(float(R.sig(_theta(1600.0) - _theta(1500.0))), 2) == 0.64 and round(float(R.sig(_theta(1500.0) - _theta(1600.0))), 2) == 0.36
    assert round(float(R.sig(_theta(1700.0) - _theta(1500.0))), 2) == 0.76 and round(float(R.sig(_theta(1500.0) - _theta(1700.0))), 2) == 0.24
    assert round(float(R.sig(0.0)), 2) == 0.50


def test_fide_rule_8_3_2_rating_change_is_k_times_score_minus_expectation():
    """Ein Sieg des Höherbewerteten (100 Punkte vorn, K = 20): +20·(1 − 0,64) = +7,2 Punkte, der Verlierer verliert ebenso viel."""
    rating = np.array([[_theta(1600.0), _theta(1500.0)]])
    true = np.array([[100.0, -100.0]])                                      # der erste gewinnt sicher (u < σ(200) = 1)
    played = np.zeros((1, 2))
    R.play_round(rating, true, played, np.array([[0]]), np.array([[1]]), np.array([[0.5]]), 20.0, None, False)
    assert S * (rating[0, 0] - _theta(1600.0)) == pytest.approx(20.0 * (1.0 - 0.64), abs=0.05)
    assert S * (rating[0, 1] - _theta(1500.0)) == pytest.approx(-20.0 * (1.0 - 0.64), abs=0.05)
    # und der Außenseiter, der gewinnt: +20·(1 − 0,36) = +12,8 Punkte
    rating = np.array([[_theta(1500.0), _theta(1600.0)]])
    R.play_round(rating, true, np.zeros((1, 2)), np.array([[0]]), np.array([[1]]), np.array([[0.5]]), 20.0, None, False)
    assert S * (rating[0, 0] - _theta(1500.0)) == pytest.approx(20.0 * (1.0 - 0.36), abs=0.05)


def test_wikipedia_example_tournament_expected_scores():
    for opp, expected in ((1609.0, 0.51), (1477.0, 0.69), (1388.0, 0.79), (1586.0, 0.54), (1720.0, 0.35)):
        assert round(float(R.sig(_theta(1613.0) - _theta(opp))), 2) == expected

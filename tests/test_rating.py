"""Elo-Schritt, Anlauf-K, K-Stufe, Untergrenze und Platzierungsschätzung: Einheiten an Handwerten."""

import math

import numpy as np
import pytest

import eli_constants as C
import eli_rating as R


def test_sigmoid_values_and_overflow_safety():
    assert R.sig(0.0) == 0.5
    assert R.sig(1.0) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), abs=1e-15)
    assert R.sig(-3.0) + R.sig(3.0) == pytest.approx(1.0, abs=1e-15)
    assert R.sig(-800.0) == 0.0 and R.sig(800.0) == 1.0


def test_400_points_are_a_factor_of_ten_in_the_odds():
    e = float(R.sig(400.0 / C.SCALE))
    assert e / (1.0 - e) == pytest.approx(10.0, rel=1e-12)


def test_fixed_k_in_theta_units():
    assert R.k_theta(np.zeros(3), np.zeros(3), 20.0, None, False) == pytest.approx(np.full(3, 20.0 / C.SCALE))


def test_start_k_decays_by_half_after_g0_games():
    k = R.k_theta(np.zeros(3), np.array([0.0, C.G0, 1e9]), 20.0, 120.0, False)
    assert k * C.SCALE == pytest.approx([120.0, 70.0, 20.0], abs=1e-6)


def test_k_step_applies_from_1600_points():
    r = np.array([99.0, 101.0]) / C.SCALE                    # 1599 und 1601 Punkte
    k = R.k_theta(r, np.zeros(2), 20.0, None, True)
    assert k * C.SCALE == pytest.approx([20.0, C.STEP_K])


def test_step_applies_on_top_of_the_start_k():
    r = np.array([101.0]) / C.SCALE
    assert R.k_theta(r, np.zeros(1), 20.0, 120.0, True)[0] * C.SCALE == pytest.approx(C.STEP_K)


def _round(ra, rb, ta, tb, u, k=20.0, k0=None, step=False, played=(0.0, 0.0)):
    Rm = np.array([[ra, rb]], dtype=float)
    true = np.array([[ta, tb]], dtype=float)
    pl = np.array([list(played)], dtype=float)
    created = R.play_round(Rm, true, pl, np.array([[0]]), np.array([[1]]), np.array([[u]]), k, k0, step)
    return Rm[0], pl[0], float(created[0])


def test_equal_players_win_moves_half_of_k():
    r, pl, created = _round(0.0, 0.0, 0.0, 0.0, u=0.2)       # u < 0,5: a gewinnt
    k = 20.0 / C.SCALE
    assert r == pytest.approx([0.5 * k, -0.5 * k]) and created == pytest.approx(0.0, abs=1e-15)
    assert list(pl) == [1.0, 1.0]


def test_a_loss_moves_the_other_way():
    r, _, _ = _round(0.0, 0.0, 0.0, 0.0, u=0.9)
    k = 20.0 / C.SCALE
    assert r == pytest.approx([-0.5 * k, 0.5 * k])


def test_result_uses_the_true_strength_and_the_expectation_the_rating():
    # wahre Differenz sehr groß: a gewinnt fast sicher (u = 0,9 < σ(30)); die Wertungen sind gleich, die Erwartung ist 0,5
    r, _, _ = _round(0.0, 0.0, 30.0, 0.0, u=0.9)
    assert r[0] == pytest.approx(0.5 * 20.0 / C.SCALE)
    # a mit höherer Wertung gewinnt gegen gleichen wahren Wert: Erwartung σ(1)
    r, _, _ = _round(1.0, 0.0, 0.0, 0.0, u=0.1)
    d = 1.0 - float(R.sig(1.0))
    assert r == pytest.approx([1.0 + 20.0 / C.SCALE * d, -20.0 / C.SCALE * d])


def test_unequal_k_creates_points():
    # a ist neu (K 120), b erfahren (K 20): a gewinnt bei Erwartung 0,5
    r, _, created = _round(0.0, 0.0, 0.0, 0.0, u=0.2, k=20.0, k0=120.0, played=(0.0, 1e9))
    ka, kb = 120.0 / C.SCALE, 20.0 / C.SCALE
    assert r == pytest.approx([0.5 * ka, -0.5 * kb])
    assert created == pytest.approx((ka - kb) * 0.5)
    assert r.sum() == pytest.approx(created)


def test_each_run_is_updated_independently():
    Rm = np.zeros((2, 2))
    true = np.zeros((2, 2))
    pl = np.zeros((2, 2))
    created = R.play_round(Rm, true, pl, np.array([[0], [0]]), np.array([[1], [1]]), np.array([[0.2], [0.9]]), 20.0, None, False)
    assert Rm[0, 0] > 0 > Rm[1, 0] and created.shape == (2,)


def test_floor_lifts_active_players_only_and_reports_the_gift():
    lo = (1350.0 - C.BASE) / C.SCALE
    Rm = np.array([[lo - 0.5, 0.2, lo - 1.0]])
    active = np.array([True, True, False])
    lift = R.apply_floor(Rm, active, 1350.0)
    assert Rm[0, 0] == pytest.approx(lo) and Rm[0, 1] == 0.2 and Rm[0, 2] == pytest.approx(lo - 1.0)
    assert lift[0] == pytest.approx(0.5)


def test_floor_exactly_at_the_floor_changes_nothing():
    lo = (1350.0 - C.BASE) / C.SCALE
    Rm = np.array([[lo]])
    assert R.apply_floor(Rm, np.array([True]), 1350.0)[0] == 0.0


def _placement(wins, r_opp, prior=1.0, true_self=0.0):
    """Eine Platzierung gegen feste Gegner mit gegebenen Ausgängen (jeder Gegner ist ein eigener Spieler; u steuert die Wahl)."""
    g = len(wins)
    Rm = np.array([r_opp], dtype=float)                       # (1, g) Gegner = Plätze 0..g-1
    true = np.full((1, g), true_self)
    stay = np.ones((1, g), dtype=bool)
    u_opp = ((np.arange(g) + 0.5) / g).reshape(1, 1, g)       # Gegner j bei Partie j
    u_win = np.where(np.array(wins, dtype=bool), 0.0, 0.999999).reshape(1, 1, g)    # 0,0 < σ → Sieg, sonst Niederlage (für σ ≤ 0,99)
    return float(R.placement_estimates(np.array([[true_self]]), Rm, true, stay, u_opp, u_win, prior)[0, 0])


def test_placement_with_even_results_against_equal_field_is_zero():
    assert _placement([1, 0] * 5, [0.0] * 10) == pytest.approx(0.0, abs=1e-9)


def test_placement_is_antisymmetric_in_wins_and_losses_against_equal_field():
    up = _placement([1, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0.0] * 10)
    down = _placement([0, 0, 0, 0, 0, 0, 1, 1, 1, 0], [0.0] * 10)
    assert up > 0 > down and up == pytest.approx(-down, abs=1e-9)


def test_placement_is_finite_with_ten_wins_and_ten_losses():
    assert 0.0 < _placement([1] * 10, [0.0] * 10) < 10.0
    assert -10.0 < _placement([0] * 10, [0.0] * 10) < 0.0


def test_placement_satisfies_the_first_order_condition():
    r_opp = np.linspace(-1.0, 1.0, 10)
    wins = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], dtype=float)
    est = _placement(list(wins), list(r_opp), prior=0.8)
    grad = float(np.sum(wins - R.sig(est - r_opp)) - est / 0.8 ** 2)
    assert grad == pytest.approx(0.0, abs=1e-9)


def test_placement_shrinks_towards_the_prior_mean():
    r_opp = [0.0] * 10
    wins = [1] * 7 + [0] * 3
    assert 0.0 < _placement(wins, r_opp, prior=0.3) < _placement(wins, r_opp, prior=1.5)


def test_placement_draws_opponents_only_from_the_staying_players():
    Rm = np.array([[0.0, 5.0, -5.0, 0.0]])
    true = np.zeros((1, 4))
    stay = np.array([[True, False, False, True]])
    u_opp = np.random.default_rng(0).random((1, 1, 10))
    u_win = np.full((1, 1, 10), 0.4)
    # Gegner nur 0 und 3, beide Wertung 0: das Ergebnis darf die Wertungen 5 und -5 der Gehenden nicht spüren (Sieg bei u < 0,5 gegen gleich starke)
    est = R.placement_estimates(np.zeros((1, 1)), Rm, true, stay, u_opp, u_win, 1.0)[0, 0]
    ref = R.placement_estimates(np.zeros((1, 1)), np.zeros((1, 4)), true, stay, u_opp, u_win, 1.0)[0, 0]
    assert est == pytest.approx(ref, abs=1e-12) and est > 0

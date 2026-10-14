"""Offenes Feld: Erhaltung der Summe, Bilanz, Abgangsregeln, Neue, Trend, Anker, Determinismus, Zufallsverbrauch und Randfälle."""

import numpy as np
import pytest

import eli_constants as C
import eli_pool as P
from conftest import small


def _identity_residual(pool: P.Pool) -> float:
    """Größte Abweichung von total_start[s+1] = total_end[s] + Zu-/Abgang + Anker, über Läufe und Saisons."""
    nxt = pool.total_start[:, 1:]
    expect = pool.total_end[:, :-1] + pool.churn_delta[:, :-1] + pool.anchor_out[:, :-1] + pool.anchor_in[:, 1:]
    return float(np.abs(nxt - expect).max())


def test_closed_field_keeps_every_rating_sum_and_has_no_drift():
    pool = P.simulate(small(churn=0.0, seasons=6), range(5))
    assert np.abs(pool.total_start).max() < 1e-12 and np.abs(pool.total_end).max() < 1e-12
    assert np.abs(pool.mean_rating).max() < 1e-12
    assert np.abs(pool.created).max() < 1e-12


def test_equal_k_keeps_the_sum_within_each_season_even_with_churn():
    pool = P.simulate(small(seasons=5), range(4))
    assert np.abs(pool.total_end - pool.total_start).max() < 1e-12


def test_sum_identity_holds_with_every_mechanism_switched_on():
    for kw in (dict(), dict(k0=120.0), dict(floor=1450.0), dict(k_steps=True), dict(placement=True), dict(anchor_join=1, anchor_stay=2),
               dict(k0=100.0, floor=1450.0, placement=True, anchor_join=0, anchor_stay=0, leave="die Schwächsten", gap=1.0, trend=0.02)):
        pool = P.simulate(small(seasons=5, **kw), range(3))
        assert _identity_residual(pool) < 1e-10, kw
        within = pool.total_end - pool.total_start - pool.created - pool.floor_lift
        assert np.abs(within).max() < 1e-10, kw


def test_floor_and_unequal_k_account_for_the_change_within_a_season():
    pool = P.simulate(small(seasons=4, k0=150.0, floor=1450.0, leave="die Schwächsten"), range(3))
    assert pool.floor_lift.sum() > 0 and np.abs(pool.created).sum() > 0


def test_same_seed_same_pool_and_different_seed_different_pool():
    m = small(seasons=4)
    a, b, c = P.simulate(m, [3]), P.simulate(m, [3]), P.simulate(m, [4])
    assert np.array_equal(a.mean_rating, b.mean_rating) and np.array_equal(a.final_rating, b.final_rating)
    assert not np.array_equal(a.mean_rating, c.mean_rating)


def test_a_run_does_not_depend_on_the_other_runs_in_the_batch():
    m = small(seasons=4, placement=True, anchor_join=1)
    both = P.simulate(m, [7, 9])
    alone = P.simulate(m, [9])
    assert np.array_equal(both.mean_rating[1], alone.mean_rating[0]) and np.array_equal(both.final_true[1], alone.final_true[0])


def test_placement_does_not_change_the_first_season_because_the_random_numbers_are_shared():
    a = P.simulate(small(seasons=3, placement=False), [5])
    b = P.simulate(small(seasons=3, placement=True), [5])
    assert np.array_equal(a.mean_rating[:, 0], b.mean_rating[:, 0])
    assert not np.array_equal(a.mean_rating[:, 2], b.mean_rating[:, 2])


def test_random_numbers_are_laid_out_independently_of_the_mechanisms():
    d = P.draws(1, 4, 3, 9)
    assert d["z"].shape == (4, 9) and d["pair"].shape == (4, 3, 9) and d["win"].shape == (4, 3, 4)
    assert d["leave"].shape == (4, 9) and d["opp"].shape == (4, 9, C.PLACE_GAMES) and d["pwin"].shape == (4, 9, C.PLACE_GAMES)
    again = P.draws(1, 4, 3, 9)
    assert all(np.array_equal(d[k], again[k]) for k in d)


def test_normals_are_standard_normal():
    z = P.normals(P.T.SplitMix64(5), 200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1.0) < 0.01


def test_founders_are_equidistant_with_the_chosen_spread():
    f = P.founder_strengths(40)
    assert f.mean() == pytest.approx(0.0, abs=1e-12)
    assert f.std() == pytest.approx(C.POP_SD, rel=1e-12)
    assert np.allclose(np.diff(f), np.diff(f)[0])


def test_the_number_of_leavers_follows_the_churn_rate():
    assert small(n=40, churn=0.15).n_out == 6 and small(n=10, churn=0.0).n_out == 0 and small(n=10, churn=0.4).n_out == 4


def test_weakest_leave_by_rating_with_random_tie_break():
    m = small(n=6, churn=0.5, leave="die Schwächsten")
    rating = np.array([[0.3, -0.5, 0.1, -0.9, 0.7, 0.0]])
    idx = P.leavers(m, rating, np.zeros((1, 6)), np.full((1, 6), 0.5))
    assert sorted(idx[0].tolist()) == [1, 3, 5]
    ties = P.leavers(m, np.zeros((1, 6)), np.zeros((1, 6)), np.array([[0.9, 0.1, 0.5, 0.3, 0.8, 0.2]]))
    assert sorted(ties[0].tolist()) == [1, 3, 5]             # gleiche Wertungen: die kleinsten Schlüssel gehen


def test_veterans_leave_by_age_with_random_tie_break():
    m = small(n=6, churn=0.5, leave="die Veteranen")
    age = np.array([[4.0, 1.0, 9.0, 2.0, 7.0, 3.0]])
    idx = P.leavers(m, np.zeros((1, 6)), age, np.full((1, 6), 0.5))
    assert sorted(idx[0].tolist()) == [0, 2, 4]
    ties = P.leavers(m, np.zeros((1, 6)), np.full((1, 6), 3.0), np.array([[0.9, 0.1, 0.5, 0.3, 0.8, 0.2]]))
    assert sorted(ties[0].tolist()) == [1, 3, 5]


def test_random_leave_follows_the_keys_and_unknown_rule_raises():
    m = small(n=6, churn=0.5, leave="zufällig")
    idx = P.leavers(m, np.zeros((1, 6)), np.zeros((1, 6)), np.array([[0.9, 0.1, 0.5, 0.3, 0.8, 0.2]]))
    assert idx[0].tolist() == [1, 5, 3]
    with pytest.raises(ValueError, match="Abgangsregel"):
        P.leavers(small(n=6, churn=0.5, leave="niemand"), np.zeros((1, 6)), np.zeros((1, 6)), np.zeros((1, 6)))


def test_field_trend_moves_every_true_strength_in_a_closed_field():
    pool = P.simulate(small(churn=0.0, seasons=5, trend=0.02), [1])
    assert pool.mean_true[0] == pytest.approx(0.02 * np.arange(1, 6), abs=1e-12)


def test_the_anchor_keeps_its_strength_while_the_field_moves():
    pool = P.simulate(small(churn=0.0, seasons=4, trend=0.05, anchor_join=1), [1])
    assert pool.final_true[0, -1] == 0.0
    assert pool.final_true[0, :-1].mean() == pytest.approx(0.2, abs=1e-12)


def test_new_players_start_below_their_strength_and_catch_up_a_quarter_per_season():
    seed, gap = 6, 1.0
    m = small(n=10, seasons=2, games=2, churn=0.1, gap=gap)
    pool = P.simulate(m, [seed])
    slot = int(np.nonzero(pool.final_age[0, :10] == 1)[0][0])           # der eine Neue nach Saison 0 hat eine Saison im Feld
    pot = C.POP_SD * P.draws(seed, 2, 2, 11)["z"][0][slot]
    assert pool.final_true[0, slot] == pytest.approx(pot - gap + C.GROWTH * gap, abs=1e-12)
    assert pool.final_age[0, :10].max() == 2 and (pool.final_age[0, :10] == 2).sum() == 9


def test_the_start_error_is_recorded_for_every_churn_and_not_after_the_last_season():
    pool = P.simulate(small(seasons=3), range(3))
    assert np.isfinite(pool.entry_mean[:, :2]).all() and np.isnan(pool.entry_mean[:, 2]).all()
    assert (pool.entry_msq[:, :2] >= pool.entry_mean[:, :2] ** 2 - 1e-12).all()
    closed = P.simulate(small(seasons=3, churn=0.0), [1])
    assert np.isnan(closed.entry_mean).all() and (closed.churn_delta == 0).all()


def test_anchor_is_absent_before_joining_and_after_leaving():
    pool = P.simulate(small(seasons=6, anchor_join=2, anchor_stay=2), [1])
    present = np.isfinite(pool.anchor[0])
    assert present.tolist() == [False, False, True, True, False, False]
    assert pool.anchor_in[0, 2] == 0.0 and pool.anchor_out[0, 3] == pytest.approx(-pool.anchor[0, 3])


def test_anchor_with_placement_starts_at_the_estimate_not_at_zero():
    pool = P.simulate(small(seasons=3, anchor_join=1, placement=True), range(10))
    assert np.abs(pool.anchor_in[:, 1]).max() > 0.0 and (pool.anchor_in[:, 0] == 0.0).all()


def test_odd_field_lets_one_player_sit_out_each_round():
    pool = P.simulate(small(n=3, churn=0.0, seasons=3), [1])
    assert abs(pool.total_end).max() < 1e-12 and np.isfinite(pool.mean_rating).all()


def test_smallest_field_and_extremes_run():
    pool = P.simulate(P.Model(n=2, seasons=1, games=1, churn=0.0), [1])
    assert pool.mean_rating.shape == (1, 1)
    pool = P.simulate(P.Model(n=10, seasons=2, games=1, churn=0.4, placement=True, anchor_join=0), [1])
    assert np.isfinite(pool.mean_rating).all()


def test_the_largest_allowed_churn_leaves_one_player_and_the_anchor_may_join_in_the_last_season():
    pool = P.simulate(P.Model(n=4, seasons=3, games=2, churn=0.75), [1])               # n_out = 3 = n − 1
    assert np.isfinite(pool.mean_rating).all()
    last = P.simulate(small(seasons=4, anchor_join=3), [1])
    assert np.isfinite(last.anchor[0, 3]) and np.isnan(last.anchor[0, :3]).all()
    first = P.simulate(small(seasons=4, anchor_join=0), [1])
    assert np.isfinite(first.anchor[0]).all()


@pytest.mark.parametrize("kw", [dict(n=1), dict(seasons=0), dict(games=0), dict(n=4, churn=1.0), dict(anchor_join=9, seasons=4), dict(anchor_join=4, seasons=4),
                                dict(anchor_join=-1, seasons=4)])
def test_invalid_models_are_rejected(kw):
    with pytest.raises(ValueError):
        P.simulate(small(**kw), [1])


def test_slots_and_properties():
    m = small(n=8)
    assert m.slots == 9 and m.n_out == 2

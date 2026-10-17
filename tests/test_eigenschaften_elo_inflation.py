"""Eigenschaftstests (metamorph) für Elo-Inflation (ORACLE-PLAYBOOK §5.2): Man verändert die Eingabe auf eine Weise, deren Wirkung auf die Ausgabe aus dem Verfahren
folgt, und prüft diese Beziehung auf Zufallsinstanzen mit festem Seed. Die vorhandenen Tests prüfen Rechenwege und feste Modellvarianten; hier stehen Aussagen, die für
jede Einstellung gelten müssen:

- Bilanzidentität: die Summe der Wertungen ändert sich nur durch Zugang, Abgang, Anker, Untergrenze und ungleiches K (auf zufälligen Modellen, auch in `balance`);
- ohne Zu- und Abgang und bei gleichem K bleiben Summe und Mittel konstant; mit Untergrenze bzw. Anlauf-K ändert sich die Summe genau um das Ausgewiesene;
- eine Runde: Umnummerieren der Plätze, Umordnen der Paare, Vertauschen von a und b (mit gespiegeltem Zufall), gemeinsame Verschiebung der Wertungen und lineares
  Skalieren von K;
- Anlauf-K gleich K ist festes K, eine Untergrenze weit unter allen Wertungen ändert nichts;
- Platzierungsschätzung: gemeinsame Verschiebung der Gegner (flacher Prior), Spiegeln der Ergebnisse und Gegnerwerte, Reihenfolge der Gegner, Monotonie;
- Untergrenze: nie senkend, idempotent, Ausweis gleich der Änderung;
- die Läufe eines Stapels sind unabhängig von Zahl und Reihenfolge der anderen Läufe.

Nicht geprüft: Umnummerieren ganzer Modelle (der Zufallsverbrauch hängt an der Platznummer, das Umnummern ist keine Eingabeänderung), K-Stufe mit gemeinsamer Verschiebung (die
Stufe hängt dokumentiert an der absoluten Wertung)."""

import numpy as np
import pytest

import eli_constants as C
import eli_evaluation as ev
import eli_pool as P
import eli_rating as R

N_INSTANCES = 30
RNG_SEED = 20261017


def _rng(salt: int) -> np.random.Generator:
    return np.random.default_rng(RNG_SEED + salt)


def _random_model(rng, **fixed) -> P.Model:
    n = int(rng.choice([6, 8, 10, 12]))
    kw = dict(n=n, seasons=int(rng.integers(3, 7)), games=int(rng.integers(2, 6)), churn=float(rng.choice([0.0, 0.1, 0.25, 0.4])),
              leave=str(rng.choice(C.LEAVE)), gap=float(rng.choice([0.0, 0.5, 1.0])), trend=float(rng.choice([-0.02, 0.0, 0.03])), k=float(rng.choice([10.0, 20.0, 40.0])),
              k0=float(rng.choice([120.0, 200.0])) if rng.random() < 0.5 else None, floor=float(rng.choice([1300.0, 1450.0])) if rng.random() < 0.5 else None,
              k_steps=bool(rng.random() < 0.3), placement=bool(rng.random() < 0.4), prior=float(rng.choice([0.5, 1.0])))
    if rng.random() < 0.6:
        kw["anchor_join"] = int(rng.integers(0, kw["seasons"] - 1))
        kw["anchor_stay"] = int(rng.integers(0, 3)) if rng.random() < 0.5 else 0
    kw.update(fixed)
    if kw["n"] * kw["churn"] > kw["n"] - 1:
        kw["churn"] = 0.25
    return P.Model(**kw)


# --- 1. Bilanzidentität ---------------------------------------------------------------------------------------------------------------------------

def test_the_rating_sum_changes_only_through_entries_exits_anchor_floor_and_unequal_k():
    rng = _rng(1)
    for _ in range(N_INSTANCES):
        model = _random_model(rng)
        pool = P.simulate(model, [int(s) for s in rng.integers(1, 10 ** 6, 2)])
        # innerhalb einer Saison: nur ungleiches K und Untergrenze
        assert np.abs(pool.total_end - pool.total_start - pool.created - pool.floor_lift).max() < 1e-10, model
        # zwischen den Saisons: nur Zugang, Abgang und Anker
        nxt = pool.total_start[:, 1:]
        expect = pool.total_end[:, :-1] + pool.churn_delta[:, :-1] + pool.anchor_out[:, :-1] + pool.anchor_in[:, 1:]
        assert np.abs(nxt - expect).max() < 1e-10, model
        # und als Summe über den Lauf (Anker, der schon in Saison 0 beitritt, zählt über anchor_in[:, 0])
        for b in range(pool.runs):
            bal = ev.balance(pool, b)
            assert bal["entries"] + bal["k"] + bal["floor"] + bal["anchor"] == pytest.approx(bal["end"], abs=1e-6), model


def test_without_entries_and_exits_an_equal_k_keeps_the_sum_and_the_field_mean_constant():
    rng = _rng(2)
    for _ in range(N_INSTANCES):
        model = _random_model(rng, churn=0.0, k0=None, floor=None, k_steps=False, placement=False, anchor_join=None, anchor_stay=0, trend=float(rng.choice([0.0, 0.03])))
        pool = P.simulate(model, [int(rng.integers(1, 10 ** 6))])
        assert np.abs(pool.mean_rating).max() < 1e-12 and np.abs(pool.total_end).max() < 1e-12 and np.abs(pool.created).max() < 1e-12
    for _ in range(N_INSTANCES):                                          # mit Anker (Start 1500, bleibt): die Summe von Feld und Anker bleibt
        model = _random_model(rng, churn=0.0, k0=None, floor=None, k_steps=False, placement=False, anchor_join=0, anchor_stay=0)
        pool = P.simulate(model, [int(rng.integers(1, 10 ** 6))])
        assert np.abs(pool.total_end).max() < 1e-12 and np.abs(pool.total_start).max() < 1e-12


def test_a_floor_adds_exactly_its_reported_points_and_never_removes_any():
    rng = _rng(3)
    for _ in range(N_INSTANCES):
        model = _random_model(rng, churn=0.0, k0=None, k_steps=False, placement=False, anchor_join=None, anchor_stay=0, floor=float(rng.choice([1450.0, 1480.0])))
        pool = P.simulate(model, [int(rng.integers(1, 10 ** 6))])
        assert (pool.floor_lift >= 0).all()
        assert pool.total_end[:, -1] == pytest.approx(pool.floor_lift.sum(axis=1), abs=1e-10)       # alle Saisons: Summe = angehobene Punkte (Start 0, kein Zu-/Abgang)
        assert (pool.final_rating >= (model.floor - C.BASE) / C.SCALE - 1e-12).all()


def test_a_start_k_equal_to_k_is_a_fixed_k_and_a_floor_far_below_everything_changes_nothing():
    rng = _rng(4)
    for _ in range(15):
        model = _random_model(rng, k0=None, floor=None)
        seed = [int(rng.integers(1, 10 ** 6))]
        base = P.simulate(model, seed)
        same_k = P.simulate(P.Model(**{**model.__dict__, "k0": model.k}), seed)
        low_floor = P.simulate(P.Model(**{**model.__dict__, "floor": 0.0}), seed)           # 0 Punkte liegt acht Einheiten unter dem Mittel
        for other in (same_k, low_floor):
            assert np.abs(other.final_rating - base.final_rating).max() < 1e-12
            assert np.abs(other.mean_rating - base.mean_rating).max() < 1e-12
        assert np.abs(low_floor.floor_lift).max() == 0.0 and np.abs(same_k.created - base.created).max() < 1e-12


# --- 2. Eine Runde ----------------------------------------------------------------------------------------------------------------------------------

def _state(rng, B=3, P_=10):
    rating = rng.normal(0.0, 0.6, (B, P_))
    true = rng.normal(0.0, 0.8, (B, P_))
    played = rng.integers(0, 30, (B, P_)).astype(float)
    order = np.stack([rng.permutation(P_) for _ in range(B)])
    m = P_ // 2
    return rating, true, played, order[:, 0:2 * m:2], order[:, 1:2 * m:2], rng.random((B, m))


def _round(rating, true, played, a, b, u, k=20.0, k0=None, step=False):
    r, p = rating.copy(), played.copy()
    created = R.play_round(r, true, p, a, b, u, k, k0, step)
    return r, p, created


def test_relabelling_the_seats_of_a_round_only_permutes_the_output():
    rng = _rng(5)
    for _ in range(N_INSTANCES):
        rating, true, played, a, b, u = _state(rng)
        perm = rng.permutation(rating.shape[1])                           # Platz q heißt danach perm[q]
        inv = np.argsort(perm)
        r1, p1, c1 = _round(rating, true, played, a, b, u, 20.0, 120.0, True)
        r2, p2, c2 = _round(rating[:, inv], true[:, inv], played[:, inv], perm[a], perm[b], u, 20.0, 120.0, True)
        assert r2[:, perm] == pytest.approx(r1, abs=1e-12) and np.array_equal(p2[:, perm], p1)
        assert c2 == pytest.approx(c1, abs=1e-12)


def test_the_order_of_the_pairs_of_a_round_and_the_swap_of_a_and_b_do_not_matter():
    rng = _rng(6)
    for _ in range(N_INSTANCES):
        rating, true, played, a, b, u = _state(rng)
        base_r, base_p, base_c = _round(rating, true, played, a, b, u, 25.0, 100.0, False)
        order = rng.permutation(a.shape[1])
        r2, p2, c2 = _round(rating, true, played, a[:, order], b[:, order], u[:, order], 25.0, 100.0, False)
        assert r2 == pytest.approx(base_r, abs=1e-12) and c2 == pytest.approx(base_c, abs=1e-12)
        # a und b tauschen: a gewinnt, wenn u < p; für b als Erstgenannten gewinnt b, wenn 1 − u < 1 − p, also u' = 1 − u
        r3, p3, c3 = _round(rating, true, played, b, a, 1.0 - u, 25.0, 100.0, False)
        assert r3 == pytest.approx(base_r, abs=1e-12) and np.array_equal(p3, base_p)
        assert c3 == pytest.approx(base_c, abs=1e-12)                # (k_b − k_a)·(−d) = (k_a − k_b)·d: dasselbe erzeugte Plus


def test_a_round_depends_only_on_rating_differences_when_k_does_not_depend_on_the_level():
    rng = _rng(7)
    for _ in range(N_INSTANCES):
        rating, true, played, a, b, u = _state(rng)
        shift = float(rng.uniform(-4.0, 4.0))
        for k0 in (None, 150.0):
            r1, _, c1 = _round(rating, true, played, a, b, u, 20.0, k0, False)
            r2, _, c2 = _round(rating + shift, true, played, a, b, u, 20.0, k0, False)
            assert r2 - shift == pytest.approx(r1, abs=1e-12) and c2 == pytest.approx(c1, abs=1e-12)


def test_a_round_is_linear_in_k_and_k0_and_the_reported_points_equal_the_change_of_the_sum():
    rng = _rng(8)
    for _ in range(N_INSTANCES):
        rating, true, played, a, b, u = _state(rng)
        f = float(rng.choice([0.5, 2.0, 5.0]))
        r1, _, c1 = _round(rating, true, played, a, b, u, 20.0, 120.0, False)
        r2, _, c2 = _round(rating, true, played, a, b, u, f * 20.0, f * 120.0, False)
        assert r2 - rating == pytest.approx(f * (r1 - rating), abs=1e-12) and c2 == pytest.approx(f * c1, abs=1e-12)
        assert (r1.sum(axis=1) - rating.sum(axis=1)) == pytest.approx(c1, abs=1e-12)            # Ausweis „erzeugte Punkte“ = tatsächliche Änderung der Summe
        r3, _, c3 = _round(rating, true, played, a, b, u, 20.0, None, False)
        assert np.abs(c3).max() < 1e-12 and (r3.sum(axis=1) - rating.sum(axis=1)) == pytest.approx(0.0, abs=1e-12)     # gleiches K: Nullsumme


def test_the_start_k_decreases_with_the_games_played_and_tends_to_k():
    rng = _rng(9)
    for _ in range(N_INSTANCES):
        k, k0 = float(rng.uniform(5, 40)), float(rng.uniform(50, 300))
        ks = R.k_theta(np.zeros(60), np.arange(60), k, k0, False) * C.SCALE
        assert (np.diff(ks) < 0).all() and ks[0] == pytest.approx(k0) and (ks > k).all()
        assert R.k_theta(np.zeros(1), np.array([1e9]), k, k0, False)[0] * C.SCALE == pytest.approx(k, rel=1e-6)
        assert R.k_theta(np.zeros(3), np.arange(3), k, None, False) * C.SCALE == pytest.approx([k] * 3)


# --- 3. Untergrenze ---------------------------------------------------------------------------------------------------------------------------------

def test_the_floor_never_lowers_a_rating_is_idempotent_and_reports_its_change():
    rng = _rng(10)
    for _ in range(N_INSTANCES):
        rating = rng.normal(0.0, 1.0, (4, 9))
        active = rng.random(9) < 0.7
        active[0] = True
        floor = float(rng.choice([1200.0, 1400.0, 1500.0]))
        before = rating.copy()
        lift = R.apply_floor(rating, active, floor)
        lo = (floor - C.BASE) / C.SCALE
        assert (rating >= before).all() and (rating[:, active] >= lo - 1e-12).all()
        assert np.array_equal(rating[:, ~active], before[:, ~active])                                # inaktive Plätze bleiben
        assert lift == pytest.approx((rating - before).sum(axis=1), abs=1e-12) and (lift >= 0).all()
        again = rating.copy()
        assert np.abs(R.apply_floor(again, active, floor)).max() == 0.0 and np.array_equal(again, rating)


# --- 4. Platzierungsschätzung ----------------------------------------------------------------------------------------------------------------------

def _placement(r_opp, wins, prior):
    """Eine Schätzung (ein Lauf, ein Neuer) aus 10 Gegnern mit gegebenen Wertungen und Ausgängen."""
    G = len(r_opp)
    stay = np.ones((1, G), dtype=bool)
    u_opp = ((np.arange(G) + 0.5) / G).reshape(1, 1, G)              # Gegner q ist Platz q
    u_win = np.where(np.asarray(wins) > 0, 0.0, 0.999999).reshape(1, 1, G)
    return float(R.placement_estimates(np.zeros((1, 1)), np.asarray(r_opp, dtype=float)[None, :], np.zeros((1, G)), stay, u_opp, u_win, prior)[0, 0])


def test_the_placement_estimate_shifts_with_the_opponents_for_a_flat_prior():
    rng = _rng(11)
    for _ in range(N_INSTANCES):
        r_opp = rng.normal(0.0, 0.8, 10)
        wins = (rng.random(10) < 0.5).astype(float)
        wins[:2] = (1.0, 0.0)                                         # mindestens ein Sieg und eine Niederlage: der Schätzer ist endlich
        c = float(rng.uniform(-1.5, 1.5))
        flat = 200.0                                                  # Prior-Standardabweichung 200: praktisch flach
        assert _placement(r_opp + c, wins, flat) == pytest.approx(_placement(r_opp, wins, flat) + c, abs=1e-4)


def test_mirroring_the_results_and_the_opponents_negates_the_placement_estimate_and_the_order_of_the_opponents_is_irrelevant():
    rng = _rng(12)
    for _ in range(N_INSTANCES):
        r_opp = rng.normal(0.0, 0.8, 10)
        wins = (rng.random(10) < 0.5).astype(float)
        prior = float(rng.choice([0.3, 0.5, 1.0, 1.5]))
        est = _placement(r_opp, wins, prior)
        assert _placement(-r_opp, 1.0 - wins, prior) == pytest.approx(-est, abs=1e-6)     # der Prior (Mittel 0) ist symmetrisch
        perm = rng.permutation(10)
        assert _placement(r_opp[perm], wins[perm], prior) == pytest.approx(est, abs=1e-6)


def test_one_more_win_never_lowers_the_placement_estimate_and_all_wins_beat_all_losses():
    rng = _rng(13)
    for _ in range(N_INSTANCES):
        r_opp = rng.normal(0.0, 0.8, 10)
        wins = (rng.random(10) < 0.5).astype(float)
        loser = np.flatnonzero(wins == 0)
        prior = float(rng.choice([0.5, 1.0]))
        if len(loser):
            more = wins.copy()
            more[loser[0]] = 1.0
            assert _placement(r_opp, more, prior) >= _placement(r_opp, wins, prior) - 1e-9
        all_wins, all_losses = _placement(r_opp, np.ones(10), prior), _placement(r_opp, np.zeros(10), prior)
        assert all_wins > all_losses                                                    # Randfälle: alles gewonnen bzw. verloren
        assert abs(all_wins) < 50.0 and abs(all_losses) < 50.0                          # bleibt wegen des Priors endlich


# --- 5. Unabhängigkeit der Läufe eines Stapels ----------------------------------------------------------------------------------------------------

def test_the_runs_of_a_batch_do_not_depend_on_the_other_runs_or_on_their_order():
    rng = _rng(14)
    for _ in range(12):
        model = _random_model(rng)
        seeds = [int(s) for s in rng.integers(1, 10 ** 6, 3)]
        batch = P.simulate(model, seeds)
        rev = P.simulate(model, seeds[::-1])
        alone = [P.simulate(model, [s]) for s in seeds]
        for idx, s in enumerate(seeds):
            for field in ("mean_rating", "total_end", "created", "final_rating", "final_true", "churn_delta"):
                # gleich bis auf die Rundung der Summen (numpy summiert je nach Form des Stapels in anderer Reihenfolge)
                assert getattr(batch, field)[idx] == pytest.approx(getattr(alone[idx], field)[0], abs=1e-12), (field, model)
                assert getattr(batch, field)[idx] == pytest.approx(getattr(rev, field)[2 - idx], abs=1e-12), (field, model)

"""Orakel für das offene Feld: eine unabhängige Partie-für-Partie-Rechnung in reinem Python (Schleifen über Spieler, Partien und Gegner, Skalare) auf denselben Zufallszahlen
gegen die vektorisierte Simulation; die Mini-Instanz aus dem Plan von Hand; die Platzierungsschätzung gegen scipy; die Bilanz als Eigenschaft über viele Läufe."""

import math

import numpy as np
import pytest

import eli_constants as C
import eli_evaluation as ev
import eli_pool as P
from conftest import hand_pool, small


def sg(x: float) -> float:
    return 0.5 * (1.0 + math.tanh(0.5 * x))


def reference(model: P.Model, seed: int) -> dict:
    """Plätze 0..n-1 Feld, n Anker. Gleiche Zufallszahlen (P.draws), aber jede Partie, jedes Aufrücken und jede Newton-Iteration einzeln mit Python-Skalaren."""
    n, S, G = model.n, model.seasons, model.games
    slots = n + 1
    d = P.draws(seed, S, G, slots)
    delta = C.POP_SD * math.sqrt(12.0 / (n * n - 1))
    true = [delta * (i - (n - 1) / 2.0) for i in range(n)] + [0.0]
    pot = list(true)
    rating = [0.0] * slots
    played = [0.0] * slots
    age = [0.0] * slots
    active = [True] * n + [False]
    level = 0.0
    out = {"mean_rating": [], "mean_true": [], "anchor": [], "created": [], "floor": []}

    def k_of(r, g):
        k = model.k
        if model.k0 is not None:
            k = model.k + (model.k0 - model.k) / (1.0 + g / C.G0)
        if model.k_steps and C.BASE + C.SCALE * r >= C.STEP_FROM:
            k = C.STEP_K
        return k / C.SCALE

    def place(ts, stay, uo, uw):
        r = 0.0
        opp = [stay[min(int(u * len(stay)), len(stay) - 1)] for u in uo]
        wins = [1.0 if uw[j] < sg(ts - true[o]) else 0.0 for j, o in enumerate(opp)]
        for _ in range(C.PLACE_ITERS):
            grad = -r / model.prior ** 2
            hess = 1.0 / model.prior ** 2
            for j, o in enumerate(opp):
                p = sg(r - rating[o])
                grad += wins[j] - p
                hess += p * (1.0 - p)
            r += grad / hess
        return r

    for s in range(S):
        if model.anchor_join is not None and s == model.anchor_join:
            active[n] = True
            true[n] = pot[n] = 0.0
            played[n] = age[n] = 0.0
            rating[n] = place(0.0, list(range(n)), d["opp"][s, n], d["pwin"][s, n]) if model.placement else 0.0
        roster = [q for q in range(slots) if active[q]]
        created = floor_lift = 0.0
        for g in range(G):
            order = sorted(roster, key=lambda q: (d["pair"][s, g, q], q))
            for m in range(len(order) // 2):
                a, b = order[2 * m], order[2 * m + 1]
                dd = (1.0 if d["win"][s, g, m] < sg(true[a] - true[b]) else 0.0) - sg(rating[a] - rating[b])
                ka, kb = k_of(rating[a], played[a]), k_of(rating[b], played[b])
                rating[a] += ka * dd
                rating[b] -= kb * dd
                played[a] += 1
                played[b] += 1
                created += (ka - kb) * dd
            if model.floor is not None:
                lo = (model.floor - C.BASE) / C.SCALE
                for q in roster:
                    if rating[q] < lo:
                        floor_lift += lo - rating[q]
                        rating[q] = lo
        for q in range(n):
            if model.trend:
                true[q] += model.trend
                pot[q] += model.trend
            true[q] += (pot[q] - true[q]) * C.GROWTH
            age[q] += 1
        level += model.trend
        out["mean_rating"].append(sum(rating[:n]) / n)
        out["mean_true"].append(sum(true[:n]) / n)
        out["created"].append(created)
        out["floor"].append(floor_lift)
        if active[n]:
            out["anchor"].append(rating[n])
            age[n] += 1
            if model.anchor_stay and s == model.anchor_join + model.anchor_stay - 1:
                active[n] = False
        else:
            out["anchor"].append(float("nan"))
        if model.n_out and s < S - 1:
            if model.leave == C.LEAVE[0]:
                key = [d["leave"][s, q] for q in range(n)]
            elif model.leave == C.LEAVE[1]:
                key = [rating[q] + 1e-9 * d["leave"][s, q] for q in range(n)]
            else:
                key = [-age[q] + 0.5 * d["leave"][s, q] for q in range(n)]
            gone = sorted(range(n), key=lambda q: (key[q], q))[:model.n_out]
            stay = [q for q in range(n) if q not in gone]
            new = {}
            for q in gone:
                p_new = level + C.POP_SD * d["z"][s, q]
                t_new = p_new - model.gap
                start = place(t_new, stay, d["opp"][s, q], d["pwin"][s, q]) if model.placement else 0.0
                new[q] = (p_new, t_new, start)
            for q, (p_new, t_new, start) in new.items():
                pot[q], true[q], rating[q] = p_new, t_new, start
                played[q] = age[q] = 0.0
    out["final_rating"], out["final_true"], out["final_age"] = rating, true, age
    return out


MODELS = [
    dict(),
    dict(leave="die Schwächsten"),
    dict(leave="die Veteranen"),
    dict(k0=120.0, leave="die Schwächsten"),
    dict(floor=1450.0),
    dict(k_steps=True, k=30.0),
    dict(gap=1.0, trend=0.03),
    dict(placement=True, prior=1.0),
    dict(placement=True, prior=0.5, leave="die Schwächsten", k0=100.0, gap=0.5),
    dict(anchor_join=1, anchor_stay=2),
    dict(anchor_join=0, placement=True, trend=-0.02, floor=1400.0, k0=80.0, k_steps=True),
    dict(n=7, anchor_join=1, churn=0.3),
]


@pytest.mark.parametrize("kw", MODELS, ids=[str(k) for k in MODELS])
def test_vectorised_simulation_equals_the_game_by_game_reference(kw):
    model = small(seasons=4, games=3, **kw)
    seeds = (3, 8)
    pool = P.simulate(model, seeds)
    for b, seed in enumerate(seeds):
        ref = reference(model, seed)
        assert pool.mean_rating[b] == pytest.approx(ref["mean_rating"], abs=1e-9)
        assert pool.mean_true[b] == pytest.approx(ref["mean_true"], abs=1e-9)
        assert pool.created[b] == pytest.approx(ref["created"], abs=1e-9)
        assert pool.floor_lift[b] == pytest.approx(ref["floor"], abs=1e-9)
        assert pool.final_rating[b] == pytest.approx(ref["final_rating"], abs=1e-9)
        assert pool.final_true[b] == pytest.approx(ref["final_true"], abs=1e-9)
        assert pool.final_age[b] == pytest.approx(ref["final_age"], abs=0.0)
        both = np.isfinite(pool.anchor[b])
        assert both.tolist() == np.isfinite(ref["anchor"]).tolist()
        assert pool.anchor[b][both] == pytest.approx(np.array(ref["anchor"])[both], abs=1e-9)


def test_mini_instance_from_the_plan_by_hand():
    """Vier Spieler mit 1520, 1480, 1510, 1490 (Summe 6000, Mittel 1500); der mit 1520 (Stärke 0) geht, ein Neuer mit 1500 und Stärke −0,5 Einheiten kommt: Summe 5980, Mittel
    1495; mittlere Stärke −0,125 Einheiten = −21,7 Punkte; Drift = −5 − (−21,7) = +16,7 Punkte (Inflation: Die Wertung des Neuen liegt über seiner Stärke)."""
    mean_rating = (5980.0 / 4 - 1500.0) / C.SCALE
    mean_true = -0.5 / 4
    pool = hand_pool(rating=(mean_rating,), true=(mean_true,), churn_delta=[[(1500.0 - 1520.0) / C.SCALE]], total_end=[[(5980.0 - 6000.0) / C.SCALE]])
    assert ev.rating_points(pool)[0, 0] == pytest.approx(-5.0)
    assert ev.level_points(pool)[0, 0] == pytest.approx(-0.125 * C.SCALE)
    assert ev.drift_points(pool)[0, 0] == pytest.approx(-5.0 + 0.125 * C.SCALE) and ev.drift_points(pool)[0, 0] == pytest.approx(16.7, abs=0.05)
    b = ev.balance(pool)
    assert b["entries"] == pytest.approx(-20.0) and b["end"] == pytest.approx(-20.0)


def test_equal_k_is_a_zero_sum_in_the_textbook_formula():
    """Lehrbuchformel in Punkten: R' = R + K (S − E) mit E = 1/(1 + 10^(−ΔR/400)); die Summe bleibt erhalten."""
    model = small(n=6, seasons=3, games=4, churn=0.0)
    seed = 2
    d = P.draws(seed, 3, 4, 7)
    delta = C.POP_SD * math.sqrt(12.0 / 35.0)
    true = [delta * (i - 2.5) for i in range(6)]
    pts = [1500.0] * 6
    for s in range(3):
        for g in range(4):
            order = sorted(range(6), key=lambda q: (d["pair"][s, g, q], q))
            for m in range(3):
                a, b = order[2 * m], order[2 * m + 1]
                e = 1.0 / (1.0 + 10.0 ** (-(pts[a] - pts[b]) / 400.0))
                w = 1.0 if d["win"][s, g, m] < sg(true[a] - true[b]) else 0.0
                pts[a] += 20.0 * (w - e)
                pts[b] -= 20.0 * (w - e)
        assert sum(pts) == pytest.approx(9000.0, abs=1e-9)
    pool = P.simulate(model, [seed])
    assert pool.final_rating[0, :6] * C.SCALE + 1500.0 == pytest.approx(pts, abs=1e-6)


def test_placement_estimate_equals_scipys_maximiser():
    optimize = pytest.importorskip("scipy.optimize")
    from eli_rating import placement_estimates
    rng = np.random.default_rng(4)
    for _ in range(8):
        r_opp = rng.normal(0.0, 0.8, 10)
        wins = (rng.random(10) < 0.5).astype(float)
        prior = float(rng.choice([0.3, 0.5, 1.0, 1.5]))
        Rm = r_opp[None, :]
        stay = np.ones((1, 10), dtype=bool)
        u_opp = ((np.arange(10) + 0.5) / 10).reshape(1, 1, 10)
        u_win = np.where(wins > 0, 0.0, 0.999999).reshape(1, 1, 10)
        est = placement_estimates(np.zeros((1, 1)), Rm, np.zeros((1, 10)), stay, u_opp, u_win, prior)[0, 0]

        def neg(r):
            p = 1.0 / (1.0 + np.exp(-(r[0] - r_opp)))
            return -(np.sum(wins * np.log(p) + (1 - wins) * np.log(1 - p)) - r[0] ** 2 / (2 * prior ** 2))

        res = optimize.minimize(neg, [0.0], method="BFGS", options={"gtol": 1e-10})
        assert est == pytest.approx(res.x[0], abs=1e-5)


def test_sum_balance_adds_up_over_many_runs():
    pool = P.simulate(small(n=12, seasons=6, k0=100.0, floor=1450.0, leave="die Schwächsten", anchor_join=1, anchor_stay=3, placement=True), range(20))
    for b in range(pool.runs):
        bal = ev.balance(pool, b)
        assert bal["entries"] + bal["k"] + bal["floor"] + bal["anchor"] == pytest.approx(bal["end"], abs=1e-6)


def test_the_anchors_own_rating_follows_the_drift_in_a_learning_field():
    """Statistische Prüfung: Wird das Feld besser, sinkt die Wertung des Ankers (konstante Stärke) um etwa die Fähigkeitszunahme minus Nachlauf."""
    pool = P.simulate(small(n=20, seasons=20, games=10, churn=0.15, trend=0.05, k=30.0, anchor_join=0), range(40))
    late = ev.anchor_points(pool)[:, -3:].mean()
    gain = ev.level_points(pool)[:, -1].mean()
    assert gain > 150 and late < -0.6 * gain

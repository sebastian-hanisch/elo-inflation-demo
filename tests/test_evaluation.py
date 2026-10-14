"""Kennzahlen, Analyse, Tabelle und Verdikte: Handwerte, Randfälle und die Grenzen der Verdikt-Schwellen."""

import numpy as np
import pytest

import eli_constants as C
import eli_evaluation as ev
import eli_pool as P
from conftest import hand_pool
from eli_scenario import Settings, make_model


# --- Formatierung -----------------------------------------------------------------------------------------------------------------------------

def test_german_number_formatting():
    assert ev.de(1.5, 1) == "1,5" and ev.de(1234.0) == "1234" and ev.pct(0.256) == "26" and ev.pct(0.256, 1) == "25,6"
    assert ev.thou(15708.4) == "15.708" and ev.thou(999.0) == "999"
    assert ev.signed(5.0) == "+5" and ev.signed(-5.0) == "−5" and ev.signed(0.0) == "+0" and ev.signed(-0.04, 1) == "−0,0"


# --- Maße -------------------------------------------------------------------------------------------------------------------------------------

def test_drift_is_rating_gain_minus_ability_gain_in_points():
    pool = hand_pool(rating=(0.1, 0.3), true=(0.0, 0.2), base=-0.05)
    assert ev.rating_points(pool)[0] == pytest.approx([0.1 * C.SCALE, 0.3 * C.SCALE])
    assert ev.level_points(pool)[0] == pytest.approx([0.05 * C.SCALE, 0.25 * C.SCALE])
    assert ev.drift_points(pool)[0] == pytest.approx([0.05 * C.SCALE, 0.05 * C.SCALE])


def test_anchor_points_keep_nan_outside_the_stay():
    pool = hand_pool(rating=(0.0, 0.0, 0.0), true=(0.0, 0.0, 0.0), anchor=(np.nan, 0.5, 0.4))
    a = ev.anchor_points(pool)[0]
    assert np.isnan(a[0]) and a[1:] == pytest.approx([0.5 * C.SCALE, 0.4 * C.SCALE])


def test_created_and_floor_points_are_cumulative():
    pool = hand_pool(rating=(0.0, 0.0, 0.0), true=(0.0, 0.0, 0.0), created=[[0.1, -0.05, 0.2]], floor_lift=[[0.0, 0.3, 0.1]])
    assert ev.created_points(pool)[0] == pytest.approx(C.SCALE * np.array([0.1, 0.05, 0.25]))
    assert ev.floor_points(pool)[0] == pytest.approx(C.SCALE * np.array([0.0, 0.3, 0.4]))


def test_entry_error_mean_and_spread_over_all_new_players():
    pool = hand_pool(rating=(0.0, 0.0), true=(0.0, 0.0), entry_mean=[[0.2, np.nan]], entry_msq=[[0.05, np.nan]])
    mean, sd = ev.entry_error(pool)
    assert mean == pytest.approx(0.2 * C.SCALE) and sd == pytest.approx(C.SCALE * np.sqrt(0.05 - 0.04))


def test_entry_error_without_new_players_is_nan():
    mean, sd = ev.entry_error(hand_pool())
    assert np.isnan(mean) and np.isnan(sd)


def test_entry_error_spread_never_goes_below_zero_by_rounding():
    pool = hand_pool(rating=(0.0,), true=(0.0,), entry_mean=[[0.3]], entry_msq=[[0.09 - 1e-12]])
    assert ev.entry_error(pool)[1] == 0.0


def test_balance_sums_the_four_items_and_the_end():
    pool = hand_pool(rating=(0.0, 0.0), true=(0.0, 0.0), churn_delta=[[0.1, 0.2]], created=[[0.05, 0.0]], floor_lift=[[0.0, 0.01]], anchor_in=[[0.0, 0.3]],
                     anchor_out=[[0.0, -0.1]], total_end=[[0.0, 0.56]])
    b = ev.balance(pool)
    assert b["entries"] == pytest.approx(0.3 * C.SCALE) and b["k"] == pytest.approx(0.05 * C.SCALE) and b["floor"] == pytest.approx(0.01 * C.SCALE)
    assert b["anchor"] == pytest.approx(0.2 * C.SCALE) and b["end"] == pytest.approx(0.56 * C.SCALE)


# --- Analyse und Tabelle ----------------------------------------------------------------------------------------------------------------------

def test_analysis_matches_a_direct_simulation():
    s = Settings(n=10, seasons=6, games=4, seed=5)
    a = ev.analyse(s)
    pool = P.simulate(make_model(s), (5,))
    assert a.drift == pytest.approx(ev.drift_points(pool)[0]) and a.rating == pytest.approx(ev.rating_points(pool)[0])
    assert a.level == pytest.approx(ev.level_points(pool)[0]) and len(a.drift) == 6


def test_anchor_join_is_clamped_into_the_run():
    assert Settings(seasons=10, join=0).anchor_join == 0
    assert Settings(seasons=10, join=99).anchor_join == 8
    assert Settings(seasons=10, join=3, use_anchor=False).anchor_join is None


def test_the_model_takes_optional_mechanisms_only_when_switched_on():
    off = make_model(Settings(use_k0=False, use_floor=False))
    assert off.k0 is None and off.floor is None
    on = make_model(Settings(use_k0=True, k0=90.0, use_floor=True, floor=1400.0, start=C.START[1]))
    assert on.k0 == 90.0 and on.floor == 1400.0 and on.placement


def test_field_table_has_a_row_per_player_and_the_anchor_only_with_anchor():
    with_anchor = ev.field_table(ev.analyse(Settings(n=10, seasons=6, games=4, use_anchor=True, join=1)))
    assert len(with_anchor) == 11 and with_anchor[-1]["Spieler"] == "Anker" and with_anchor[-1]["wahre Stärke"] == 1500
    without = ev.field_table(ev.analyse(Settings(n=10, seasons=6, games=4, use_anchor=False)))
    assert len(without) == 10 and all(r["Spieler"] != "Anker" for r in without)
    row = with_anchor[0]
    assert row["Abweichung"] == row["Wertung"] - row["wahre Stärke"]


def test_founders_have_played_all_seasons_in_a_closed_field():
    t = ev.field_table(ev.analyse(Settings(n=10, seasons=6, games=4, churn=0.0, use_anchor=False)))
    assert {r["Saisons im Feld"] for r in t} == {6}


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def _analysis_with_drift(end: float, with_anchor: bool = True) -> ev.Analysis:
    n = 3
    z = np.zeros(n)
    anchor = np.array([np.nan, 0.0, 12.0]) if with_anchor else np.full(n, np.nan)
    return ev.Analysis(Settings(seasons=n), None, np.array([0.0, 0.0, end]), z.copy(), z.copy(), anchor, z.copy(), z.copy(), (0.0, 100.0))


def test_verdict_boundary_between_noise_and_inflation():
    below = ev.verdict_analysis(_analysis_with_drift(2 * ev.NOISE - 0.01))
    at = ev.verdict_analysis(_analysis_with_drift(2 * ev.NOISE))
    assert below[0] == "info" and "Rauschen" in below[1]
    assert at[0] == "warning" and "Inflation" in at[1]
    assert ev.verdict_analysis(_analysis_with_drift(-2 * ev.NOISE))[1].count("Deflation") == 1
    assert ev.verdict_analysis(_analysis_with_drift(-2 * ev.NOISE + 0.01))[0] == "info"


def test_verdict_names_the_anchor_only_when_it_played():
    assert "Anker" in ev.verdict_analysis(_analysis_with_drift(0.0, True))[1]
    assert "Anker" not in ev.verdict_analysis(_analysis_with_drift(0.0, False))[1]
    assert "+12 Punkten" in ev.verdict_analysis(_analysis_with_drift(0.0, True))[1]


# --- Experimente: Form und Reihenfolge --------------------------------------------------------------------------------------------------------

def test_every_sweep_returns_the_documented_rows():
    assert [r["key"] for r in ev.sweep_who()] == ["closed", "random", "weak", "veterans", "youth"]
    assert [r["key"] for r in ev.sweep_field()] == ["still", "learns", "ages"]
    assert [r["key"] for r in ev.sweep_start_k()] == ["random", "random_k0", "weak", "weak_k0"]
    assert [r["key"] for r in ev.sweep_anchor()] == ["closed", "weak", "youth", "weak_k0", "learns", "ages"]
    assert [r["key"] for r in ev.sweep_nothing()] == ["weak", "weak_floor", "weak_steps", "random", "random_floor", "random_steps"]
    rows = ev.sweep_placement()
    assert [r["key"] for r in rows] == ["random_start", "random_place", "youth_start", "youth_place", "weak_start", "weak_place", "weak_k0_start", "weak_k0_place"]
    assert [r["placement"] for r in rows] == [False, True] * 4
    prior = ev.sweep_prior()
    assert [(r["scenario"], r["prior"]) for r in prior] == [("youth", 0.5), ("youth", 1.0), ("youth", 2.0), ("weak", 0.5), ("weak", 1.0), ("weak", 2.0)]
    assert [r["scenario"] for r in ev.sweep_placement_anchor()] == ["youth", "weak_k0"]


def test_experiments_run_over_the_sixty_fixed_instances_of_forty_seasons():
    row = ev.sweep_who()[1]
    assert len(row["series"]) == C.EXP_SEASONS and row["d40"] == row["series"][-1] and row["d10"] == row["series"][9] and row["d20"] == row["series"][19]
    assert ev._pool(ev._model(leave=C.LEAVE[0])).runs == 60


def test_anchor_experiment_measures_after_the_stay_and_in_the_right_era():
    m = ev._model(seasons=2 + C.ANCHOR_STAY, anchor_join=2, anchor_stay=C.ANCHOR_STAY, leave=C.LEAVE[1])
    pool = ev._pool(m)
    assert np.isnan(pool.anchor[:, 1]).all() and np.isfinite(pool.anchor[:, 2:]).all()
    value = ev.anchor_after({"leave": C.LEAVE[1]}, 2)
    assert value == pytest.approx(ev.anchor_points(pool)[:, 2 + C.ANCHOR_STAY - 1])


def test_join_max_leaves_two_seasons_for_the_anchor():
    from eli_scenario import join_max
    assert join_max(40) == 38 and join_max(5) == 3


def test_field_table_labels_and_ability_relative_to_the_founders():
    model = P.Model(n=2, seasons=2, games=1, churn=0.0)
    pool = hand_pool(model=model, base=0.2, final_rating=[[0.1, -0.1, 0.0]], final_true=[[0.5, 0.2, 0.0]], final_age=[[2, 1, 0]], anchor=[np.nan, 0.0])
    a = ev.Analysis(Settings(n=2, seasons=2, use_anchor=True, join=0), pool, np.zeros(2), np.zeros(2), np.zeros(2), np.array([np.nan, 0.0]), np.zeros(2), np.zeros(2), (0.0, 0.0))
    rows = ev.field_table(a)
    assert [r["Spieler"] for r in rows] == ["1", "2", "Anker"]
    assert rows[0]["wahre Stärke"] == round(1500 + C.SCALE * 0.3) and rows[0]["Wertung"] == round(1500 + C.SCALE * 0.1)
    assert rows[1]["Saisons im Feld"] == 1 and rows[0]["Abweichung"] == rows[0]["Wertung"] - rows[0]["wahre Stärke"]


def _rows(keys, **vals):
    return [{"key": k, "label": k, **{f: v for f, v in vals.items()}} for k in keys]


def test_anchor_verdict_quotes_the_largest_standard_error():
    rows = []
    for k in ("closed", "weak", "youth", "weak_k0", "learns", "ages"):
        rows.append({"key": k, "label": k, "a2": 0.0, "a15": 0.0, "a30": 0.0, "se2": 1.0, "se15": 2.0, "se30": 3.0, "shift": 0.0})
    rows[1]["se15"] = 7.0
    assert "±7 Punkten" in ev.verdict_anchor(rows)[1]


def test_placement_verdict_quotes_the_youth_rows_not_the_other_scenarios():
    def row(key, d40, **kw):
        return {"key": key, "d40": d40, "entry_mean": 1.0, "entry_sd": 2.0, **kw}
    rows = [row("youth_start", 50.0, entry_mean=173.0), row("youth_place", -150.0, entry_mean=7.0, entry_sd=103.0), row("weak_start", -25.0), row("weak_place", 34.0),
            row("weak_k0_start", 61.0), row("weak_k0_place", 127.0)]
    prior = [{"scenario": s, "prior": p, "d40": d} for s, p, d in (("youth", 0.5, -7.0), ("youth", 1.0, -150.0), ("youth", 2.0, -1314.0), ("weak", 0.5, 111.0),
                                                                    ("weak", 1.0, 222.0), ("weak", 2.0, 333.0))]
    anchor = [{"scenario": "weak_k0", "start": (1.0, 2.0), "place": (3.0, 4.0)}, {"scenario": "youth", "start": (55.0, 52.0), "place": (15.0, -114.0)}]
    text = ev.verdict_placement(rows, prior, anchor)[1]
    assert "−7 bei 0,5" in text and "−1314 bei 2,0" in text and "+15 → −114 gegen +55 → +52" in text
    assert "111" not in text

"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Zufallsstrom hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import numpy as np
import pytest

import eli_constants as C
import eli_evaluation as ev


@pytest.fixture(scope="module")
def who():
    return {r["key"]: r for r in ev.sweep_who()}


@pytest.fixture(scope="module")
def field():
    return {r["key"]: r for r in ev.sweep_field()}


@pytest.fixture(scope="module")
def start_k():
    return {r["key"]: r for r in ev.sweep_start_k()}


@pytest.fixture(scope="module")
def anchor():
    return {r["key"]: r for r in ev.sweep_anchor()}


@pytest.fixture(scope="module")
def nothing():
    return {r["key"]: r for r in ev.sweep_nothing()}


@pytest.fixture(scope="module")
def placement():
    return {r["key"]: r for r in ev.sweep_placement()}


@pytest.fixture(scope="module")
def prior():
    return {(r["scenario"], r["prior"]): r for r in ev.sweep_prior()}


@pytest.fixture(scope="module")
def placement_anchor():
    return {r["scenario"]: r for r in ev.sweep_placement_anchor()}


# --- Experiment 1: wer geht, wer kommt --------------------------------------------------------------------------------------------------------

def test_a_closed_field_has_no_drift(who):
    assert abs(who["closed"]["d40"]) < 1e-9 and who["closed"]["sd"] < 1e-9


def test_random_and_veteran_leave_move_nothing_measurable(who):
    for key in ("random", "veterans"):
        assert abs(who[key]["d40"]) < 3 * who[key]["se"] + 1.0, key
    assert who["random"]["d40"] == pytest.approx(-3.6, abs=1.0) and who["veterans"]["d40"] == pytest.approx(0.6, abs=1.0)


def test_the_weakest_leaving_deflates_by_about_25_points(who):
    w = who["weak"]
    assert w["d10"] == pytest.approx(-22.2, abs=1.0) and w["d20"] == pytest.approx(-21.0, abs=1.0) and w["d40"] == pytest.approx(-24.6, abs=1.0)
    assert w["level"] == pytest.approx(184.1, abs=1.0) and w["rating"] == pytest.approx(159.6, abs=1.0)


def test_learning_newcomers_started_at_1500_inflate_by_about_50_points(who):
    y = who["youth"]
    assert y["d10"] == pytest.approx(55.9, abs=1.0) and y["d20"] == pytest.approx(53.1, abs=1.0) and y["d40"] == pytest.approx(50.1, abs=1.0)
    assert y["level"] == pytest.approx(-49.8, abs=1.0) and abs(y["rating"]) < 3 * y["se"] + 1.0
    assert y["entry_mean"] == pytest.approx(172.9, abs=1.0) and y["entry_sd"] == pytest.approx(138.9, abs=1.0)


def test_a_single_run_scatters_by_about_the_stated_noise(who):
    assert who["random"]["sd"] == pytest.approx(ev.NOISE, abs=1.0)
    assert 10 < who["weak"]["sd"] < 20 and 10 < who["youth"]["sd"] < 20


def test_the_rating_sum_is_stationary_when_leavers_and_newcomers_carry_the_same():
    """Zufälliger Abgang, Zugang bei 1500: Die Gehenden nehmen im Mittel 1500 mit, die Kommenden bringen 1500; die Summe der Wertungen bleibt stehen."""
    pool = ev._pool(ev._model(leave=C.LEAVE[0]))
    assert abs(np.diff(pool.total_start[:, 5:], axis=1).mean() * C.SCALE) < 10.0
    assert abs(pool.churn_delta[:, :-1].mean() * C.SCALE) < 10.0


def test_the_rating_mean_approaches_the_level_where_leavers_take_what_newcomers_bring():
    """Gehen die Schwächsten, nehmen sie weniger mit, als die Neuen bringen: Das Mittel steigt, mit abnehmendem Schritt (bei 40 Saisons noch nicht am Ende)."""
    pool = ev._pool(ev._model(leave=C.LEAVE[1]))
    steps = [np.diff(pool.total_start[:, a:b], axis=1).mean() * C.SCALE for a, b in ((0, 10), (10, 20), (20, 30), (30, 40))]
    assert steps[0] > steps[1] > steps[2] > steps[3] > 0
    assert steps[3] < 0.2 * steps[0]
    assert pool.churn_delta[:, :-1].mean() > 0                                          # die Kommenden bringen mehr, als die Gehenden mitnehmen


# --- Experiment 2: das Feld lernt oder altert -------------------------------------------------------------------------------------------------

def test_a_learning_field_deflates_and_an_ageing_field_inflates(field):
    assert field["learns"]["d40"] == pytest.approx(-142.6, abs=1.5) and field["ages"]["d40"] == pytest.approx(135.4, abs=1.5)
    assert field["learns"]["d10"] == pytest.approx(-31.9, abs=1.0) and field["learns"]["d20"] == pytest.approx(-70.5, abs=1.0)
    assert field["ages"]["d10"] == pytest.approx(37.6, abs=1.0) and field["ages"]["d20"] == pytest.approx(68.5, abs=1.0)
    assert field["learns"]["level"] == pytest.approx(142.7, abs=1.0) and field["ages"]["level"] == pytest.approx(-135.3, abs=1.0)
    assert abs(field["learns"]["rating"]) < 3 * field["learns"]["se"] + 1.0 and abs(field["ages"]["rating"]) < 3 * field["ages"]["se"] + 1.0


# --- Experiment 3: Anlauf-K -------------------------------------------------------------------------------------------------------------------

def test_start_k_changes_little_with_random_leave_and_much_with_selection(start_k):
    assert start_k["random"]["d40"] == pytest.approx(-3.6, abs=1.0) and start_k["random_k0"]["d40"] == pytest.approx(-2.3, abs=1.0)
    assert start_k["weak"]["d40"] == pytest.approx(-24.6, abs=1.0) and start_k["weak_k0"]["d40"] == pytest.approx(60.7, abs=1.5)
    assert start_k["weak_k0"]["created"] == pytest.approx(6596.0, abs=60.0) and abs(start_k["random_k0"]["created"]) < 300
    assert start_k["weak"]["created"] == 0.0


# --- Experiment 4: der Anker ------------------------------------------------------------------------------------------------------------------

def test_the_anchor_reads_the_field_level_in_a_learning_or_ageing_field(anchor):
    assert anchor["learns"]["a2"] == pytest.approx(-1.1, abs=2.0) and anchor["learns"]["a30"] == pytest.approx(-96.1, abs=3.0)
    assert anchor["ages"]["a2"] == pytest.approx(19.8, abs=2.0) and anchor["ages"]["a30"] == pytest.approx(80.2, abs=3.0)
    assert anchor["learns"]["shift"] < -75 and anchor["ages"]["shift"] > 45


def test_the_anchor_in_the_middle_era_and_with_a_shorter_stay(anchor):
    assert anchor["learns"]["a15"] == pytest.approx(-54.1, abs=3.0) and anchor["ages"]["a15"] == pytest.approx(44.3, abs=3.0)
    assert anchor["youth"]["a15"] == pytest.approx(55.2, abs=3.0)
    short = ev.anchor_after({"leave": C.LEAVE[0], "trend": C.LEARN_STEP}, 30, stay=2)
    assert short.mean() == pytest.approx(-60.0, abs=4.0) and short.mean() > anchor["learns"]["a30"] + 20


def test_the_anchor_shift_in_the_other_cases_is_inside_the_noise(anchor):
    others = [anchor[k] for k in ("closed", "weak", "youth", "weak_k0")]
    assert max(abs(r["shift"]) for r in others) < 17
    assert min(r["shift"] for r in others) == pytest.approx(-13.6, abs=2.0) and max(r["shift"] for r in others) == pytest.approx(15.6, abs=2.0)
    assert max(r[f"se{e}"] for r in anchor.values() for e in C.ANCHOR_ERAS) < 9


def test_the_anchor_in_the_youth_field_is_inflated_from_the_start(anchor):
    assert anchor["youth"]["a2"] == pytest.approx(54.8, abs=2.0) and anchor["youth"]["a30"] == pytest.approx(51.6, abs=2.0)


def test_the_anchor_is_noisy_in_a_single_run(anchor):
    pool = ev._pool(ev._model(seasons=2 + C.ANCHOR_STAY, anchor_join=2, anchor_stay=C.ANCHOR_STAY, churn=0.0))
    sd = float(ev.anchor_points(pool)[:, 2 + C.ANCHOR_STAY - 1].std())
    assert sd == pytest.approx(ev.NOISE_ANCHOR, abs=6.0)


# --- Experiment 5: Untergrenze und K-Stufe ----------------------------------------------------------------------------------------------------

def test_floor_and_k_steps_do_not_move_the_weakest_leave_scenario(nothing):
    assert nothing["weak"]["d40"] == pytest.approx(-24.6, abs=1.0)
    assert abs(nothing["weak_floor"]["d40"] - nothing["weak"]["d40"]) < 0.5 and nothing["weak_floor"]["floor"] == pytest.approx(5.1, abs=2.0)
    assert nothing["weak_steps"]["d40"] == pytest.approx(-26.8, abs=1.0) and abs(nothing["weak_steps"]["d40"] - nothing["weak"]["d40"]) < 3


def test_a_floor_that_bites_creates_points_and_inflates(nothing):
    r = nothing["random_floor"]
    assert r["floor"] == pytest.approx(15708.0, abs=150.0) and r["d40"] == pytest.approx(56.9, abs=1.5)
    assert nothing["random_steps"]["d40"] == pytest.approx(-5.4, abs=1.0) and nothing["random_steps"]["created"] < 0


# --- Experiment 6: Platzierungspartien --------------------------------------------------------------------------------------------------------

def test_placement_removes_the_entry_error_of_the_learning_youth(placement):
    assert placement["youth_start"]["entry_mean"] == pytest.approx(172.9, abs=1.0) and placement["youth_place"]["entry_mean"] == pytest.approx(6.8, abs=1.5)
    assert placement["youth_place"]["entry_sd"] == pytest.approx(103.4, abs=1.5) and placement["random_place"]["entry_sd"] == pytest.approx(95.8, abs=1.5)


def test_placement_flips_the_sign_of_the_youth_drift(placement):
    assert placement["youth_start"]["d40"] == pytest.approx(50.1, abs=1.0) and placement["youth_place"]["d40"] == pytest.approx(-149.6, abs=2.0)
    assert placement["youth_place"]["d10"] == pytest.approx(-38.7, abs=1.5) and placement["youth_place"]["d20"] == pytest.approx(-93.6, abs=2.0)
    assert abs(placement["random_place"]["d40"]) < 3 * placement["random_place"]["se"] + 1.0


def test_placement_makes_the_selected_scenarios_worse(placement):
    assert placement["weak_start"]["d40"] == pytest.approx(-24.6, abs=1.0) and placement["weak_place"]["d40"] == pytest.approx(33.7, abs=1.5)
    assert placement["weak_k0_start"]["d40"] == pytest.approx(60.7, abs=1.5) and placement["weak_k0_place"]["d40"] == pytest.approx(127.2, abs=2.0)


def test_placement_depends_on_the_prior_and_diverges_for_large_ones(prior):
    assert prior[("youth", 0.5)]["d40"] == pytest.approx(-7.2, abs=1.5) and prior[("youth", 1.0)]["d40"] == pytest.approx(-149.6, abs=2.0)
    assert prior[("youth", 2.0)]["d40"] == pytest.approx(-1313.8, abs=40.0) and prior[("youth", 2.0)]["entry_sd"] > 800
    assert prior[("weak", 0.5)]["d40"] == pytest.approx(-7.2, abs=1.5) and prior[("weak", 2.0)]["d40"] == pytest.approx(78.2, abs=3.0)


def test_the_anchor_loses_points_under_placement_in_the_youth_field(placement_anchor):
    y = placement_anchor["youth"]
    assert y["start"][0] == pytest.approx(54.8, abs=2.0) and y["start"][1] == pytest.approx(51.6, abs=2.0)
    assert y["place"][0] == pytest.approx(14.7, abs=3.0) and y["place"][1] == pytest.approx(-113.9, abs=4.0)
    k = placement_anchor["weak_k0"]
    assert k["start"][1] == pytest.approx(41.6, abs=3.0) and k["place"][1] == pytest.approx(98.0, abs=4.0)


# --- Verdikte nennen die Zahlen aus den Daten -------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(who, field, start_k, anchor, nothing):
    assert "**−25**" in ev.verdict_who(list(who.values()))[1] and "**+50**" in ev.verdict_who(list(who.values()))[1]
    assert "**−143**" in ev.verdict_field(list(field.values()))[1] and "**+135**" in ev.verdict_field(list(field.values()))[1]
    assert "**+61**" in ev.verdict_start_k(list(start_k.values()))[1] and "6.596" in ev.verdict_start_k(list(start_k.values()))[1]
    assert "**−1 → −96**" in ev.verdict_anchor(list(anchor.values()))[1]
    assert "**+57**" in ev.verdict_nothing(list(nothing.values()))[1] and "15.708" in ev.verdict_nothing(list(nothing.values()))[1]

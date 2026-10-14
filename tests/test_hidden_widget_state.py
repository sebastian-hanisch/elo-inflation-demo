"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück. Die Obergrenze des Beitritts hängt von den Saisons ab."""

import os

from streamlit.testing.v1 import AppTest

import eli_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 600
HIDDEN = {"k0_slider", "floor_slider", "prior_slider", "join_slider"}


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def widget_keys(at):
    return {s.key for s in at.slider}


def test_widgets_are_only_drawn_when_their_switch_is_on():
    at = fresh()
    assert widget_keys(at) & HIDDEN == {"join_slider"}              # der Anker ist voreingestellt
    at.checkbox(key="use_k0_check").set_value(True).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == {"join_slider", "k0_slider"}
    at.checkbox(key="use_floor_check").set_value(True).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == {"join_slider", "k0_slider", "floor_slider"}
    at.selectbox(key="start_select").set_value(C.START[1]).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == HIDDEN
    at.checkbox(key="use_anchor_check").set_value(False).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == HIDDEN - {"join_slider"}
    assert not at.exception


def test_hidden_values_survive_and_are_not_written_into_hidden_widgets():
    at = fresh()
    at.checkbox(key="use_k0_check").set_value(True).run(timeout=TIMEOUT)
    at.slider(key="k0_slider").set_value(200).run(timeout=TIMEOUT)
    at.checkbox(key="use_k0_check").set_value(False).run(timeout=TIMEOUT)
    assert "k0_slider" not in at.session_state
    at.checkbox(key="use_k0_check").set_value(True).run(timeout=TIMEOUT)
    assert at.slider(key="k0_slider").value == 200
    assert not at.exception


def test_join_follows_a_shrinking_number_of_seasons():
    at = fresh()
    at.slider(key="join_slider").set_value(35).run(timeout=TIMEOUT)
    at.slider(key="seasons_slider").set_value(10).run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="join_slider").value <= 8


def test_preset_sets_values_and_hidden_widgets_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Das Feld lernt"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="leave_select").value == C.LEAVE[0] and abs(at.slider(key="trend_slider").value - 0.02) < 1e-9
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Geschlossenes Feld"]].click().run(timeout=TIMEOUT)
    assert at.slider(key="churn_slider").value == 0.0 and abs(at.slider(key="trend_slider").value) < 1e-9
    assert not at.exception


def test_the_anchor_switch_removes_the_anchor_from_the_table():
    at = fresh()
    assert any("Anker" in str(row) for row in at.dataframe[0].value.to_dict("records"))
    at.checkbox(key="use_anchor_check").set_value(False).run(timeout=TIMEOUT)
    assert not any(r["Spieler"] == "Anker" for r in at.dataframe[0].value.to_dict("records"))
    assert not at.exception

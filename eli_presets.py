"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Ausblendbare Regler (Anlauf-K, Untergrenze, Prior, Beitritt des Ankers je nach Schalter) folgen dem KEPT-Muster
(DEMO-PLAYBOOK §3): der Wert eines nicht gezeichneten Reglers wird nie in sein Widget geschrieben, sondern in KEPT abgelegt und unmittelbar vor dem Zeichnen per
`seed_widget` eingesetzt. Die Obergrenze des Beitritts hängt von den Saisons ab; `clamp_kept` begrenzt vorher."""

import random
from dataclasses import dataclass

import streamlit as st

import eli_constants as C


def _bool(value) -> bool:
    return str(value).lower() in ("1", "true", "ja")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "seasons_slider": SettingSpec("ss", int, C.DEFAULT_SEASONS, C.SEASONS_MIN, C.SEASONS_MAX),
    "games_slider": SettingSpec("g", int, C.DEFAULT_GAMES, C.GAMES_MIN, C.GAMES_MAX),
    "churn_slider": SettingSpec("ch", float, C.DEFAULT_CHURN, C.CHURN_MIN, C.CHURN_MAX),
    "leave_select": SettingSpec("lv", str, C.DEFAULT_LEAVE, choices=C.LEAVE),
    "gap_slider": SettingSpec("gp", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "trend_slider": SettingSpec("tr", float, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "use_k0_check": SettingSpec("uk", _bool, False),
    "k0_slider": SettingSpec("k0", int, C.DEFAULT_K0, C.K0_MIN, C.K0_MAX),
    "use_floor_check": SettingSpec("uf", _bool, False),
    "floor_slider": SettingSpec("fl", int, C.DEFAULT_FLOOR, C.FLOOR_MIN, C.FLOOR_MAX),
    "k_steps_check": SettingSpec("ks", _bool, False),
    "start_select": SettingSpec("st", str, C.START[0], choices=C.START),
    "prior_slider": SettingSpec("pr", float, C.DEFAULT_PRIOR, C.PRIOR_MIN, C.PRIOR_MAX),
    "use_anchor_check": SettingSpec("ua", _bool, True),
    "join_slider": SettingSpec("j", int, C.DEFAULT_JOIN, C.JOIN_MIN, C.SEASONS_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "seasons": "seasons_slider", "games": "games_slider", "churn": "churn_slider", "leave": "leave_select", "gap": "gap_slider",
               "trend": "trend_slider", "k": "k_slider", "use_k0": "use_k0_check", "k0": "k0_slider", "use_floor": "use_floor_check", "floor": "floor_slider",
               "k_steps": "k_steps_check", "start": "start_select", "prior": "prior_slider", "use_anchor": "use_anchor_check", "join": "join_slider", "seed": "seed_input"}
KEPT_KEYS = ("k0_slider", "floor_slider", "prior_slider", "join_slider")


def _kept() -> dict:
    return st.session_state.setdefault("kept_values", {})


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key in KEPT_KEYS:
            _kept().setdefault(state_key, spec.default)
        elif state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines ausblendbaren Reglers: seinen zuletzt gewählten Wert einsetzen."""
    if state_key not in st.session_state:
        st.session_state[state_key] = _kept().get(state_key, SETTING_SPECS[state_key].default)


def clamp_kept(state_key, lo, hi):
    """Begrenzt den gemerkten (und, falls gezeichnet, den Widget-)Wert auf [lo, hi]; nötig, wenn die Grenzen von anderen Reglern abhängen."""
    kept = _kept()
    kept[state_key] = min(max(kept.get(state_key, SETTING_SPECS[state_key].default), lo), hi)
    if state_key in st.session_state:
        st.session_state[state_key] = min(max(st.session_state[state_key], lo), hi)


def remember(state_key, value):
    _kept()[state_key] = value


def current_value(state_key):
    """Aktueller Wert, auch wenn der Regler in diesem Lauf nicht gezeichnet wird (dann der gemerkte)."""
    if state_key in KEPT_KEYS:
        return st.session_state.get(state_key, _kept().get(state_key, SETTING_SPECS[state_key].default))
    return st.session_state.get(state_key, SETTING_SPECS[state_key].default)


def _store(state_key, value):
    if state_key in KEPT_KEYS:
        _kept()[state_key] = value
        st.session_state.pop(state_key, None)    # beim nächsten Zeichnen aus KEPT einsetzen
    else:
        st.session_state[state_key] = value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        _store(state_key, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            _store(state_key, C.PRESETS[name][key])


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)

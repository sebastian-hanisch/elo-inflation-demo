"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Adressparameter."""

import eli_constants as C
import eli_presets as P
from eli_scenario import Settings

PRESET_FIELDS = {"n": "n", "seasons": "seasons", "games": "games", "churn": "churn", "leave": "leave", "gap": "gap", "trend": "trend", "k": "k", "use_k0": "use_k0", "k0": "k0",
                 "use_floor": "use_floor", "floor": "floor", "k_steps": "k_steps", "start": "start", "prior": "prior", "use_anchor": "use_anchor", "join": "join", "seed": "seed"}


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Die Schwächsten gehen"]
    assert Settings(**{field: p[key] for key, field in PRESET_FIELDS.items()}) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(PRESET_FIELDS.values()) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_parameters_are_unique():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_hidden_widgets_are_exactly_the_switch_dependent_ones():
    assert set(P.KEPT_KEYS) == {"k0_slider", "floor_slider", "prior_slider", "join_slider"}


def test_each_preset_shows_its_mechanism():
    assert C.PRESETS["Geschlossenes Feld"]["churn"] == 0.0
    assert C.PRESETS["Das Feld lernt"]["trend"] > 0
    assert C.PRESETS["Neue unter ihrer Startwertung"]["gap"] > 0
    assert C.PRESETS["Die Schwächsten gehen"]["leave"] == C.LEAVE[1]


def test_bool_caster_reads_the_address_values():
    assert P._bool("True") and P._bool("1") and P._bool("true") and not P._bool("False") and not P._bool("0") and not P._bool("")


def test_sweep_instances_are_fixed_and_distinct():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60


def test_the_prior_slider_ends_below_the_diverging_range():
    assert C.PRIOR_MAX < max(C.PRIOR_GRID) and C.PRIOR_MIN > 0

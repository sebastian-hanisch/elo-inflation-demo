"""Kennzahlen, Analyse des gezeigten Laufs, sechs Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Drift (Punkte) = Änderung des Wertungsmittels der n Feldspieler minus SCALE mal Änderung ihrer mittleren wahren Stärke, gegenüber dem Beginn. Positiv heißt Inflation (die
Wertungen liegen über der Fähigkeit), negativ Deflation. Die absolute Fähigkeit gibt es im Modell nur per Konvention: Stärke 0 = Mittel der Gründer = 1500 Punkte; nachprüfbar ist
sie über den Anker, einen Spieler konstanter Stärke 0."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import eli_constants as C
import eli_pool as P
from eli_scenario import Settings, make_model

NAN = float("nan")
NOISE = 17.0                  # Streuung (Punkte) der Drift eines einzelnen Laufs nach 40 Saisons im Feld mit zufälligem Abgang (Experiment 1)
NOISE_ANCHOR = 40.0           # Streuung (Punkte) der Wertung des Ankers je Lauf nach 4 Saisons (Experiment 4)


def de(x: float, digits: int = 0) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def thou(x: float) -> str:
    """Ganzzahl mit Punkt als Tausendertrenner."""
    return f"{round(x):,}".replace(",", ".")


def signed(x: float, digits: int = 0) -> str:
    """Mit Vorzeichen und echtem Minuszeichen, damit „+46“ und „−147“ gleich breit lesbar sind."""
    s = f"{x:+.{digits}f}".replace(".", ",")
    return s.replace("-", "−")


# --- Maße -------------------------------------------------------------------------------------------------------------------------------------

def level_points(pool: P.Pool) -> np.ndarray:
    """(Läufe, Saisons): Änderung der mittleren wahren Stärke des Feldes gegenüber den Gründern, in Punkten."""
    return C.SCALE * (pool.mean_true - pool.base_true[:, None])


def rating_points(pool: P.Pool) -> np.ndarray:
    """(Läufe, Saisons): Wertungsmittel des Feldes minus 1500, in Punkten."""
    return C.SCALE * pool.mean_rating


def drift_points(pool: P.Pool) -> np.ndarray:
    """(Läufe, Saisons): Wertungsmittel minus Fähigkeit, beides gegenüber dem Beginn, in Punkten."""
    return rating_points(pool) - level_points(pool)


def anchor_points(pool: P.Pool) -> np.ndarray:
    """(Läufe, Saisons): Wertung des Ankers minus 1500, in Punkten; NaN außerhalb seiner Zeit."""
    return C.SCALE * pool.anchor


def created_points(pool: P.Pool) -> np.ndarray:
    """(Läufe, Saisons): kumulierte Punkte, die ungleiches K (Anlauf-K, K-Stufe) erzeugt hat."""
    return C.SCALE * np.cumsum(pool.created, axis=1)


def floor_points(pool: P.Pool) -> np.ndarray:
    """(Läufe, Saisons): kumulierte Punkte, die die Untergrenze erzeugt hat."""
    return C.SCALE * np.cumsum(pool.floor_lift, axis=1)


def balance(pool: P.Pool, run: int = 0) -> dict:
    """Bilanz der Wertungssumme eines Laufs in Punkten über 1500 je Spieler: Zu- und Abgang netto, ungleiches K, Untergrenze, Anker (Beitritt und Austritt), Summe am Ende.
    Es gilt end = entries + k + floor + anchor (Summe der Wertungen ändert sich nur durch diese vier Dinge)."""
    s = C.SCALE
    return {"entries": s * float(pool.churn_delta[run].sum()), "k": s * float(pool.created[run].sum()), "floor": s * float(pool.floor_lift[run].sum()),
            "anchor": s * float(pool.anchor_in[run].sum() + pool.anchor_out[run].sum()), "end": s * float(pool.total_end[run, -1])}


def entry_error(pool: P.Pool):
    """(Mittel, Streuung) von Startwertung minus wahrer Stärke der Neuen, in Punkten, über alle Neuen aller Läufe; (NaN, NaN) ohne Zugang."""
    m = pool.entry_mean[~np.isnan(pool.entry_mean)]
    if not len(m):
        return NAN, NAN
    msq = pool.entry_msq[~np.isnan(pool.entry_msq)]
    mean = float(m.mean())
    return C.SCALE * mean, C.SCALE * math.sqrt(max(float(msq.mean()) - mean ** 2, 0.0))


# --- Der gezeigte Lauf ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    pool: P.Pool                 # ein Lauf
    drift: np.ndarray            # (Saisons,)
    level: np.ndarray
    rating: np.ndarray
    anchor: np.ndarray           # NaN außerhalb der Zeit des Ankers (alles NaN ohne Anker)
    created: np.ndarray
    floor: np.ndarray
    entry: tuple                 # (Mittel, Streuung) des Startfehlers


def analyse(s: Settings) -> Analysis:
    pool = P.simulate(make_model(s), (s.seed,))
    return Analysis(s, pool, drift_points(pool)[0], level_points(pool)[0], rating_points(pool)[0], anchor_points(pool)[0], created_points(pool)[0], floor_points(pool)[0],
                    entry_error(pool))


def field_table(a: Analysis) -> list:
    """Spieler am Ende des Laufs: Platz, Saisons im Feld, Wertung, wahre Stärke und Abweichung (alles in Punkten um 1500; die Fähigkeit relativ zu den Gründern)."""
    p = a.pool
    rows = []
    for q in range(p.model.n + 1):
        if q == p.model.n and a.settings.anchor_join is None:
            continue
        rating = C.BASE + C.SCALE * float(p.final_rating[0, q])
        truth = C.BASE + C.SCALE * float(p.final_true[0, q] - p.base_true[0])
        rows.append({"Spieler": "Anker" if q == p.model.n else str(q + 1), "Saisons im Feld": int(p.final_age[0, q]), "Wertung": round(rating), "wahre Stärke": round(truth),
                     "Abweichung": round(rating - truth)})
    return rows


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

def _model(**kw) -> P.Model:
    return P.Model(**{"n": C.EXP_N, "seasons": C.EXP_SEASONS, "games": C.EXP_GAMES, "churn": C.EXP_CHURN, "k": C.EXP_K, **kw})


@functools.lru_cache(maxsize=128)
def _pool(model: P.Model) -> P.Pool:
    return P.simulate(model, C.SWEEP_SEEDS)


def _drift_row(key: str, label: str, model: P.Model) -> dict:
    pool = _pool(model)
    dr = drift_points(pool)
    mean = dr.mean(axis=0)
    marks = {f"d{m}": float(mean[min(m, model.seasons) - 1]) for m in C.EXP_MARKS}
    ent_mean, ent_sd = entry_error(pool)
    return {"key": key, "label": label, "series": mean, **marks, "sd": float(dr[:, -1].std()), "se": float(dr[:, -1].std() / math.sqrt(pool.runs)),
            "level": float(level_points(pool)[:, -1].mean()), "rating": float(rating_points(pool)[:, -1].mean()), "created": float(created_points(pool)[:, -1].mean()),
            "floor": float(floor_points(pool)[:, -1].mean()), "entry_mean": ent_mean, "entry_sd": ent_sd}


def sweep_who() -> list:
    """Experiment 1: Wer geht, wer kommt (Abgangsregel und lernende Neue)."""
    return [_drift_row("closed", "Geschlossenes Feld", _model(churn=0.0)),
            _drift_row("random", "Zufälliger Abgang", _model(leave=C.LEAVE[0])),
            _drift_row("weak", "Die Schwächsten gehen", _model(leave=C.LEAVE[1])),
            _drift_row("veterans", "Die Veteranen gehen", _model(leave=C.LEAVE[2])),
            _drift_row("youth", "Lernende Jugend", _model(leave=C.LEAVE[0], gap=1.0))]


def sweep_field() -> list:
    """Experiment 2: Das ganze Feld lernt oder altert."""
    return [_drift_row("still", "Feld bleibt gleich", _model(leave=C.LEAVE[0])),
            _drift_row("learns", "Das Feld lernt", _model(leave=C.LEAVE[0], trend=C.LEARN_STEP)),
            _drift_row("ages", "Das Feld altert", _model(leave=C.LEAVE[0], trend=-C.LEARN_STEP))]


def sweep_start_k() -> list:
    """Experiment 3: Anlauf-K, mit und ohne Auslese."""
    return [_drift_row("random", "Zufälliger Abgang, festes K", _model(leave=C.LEAVE[0])),
            _drift_row("random_k0", "Zufälliger Abgang, Anlauf-K", _model(leave=C.LEAVE[0], k0=C.EXP_K0)),
            _drift_row("weak", "Die Schwächsten gehen, festes K", _model(leave=C.LEAVE[1])),
            _drift_row("weak_k0", "Die Schwächsten gehen, Anlauf-K", _model(leave=C.LEAVE[1], k0=C.EXP_K0))]


def _anchor_scenarios() -> tuple:
    return (("closed", "Geschlossenes Feld", dict(churn=0.0)),
            ("weak", "Die Schwächsten gehen", dict(leave=C.LEAVE[1])),
            ("youth", "Lernende Jugend", dict(leave=C.LEAVE[0], gap=1.0)),
            ("weak_k0", "Anlauf-K, die Schwächsten gehen", dict(leave=C.LEAVE[1], k0=C.EXP_K0)),
            ("learns", "Das Feld lernt", dict(leave=C.LEAVE[0], trend=C.LEARN_STEP)),
            ("ages", "Das Feld altert", dict(leave=C.LEAVE[0], trend=-C.LEARN_STEP)))


def anchor_after(kw: dict, era: int, stay: int = C.ANCHOR_STAY, **extra) -> np.ndarray:
    """Wertung des Ankers (Punkte um 1500) am Ende seiner `stay` Saisons, wenn er nach `era` Saisons beitritt: ein Wert je Lauf."""
    model = _model(seasons=era + stay, anchor_join=era, anchor_stay=stay, **kw, **extra)
    return anchor_points(_pool(model))[:, era + stay - 1]


def sweep_anchor() -> list:
    """Experiment 4: Der Anker über die Zeit."""
    rows = []
    for key, label, kw in _anchor_scenarios():
        vals = {era: anchor_after(kw, era) for era in C.ANCHOR_ERAS}
        row = {"key": key, "label": label}
        for era, v in vals.items():
            row[f"a{era}"] = float(v.mean())
            row[f"se{era}"] = float(v.std() / math.sqrt(len(v)))
        first, last = C.ANCHOR_ERAS[0], C.ANCHOR_ERAS[-1]
        row["shift"] = row[f"a{last}"] - row[f"a{first}"]
        row["shift_se"] = math.hypot(row[f"se{last}"], row[f"se{first}"])
        rows.append(row)
    return rows


def sweep_nothing() -> list:
    """Experiment 5: Untergrenze und K-Stufe."""
    return [_drift_row("weak", "Die Schwächsten gehen", _model(leave=C.LEAVE[1])),
            _drift_row("weak_floor", "… mit Untergrenze 1350", _model(leave=C.LEAVE[1], floor=C.EXP_FLOOR)),
            _drift_row("weak_steps", "… mit K-Stufe ab 1600", _model(leave=C.LEAVE[1], k_steps=True)),
            _drift_row("random", "Zufälliger Abgang", _model(leave=C.LEAVE[0])),
            _drift_row("random_floor", "… mit Untergrenze 1450", _model(leave=C.LEAVE[0], floor=1450)),
            _drift_row("random_steps", "… mit K-Stufe ab 1600", _model(leave=C.LEAVE[0], k_steps=True))]


def _placement_scenarios() -> tuple:
    return (("random", "Zufälliger Abgang", dict(leave=C.LEAVE[0])),
            ("youth", "Lernende Jugend", dict(leave=C.LEAVE[0], gap=1.0)),
            ("weak", "Die Schwächsten gehen", dict(leave=C.LEAVE[1])),
            ("weak_k0", "Anlauf-K, die Schwächsten gehen", dict(leave=C.LEAVE[1], k0=C.EXP_K0)))


def sweep_placement() -> list:
    """Experiment 6: Startwertung aus Platzierungspartien, je Szenario ohne und mit (Prior 1,0)."""
    rows = []
    for key, label, kw in _placement_scenarios():
        rows.append({**_drift_row(key + "_start", label, _model(**kw)), "scenario": key, "name": label, "placement": False})
        rows.append({**_drift_row(key + "_place", label, _model(placement=True, prior=C.DEFAULT_PRIOR, **kw)), "scenario": key, "name": label, "placement": True})
    return rows


def sweep_prior() -> list:
    """Experiment 6, zweiter Teil: Drift nach 40 Saisons in Abhängigkeit vom Prior der Platzierung."""
    rows = []
    for key, label, kw in _placement_scenarios()[1:3]:
        for prior in C.PRIOR_GRID:
            rows.append({**_drift_row(f"{key}_prior{prior}", label, _model(placement=True, prior=prior, **kw)), "scenario": key, "name": label, "prior": prior})
    return rows


def sweep_placement_anchor() -> list:
    """Experiment 6, dritter Teil: Anker (Beitritt nach 2 und nach 30 Saisons) ohne und mit Platzierung."""
    rows = []
    for key, label, kw in _placement_scenarios()[1:2] + _placement_scenarios()[3:]:
        row = {"scenario": key, "name": label}
        for placement in (False, True):
            vals = {era: anchor_after(kw, era, placement=placement, prior=C.DEFAULT_PRIOR) for era in (C.ANCHOR_ERAS[0], C.ANCHOR_ERAS[-1])}
            tag = "place" if placement else "start"
            row[tag] = (float(vals[C.ANCHOR_ERAS[0]].mean()), float(vals[C.ANCHOR_ERAS[-1]].mean()))
        rows.append(row)
    return rows


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis):
    s = a.settings
    end = float(a.drift[-1])
    text = (f"Nach {s.seasons} Saisons liegt das Wertungsmittel des Feldes {signed(a.rating[-1])} Punkte gegenüber dem Start, die mittlere Fähigkeit {signed(a.level[-1])}: "
            f"**Drift {signed(end)} Punkte**.")
    if np.isfinite(a.anchor).any():
        last = int(np.nonzero(np.isfinite(a.anchor))[0][-1])
        text += f" Der Anker (konstante Stärke) steht bei {signed(a.anchor[last])} Punkten um 1500 und rauscht je Lauf um etwa ±{de(NOISE_ANCHOR)}."
    if abs(end) < 2 * NOISE:
        return "info", text + f" Das liegt im Rauschen eines einzelnen Laufs (rund ±{de(NOISE)} Punkte); die Experimente unten mitteln über 60 Läufe."
    if end > 0:
        return "warning", text + " **Inflation:** Die Wertungen im Feld liegen über dem, was die Fähigkeit hergibt."
    return "warning", text + " **Deflation:** Die Wertungen im Feld bleiben hinter der Fähigkeit zurück."


def verdict_who(rows):
    r = {x["key"]: x for x in rows}
    return "info", (f"Nach {C.EXP_SEASONS} Saisons (Mittel über 60 Läufe): im geschlossenen Feld **{signed(r['closed']['d40'])}**, bei zufälligem Abgang {signed(r['random']['d40'])} und "
                    f"bei Abgang der Veteranen {signed(r['veterans']['d40'])}: kein messbarer Unterschied (der Mittelwert hat einen Fehler von etwa ±{de(r['random']['se'])}). "
                    f"Gehen die **Schwächsten**, sind es **{signed(r['weak']['d40'])}** (die Fähigkeit steigt um {signed(r['weak']['level'])}, die Wertung nur um "
                    f"{signed(r['weak']['rating'])}); kommen **lernende Neue** bei 1500, sind es **{signed(r['youth']['d40'])}**. Das Wertungsmittel steigt oder fällt, bis die "
                    "Gehenden im Mittel so viel mitnehmen, wie die Kommenden mitbringen.")


def verdict_field(rows):
    r = {x["key"]: x for x in rows}
    return "info", (f"Wird das ganze Feld je Saison um {de(C.LEARN_STEP, 2)} Einheiten besser, **fällt** die Drift auf **{signed(r['learns']['d40'])}** Punkte, wird es schlechter, "
                    f"**steigt** sie auf **{signed(r['ages']['d40'])}**; die Wertungsmittel selbst bleiben bei {signed(r['learns']['rating'])} bzw. {signed(r['ages']['rating'])} Punkten. "
                    "Die Elo-Summe kennt nur Differenzen: Sie kann nicht sehen, dass alle besser geworden sind.")


def verdict_start_k(rows):
    r = {x["key"]: x for x in rows}
    return "warning", (f"Mit Anlauf-K {de(C.EXP_K0)} und **zufälligem Abgang** ändert sich fast nichts ({signed(r['random']['d40'])} gegen {signed(r['random_k0']['d40'])} Punkte). "
                       f"**Gehen die Schwächsten**, wächst die Drift von {signed(r['weak']['d40'])} auf **{signed(r['weak_k0']['d40'])}**, und die Partien erzeugen im Mittel "
                       f"{thou(r['weak_k0']['created'])} Punkte. Das große K verstärkt, was die Auslese ohnehin tut: Wer durch Glück hoch gerutscht ist, bleibt, wer durch Pech "
                       "gefallen ist, geht.")


def verdict_anchor(rows):
    r = {x["key"]: x for x in rows}
    first, last = C.ANCHOR_ERAS[0], C.ANCHOR_ERAS[-1]
    se = max(x[f"se{e}"] for x in rows for e in C.ANCHOR_ERAS)
    others = [x["shift"] for x in rows if x["key"] not in ("learns", "ages")]
    return "info", (f"Derselbe Spieler (Stärke konstant), Beitritt nach {first} oder nach {last} Saisons, Wertung nach {C.ANCHOR_STAY} Saisons: im Feld, das **lernt**, "
                    f"**{signed(r['learns'][f'a{first}'])} → {signed(r['learns'][f'a{last}'])}**, im Feld, das **altert**, **{signed(r['ages'][f'a{first}'])} → "
                    f"{signed(r['ages'][f'a{last}'])}**; in den übrigen Fällen ist die Verschiebung klein ({signed(min(others))} bis {signed(max(others))}). Der Anker liest die Drift "
                    f"ab, mit Nachlauf: Bei der lernenden Jugend steht er von Anfang an bei {signed(r['youth'][f'a{first}'])} ({signed(r['youth'][f'a{last}'])} nach {last} Saisons). "
                    f"Der Fehler jedes Mittelwerts liegt bei etwa ±{de(se)} Punkten, Unterschiede unter etwa 15 Punkten sind nicht zu deuten.")


def verdict_nothing(rows):
    r = {x["key"]: x for x in rows}
    return "info", (f"Untergrenze 1350 und K-Stufe ab 1600 verändern die Drift **nicht**: {signed(r['weak']['d40'])} gegen {signed(r['weak_floor']['d40'])} und "
                    f"{signed(r['weak_steps']['d40'])}. Die Untergrenze greift kaum (im Mittel {thou(r['weak_floor']['floor'])} Punkte), weil die Schwächsten meist gehen, bevor sie sie erreichen; die K-Stufe bewegt nur wenige Spieler. "
                    f"**Bleiben die Schwachen** (zufälliger Abgang) und liegt die Untergrenze bei 1450, erzeugt sie {thou(r['random_floor']['floor'])} Punkte und die Drift steigt "
                    f"auf **{signed(r['random_floor']['d40'])}**: Die Untergrenze wirkt, wo sie greift. Gemessen bei diesen Parametern, kein Beleg für Unschädlichkeit.")


def verdict_placement(rows, prior_rows, anchor_rows):
    r = {x["key"]: x for x in rows}
    yp = {x["prior"]: x for x in prior_rows if x["scenario"] == "youth"}
    ya = next(x for x in anchor_rows if x["scenario"] == "youth")
    return "warning", (f"Die Platzierung beseitigt den **Eintrittsfehler** der lernenden Jugend ({signed(r['youth_start']['entry_mean'])} → {signed(r['youth_place']['entry_mean'])} "
                       f"Punkte), aber die Drift **kehrt das Vorzeichen** um: {signed(r['youth_start']['d40'])} → **{signed(r['youth_place']['d40'])}**. Der Grund: Das Wertungsmittel "
                       "richtet sich nach dem, was die Neuen mitbringen; kommen sie mit ihrer wahren, niedrigen Stärke, sinkt es auf deren Niveau, und der Anker verliert "
                       f"({signed(ya['place'][0])} → {signed(ya['place'][1])} gegen {signed(ya['start'][0])} → {signed(ya['start'][1])} bei Start 1500). Wo ausgelesen wird, "
                       f"wächst die Drift: Schwächsten {signed(r['weak_start']['d40'])} → {signed(r['weak_place']['d40'])}, Anlauf-K {signed(r['weak_k0_start']['d40'])} → "
                       f"{signed(r['weak_k0_place']['d40'])}. Die Schätzung streut um {de(r['youth_place']['entry_sd'])} Punkte und hängt am Prior ({signed(yp[0.5]['d40'])} bei 0,5, "
                       f"{signed(yp[1.0]['d40'])} bei 1,0, {signed(yp[2.0]['d40'])} bei 2,0): ein Gegenmittel mit Nebenwirkungen, nicht von der Stange.")

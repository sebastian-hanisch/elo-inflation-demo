"""Elo-Inflation – wenn die Skala wandert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zusatzstück der Ranking-Linie hinter Elo: Eine Elo-Wertung gilt nur relativ zum Feld. Gehen und kommen Spieler, kann sich die Skala verschieben, ohne dass sich ein einzelner
Spieler ändert. Sechs Experimente messen, wann das geschieht (wer geht, wer kommt, lernt das Feld), woran man es erkennt (ein Anker konstanter Stärke) und was Anlauf-K,
Untergrenze, K-Stufe und Platzierungspartien daran ändern. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import eli_constants as C
import eli_evaluation as ev
from eli_pdf_export import build_pdf
from eli_presets import (
    apply_preset,
    bounds,
    clamp_kept,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from eli_scenario import Settings, join_max
from eli_visualization import (
    build_anchor,
    build_balance,
    build_drift,
    build_drift_lines,
    build_k_effects,
    build_overview,
    build_placement,
    build_prior,
)

st.set_page_config(page_title="Elo-Inflation – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _who():
    return ev.sweep_who()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _field():
    return ev.sweep_field()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _start_k():
    return ev.sweep_start_k()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _anchor():
    return ev.sweep_anchor()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _nothing():
    return ev.sweep_nothing()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _placement():
    return ev.sweep_placement(), ev.sweep_prior(), ev.sweep_placement_anchor()


st.title("📈 Elo-Inflation: wenn die Skala wandert")
st.markdown(
    """
Eine Elo-Wertung sagt nur, wie ein Spieler **gegenüber dem Feld** steht. Gehen und kommen Spieler, kann sich die Skala verschieben, **ohne dass sich ein einzelner Spieler
ändert**: Das ist die **Inflation** (die Zahlen steigen schneller als die Fähigkeit) oder die **Deflation** (sie bleiben zurück). Diese Demo ist ein **Zusatzstück der
Ranking-Linie** hinter dem Elo-Stück und misst in einem simulierten Feld, wann das geschieht, woran man es erkennt (ein **Anker** konstanter Stärke) und was Anlauf-K,
Untergrenze, K-Stufe und **Platzierungspartien** daran ändern.

Weiter unten: sechs **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Warum die Skala wandert", expanded=True):
    st.markdown(
        """
1. **Die Summe ist die Bilanz.** Eine Partie verschiebt Punkte von einem Spieler zum anderen, die Summe aller Wertungen bleibt gleich (bei gleichem K). Sie ändert sich nur durch
   **vier Dinge**: die Startwertungen der Neuen (plus), die Wertungen der Gehenden (minus), Punkte aus **ungleichem K** (Anlauf-K, K-Stufe) und aus einer **Untergrenze**.
2. **Das Mittel richtet sich nach dem Zugang.** Das Wertungsmittel steigt oder fällt, bis die Gehenden im Mittel so viel mitnehmen, wie die Kommenden mitbringen. Kommen alle bei
   1500, strebt das Mittel dorthin, gleichgültig, wie stark sie sind.
3. **Die Fähigkeit steht nirgends.** Elo kennt nur Differenzen. Ob das Feld besser oder schlechter wurde, sieht man nur an einem **Anker**, einem Spieler konstanter
   Stärke, oder an einer externen Messlatte. In der Simulation ist die Fähigkeit bekannt, ihr Nullpunkt (1500) ist eine Konvention.
4. **Drift** heißt hier: Änderung des Wertungsmittels minus Änderung der mittleren Fähigkeit des Feldes, in Punkten. Positiv ist Inflation, negativ Deflation.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Feld und Saisons**")
    n = st.slider("Spieler im Feld n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="So viele Spieler sind jederzeit im Feld (ohne den Anker).")
    seasons = st.slider("Saisons", *bounds("seasons_slider"), key="seasons_slider", help="Nach jeder Saison gehen Spieler, und Neue kommen.")
    games = st.slider("Runden je Saison", *bounds("games_slider"), key="games_slider", help="Jeder Spieler spielt je Runde einmal, zufällig gepaart (bei ungeradem Feld setzt einer aus).")
    churn = st.slider("Abgang je Saison (Anteil)", *bounds("churn_slider"), step=C.CHURN_STEP, key="churn_slider", format="%.2f",
                      help="Anteil des Feldes, der nach jeder Saison geht; ebenso viele Neue kommen. 0 = geschlossenes Feld.")
    st.markdown("**Wer geht, wer kommt**")
    leave = st.selectbox("Wer geht?", C.LEAVE, key="leave_select", help="Zufällig, die mit der niedrigsten Wertung oder die mit den meisten Saisons im Feld.")
    gap = st.slider("Neue starten so weit unter ihrer späteren Stärke", *bounds("gap_slider"), step=C.GAP_STEP, key="gap_slider",
                    help="θ-Einheiten (1 = 174 Punkte). Sie holen je Saison ein Viertel des Abstands auf („lernende Jugend“). 0 = Neue kommen mit ihrer Stärke.")
    trend = st.slider("Das ganze Feld wird je Saison besser (+) oder schlechter (−)", *bounds("trend_slider"), step=C.TREND_STEP, key="trend_slider", format="%.2f",
                      help="θ-Einheiten je Saison für alle; Neue kommen auf dem Niveau des Feldes. Der Anker bleibt, wo er ist.")
    st.markdown("**Elo**")
    k = st.slider("K in Elo-Punkten", *bounds("k_slider"), key="k_slider", help="Fester Verstärkungsfaktor: Wertungsänderung je Partie ist K mal (Ergebnis minus Erwartung).")
    use_k0 = st.checkbox("Anlauf-K für Neue", key="use_k0_check", help="Neue spielen anfangs mit größerem K, nach 10 Partien ist der Aufschlag halbiert.")
    if use_k0:
        seed_widget("k0_slider")
        k0 = st.slider("Anlauf-K in Punkten", *bounds("k0_slider"), key="k0_slider", help="K eines Neuen vor der ersten Partie.")
        remember("k0_slider", k0)
    use_floor = st.checkbox("Untergrenze der Wertung", key="use_floor_check", help="Keine Wertung fällt unter diese Zahl (die Differenz wird gutgeschrieben).")
    if use_floor:
        seed_widget("floor_slider")
        floor = st.slider("Untergrenze in Punkten", *bounds("floor_slider"), step=C.FLOOR_STEP, key="floor_slider")
        remember("floor_slider", floor)
    k_steps = st.checkbox("K = 10 ab 1600 Punkten", key="k_steps_check", help="Wer 1600 oder mehr hat, spielt mit kleinerem K (die übliche Stufung).")
    st.markdown("**Startwertung der Neuen**")
    start = st.selectbox("Neue starten bei", C.START, key="start_select", help="1500 oder die Schätzung aus 10 Partien gegen zufällige Mitglieder des Feldes (deren Wertungen ändern sich dabei nicht).")
    if start == C.START[1]:
        seed_widget("prior_slider")
        prior = st.slider("Prior der Platzierung (Standardabweichung)", *bounds("prior_slider"), step=C.PRIOR_STEP, key="prior_slider", format="%.1f",
                          help="Wie weit eine Schätzung vom Mittel 1500 abweichen darf (θ-Einheiten). Darüber (bei 2,0 und mehr) wächst die Drift unbegrenzt; siehe Experiment 6.")
        remember("prior_slider", prior)
    st.markdown("**Referenzspieler (Anker)**")
    use_anchor = st.checkbox("Anker mitspielen lassen", key="use_anchor_check", help="Ein Spieler mit konstanter Stärke (1500 per Konvention), der nie geht.")
    if use_anchor:
        clamp_kept("join_slider", C.JOIN_MIN, join_max(int(seasons)))
        seed_widget("join_slider")
        join = st.slider("Anker tritt bei nach Saison", C.JOIN_MIN, join_max(int(seasons)), key="join_slider", help="Anzahl der Saisons, die vor seinem Beitritt gespielt sind; er startet bei 1500.")
        remember("join_slider", join)
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

k0_v = int(current_value("k0_slider"))
floor_v = int(current_value("floor_slider"))
prior_v = round(float(current_value("prior_slider")), 1)
join_v = int(current_value("join_slider"))
sync_query_params({"n_slider": int(n), "seasons_slider": int(seasons), "games_slider": int(games), "churn_slider": round(float(churn), 2), "leave_select": leave,
                   "gap_slider": round(float(gap), 2), "trend_slider": round(float(trend), 2), "k_slider": int(k), "use_k0_check": bool(use_k0), "k0_slider": k0_v,
                   "use_floor_check": bool(use_floor), "floor_slider": floor_v, "k_steps_check": bool(k_steps), "start_select": start, "prior_slider": prior_v,
                   "use_anchor_check": bool(use_anchor), "join_slider": join_v, "seed_input": int(seed)})

settings = Settings(int(n), int(seasons), int(games), round(float(churn), 2), leave, round(float(gap), 2), round(float(trend), 2), float(k), bool(use_k0), float(k0_v),
                    bool(use_floor), float(floor_v), bool(k_steps), start, prior_v, bool(use_anchor), join_v, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Skala im gezeigten Lauf")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Drift nach {settings.seasons} Saisons", f"{ev.signed(a.drift[-1])} Punkte", delta=f"Fähigkeit {ev.signed(a.level[-1])}, Wertung {ev.signed(a.rating[-1])}", delta_color="off",
          help=f"Änderung des Wertungsmittels minus Änderung der mittleren wahren Stärke des Feldes, gegenüber dem Beginn. Ein einzelner Lauf streut um rund ±{ev.de(ev.NOISE)} Punkte.")
anchor_seen = [i for i, v in enumerate(a.anchor) if v == v]
if anchor_seen:
    last = anchor_seen[-1]
    m2.metric("Anker am Ende", ev.de(C.BASE + a.anchor[last]), delta=f"{ev.signed(a.anchor[last])} gegen seine Stärke 1500", delta_color="off",
              help=f"Ein Spieler konstanter Stärke (1500 per Konvention). Seine Wertung rauscht je Lauf um rund ±{ev.de(ev.NOISE_ANCHOR)} Punkte; die Experimente mitteln über 60 Läufe.")
else:
    m2.metric("Anker am Ende", "–", delta="kein Anker", delta_color="off")
extra = float(a.created[-1] + a.floor[-1])
m3.metric("Punkte aus K und Untergrenze", ev.signed(extra), delta=f"{ev.signed(extra / settings.n, 1)} je Spieler", delta_color="off",
          help="Punkte, die die Partien (Anlauf-K, K-Stufe) und die Untergrenze zur Summe der Wertungen hinzugefügt haben; bei festem K und ohne Untergrenze null.")
if a.entry[0] == a.entry[0]:
    m4.metric("Startfehler der Neuen", f"{ev.signed(a.entry[0])} ± {ev.de(a.entry[1])}", delta="Startwertung minus wahre Stärke, Punkte", delta_color="off",
              help="Mittel und Streuung über alle Neuen des Laufs. Bei Start 1500 ist das Mittel null, wenn Neue so stark sind wie das Feld.")
else:
    m4.metric("Startfehler der Neuen", "–", delta="kein Zugang", delta_color="off")

st.plotly_chart(build_overview(a), width="stretch", key="overview_chart")
st.caption("Wertungsmittel der Feldspieler und ihre mittlere Fähigkeit nach jeder Saison (um 1500; die Fähigkeit relativ zu den Gründern); die Lücke zwischen beiden ist die Drift. "
           "Der Anker (violett) ist ein Spieler konstanter Stärke und rauscht.")
st.plotly_chart(build_drift(a), width="stretch", key="drift_chart")
st.caption("Drift je Saison; gestrichelt die Punkte aus ungleichem K und Untergrenze (kumuliert, je Feldspieler), falls vorhanden.")
st.plotly_chart(build_balance(a), width="stretch", key="balance_chart")
st.caption("Bilanz der Wertungssumme über den ganzen Lauf: Sie ändert sich nur durch Zu- und Abgang, ungleiches K, die Untergrenze und den Anker. Die Summe am Ende ist die Summe der vier Posten.")

rows_t = ev.field_table(a)
st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
st.download_button("📄 PDF des aktuellen Laufs", data=build_pdf(a), file_name="elo-inflation.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wo steht die Skala?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wer geht, wer kommt")
st.caption("40 Spieler (feste Instanz, unabhängig vom Regler), 40 Saisons zu 20 Runden, K = 20, 15 % Abgang je Saison, Neue bei 1500; „lernende Jugend“: Neue sind 1 Einheit unter ihrer "
           "späteren Stärke und holen ein Viertel je Saison auf; Mittel über 60 feste Läufe.")
rows1 = _who()
st.plotly_chart(build_drift_lines(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_who(rows1))
st.dataframe(pd.DataFrame([{"Szenario": r["label"], "Drift nach 10 Saisons": round(r["d10"]), "nach 20": round(r["d20"]), "nach 40": round(r["d40"]),
                            "Fähigkeit (Punkte)": round(r["level"]), "Wertungsmittel (Punkte)": round(r["rating"]), "Streuung eines Laufs": round(r["sd"])} for r in rows1]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Das ganze Feld lernt oder altert")
st.caption("Dieselbe Instanz, zufälliger Abgang; alle Spieler werden je Saison um 0,02 Einheiten besser bzw. schlechter, Neue kommen auf dem Niveau des Feldes; der Anker "
           "(Experiment 4) bleibt, wo er ist.")
rows2 = _field()
st.plotly_chart(build_drift_lines(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_field(rows2))

st.subheader("🔬 Experiment 3: Anlauf-K erzeugt Punkte")
st.caption("Dieselbe Instanz mit festem K = 20 gegen Anlauf-K 120 (nach 10 Partien halbiert), je bei zufälligem Abgang und wenn die Schwächsten gehen.")
rows3 = _start_k()
st.plotly_chart(build_k_effects(rows3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_start_k(rows3))
st.dataframe(pd.DataFrame([{"Szenario": r["label"], "Drift nach 40 Saisons": round(r["d40"]), "Punkte aus ungleichem K (kumuliert)": round(r["created"]),
                            "Wertungsmittel (Punkte)": round(r["rating"])} for r in rows3]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Der Anker über die Zeit")
st.caption("Ein Spieler konstanter Stärke (1500 per Konvention) tritt nach 2, 15 oder 30 Saisons bei und spielt 4 Saisons (80 Partien); gezeigt ist seine Wertung danach, Mittel über 60 "
           "feste Läufe, Fehlerbalken: Standardfehler.")
rows4 = _anchor()
st.plotly_chart(build_anchor(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_anchor(rows4))
st.dataframe(pd.DataFrame([{"Szenario": r["label"], f"Beitritt nach {C.ANCHOR_ERAS[0]}": round(r[f"a{C.ANCHOR_ERAS[0]}"]), f"nach {C.ANCHOR_ERAS[1]}": round(r[f"a{C.ANCHOR_ERAS[1]}"]),
                            f"nach {C.ANCHOR_ERAS[2]}": round(r[f"a{C.ANCHOR_ERAS[2]}"]), "Verschiebung": round(r["shift"]), "± Standardfehler": round(r["shift_se"])}
                           for r in rows4]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 5: Untergrenze und K-Stufe")
st.caption("Dieselbe Instanz; Untergrenze 1350 (und 1450) und „K = 10 ab 1600“, je bei Abgang der Schwächsten und bei zufälligem Abgang.")
rows5 = _nothing()
st.plotly_chart(build_k_effects(rows5), width="stretch", key="exp5_chart")
verdict(*ev.verdict_nothing(rows5))
st.dataframe(pd.DataFrame([{"Szenario": r["label"], "Drift nach 40 Saisons": round(r["d40"]), "± Standardfehler": round(r["se"], 1), "Punkte aus der Untergrenze": round(r["floor"]),
                            "Punkte aus ungleichem K": round(r["created"])} for r in rows5]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 6: Startwertung aus Platzierungspartien")
st.caption("Ein Neuer startet nicht bei 1500, sondern bei der Schätzung aus 10 Partien gegen zufällige Mitglieder des Feldes (Maximum Likelihood mit Normal-Prior, Standardabweichung 1,0 "
           "Einheiten); dieselbe Instanz, 60 feste Läufe, dieselben Zufallszahlen mit und ohne Platzierung.")
rows6, prior_rows, anchor_rows6 = _placement()
st.plotly_chart(build_placement(rows6), width="stretch", key="exp6_chart")
verdict(*ev.verdict_placement(rows6, prior_rows, anchor_rows6))
st.dataframe(pd.DataFrame([{"Szenario": r["name"], "Start": "Platzierung" if r["placement"] else "1500", "Drift nach 10": round(r["d10"]), "nach 20": round(r["d20"]),
                            "nach 40": round(r["d40"]), "Startfehler Mittel": round(r["entry_mean"]), "Startfehler Streuung": round(r["entry_sd"])} for r in rows6]),
             hide_index=True, width="stretch")
st.markdown("**Empfindlich gegen den Prior:** Dieselbe Platzierung mit Prior-Standardabweichung 0,5, 1,0 und 2,0 (Drift nach 40 Saisons):")
st.plotly_chart(build_prior(prior_rows), width="stretch", key="exp6_prior_chart")
st.caption("Bei 2,0 und mehr ist das Maximum Likelihood nahezu ungebremst (10 Siege aus 10 Partien oder 0 aus 10 haben keine endliche Schätzung ohne Prior); die Wertungen der Lernenden laufen "
           "weg. Der Regler der Demo endet deshalb bei 1,5.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die absolute Fähigkeit ist definiert** | Im Modell gehen nur Differenzen in die Gewinnchance ein; der Nullpunkt (1500) ist eine Konvention. „Inflation“ ist nur gegen einen Anker oder eine externe Messlatte eine Aussage. | – |
| **Zu- und Abgang sind stilisiert** | Feste Abgangsquote, feste Verteilungen, keine Rückkehrer, keine Wertungsperioden, keine Mindestpartienzahl; reale Felder sind heterogener. | – |
| **Alle spielen gegen alle** | Zufällige Paarung im ganzen Feld; in getrennten Ligen und Regionen kann jede Gruppe für sich driften. | – |
| **Untergrenze und K-Stufe wirken** | Bei diesen Parametern bewegen sie wenig, wenn die Schwächsten gehen (Experiment 5); das ist kein Beleg, dass sie unschädlich sind. | – |
| **Der Anker ist genau** | Seine Wertung rauscht um etwa ±40 Punkte je Lauf und läuft seinem Ziel nach; Verschiebungen unter etwa 15 Punkten sind nicht zu deuten. | – |
| **Platzierung ist ein Gegenmittel** | Sie beseitigt den Eintrittsfehler, aber nicht die Drift, und hängt am Prior (Experiment 6); andere Verfahren (Unsicherheit statt Platzierung) sind hier nicht gemessen. | Glicko und TrueSkill |
| **Ein Verfahren** | Nur Elo; wie andere Verfahren über die Zeit driften, vergleicht ein eigenes Stück. | Welches Verfahren, gemessen woran? |
| **Alles ist simuliert** | Gründerstärken aus dem Generator, danach Abgang, Zugang und Lernen; die Zahlen sind Simulation, keine Aussage über Schach oder einen anderen Sport. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Skala.** Wertungen $R$ in logistischen Einheiten; $1$ Einheit $= 400/\ln 10 \approx 173{,}7$ Elo-Punkte, $0 = 1500$. Die Erwartung ist $E_{ij} = \sigma(R_i - R_j)$,
die Gewinnchance $P_{ij} = \sigma(s_i - s_j)$ mit den wahren Stärken $s$.

**Elo.** Nach jeder Partie $R_i \leftarrow R_i + k_i\,(S - E)$, $R_j \leftarrow R_j - k_j\,(S - E)$ mit $k = K/173{,}7$. Anlauf-K: $K_i = K + (K_0 - K)/(1 + g_i/10)$ nach $g_i$
Partien. Untergrenze: $R \leftarrow \max(R, F)$ nach jeder Runde.

**Bilanz.** Die Summe $\Sigma$ der Wertungen ändert sich je Partie um $(k_i - k_j)(S - E)$, bei gleichem K also gar nicht; Zu- und Abgang ändern sie um die Startwertungen der Neuen
minus die Wertungen der Gehenden, die Untergrenze um die Gutschrift. Im Gleichgewicht ($\Sigma$ stationär) gilt: Mittel der Wertungen der Gehenden = Mittel der Startwertungen der
Neuen + erzeugte Punkte je Wechsel.

**Drift.** $D_t = \bar R_t - \bar s_t$ in Punkten, mit $\bar R_t$ dem Wertungsmittel der $n$ Feldspieler nach Saison $t$ und $\bar s_t$ ihrer mittleren wahren Stärke, beide gegenüber
dem Beginn ($\bar s_0 = 0$ per Konvention).

**Anker.** Ein Spieler mit $s = 0$ und Startwertung 1500. Im Gleichgewicht gilt etwa $R_{\text{Anker}} \approx \bar R - \bar s$, also die Drift; er läuft dem Ziel mit
Geschwindigkeit $\sim k/4$ je Partie nach.

**Platzierung.** Ein Neuer mit Stärke $s$ spielt $G = 10$ Partien gegen zufällige Mitglieder $o$ des Feldes mit Wertung $R_o$, Ausgänge $w_o$. Startwertung $\hat r$ maximiert
$\sum_o \bigl[w_o \ln \sigma(r - R_o) + (1 - w_o) \ln(1 - \sigma(r - R_o))\bigr] - r^2/(2\tau^2)$ (Prior-Standardabweichung $\tau$); strikt konkav, Newton-Verfahren.

**Zufall.** Normalverteilung per Box-Muller aus der Gleichverteilten des Generators (SplitMix64); Paarung durch Sortieren zufälliger Schlüssel je Runde; alle Zufallszahlen eines
Laufs stehen in fester Reihenfolge fest, unabhängig von Platzierung und Abgangsregel.

**Literatur.**
- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present.* Arco (die Aktualisierungsformel nach der üblichen Darstellung; das Buch wurde nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- Empirische Befunde zur Inflation und Deflation von Schachwertungen wurden nicht herangezogen; die Zahlen dieser Demo sind Simulation.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)

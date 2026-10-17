# Elo-Inflation: wenn die Skala wandert – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-elo-inflation-demo.streamlit.app/)**

Zusatzstück der **Ranking-Linie** (Konzepte-Demo, Methodenreihe), Fortsetzung von `elo-demo`: Eine Elo-Wertung gilt nur **relativ zum Feld**. Gehen und kommen Spieler, kann sich
die Skala verschieben, ohne dass sich ein einzelner Spieler ändert. Gemessen wird in einem simulierten, **offenen Feld** (Abgang und Zugang, Lernen und Altern), wann das
geschieht, woran man es erkennt (ein **Anker** konstanter Stärke) und was Anlauf-K, Untergrenze, K-Stufe und **Platzierungspartien** daran ändern. Alles ist simuliert, es gibt
keinen echten Datensatz; die Gründerstärken und der Zufallsstrom kommen aus dem Generator der Linie (Version 1, wortgleich).

## Kernfrage

Wann steigen oder fallen Elo-Wertungen schneller als die Fähigkeit, und woran sieht man es? **Die Summe der Wertungen ist die Bilanz:** Eine Partie verschiebt nur Punkte, die
Summe ändert sich durch die Startwertungen der Neuen, die Wertungen der Gehenden, ungleiches K und eine Untergrenze, sonst durch nichts. Das **Wertungsmittel richtet sich
deshalb nach Zugang und Abgang**; bei unselektivem Abgang bleibt es bei der Startwertung der Neuen, unabhängig von der Fähigkeit. Gemessen (40 Spieler, 40 Saisons, 15 % Abgang, Mittel über 60 Läufe): ohne Zu- und Abgang gibt es keine Drift, bei
zufälligem Abgang und bei Abgang der Veteranen keine messbare (−4 und +1 Punkte). **Gehen die Schwächsten**, fällt die Drift auf **−25** Punkte (die Fähigkeit steigt um 184, die
Wertung nur um 160); kommen **lernende Neue** bei 1500, steigt sie auf **+50**. **Lernt das ganze Feld** (+0,02 Einheiten je Saison), sind es **−143**, **altert es**, **+135**, und das
Wertungsmittel bleibt bei null: Elo kennt nur Differenzen. Ein **Anker** (Spieler konstanter Stärke) liest das ab, mit Nachlauf und Rauschen (Mittel über 60 Läufe): −1 → −96 Punkte, wenn er nach 2 bzw. nach 30 Saisons zu einem
lernenden Feld stößt; +20 → +80 bei einem alternden. **Anlauf-K** ändert die Skala stark, wo die Startwertung falsch liegt: bei Auslese +6.596 Punkte (−25 → +61), bei zufälligem Abgang kaum; die **Untergrenze** wirkt erst, wo Schwache bleiben (+57).
**Platzierungspartien** beseitigen den Eintrittsfehler (+173 → +7 Punkte), aber nicht die Drift: Sie dreht bei der lernenden Jugend auf −150, bei Auslese ins Plus, und hängt am Prior.

## Modell

- **Feld:** n Spieler (Gründerstärken `make_strengths`, gleichabständig, Mittel 0 = 1500, Streuung 0,8 Einheiten ≈ 139 Punkte) und ein Platz für den Anker. Je Saison `games`
  Runden; in jeder Runde spielen alle aktiven Plätze in zufälliger Paarung (bei ungerader Zahl setzt einer aus), P(i schlägt j) = σ(s_i − s_j) mit den wahren Stärken s.
- **Elo:** R_i += k (S − E), E = σ(R_i − R_j), k = K/173,7 (400 Punkte = Faktor 10 der Gewinnchancen); optional **Anlauf-K** K_i = K + (K₀ − K)/(1 + g_i/10), **K-Stufe**
  (ab 1600 Punkten K = 10), **Untergrenze** (nach jeder Runde R ← max(R, F)).
- **Abgang und Zugang:** nach jeder Saison gehen `round(Abgangsquote · n)` Spieler (zufällig, die Schwächsten nach Wertung oder die Veteranen nach Saisons im Feld), ebenso viele
  Neue kommen. Ein Neuer hat die Stärke N(Niveau, 0,8) minus einen Abstand `gap` und holt je Saison ein Viertel des Abstands auf („lernende Jugend“). Mit `trend` ≠ 0 ändern sich
  alle Stärken je Saison um denselben Betrag, Neue kommen auf dem Niveau des Feldes; der Anker bleibt.
- **Startwertung:** 1500 oder die **Schätzung aus 10 Platzierungspartien** gegen zufällige Mitglieder des Feldes (deren Wertungen ändern sich dabei nicht): Maximum Likelihood mit
  Normal-Prior (Mittel 1500, Standardabweichung τ), Newton-Verfahren (strikt konkav).
- **Anker:** ein Spieler der Stärke 0 (= 1500 per Konvention), tritt nach einer wählbaren Zahl Saisons bei, startet bei 1500 (oder bei der Platzierung) und geht nie (im
  Experiment 4 nach vier Saisons).
- **Maße:** *Drift* = Änderung des Wertungsmittels der n Feldspieler minus Änderung ihrer mittleren wahren Stärke (Punkte, gegenüber dem Beginn); *Anker-Wertung* nach seinen
  Saisons; *Bilanz* der Wertungssumme (Zu- und Abgang netto, ungleiches K, Untergrenze, Anker); *Startfehler* der Neuen (Startwertung minus wahre Stärke).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 40 Spielern, 40 Saisons zu 20 Runden,
  K = 20, 15 % Abgang, unabhängig vom Regler. **Dieselben Zufallszahlen** gehören zu Läufen mit und ohne Platzierung, mit und ohne Anlauf-K usw. (der Verbrauch je Lauf hat eine
  feste Reihenfolge und Größe), die Vergleiche sind also gepaart.
- **Vektorisierung:** Die Paare einer Runde sind disjunkt, ihre Aktualisierungen also unabhängig; die Simulation rechnet alle Läufe und alle Paare einer Runde auf einmal und
  ist im Orakel an einer Partie-für-Partie-Rechnung in reinem Python auf denselben Zufallszahlen geprüft (12 Modellvarianten).
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie); Normalverteilung per Box-Muller.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 40 Spieler, 40 Saisons zu 20 Runden, K = 20, 15 % Abgang, Drift in Punkten (Mittel), wenn nicht anders genannt.

| Frage | Befund | Test |
|---|---|---|
| Geschlossenes Feld | Drift 0 in jedem Lauf (Nullsumme). | `test_a_closed_field_has_no_drift` |
| Zufälliger Abgang, Abgang der Veteranen | **−4 und +1** nach 40 Saisons, innerhalb des Fehlers des Mittels (±2). | `test_random_and_veteran_leave_move_nothing_measurable` |
| Die Schwächsten gehen | **−22 / −21 / −25** nach 10 / 20 / 40 Saisons; Fähigkeit +184, Wertungsmittel +160. | `test_the_weakest_leaving_deflates_by_about_25_points` |
| Lernende Jugend (Neue 1 Einheit unter ihrer Spitze, bei 1500) | **+56 / +53 / +50**; Fähigkeit −50, Wertungsmittel ±0, Startfehler +173 ± 139. | `test_learning_newcomers_started_at_1500_inflate_by_about_50_points` |
| Das ganze Feld lernt / altert (±0,02 je Saison) | **−32 / −71 / −143** bzw. **+38 / +69 / +135**; das Wertungsmittel bleibt bei ±0. | `test_a_learning_field_deflates_and_an_ageing_field_inflates` |
| Anlauf-K 120 | zufälliger Abgang −4 → −2; **die Schwächsten gehen −25 → +61**, 6.596 Punkte aus ungleichem K. | `test_start_k_changes_little_with_random_leave_and_much_with_selection` |
| Anker (Beitritt nach 2 / 15 / 30 Saisons, 4 Saisons = 80 Partien) | Feld lernt **−1 / −54 / −96**; altert **+20 / +44 / +80**; lernende Jugend +55 / +55 / +52; übrige Verschiebungen −14 bis +16 (Standardfehler bis ±9). | `test_the_anchor_reads_the_field_level_in_a_learning_or_ageing_field`, `test_the_anchor_shift_in_the_other_cases_is_inside_the_noise`, `test_the_anchor_in_the_youth_field_is_inflated_from_the_start` |
| Untergrenze 1350 / K-Stufe ab 1600 (die Schwächsten gehen) | −25 → −25 (die Untergrenze liefert 5 Punkte) / **−27**: kein Effekt. | `test_floor_and_k_steps_do_not_move_the_weakest_leave_scenario` |
| Untergrenze 1450, zufälliger Abgang (Schwache bleiben) | **+57**, 15.708 Punkte aus der Untergrenze; K-Stufe −5. | `test_a_floor_that_bites_creates_points_and_inflates` |
| Platzierung: Startfehler der lernenden Jugend | **+173 → +7 Punkte**, Streuung 139 → 103. | `test_placement_removes_the_entry_error_of_the_learning_youth` |
| Platzierung: Drift der lernenden Jugend | **+50 → −150** (nach 10 / 20 Saisons −39 / −94); bei zufälligem Abgang ±0. | `test_placement_flips_the_sign_of_the_youth_drift` |
| Platzierung bei Auslese | die Schwächsten gehen **−25 → +34**; Anlauf-K **+61 → +127**. | `test_placement_makes_the_selected_scenarios_worse` |
| Platzierung: Prior 0,5 / 1,0 / 2,0 (lernende Jugend, Drift nach 40 Saisons) | **−7 / −150 / −1.314**; bei 2,0 streut der Startfehler um mehr als 1.000 Punkte: die Schätzung läuft weg. | `test_placement_depends_on_the_prior_and_diverges_for_large_ones` |
| Anker unter Platzierung (lernende Jugend, nach 2 / 30 Saisons) | **+15 → −114** gegen +55 → +52 bei Start 1500. | `test_the_anchor_loses_points_under_placement_in_the_youth_field` |
| Rauschen eines Laufs | Die Drift eines einzelnen Laufs streut um ±17 Punkte, die Wertung des Ankers um ±40. | `test_a_single_run_scatters_by_about_the_stated_noise`, `test_the_anchor_is_noisy_in_a_single_run` |
| Bilanz | Bei zufälligem Abgang ist die Summe stationär (±10 Punkte je Saison); gehen die Schwächsten, steigt das Mittel mit abnehmendem Schritt, auch nach 40 Saisons noch. | `test_the_rating_sum_is_stationary_when_leavers_and_newcomers_carry_the_same`, `test_the_rating_mean_approaches_the_level_where_leavers_take_what_newcomers_bring` |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Vorab-Zahlen verschieben sich um wenige Punkte** (+46 / −30 / +58 / −147 / +131 gegen **+50 / −25 / +61 / −143 / +135**): andere Gründerstärken (gleichabständig aus dem
  Generator statt zufällig gezogen), anderer Zufallsstrom, 60 feste Läufe, Untergrenze nach jeder Runde statt nach jeder Partie. Die Aussagen sind dieselben.
- **Anlauf-K wirkt stark nur, wo die Startwertung falsch liegt.** Der Plan nannte „+58, wenn ein Anlauf-K Punkte erzeugt“ ohne Bedingung; bei zufälligem Abgang bleibt es bei ±0 (−4 gegen −2), bei Auslese sind es +6.596 Punkte. Bei lernender Jugend ohne Auslese vernichtet Anlauf-K 120 Punkte (−20.988, Drift +50 → −52; Zusatzmessung der Prüfung mit 60 Läufen, nicht im Test).
- **Die Untergrenze ist nicht nur ein negativer Befund.** Bei 1350 greift sie kaum (im Mittel 5 Punkte), weil die Schwächsten meist vorher gehen; bleiben die Schwachen und liegt sie bei 1450,
  erzeugt sie 15.708 Punkte und die Drift +57. Der Plan hatte „greift nicht“ als Befund; das gilt nur für diese Parameter.
- **Der Anker bleibt vier Saisons** (80 Partien) statt zwei (40): Mit 40 Partien läuft er seinem Ziel nur zu etwa 60 % nach (lernendes Feld nach 30 Saisons: −60 statt −96). Der Nachlauf
  steht im Verdikt; der Anker liest die Drift ab, aber verzögert und mit ±40 Punkten Rauschen je Lauf.
- **Platzierung: gemischter Befund statt Gegenmittel**, wie in der Vorab-Messung (30 Läufe); Prior-Regler bis 1,5 statt 2,0, weil die Schätzung der lernenden Jugend bei 2,0 weglief
  (Drift −1.314); schon ab 1,5 wächst sie schnell (−302 gegen −150 bei 1,0). Die Vorab-Zahl +46 war die Summe zweier Wirkungen: Eintrittsfehler (plus) und das Lernen der Etablierten (minus).
- **Neu gegenüber dem Plan:** die Bilanz der Wertungssumme als Erklärung (vier Posten, im Orakel exakt geprüft) und der Satz, dass das Wertungsmittel sich nach dem Zugang richtet.
  Die geplante Funktion `rank_with_tolerance` ist nicht umgesetzt: Die Demo vergleicht Mittel mit Standardfehlern, keine Rangfolgen.

## Abgrenzung

- **Elo** (Stück 7, `elo-demo`): ein festes Feld und Anlauf-K; dort ist der Bruch der Nullsumme der erste Befund. Dieses Stück ist die Fortsetzung mit offenem Feld, Zeit und Anker.
- **Bradley-Terry** (Stück 5): Dort ist die Skala eines Schätzers über alle Partien durch λ und den Mittelwert null festgelegt; Vergleiche über die Zeit hat es nicht.
- **Glicko und TrueSkill** (Stück 8): Die Unsicherheit statt der Platzierung als Antwort auf Neue; hier nicht gemessen.
- **Nachfolger (nur beim Namen):** Welches Verfahren, gemessen woran?

## Ehrliche Grenzen

- **Die absolute Fähigkeit ist eine Konvention:** Im Modell gehen nur Differenzen in die Gewinnchance ein. „Inflation“ ist nur gegen einen Anker oder eine externe Messlatte eine
  Aussage; die Drift gegen die mittlere Fähigkeit ist dieselbe Aussage in der Sprache der Simulation.
- **Zu- und Abgang sind stilisiert:** feste Quote, feste Verteilungen, keine Rückkehrer, keine Wertungsperioden, keine Mindestpartienzahl, zufällige Paarung im ganzen Feld.
- **Das Rauschen des Ankers ist groß** (±40 Punkte je Lauf, ±5 bis ±9 im Mittel über 60 Läufe): Verschiebungen unter etwa 15 Punkten sind nicht zu deuten.
- **Untergrenze und K-Stufe:** bei diesen Parametern wenig Wirkung, kein Beleg für Unschädlichkeit; die K-Stufe bewegt in 40 Saisons kaum jemanden über 1600.
- **Platzierung ist eine Variante:** zehn Partien gegen zufällige Mitglieder, ein Prior bei 1500; mehr Partien (20 und 40) verschlechtern die Drift in der Vorab-Messung (30 Läufe,
  nicht in der App). Andere Verfahren sind nicht gemessen.
- **Ein Feld von 10 bis 100, 60 Läufe je Zelle, ein Generator;** Unterschiede unter etwa ±5 Punkten sind Rauschen.
- **Alles simuliert:** keine Aussage über Schach oder einen anderen Sport.

## Quellen

- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present.* Arco Publishing, New York (die Aktualisierungsformel nach der üblichen Darstellung; das Buch wurde nicht
  eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- **Nicht herangezogen:** empirische Arbeiten zur Inflation und Deflation von Schachwertungen (z. B. Messungen der FIDE-Verteilungen). Die Zahlen dieser Demo sind Simulation; sie
  zeigen Mechanismen, keine Größenordnungen der Wirklichkeit.

## Tests

169 Tests, rund 140 Sekunden: Einheiten gegen Handwerte (`test_rating.py`: Elo-Schritt, Anlauf-K, K-Stufe, Untergrenze, Platzierung; `test_pool.py`: Summenerhaltung, Bilanz,
Abgangsregeln, Neue, Trend, Anker, Zufallsverbrauch, Randfälle), Generator (`test_generator.py`, wortgleich), **Orakel** (`test_oracle_elo_inflation.py`: eine unabhängige
Partie-für-Partie-Rechnung in reinem Python auf denselben Zufallszahlen gegen die vektorisierte Simulation in 12 Modellvarianten; Elo gegen die Lehrbuchformel 10^(ΔR/400);
die Platzierungsschätzung gegen den Maximierer von scipy; die Mini-Instanz des Plans von Hand; die Bilanz über 20 Läufe), Maße und Verdikte (`test_evaluation.py`), alle Zahlen
dieses README (`test_claims.py`), Presets, ausblendbare Regler (`test_hidden_widget_state.py`), PDF, Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die vier `eli_`-Kernmodule): alle **146 Mutationsstellen** einzeln gegen die Einheitentests ohne
App-Tests geprüft. Im ersten Lauf überlebten 24, davon **16 echte Lücken** (der erlaubte Rand der Abgänge und des Ankerbeitritts, die Spieler-Beschriftung und die Fähigkeit relativ
zu den Gründern in der Tabelle, der größte Standardfehler und die Prior-Zeilen im Verdikt, die Obergrenze des Beitritts) und 8 äquivalente; die Lücken sind mit Handwert-Tests
geschlossen, eine tote Hilfsfunktion entfernt. Im zweiten Lauf: **138 getötet, 8 überlebt, alle äquivalent**: die Schutzgrenze `cnt − 1` der Gegnerwahl (die Gleichverteilte ist
kleiner als 1), Vergleiche bei stetig verteilten Zufallszahlen (`u < p` gegen `u <= p`), die Gleichheit der K-Stufe bei genau 1600 Punkten (bei Gleitkommazahlen nicht erreichbar) und
die Vorzeichen-Schwelle im Verdikt, an der der Betrag der Drift (mindestens das Doppelte der Streuung) nie null ist.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie; liefert die Gründerstärken und den Zufallsstrom) |
| `eli_rating.py` | Elo-Schritt, Anlauf-K, K-Stufe, Untergrenze, Platzierungsschätzung (vektorisiert) |
| `eli_pool.py` | offenes Feld: Saisons, Abgang und Zugang, Lernen und Altern, Anker; `Model`, `Pool` |
| `eli_scenario.py` | Reglerwerte (`Settings`) und das Modell |
| `eli_evaluation.py` | Maße, Bilanz, Analyse, sechs Experimente über die festen Läufe, Verdikte |
| `eli_presets.py`, `eli_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `eli_visualization.py`, `eli_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Wertungsperioden und Mindestpartienzahlen** (wie reale Verbände sie führen).
- **Rückkehrer** und **getrennte Ligen**, in denen jede Gruppe für sich driften kann.
- **Eine Korrektur der Drift** (Neuberechnung der Skala gegen den Anker): würde die Konvention als Entscheidung des Betreibers zeigen, ist eine eigene Frage.
- **Glicko-artige Unsicherheit als Antwort auf Neue:** Stück „Glicko und TrueSkill“.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).

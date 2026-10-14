"""PDF-Export der aktuellen Konfiguration und der Bilanz (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import eli_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("φ", "phi"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Elo-Inflation: wenn die Skala wandert"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(f"Spieler: {s.n}, Saisons: {s.seasons} zu je {s.games} Runden, Abgang {ev.pct(s.churn)} % je Saison ({s.leave}), Seed: {s.seed}."),
                   new_x="LMARGIN", new_y="NEXT")
    extras = [f"K = {s.k}"]
    if s.use_k0:
        extras.append(f"Anlauf-K {s.k0:.0f}")
    if s.use_floor:
        extras.append(f"Untergrenze {s.floor:.0f}")
    if s.k_steps:
        extras.append("K-Stufe ab 1600")
    extras.append(f"Start: {s.start}" + (f" (Prior {ev.de(s.prior, 1)})" if s.placement else ""))
    if s.gap:
        extras.append(f"Neue {ev.de(s.gap, 2)} unter ihrer Spitze")
    if s.trend:
        extras.append(f"Feld {ev.signed(s.trend, 2)} je Saison")
    pdf.multi_cell(0, 5.5, _latin("; ".join(extras) + "."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    b = ev.balance(a.pool)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _latin("Bilanz der Wertungssumme (Punkte über 1500 je Spieler, über alle Spieler)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for label, key in (("Zu- und Abgang netto", "entries"), ("ungleiches K", "k"), ("Untergrenze", "floor"), ("Anker", "anchor"), ("Summe am Ende", "end")):
        pdf.cell(80, 6, _latin(label), border=1)
        pdf.cell(40, 6, _latin(ev.signed(b[key])), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Spieler", "Saisons", "Wertung", "wahre Stärke", "Abweichung")
    widths = (30, 30, 30, 40, 35)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for row in ev.field_table(a):
        for value, w in zip(row.values(), widths):
            pdf.cell(w, 6, _latin(str(value)), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Gründerstärken aus dem Generator der Ranking-Linie, danach Abgang, Zugang und Lernen); die wahre Stärke ist nur in der Simulation bekannt, "
        "ihr Nullpunkt (1500) ist eine Konvention. Elo-Punkte: 400 Punkte entsprechen dem Faktor 10 der Gewinnchancen. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."),
        new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import eli_constants as C
import eli_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_overview(a: ev.Analysis):
    """Wertungsmittel des Feldes und mittlere Fähigkeit (beide gegenüber dem Start, in Punkten um 1500) je Saison; der Anker als eigene Linie. Die Lücke ist die Drift."""
    x = np.arange(1, len(a.rating) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=C.BASE + a.rating, mode="lines", name="Wertungsmittel des Feldes", line=dict(color=C.COLOR_RATING, width=3),
                             hovertemplate="Saison %{x}: Wertungsmittel %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=C.BASE + a.level, mode="lines", name="mittlere Fähigkeit des Feldes (nur in der Simulation bekannt)",
                             line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), hovertemplate="Saison %{x}: Fähigkeit %{y:.0f}<extra></extra>"))
    if np.isfinite(a.anchor).any():
        fig.add_trace(go.Scatter(x=x, y=C.BASE + a.anchor, mode="lines", name="Anker (konstante Stärke 1500)", line=dict(color=C.COLOR_ANCHOR, width=1.8), connectgaps=False,
                                 hovertemplate="Saison %{x}: Anker %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title_text="Saison")
    fig.update_yaxes(title_text="Punkte (Elo-Skala)")
    return _base(fig, height=400)


def build_drift(a: ev.Analysis):
    """Drift je Saison (Wertungsmittel minus Fähigkeit) und kumulierte Punkte aus ungleichem K und Untergrenze, je Feldspieler."""
    x = np.arange(1, len(a.drift) + 1)
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    fig.add_trace(go.Scatter(x=x, y=a.drift, mode="lines", name="Drift (Wertung − Fähigkeit)", line=dict(color=C.COLOR_RATING, width=3),
                             hovertemplate="Saison %{x}: Drift %{y:.0f} Punkte<extra></extra>"))
    n = a.pool.model.n
    extra = (a.created + a.floor) / n
    if np.any(np.abs(extra) > 1e-9):
        fig.add_trace(go.Scatter(x=x, y=extra, mode="lines", name="Punkte aus ungleichem K und Untergrenze (kumuliert, je Spieler)", line=dict(color=C.COLOR_CREATED, width=2, dash="dash"),
                                 hovertemplate="Saison %{x}: %{y:.0f} Punkte je Spieler<extra></extra>"))
    fig.update_xaxes(title_text="Saison")
    fig.update_yaxes(title_text="Punkte")
    return _base(fig, height=300)


def build_balance(a: ev.Analysis):
    """Bilanz der Wertungssumme (Punkte über 1500 je Spieler, über alle Spieler): was sie von 0 auf den Endwert gebracht hat."""
    b = ev.balance(a.pool)
    names = ["Zu- und Abgang (Startwertungen − Wertungen der Gehenden)", "ungleiches K", "Untergrenze", "Anker"]
    values = [b["entries"], b["k"], b["floor"], b["anchor"]]
    fig = go.Figure(go.Waterfall(x=names + ["Summe am Ende"], y=values + [0], measure=["relative"] * len(values) + ["total"], text=[ev.signed(v) for v in values] + [ev.signed(b["end"])],
                                 textposition="outside", increasing=dict(marker=dict(color=C.COLOR_CREATED)), decreasing=dict(marker=dict(color=C.COLOR_RATING)),
                                 totals=dict(marker=dict(color=C.COLOR_ANCHOR)), hovertemplate="%{x}: %{y:.0f} Punkte<extra></extra>"))
    fig.update_yaxes(title_text="Summe der Wertungen über 1500 · Spielerzahl (Punkte)")
    return _base(fig, height=320)


def build_drift_lines(rows, keys=None):
    """Mittlere Drift (60 Läufe) je Saison für mehrere Szenarien."""
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    sel = [r for r in rows if keys is None or r["key"] in keys]
    for k, r in enumerate(sel):
        x = np.arange(1, len(r["series"]) + 1)
        fig.add_trace(go.Scatter(x=x, y=r["series"], mode="lines", name=r["label"], line=dict(color=C.PALETTE[k % len(C.PALETTE)], width=2.5),
                                 hovertemplate=r["label"] + ", Saison %{x}: %{y:.0f} Punkte<extra></extra>"))
    fig.update_xaxes(title_text="Saison")
    fig.update_yaxes(title_text="Drift in Punkten (Mittel über 60 Läufe)")
    return _base(fig, height=360)


def build_anchor(rows):
    """Anker-Wertung (Punkte um 1500) nach 4 Saisons, je Beitrittszeit und Szenario; Fehlerbalken: Standardfehler des Mittels über 60 Läufe."""
    shades = ("#9cc5bf", "#5f9d95", "#2F6B65")
    fig = go.Figure()
    for k, era in enumerate(C.ANCHOR_ERAS):
        fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[r[f"a{era}"] for r in rows], error_y=dict(type="data", array=[r[f"se{era}"] for r in rows]), name=f"Beitritt nach {era} Saisons",
                             marker_color=shades[k], hovertemplate="%{x}, Beitritt nach " + str(era) + ": %{y:.0f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Wertung des Ankers minus 1500 (Punkte)")
    return _base(fig)


def build_k_effects(rows):
    """Drift nach 40 Saisons je Szenario (Balken) mit Standardfehler."""
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=[r["d40"] for r in rows], error_y=dict(type="data", array=[r["se"] for r in rows]),
                           marker_color=[C.PALETTE[k % len(C.PALETTE)] for k in range(len(rows))], text=[ev.signed(r["d40"]) for r in rows], textposition="outside",
                           hovertemplate="%{x}: %{y:.0f} Punkte<extra></extra>"))
    fig.update_yaxes(title_text="Drift nach 40 Saisons (Punkte)")
    return _base(fig)


def build_placement(rows):
    """Drift nach 40 Saisons je Szenario, Start bei 1500 gegen Platzierungspartien."""
    names = []
    for r in rows:
        if r["name"] not in names:
            names.append(r["name"])
    fig = go.Figure()
    for placement, label, color in ((False, "Start bei 1500", "#9cc5bf"), (True, "Platzierungspartien", "#2F6B65")):
        sel = [next(r for r in rows if r["name"] == n and r["placement"] == placement) for n in names]
        fig.add_trace(go.Bar(x=names, y=[r["d40"] for r in sel], error_y=dict(type="data", array=[r["se"] for r in sel]), name=label, marker_color=color,
                             text=[ev.signed(r["d40"]) for r in sel], textposition="outside", hovertemplate=label + ", %{x}: %{y:.0f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Drift nach 40 Saisons (Punkte)")
    return _base(fig)


def build_prior(rows):
    """Drift nach 40 Saisons in Abhängigkeit vom Prior der Platzierung."""
    fig = go.Figure()
    for k, name in enumerate(dict.fromkeys(r["name"] for r in rows)):
        sel = [r for r in rows if r["name"] == name]
        fig.add_trace(go.Bar(x=[ev.de(r["prior"], 1) for r in sel], y=[r["d40"] for r in sel], name=name, marker_color=C.PALETTE[k % len(C.PALETTE)],
                             text=[ev.signed(r["d40"]) for r in sel], textposition="outside", hovertemplate=name + ", Prior %{x}: %{y:.0f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Prior-Standardabweichung der Platzierung (θ-Einheiten)", type="category")
    fig.update_yaxes(title_text="Drift nach 40 Saisons (Punkte)")
    return _base(fig)

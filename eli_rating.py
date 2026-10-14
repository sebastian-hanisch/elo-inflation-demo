"""Elo-Schritt, Anlauf-K, K-Stufe, Untergrenze und Platzierungsschätzung. Alles vektorisiert über Läufe (erste Achse) und Spielerplätze (zweite Achse), in θ-Einheiten
(1 Einheit = 173,7 Elo-Punkte, 0 = 1500). Die Paare einer Runde sind disjunkt, ihre Aktualisierungen also unabhängig voneinander und gleich der Partie-für-Partie-Rechnung."""

import numpy as np

import eli_constants as C


def sig(x):
    """Gewinnerwartung E = σ(a − b), überlaufsicher."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def k_theta(R, played, k: float, k0, step: bool):
    """K in θ-Einheiten je Spieler. Fest k Punkte; mit Anlauf-K `k0` ist K = k + (k0 − k) / (1 + Partien / G0); mit K-Stufe gilt ab 1600 Punkten K = 10."""
    kk = np.full(np.shape(R), float(k))
    if k0 is not None:
        kk = k + (k0 - k) / (1.0 + np.asarray(played, dtype=float) / C.G0)
    if step:
        kk = np.where(C.BASE + C.SCALE * np.asarray(R, dtype=float) >= C.STEP_FROM, C.STEP_K, kk)
    return kk / C.SCALE


def play_round(R, true, played, a, b, u, k: float, k0, step: bool):
    """Eine Runde: Spieler a[:, m] gegen b[:, m]; a gewinnt, wenn u < σ(wahr_a − wahr_b). R und played werden verändert.

    Rückgabe: je Lauf die Summe der Punkte, die diese Runde erzeugt (Summe der Änderungen beider Spieler = (k_a − k_b)(Ergebnis − Erwartung)), in θ-Einheiten."""
    rows = np.arange(R.shape[0])[:, None]
    ra, rb = R[rows, a], R[rows, b]
    d = (u < sig(true[rows, a] - true[rows, b])).astype(float) - sig(ra - rb)
    ka = k_theta(ra, played[rows, a], k, k0, step)
    kb = k_theta(rb, played[rows, b], k, k0, step)
    R[rows, a] = ra + ka * d
    R[rows, b] = rb - kb * d
    played[rows, a] += 1
    played[rows, b] += 1
    return ((ka - kb) * d).sum(axis=1)


def apply_floor(R, active, floor: float):
    """Untergrenze (Punkte) für die aktiven Plätze. R wird verändert; Rückgabe je Lauf: die Punkte, die dadurch hinzukommen (θ-Einheiten)."""
    lo = (float(floor) - C.BASE) / C.SCALE
    new = np.where(active[None, :], np.maximum(R, lo), R)
    lift = (new - R).sum(axis=1)
    R[:] = new
    return lift


def placement_estimates(true_self, R, true, stay, u_opp, u_win, prior_sd: float, iters: int = C.PLACE_ITERS):
    """Schätzung der Wertung neuer Spieler aus Platzierungspartien gegen zufällige Mitglieder des Feldes (deren Wertungen sich dabei nicht ändern).

    true_self (B, E): wahre Stärke der E Neuen; R, true (B, P): Wertung und Stärke aller Plätze; stay (B, P): wer als Gegner zur Verfügung steht;
    u_opp, u_win (B, E, G): Gleichverteilte für Gegnerwahl und Ausgang. Maximum Likelihood mit Normal-Prior (Mittel 0 = 1500, Standardabweichung prior_sd):
    r maximiert Σ [w log σ(r − R_o) + (1 − w) log(1 − σ(r − R_o))] − r²/(2 prior_sd²); Newton-Verfahren (strikt konkav)."""
    B, E, G = u_opp.shape
    order = np.argsort(~stay, axis=1, kind="stable")                    # die Plätze der Bleibenden zuerst, in Platzreihenfolge
    cnt = np.maximum(stay.sum(axis=1), 1)[:, None, None]
    pick = np.minimum((u_opp * cnt).astype(int), cnt - 1)
    opp = np.take_along_axis(order, pick.reshape(B, E * G), axis=1)
    r_opp = np.take_along_axis(R, opp, axis=1).reshape(B, E, G)
    t_opp = np.take_along_axis(true, opp, axis=1).reshape(B, E, G)
    wins = (u_win < sig(true_self[:, :, None] - t_opp)).astype(float)
    r = np.zeros((B, E))
    prec = 1.0 / prior_sd ** 2
    for _ in range(iters):
        p = sig(r[:, :, None] - r_opp)
        r = r + ((wins - p).sum(axis=2) - r * prec) / ((p * (1.0 - p)).sum(axis=2) + prec)
    return r

"""PDF-Export: erzeugt für jede Konfiguration gültige PDF-Bytes; der Latin-1-Filter ersetzt alle Zeichen außerhalb der Kernschrift."""

import eli_constants as C
import eli_evaluation as ev
import eli_pdf_export as X
from eli_scenario import Settings


def test_pdf_is_built_for_every_mechanism():
    for s in (Settings(n=10, seasons=6, games=4), Settings(n=10, seasons=6, games=4, use_k0=True, use_floor=True, k_steps=True),
              Settings(n=10, seasons=6, games=4, start=C.START[1], gap=1.0, trend=0.02, use_anchor=False), Settings(n=10, seasons=6, games=4, churn=0.0)):
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("τ ≈ 0,5; σ → φ µ θ − **fett** €")
    assert out == "Tau ca. 0,5; sigma -> phi mu theta - fett EUR"
    assert out.encode("latin-1")

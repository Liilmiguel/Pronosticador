"""Pipeline completo por línea de comandos:
  1. scraping (opcional, --scrape)
  2. variables + Gradient Boosting
  3. predicciones fuera de muestra (walk-forward) desde --desde → stacker y backtest

Uso:  .venv/bin/python entrenar.py [--scrape] [--desde 2019]
"""
import argparse
import time

from predictor.config import CURRENT_SEASON
from predictor.models import build_oos, load_or_build, load_or_build_national
from predictor.scraper import load_table, scrape_all


def bar(p, msg):
    print(f"\r[{p:5.1%}] {msg:<70}", end="", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", action="store_true", help="descargar/actualizar datos primero")
    ap.add_argument("--desde", type=int, default=2019, help="primera temporada del backtest walk-forward")
    ap.add_argument("--api-football", action="store_true", help="avanzar la descarga de stats de selecciones")
    a = ap.parse_args()
    t0 = time.time()
    if a.scrape:
        print(scrape_all(progress=bar))
    if a.api_football:
        from predictor import apifootball
        from predictor.national import load as load_intl
        intl = load_intl()
        print(apifootball.sync(set(intl["home"]) | set(intl["away"]), progress=bar))
    fb, F, gbm, stats = load_or_build(load_table("matches"), progress=bar)
    load_or_build_national(progress=bar)
    print(f"\nModelo listo ({time.time() - t0:.0f}s). Generando predicciones fuera de muestra…")
    oos = build_oos(F, fb.feature_cols, list(range(a.desde, CURRENT_SEASON + 1)), progress=bar)
    print(f"\nListo: {len(oos):,} partidos fuera de muestra ({time.time() - t0:.0f}s)")

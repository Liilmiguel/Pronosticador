"""Descarga diaria de estadísticas de selecciones (API-Football) y reentrenamiento del modelo de selecciones.

La ejecuta launchd una vez al día (ver ~/Library/LaunchAgents/com.predictor-apuestas.selecciones.plist).
No toca la base de clubes, así que no obliga a reentrenar los modelos de ligas.
Registro en data/logs/selecciones.log.
"""
import sys
import traceback
from datetime import datetime

from predictor import apifootball
from predictor.config import DATA_DIR
from predictor.models import load_or_build_national
from predictor.national import load

LOG = DATA_DIR / "logs" / "selecciones.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


if __name__ == "__main__":
    try:
        if not apifootball.get_key():
            log("sin clave de API-Football; nada que hacer")
            sys.exit(0)
        intl = load()
        res = apifootball.sync(set(intl["home"]) | set(intl["away"]))
        log(f"sync: {res}")
        if res.get("partidos_nuevos"):
            load_or_build_national()  # la caché cambia sola al cambiar el CSV de estadísticas
            log(f"modelo de selecciones reentrenado ({len(apifootball.load()) // 2} partidos API-Football)")
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        sys.exit(1)

"""Conector API-Football (api-sports.io): estadísticas de eliminatorias, amistosos, Nations League y Mundial.

API oficial con plan gratuito (100 solicitudes/día, 10 por minuto). Requiere una clave propia:
https://dashboard.api-football.com/register → copiar la "API Key" en la app (barra lateral) o en
data/.api_football_key. La descarga es incremental y respeta el presupuesto diario: cada ejecución
avanza un poco y retoma donde quedó.
"""
from __future__ import annotations

import difflib
import json
import os
import time

import pandas as pd
import requests

from .config import DATA_DIR, RAW_DIR

BASE = "https://v3.football.api-sports.io"
KEY_FILE = DATA_DIR / ".api_football_key"
STATS_CSV = RAW_DIR / "intl" / "apifootball_stats.csv"
STATE_FILE = RAW_DIR / "intl" / "apifootball_state.json"
FIRST_SEASON = 2018
# competiciones de selecciones absolutas masculinas que interesan
WANTED = ("World Cup", "Friendlies", "UEFA Nations League", "CONCACAF Nations League", "Euro Championship",
          "Africa Cup of Nations", "Asian Cup", "Copa America", "CONCACAF Gold Cup")
EXCLUDE = ("Women", "U17", "U19", "U20", "U21", "U23", "Olympic", "Clubs", "Club", "Beach", "Futsal")
NAME_MAP = {"USA": "United States", "Korea Republic": "South Korea", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
            "Congo DR": "DR Congo", "Cape Verde Islands": "Cape Verde", "China": "China PR", "Chinese Taipei": "Taiwan",
            "Curacao": "Curaçao", "Türkiye": "Turkey", "Ireland": "Republic of Ireland", "FYR Macedonia": "North Macedonia",
            "Macedonia": "North Macedonia", "Czechia": "Czech Republic", "Swaziland": "Eswatini",
            "Sao Tome and Principe": "São Tomé and Príncipe", "Timor-Leste": "Timor-Leste", "Macao": "Macau",
            "Rep. Of Ireland": "Republic of Ireland", "Rep. of Ireland": "Republic of Ireland"}
COUNT_STATS = ["corners", "shots", "sot", "fouls"]
FINISHED = {"FT", "AET", "PEN"}


class ApiFootballError(RuntimeError):
    pass


def get_key() -> str | None:
    if os.environ.get("API_FOOTBALL_KEY"):
        return os.environ["API_FOOTBALL_KEY"].strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip() or None
    return None


def save_key(key: str) -> None:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key.strip())
    KEY_FILE.chmod(0o600)


class Client:
    def __init__(self, key: str, budget: int = 95, min_interval: float = 6.5):
        self.s = requests.Session()
        self.s.headers["x-apisports-key"] = key
        self.budget, self.min_interval, self.calls, self._last = budget, min_interval, 0, 0.0

    def get(self, path: str, **params):
        if self.calls >= self.budget:
            raise ApiFootballError("Presupuesto de solicitudes de hoy agotado")
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.s.get(f"{BASE}/{path}", params=params, timeout=30)
        self._last, self.calls = time.time(), self.calls + 1
        if r.status_code != 200:
            raise ApiFootballError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        errs = data.get("errors")
        if errs:
            raise ApiFootballError(str(errs))
        remaining = r.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None and int(remaining) <= 1:
            self.budget = self.calls  # no gastar más hoy
        return data.get("response", [])


def _state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"leagues": None, "fixtures": {},
                                                                          "no_stats": [], "denied": []}


def _save_state(st: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st))


def _map_name(name: str, known: set) -> str:
    name = NAME_MAP.get(name, name)
    if name in known or not known:
        return name
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.88)
    return close[0] if close else name


def parse_statistics(resp: list, fx: dict, known: set) -> list[dict]:
    """Convierte la respuesta de /fixtures/statistics en dos filas (una por selección)."""
    if len(resp) != 2:
        return []
    vals = {}
    for side in resp:
        st = {s["type"]: s["value"] for s in side.get("statistics", [])}
        if all(v is None for v in st.values()):
            return []
        num = lambda k: float(st[k]) if st.get(k) is not None else float("nan")
        vals[side["team"]["id"]] = {
            "corners": num("Corner Kicks"),
            "cards": (0 if st.get("Yellow Cards") is None else num("Yellow Cards"))
                     + 2 * (0 if st.get("Red Cards") is None else num("Red Cards")),
            "shots": num("Total Shots"), "sot": num("Shots on Goal"), "fouls": num("Fouls"),
            "xg": float(st["expected_goals"]) if st.get("expected_goals") not in (None, "") else float("nan")}
    home, away = fx["teams"]["home"], fx["teams"]["away"]
    if home["id"] not in vals or away["id"] not in vals:
        return []
    comp = f"{fx['league']['name']} {fx['league']['season']}"
    rows = []
    for t, o, is_home in ((home, away, 1), (away, home, 0)):
        rows.append({"match_id": f"apif_{fx['fixture']['id']}", "date": fx["fixture"]["date"][:10],
                     "competition": comp, "stage": fx["league"].get("round"),
                     "referee": fx["fixture"].get("referee"), "team": _map_name(t["name"], known),
                     "opp": _map_name(o["name"], known), "is_home": is_home,
                     "gf": fx["goals"]["home" if is_home else "away"], "ga": fx["goals"]["away" if is_home else "home"],
                     **vals[t["id"]], **{f"{k}_opp": v for k, v in vals[o["id"]].items()}, "source": "API-Football"})
    return rows


def sync(known_names: set, budget: int = 95, progress=None) -> dict:
    """Avanza la descarga incremental. Devuelve un resumen (solicitudes usadas, partidos nuevos…)."""
    key = get_key()
    if not key:
        raise ApiFootballError("Falta la clave de API-Football")
    cli, st = Client(key, budget), _state()
    existing = pd.read_csv(STATS_CSV) if STATS_CSV.exists() else pd.DataFrame(columns=["match_id"])
    have = set(existing["match_id"].astype(str))
    new_rows, msg = [], "ok"
    try:
        if st["leagues"] is None:
            leagues = []
            for item in cli.get("leagues", country="World"):
                name = item["league"]["name"]
                if item["country"]["name"] != "World" or not any(w in name for w in WANTED) \
                        or any(x in name for x in EXCLUDE):
                    continue
                seasons = [s["year"] for s in item["seasons"] if s["year"] >= FIRST_SEASON
                           and s.get("coverage", {}).get("fixtures", {}).get("statistics_fixtures")]
                if seasons:
                    leagues.append({"id": item["league"]["id"], "name": name, "seasons": seasons})
            st["leagues"] = leagues
            _save_state(st)
        # 1) listas de partidos por competición y temporada (la temporada en curso se refresca siempre)
        this_year = pd.Timestamp.today().year
        for lg in st["leagues"]:
            for season in sorted(lg["seasons"], reverse=True):
                k = f"{lg['id']}_{season}"
                if k in st["denied"] or (k in st["fixtures"] and season < this_year - 1):
                    continue
                try:
                    fxs = cli.get("fixtures", league=lg["id"], season=season)
                except ApiFootballError as e:
                    if "plan" in str(e).lower() or "season" in str(e).lower():
                        st["denied"].append(k)  # el plan gratuito no cubre esa temporada
                        continue
                    raise
                st["fixtures"][k] = [f for f in fxs if f["fixture"]["status"]["short"] in FINISHED]
                _save_state(st)
        # 2) estadísticas de cada partido terminado (más recientes primero)
        todo = [f for fl in st["fixtures"].values() for f in fl
                if f"apif_{f['fixture']['id']}" not in have and f["fixture"]["id"] not in st["no_stats"]]
        todo.sort(key=lambda f: f["fixture"]["date"], reverse=True)
        for i, fx in enumerate(todo):
            rows = parse_statistics(cli.get("fixtures/statistics", fixture=fx["fixture"]["id"]), fx, known_names)
            if rows:
                new_rows += rows
            else:
                st["no_stats"].append(fx["fixture"]["id"])
            if progress:
                progress(min(1.0, cli.calls / max(cli.budget, 1)), f"API-Football: {len(new_rows) // 2} partidos nuevos")
    except ApiFootballError as e:
        msg = str(e)
    finally:
        _save_state(st)
        if new_rows:
            out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(STATS_CSV, index=False)
    pending = sum(1 for fl in st["fixtures"].values() for f in fl
                  if f"apif_{f['fixture']['id']}" not in have | {r["match_id"] for r in new_rows}
                  and f["fixture"]["id"] not in st["no_stats"])
    return {"solicitudes": cli.calls, "partidos_nuevos": len(new_rows) // 2, "pendientes": pending,
            "competiciones": len(st["leagues"] or []), "mensaje": msg}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Algunas competiciones (p. ej. eliminatorias africanas) solo traen tarjetas: si tiros y faltas valen 0
    para ambos equipos, córners/tiros/faltas no fueron registrados (no son ceros reales)."""
    df = df.copy()
    df["team"] = df["team"].replace(NAME_MAP)
    df["opp"] = df["opp"].replace(NAME_MAP)
    empty = (df["shots"].fillna(0) == 0) & (df["shots_opp"].fillna(0) == 0) & \
            (df["fouls"].fillna(0) == 0) & (df["fouls_opp"].fillna(0) == 0)
    for c in COUNT_STATS:
        df.loc[empty, [c, f"{c}_opp"]] = float("nan")
    return df


def load() -> pd.DataFrame:
    if not STATS_CSV.exists():
        return pd.DataFrame()
    df = clean(pd.read_csv(STATS_CSV))
    df["date"] = pd.to_datetime(df["date"])
    return df

"""Córners, tarjetas, remates y faltas para selecciones nacionales.

Fuentes:
- StatsBomb Open Data (github.com/statsbomb/open-data, uso libre con atribución): Mundiales 2018 y 2022,
  Euro 2020 y 2024, Copa América 2024 y Copa Africana 2023 (~314 partidos, solo tiempo reglamentario).
- API-Football (opcional, con clave propia): eliminatorias, amistosos, Nations League, Mundial 2026 y lo más
  reciente; ver apifootball.py.

Modelo: con pocos partidos por selección se usa una regresión Poisson con contracción (shrinkage):
log E[stat] = f(diferencia de Elo, cancha neutral, promedio propio a favor y del rival en contra, ambos
contraídos hacia la media global). Dispersión con binomial negativa. Validación dejando fuera un torneo.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import PoissonRegressor

from .config import RAW_DIR
from .stats_markets import total_dist

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITIONS = [(43, 3, "Mundial 2018"), (43, 106, "Mundial 2022"), (55, 43, "Euro 2020"), (55, 282, "Euro 2024"),
                (223, 282, "Copa América 2024"), (1267, 107, "Copa Africana 2023")]
STATS = {"corners": "Córners", "cards": "Tarjetas", "shots": "Remates", "sot": "Remates al arco", "fouls": "Faltas"}
CACHE = RAW_DIR / "intl" / "statsbomb_stats.csv"
SHRINK_K = 4.0  # partidos "virtuales" de media global al contraer el promedio de cada selección

# StatsBomb → nombres del dataset de resultados internacionales
NAME_MAP = {"Korea Republic": "South Korea", "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast",
            "Türkiye": "Turkey", "Czechia": "Czech Republic", "Cape Verde Islands": "Cape Verde",
            "Congo DR": "DR Congo", "Republic of Ireland": "Republic of Ireland", "North Macedonia": "North Macedonia",
            "United States": "United States", "Equatorial Guinea": "Equatorial Guinea", "Gambia": "Gambia"}
ON_TARGET = {"Goal", "Saved", "Saved To Post"}


# --------------------------------------------------------------------------- descarga
def _get(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _aggregate(match: dict, comp_label: str) -> list[dict]:
    events = _get(f"{BASE}/events/{match['match_id']}.json")
    home, away = match["home_team"]["home_team_name"], match["away_team"]["away_team_name"]
    agg = {t: dict(corners=0, cards=0, shots=0, sot=0, fouls=0, xg=0.0) for t in (home, away)}
    for e in events:
        if e.get("period", 1) > 2:  # solo tiempo reglamentario
            continue
        t, typ = e["team"]["name"], e["type"]["name"]
        if t not in agg:
            continue
        a = agg[t]
        if typ == "Pass" and e.get("pass", {}).get("type", {}).get("name") == "Corner":
            a["corners"] += 1
        elif typ == "Shot":
            a["shots"] += 1
            a["xg"] += e["shot"].get("statsbomb_xg", 0.0)
            a["sot"] += e["shot"]["outcome"]["name"] in ON_TARGET
        elif typ in ("Foul Committed", "Bad Behaviour"):
            if typ == "Foul Committed":
                a["fouls"] += 1
            card = e.get("foul_committed", e.get("bad_behaviour", {})).get("card", {}).get("name")
            a["cards"] += {"Yellow Card": 1, "Second Yellow": 3, "Red Card": 2}.get(card, 0)
    rows = []
    for team, opp, is_home in ((home, away, 1), (away, home, 0)):
        rows.append({"match_id": match["match_id"], "date": match["match_date"], "competition": comp_label,
                     "stage": match.get("competition_stage", {}).get("name"),
                     "referee": (match.get("referee") or {}).get("name"),
                     "team": NAME_MAP.get(team, team), "opp": NAME_MAP.get(opp, opp), "is_home": is_home,
                     "gf": match["home_score"] if is_home else match["away_score"],
                     "ga": match["away_score"] if is_home else match["home_score"],
                     **agg[team], **{f"{k}_opp": v for k, v in agg[opp].items()}})
    return rows


def download(progress=None, workers: int = 8) -> pd.DataFrame:
    """Descarga incremental: solo procesa partidos que no estén ya en la caché local."""
    done = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["match_id"])
    seen = set(done["match_id"])
    todo = []
    for cid, sid, label in COMPETITIONS:
        for m in _get(f"{BASE}/matches/{cid}/{sid}.json"):
            if m["match_id"] not in seen:
                todo.append((m, label))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(lambda x: _aggregate(*x), todo)):
            rows += r
            if progress:
                progress((i + 1) / max(len(todo), 1), f"StatsBomb: {i + 1}/{len(todo)} partidos")
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if rows else done
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


def load() -> pd.DataFrame:
    """StatsBomb + API-Football (si hay). Si un partido está en ambas, se queda el de StatsBomb."""
    from . import apifootball
    frames = []
    if CACHE.exists():
        frames.append(pd.read_csv(CACHE).assign(source="StatsBomb"))
    api = apifootball.load()
    if not api.empty:
        frames.append(api)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(["date", "team", "opp"], keep="first").reset_index(drop=True)


def data_version() -> str:
    """Huella de los archivos de estadísticas (para invalidar la caché del modelo)."""
    from . import apifootball
    return "-".join(f"{p.stat().st_mtime:.0f}" for p in (CACHE, apifootball.STATS_CSV) if p.exists())


# --------------------------------------------------------------------------- modelo
@dataclass
class NationalStatsModel:
    glob: dict = field(default_factory=dict)       # media global por estadística
    team_for: dict = field(default_factory=dict)   # (stat, team) -> (suma, n)
    team_against: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    alpha: dict = field(default_factory=dict)
    validation: pd.DataFrame | None = None
    data: pd.DataFrame | None = None

    @staticmethod
    def _rates(d: pd.DataFrame, stat: str):
        f = d.groupby("team")[stat].agg(["sum", "count"])
        a = d.groupby("opp")[stat].agg(["sum", "count"])  # lo que concede el rival = lo que hace el equipo vs él
        return ({t: (r["sum"], r["count"]) for t, r in f.iterrows()},
                {t: (r["sum"], r["count"]) for t, r in a.iterrows()})

    @staticmethod
    def _shrunk(tbl, team, g, exclude=0.0, exclude_n=0):
        s, n = tbl.get(team, (0.0, 0))
        s, n = s - exclude, n - exclude_n
        return (s + SHRINK_K * g) / (n + SHRINK_K)

    def _X(self, d, stat, g, f_tbl, a_tbl, loo: bool) -> np.ndarray:
        """Variables: diferencia de Elo propia, neutral, local, log(ataque propio), log(concesión del rival)."""
        tf = np.array([self._shrunk(f_tbl, t, g, v if loo else 0, 1 if loo else 0)
                       for t, v in zip(d["team"], d[stat])])
        oa = np.array([self._shrunk(a_tbl, o, g, v if loo else 0, 1 if loo else 0)
                       for o, v in zip(d["opp"], d[stat])])
        return np.column_stack([d["elo_diff"] / 400, d["neutral"], d["is_home"] * (1 - d["neutral"]),
                                np.log(np.maximum(tf, 0.1) / g), np.log(np.maximum(oa, 0.1) / g)])

    def build(self, sb: pd.DataFrame, nat) -> "NationalStatsModel":
        """`nat` = NationalModel ya entrenado (aporta el Elo previo a cada partido)."""
        F = nat.F[["date", "home", "away", "h_elo", "a_elo", "neutral"]]
        key = pd.concat([F.rename(columns={"home": "team", "away": "opp", "h_elo": "elo_t", "a_elo": "elo_o"}),
                         F.rename(columns={"away": "team", "home": "opp", "a_elo": "elo_t", "h_elo": "elo_o"})])
        d = sb.merge(key, on=["date", "team", "opp"], how="left")
        # partidos sin cruce exacto (nombres raros): Elo actual como respaldo
        miss = d["elo_t"].isna()
        d.loc[miss, "elo_t"] = d.loc[miss, "team"].map(nat.elo).fillna(1500)
        d.loc[miss, "elo_o"] = d.loc[miss, "opp"].map(nat.elo).fillna(1500)
        d["neutral"] = d["neutral"].fillna(1)
        d["elo_diff"] = d["elo_t"] - d["elo_o"] + np.where(d["neutral"] == 0, 100 * (2 * d["is_home"] - 1), 0)
        self.data = d
        val = []
        full = d
        for stat in STATS:
            d = full.dropna(subset=[stat])  # cada estadística usa los partidos que la registran
            g = float(d[stat].mean())
            self.glob[stat] = g
            # validación: se deja fuera cada torneo completo
            preds = pd.Series(np.nan, index=d.index)
            base = pd.Series(np.nan, index=d.index)
            for comp in d["competition"].unique():
                tr, te = d[d["competition"] != comp], d[d["competition"] == comp]
                gt = float(tr[stat].mean())
                ft, at = self._rates(tr, stat)
                m = PoissonRegressor(alpha=0.5, max_iter=500).fit(self._X(tr, stat, gt, ft, at, True), tr[stat])
                preds[te.index] = m.predict(self._X(te, stat, gt, ft, at, False))
                base[te.index] = gt
            y = d[stat].to_numpy(float)
            mu = preds.to_numpy()
            self.alpha[stat] = float(max(0.0, np.mean((y - mu) ** 2 - mu) / np.mean(mu ** 2)))
            val.append({"estadística": STATS[stat], "MAE modelo": float(np.mean(np.abs(y - mu))),
                        "MAE media global": float(np.mean(np.abs(y - base.to_numpy()))),
                        "media por equipo": g, "n": len(y)})
            # modelo final con todos los datos
            ft, at = self._rates(d, stat)
            self.team_for[stat], self.team_against[stat] = ft, at
            self.models[stat] = PoissonRegressor(alpha=0.5, max_iter=500).fit(self._X(d, stat, g, ft, at, True), d[stat])
        self.validation = pd.DataFrame(val)
        self.data = full
        return self

    def expected(self, team: str, opp: str, elo_t: float, elo_o: float, neutral: bool, is_home: int) -> dict:
        out = {}
        diff = elo_t - elo_o + (0 if neutral else 100 * (2 * is_home - 1))
        row = pd.DataFrame({"team": [team], "opp": [opp], "elo_diff": [diff], "neutral": [int(neutral)],
                            "is_home": [is_home]})
        for stat, m in self.models.items():
            row[stat] = 0
            X = self._X(row, stat, self.glob[stat], self.team_for[stat], self.team_against[stat], False)
            out[stat] = float(m.predict(X)[0])
        return out

    def coverage(self, team: str) -> int:
        return int((self.data["team"] == team).sum()) if self.data is not None else 0

    def sources(self) -> pd.Series:
        return self.data.drop_duplicates("match_id")["source"].value_counts()

    def team_table(self, team: str) -> pd.Series:
        d = self.data[self.data["team"] == team]
        if d.empty:
            return pd.Series(dtype=float)
        cols = {"corners": "Córners a favor", "corners_opp": "Córners en contra", "cards": "Tarjetas",
                "cards_opp": "Tarjetas del rival", "shots": "Remates", "shots_opp": "Remates en contra",
                "sot": "Remates al arco", "sot_opp": "Remates al arco en contra", "fouls": "Faltas",
                "xg": "xG a favor", "xg_opp": "xG en contra"}
        return d[list(cols)].mean(skipna=True).rename(cols)

    def total_probs(self, mu_h, mu_a, stat, line) -> float:
        dist = total_dist(mu_h, mu_a, self.alpha[stat], self.alpha[stat])
        return float(1 - dist[: int(line) + 1].sum())

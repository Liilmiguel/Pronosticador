"""Selecciones nacionales: ~49.000 partidos internacionales desde 1872 (dataset abierto martj42/international_results).

- Elo estilo "World Football Elo": K según importancia del torneo (Mundial 60 … amistoso 20),
  localía solo si la cancha no es neutral, multiplicador por diferencia de goles.
- Forma reciente (media exponencial de goles a favor/en contra y puntos), experiencia, H2H, descanso.
- Goles esperados de cada selección con regresión Poisson (Gradient Boosting) sobre Elo y forma; la matriz
  de marcadores usa la corrección Dixon-Coles → todos los mercados de goles. (Un Dixon-Coles clásico por
  equipo funciona mal aquí: ~240 selecciones que casi no se cruzan entre confederaciones.)
- Gradient Boosting 1X2 (partidos desde 1990) validado en los últimos 2 años.
"""
from __future__ import annotations

import io
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .config import DB_PATH, RAW_DIR
from .dixon_coles import markets_from_matrix, score_matrix

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ELO_URL = "https://www.eloratings.net"
ELO_NAME_FIX = {"AS": "American Samoa", "CZ": "Czech Republic", "IE": "Republic of Ireland", "MO": "Macau",
                "ST": "São Tomé and Príncipe", "TL": "Timor-Leste", "VI": "United States Virgin Islands"}
ELO_TOURNAMENT = {"Friendly": "Friendly", "Friendly tournament": "Friendly", "FIFA Series": "Friendly",
                  "World Cup": "FIFA World Cup", "World Cup qualifier": "FIFA World Cup qualification",
                  "African Nations Cup": "African Cup of Nations",
                  "African Nations Cup qualifier": "African Cup of Nations qualification",
                  "European Championship": "UEFA Euro", "European Championship qualifier": "UEFA Euro qualification",
                  "Copa America": "Copa América", "Asian Cup": "AFC Asian Cup",
                  "Asian Cup qualifier": "AFC Asian Cup qualification"}
HOME_ADV = 100.0
K_BY_LEVEL = {4: 60.0, 3: 50.0, 2: 40.0, 1: 30.0, 0: 20.0}
LEVEL_NAMES = {4: "Mundial (fase final)", 3: "Torneo continental (fase final)",
               2: "Eliminatorias / Nations League", 1: "Otro torneo", 0: "Amistoso"}
CONTINENTAL = {"UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
               "Confederations Cup", "CONCACAF Championship", "Oceania Nations Cup"}
FORM_HL = 5
FEATURES = ["h_elo", "a_elo", "elo_diff", "elo_p", "neutral", "level", "h_gf", "h_ga", "h_pts", "a_gf", "a_ga",
            "a_pts", "h_n", "a_n", "h_rest", "a_rest", "h2h_gd", "h2h_n", "d_gd"]


def tournament_level(t: str) -> int:
    t = str(t)
    if t == "FIFA World Cup":
        return 4
    if t in CONTINENTAL:
        return 3
    if "qualification" in t or "Nations League" in t:
        return 2
    if t == "Friendly":
        return 0
    return 1


def _elo_tournament(name: str) -> str:
    if name in ELO_TOURNAMENT:
        return ELO_TOURNAMENT[name]
    if "Nations League" in name:  # "European Nations League B" → UEFA Nations League
        return "UEFA Nations League" if name.startswith("European") else "CONCACAF Nations League" \
            if name.startswith("CONCACAF") else name
    return name.replace(" qualifier", " qualification")


def download_recent(after: pd.Timestamp, known_names: set) -> pd.DataFrame:
    """Partidos recientes desde eloratings.net (se actualiza a los pocos días de cada fecha FIFA)."""
    H = {"User-Agent": "Mozilla/5.0 (predictor-apuestas; uso personal)"}
    get = lambda path: requests.get(f"{ELO_URL}/{path}", headers=H, timeout=60).content.decode("utf-8", "replace")
    teams, tours = {}, {}
    for line in get("en.teams.tsv").splitlines():
        parts = line.split("\t")
        if len(parts) > 1:
            hit = [a for a in parts[1:] if a in known_names]
            teams[parts[0]] = ELO_NAME_FIX.get(parts[0], hit[0] if hit else parts[1])
    for line in get("en.tournaments.tsv").splitlines():
        parts = line.split("\t")
        if len(parts) > 1:
            tours[parts[0]] = parts[1]
    rows = []
    for year in range(after.year, pd.Timestamp.today().year + 1):
        for line in get(f"{year}_results.tsv").splitlines():
            r = line.split("\t")
            if len(r) < 9 or not r[0].isdigit():
                continue
            date = pd.Timestamp(int(r[0]), int(r[1]), int(r[2]))
            if date <= after:
                continue
            loc = r[8].strip()
            rows.append({"date": date, "home": teams.get(r[3], r[3]), "away": teams.get(r[4], r[4]),
                         "fthg": int(r[5]), "ftag": int(r[6]),
                         "tournament": _elo_tournament(tours.get(r[7], r[7])), "city": None,
                         "country": teams.get(loc, loc) if loc else teams.get(r[3], r[3]),
                         "neutral": int(bool(loc) and loc != r[3]), "source": "eloratings.net"})
    return pd.DataFrame(rows)


def download() -> pd.DataFrame:
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    (RAW_DIR / "intl").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "intl" / "results.csv").write_bytes(r.content)
    df = pd.read_csv(io.BytesIO(r.content))
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df = df.rename(columns={"home_team": "home", "away_team": "away", "home_score": "fthg", "away_score": "ftag"})
    df["source"] = "martj42"
    try:  # completar con lo más reciente (Nations League en curso, última fecha FIFA…)
        recent = download_recent(df["date"].max(), set(df["home"]) | set(df["away"]))
        df = pd.concat([df, recent], ignore_index=True).drop_duplicates(["date", "home", "away"], keep="first")
    except requests.RequestException:
        pass
    with sqlite3.connect(DB_PATH) as con:
        df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_sql("intl", con, if_exists="replace", index=False)
    return df


def load() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        try:
            df = pd.read_sql("SELECT * FROM intl", con)
        except Exception:
            return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df


@dataclass
class NationalModel:
    elo: dict = field(default_factory=dict)
    form: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)
    h2h: dict = field(default_factory=dict)
    gbm: HistGradientBoostingClassifier | None = None
    goals: tuple | None = None
    rho: float = -0.05
    w_gbm: float = 0.5
    validation: pd.DataFrame | None = None
    F: pd.DataFrame | None = None

    # ------------------------------------------------------------------ variables
    def _team_feats(self, team: str, side: str, date) -> dict:
        f = self.form.get(team, (np.nan, np.nan, np.nan))
        last = self.last.get(team)
        return {f"{side}_elo": self.elo.get(team, 1500.0), f"{side}_gf": f[0], f"{side}_ga": f[1], f"{side}_pts": f[2],
                f"{side}_n": min(self.n.get(team, 0), 200),
                f"{side}_rest": min((date - last).days, 365) if last is not None else np.nan}

    def _match_feats(self, home, away, date, neutral, level) -> dict:
        f = {**self._team_feats(home, "h", date), **self._team_feats(away, "a", date)}
        f["neutral"], f["level"] = int(neutral), int(level)
        f["elo_diff"] = f["h_elo"] - f["a_elo"] + (0 if neutral else HOME_ADV)
        f["elo_p"] = 1 / (1 + 10 ** (-f["elo_diff"] / 400))
        hist = self.h2h.get(tuple(sorted((home, away))), [])
        sign = 1 if home < away else -1
        f["h2h_n"] = len(hist)
        f["h2h_gd"] = sign * float(np.mean(hist)) if hist else np.nan
        f["d_gd"] = (f["h_gf"] - f["h_ga"]) - (f["a_gf"] - f["a_ga"])
        return f

    def build(self, df: pd.DataFrame, progress=None) -> "NationalModel":
        df = df.dropna(subset=["fthg", "ftag"]).sort_values("date", kind="stable").reset_index(drop=True)
        df["level"] = df["tournament"].map(tournament_level)
        a = 1 - 0.5 ** (1 / FORM_HL)
        h2h = defaultdict(lambda: deque(maxlen=8))
        self.h2h = h2h
        rows = []
        for r in df.itertuples(index=False):
            rows.append(self._match_feats(r.home, r.away, r.date, r.neutral, r.level))
            hg, ag = int(r.fthg), int(r.ftag)
            f = rows[-1]
            exp_h = f["elo_p"]
            score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            gd = abs(hg - ag)
            mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11.0 + gd) / 8.0
            delta = K_BY_LEVEL[r.level] * mult * (score - exp_h)
            self.elo[r.home] = f["h_elo"] + delta
            self.elo[r.away] = f["a_elo"] - delta
            for team, gf, ga in ((r.home, hg, ag), (r.away, ag, hg)):
                pts = 3 if gf > ga else 1 if gf == ga else 0
                old = self.form.get(team)
                self.form[team] = (gf, ga, pts) if old is None else tuple(
                    (1 - a) * o + a * x for o, x in zip(old, (gf, ga, pts)))
                self.last[team] = r.date
                self.n[team] = self.n.get(team, 0) + 1
            h2h[tuple(sorted((r.home, r.away)))].append((1 if r.home < r.away else -1) * (hg - ag))
        F = pd.concat([df, pd.DataFrame(rows).drop(columns=["neutral", "level"])], axis=1)
        F["y_res"] = np.select([F["fthg"] > F["ftag"], F["fthg"] == F["ftag"]], [0, 1], 2)
        self.F = F
        self.h2h = {k: list(v) for k, v in h2h.items()}
        if progress:
            progress(0.5, "Selecciones: validando modelos…")
        self._validate(F)
        if progress:
            progress(0.8, "Selecciones: entrenando modelo final…")
        self.gbm = self._fit_gbm(F[F["date"] >= "1990-01-01"])
        self.goals = self._fit_goals(F[F["date"] >= "1990-01-01"])
        return self

    @staticmethod
    def _fit_gbm(d):
        age = (d["date"].max() - d["date"]).dt.days / 365.25
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=20, min_samples_leaf=100,
                                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                                              random_state=0).fit(d[FEATURES].to_numpy(np.float32), d["y_res"],
                                                                  sample_weight=(0.5 ** (age / 10)).to_numpy())

    @staticmethod
    def _fit_goals(d):
        age = (d["date"].max() - d["date"]).dt.days / 365.25
        w = (0.5 ** (age / 10)).to_numpy()
        X = d[FEATURES].to_numpy(np.float32)
        mk = lambda: HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=20,
                                                   min_samples_leaf=100, early_stopping=True, random_state=0)
        return mk().fit(X, d["fthg"], sample_weight=w), mk().fit(X, d["ftag"], sample_weight=w)

    def _goal_probs(self, goals, X: pd.DataFrame):
        A = X[FEATURES].to_numpy(np.float32)
        lam, mu = goals[0].predict(A), goals[1].predict(A)
        mks = [markets_from_matrix(score_matrix(l, m, self.rho)) for l, m in zip(lam, mu)]
        return np.array([[m["1"], m["X"], m["2"]] for m in mks]), lam, mu

    def _validate(self, F):
        cutoff = F["date"].max() - pd.Timedelta(days=730)
        tr, te = F[(F["date"] >= "1990-01-01") & (F["date"] < cutoff)], F[F["date"] >= cutoff].copy()
        y = te["y_res"].to_numpy()
        gbm = self._fit_gbm(tr)
        P_g = gbm.predict_proba(te[FEATURES].to_numpy(np.float32))
        elo_lr = LogisticRegression(max_iter=1000).fit(tr[["elo_diff", "neutral"]], tr["y_res"])
        P_e = elo_lr.predict_proba(te[["elo_diff", "neutral"]])
        P_d, lam, mu = self._goal_probs(self._fit_goals(tr), te)
        yh, ya = te["fthg"].to_numpy(float), te["ftag"].to_numpy(float)
        self.goal_validation = {"MAE goles local": float(np.mean(np.abs(yh - lam))),
                                "MAE goles visita": float(np.mean(np.abs(ya - mu))),
                                "goles reales/partido": float((yh + ya).mean()),
                                "goles predichos/partido": float((lam + mu).mean())}

        def ll(P):
            return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))

        best = min((ll(w * P_g + (1 - w) * P_d), w) for w in np.linspace(0, 1, 11))
        self.w_gbm = float(best[1])
        P_ens = self.w_gbm * P_g + (1 - self.w_gbm) * P_d
        rows = [("Ensamble", P_ens), ("Gradient Boosting", P_g), ("Poisson de goles", P_d), ("Solo Elo", P_e),
                ("Frecuencias base", np.tile(np.bincount(tr["y_res"], minlength=3) / len(tr), (len(y), 1)))]
        self.validation = pd.DataFrame([{"modelo": n, "log loss": ll(P), "acierto": float(np.mean(P.argmax(1) == y)),
                                         "n": len(y)} for n, P in rows])

    # ------------------------------------------------------------------ predicción
    def predict(self, home: str, away: str, neutral: bool, level: int, date=None) -> dict:
        date = pd.Timestamp(date) if date is not None else pd.Timestamp.today().normalize()
        f = self._match_feats(home, away, date, neutral, level)
        p_g = self.gbm.predict_proba(pd.DataFrame([f])[FEATURES].to_numpy(np.float32))[0]
        X = pd.DataFrame([f])[FEATURES].to_numpy(np.float32)
        lam, mu = float(self.goals[0].predict(X)[0]), float(self.goals[1].predict(X)[0])
        mat = score_matrix(lam, mu, self.rho)
        mk = markets_from_matrix(mat)
        p_d = np.array([mk["1"], mk["X"], mk["2"]])
        p = self.w_gbm * p_g + (1 - self.w_gbm) * p_d
        return {"p": p, "gbm": p_g, "dc": p_d, "markets": mk, "features": f, "matrix": mat}

    def ranking(self, active_years: int = 2) -> pd.DataFrame:
        cut = self.F["date"].max() - pd.Timedelta(days=365 * active_years)
        rows = [{"Selección": t, "Elo": e, "Partidos": self.n.get(t, 0), "Último partido": self.last.get(t),
                 "Goles a favor (forma)": self.form[t][0], "Goles en contra (forma)": self.form[t][1]}
                for t, e in self.elo.items() if self.last.get(t) is not None and self.last[t] >= cut]
        df = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
        df.index += 1
        return df

    def teams(self) -> list[str]:
        cut = self.F["date"].max() - pd.Timedelta(days=365 * 4)
        return sorted(t for t, d in self.last.items() if d >= cut)

    def last_matches(self, team: str, n: int = 8) -> pd.DataFrame:
        F = self.F
        d = F[(F["home"] == team) | (F["away"] == team)].tail(n)
        return d[["date", "home", "fthg", "ftag", "away", "tournament"]]

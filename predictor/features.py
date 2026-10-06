"""Ingeniería de variables pre-partido (sin fuga de información: solo usa datos anteriores a cada partido).

Factores similares a los que usan las casas de apuestas:
- Rating Elo por país (con ventaja local, multiplicador por diferencia de goles y regresión entre temporadas).
- Forma reciente (media exponencial corta y larga) de goles, tiros, tiros a puerta, córners, tarjetas,
  faltas y puntos.
- Árbitro: tarjetas y faltas por partido que suele sancionar.
- Forma como local del local y como visitante del visitante.
- Descanso (días desde el último partido) y experiencia (partidos en la base).
- Historial de enfrentamientos directos.
- Contexto de la liga (promedio de goles local/visita, % de victorias locales y empates).
- Probabilidades implícitas del mercado (cuotas sin margen), opcionales.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .betting import devig_matrix

ELO_K = 20.0
ELO_HOME = 60.0
ELO_REGRESS = 0.2
SHORT_HL, LONG_HL, VENUE_HL = 3, 10, 5

TEAM_STATS = ["gf", "ga", "sf", "sa", "sotf", "sota", "cf", "ca", "kf", "ka", "ff", "fa", "pts"]
REF_HL = 15
VENUE_STATS = ["gf", "ga", "pts"]


def elo_base(tier: int) -> float:
    return 1500.0 - 80.0 * (tier - 1)


def _gd_mult(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11.0 + gd) / 8.0


def add_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Tarjetas al estilo de las casas: amarilla = 1, roja = 2."""
    for s in ("h", "a"):
        df[f"{s}_cards"] = df[f"{s}_yellow"] + 2 * df[f"{s}_red"].fillna(0)
    return df


def market_probs(df: pd.DataFrame, prefix: str = "odds") -> np.ndarray:
    o = df[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]].to_numpy(float)
    ok = np.all(o > 1.0, axis=1)
    p = np.full(o.shape, np.nan)
    p[ok] = devig_matrix(o[ok])
    return p


def market_o25(df: pd.DataFrame) -> np.ndarray:
    o = df[["odds_o25", "odds_u25"]].to_numpy(float)
    ok = np.all(o > 1.0, axis=1)
    p = np.full(len(df), np.nan)
    p[ok] = devig_matrix(o[ok])[:, 0]
    return p


class FeatureBuilder:
    """Calcula variables para el histórico y guarda el 'estado' de cada equipo para predecir partidos futuros."""

    def __init__(self):
        self.elo: dict = {}
        self.team_season: dict = {}
        self.team_state: pd.DataFrame | None = None
        self.venue_state: pd.DataFrame | None = None
        self.league_state: pd.DataFrame | None = None
        self.h2h: dict = defaultdict(lambda: deque(maxlen=6))
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------ histórico
    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        df = matches.dropna(subset=["fthg", "ftag"]).sort_values(["date", "league", "home"]).reset_index(drop=True)
        df["mid"] = np.arange(len(df))
        add_cards(df)
        feats = pd.DataFrame(index=df.index)

        # --- Elo + enfrentamientos directos (bucle cronológico)
        elo_h, elo_a, h2h_gd, h2h_n = (np.empty(len(df)) for _ in range(4))
        cols = df[["country", "tier", "season", "home", "away", "fthg", "ftag"]].itertuples(index=False)
        for k, (country, tier, season, home, away, hg, ag) in enumerate(cols):
            kh, ka = (country, home), (country, away)
            rh, ra = self._elo_pre(kh, tier, season), self._elo_pre(ka, tier, season)
            elo_h[k], elo_a[k] = rh, ra
            hist = self.h2h[(country, *sorted((home, away)))]
            sign = 1 if home < away else -1
            h2h_n[k] = len(hist)
            h2h_gd[k] = sign * np.mean(hist) if hist else np.nan
            exp_h = 1.0 / (1.0 + 10 ** (-(rh + ELO_HOME - ra) / 400.0))
            score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = ELO_K * _gd_mult(int(hg - ag)) * (score - exp_h)
            self.elo[kh], self.elo[ka] = rh + delta, ra - delta
            hist.append(sign * (hg - ag))
        feats["h_elo"], feats["a_elo"] = elo_h, elo_a
        feats["elo_diff"] = elo_h + ELO_HOME - elo_a
        feats["elo_p_home"] = 1.0 / (1.0 + 10 ** (-feats["elo_diff"] / 400.0))
        feats["h2h_gd"], feats["h2h_n"] = h2h_gd, h2h_n

        # --- forma por equipo (formato largo: una fila por equipo y partido)
        long = self._long(df)
        g = long.groupby("tkey", sort=False)
        post = {}
        for c in TEAM_STATS:
            for tag, hl in (("s", SHORT_HL), ("l", LONG_HL)):
                post[f"{c}_{tag}"] = g[c].transform(lambda s, hl=hl: s.ewm(halflife=hl, min_periods=1).mean())
        post = pd.DataFrame(post, index=long.index)
        post["n"] = g.cumcount() + 1
        post["last_date"] = long["date"]
        pre = post.groupby(long["tkey"], sort=False).shift()
        pre["rest"] = (long["date"] - pre.pop("last_date")).dt.days.clip(upper=30)
        pre["n"] = pre["n"].fillna(0).clip(upper=100)
        self.team_state = post.assign(tkey=long["tkey"]).groupby("tkey").last()

        # --- forma en la condición (local en casa / visita fuera)
        gv = long.groupby(["tkey", "is_home"], sort=False)
        vpost = pd.DataFrame({f"v{c}": gv[c].transform(lambda s: s.ewm(halflife=VENUE_HL, min_periods=1).mean())
                              for c in VENUE_STATS}, index=long.index)
        vpre = vpost.groupby([long["tkey"], long["is_home"]], sort=False).shift()
        self.venue_state = vpost.assign(tkey=long["tkey"], is_home=long["is_home"]).groupby(["tkey", "is_home"]).last()

        team_pre = pd.concat([pre, vpre], axis=1)
        for side, flag in (("h", 1), ("a", 0)):
            part = team_pre[long["is_home"] == flag].set_index(long.loc[long["is_home"] == flag, "mid"])
            part = part.reindex(df["mid"]).set_axis(df.index)
            feats[[f"{side}_{c}" for c in part.columns]] = part.to_numpy()

        # --- contexto de liga (últimos ~380 partidos)
        lg = df.groupby("league", sort=False)
        lpost = pd.DataFrame({
            "lg_hg": lg["fthg"].transform(lambda s: s.rolling(380, min_periods=30).mean()),
            "lg_ag": lg["ftag"].transform(lambda s: s.rolling(380, min_periods=30).mean()),
            "lg_hw": (df["fthg"] > df["ftag"]).groupby(df["league"]).transform(lambda s: s.rolling(380, min_periods=30).mean()),
            "lg_dr": (df["fthg"] == df["ftag"]).groupby(df["league"]).transform(lambda s: s.rolling(380, min_periods=30).mean()),
            "lg_cor": (df["h_corners"] + df["a_corners"]).groupby(df["league"]).transform(
                lambda s: s.rolling(380, min_periods=30).mean()),
            "lg_crd": (df["h_cards"] + df["a_cards"]).groupby(df["league"]).transform(
                lambda s: s.rolling(380, min_periods=30).mean()),
        })
        feats[lpost.columns] = lpost.groupby(df["league"]).shift().to_numpy()
        self.league_state = lpost.assign(league=df["league"]).groupby("league").last()

        # --- árbitro: tarjetas y faltas que suele sancionar (media exponencial de sus partidos previos)
        ref = df["referee"].astype(str).str.strip()
        valid = ~ref.isin(["", "None", "nan", "NaN"]) & df["h_cards"].notna()
        rk = (df["country"] + "|" + ref)[valid]
        rd = pd.DataFrame({"cards": (df["h_cards"] + df["a_cards"])[valid],
                           "fouls": (df["h_fouls"] + df["a_fouls"])[valid], "rk": rk})
        gr = rd.groupby("rk", sort=False)
        rpost = pd.DataFrame({c: gr[c].transform(lambda s: s.ewm(halflife=REF_HL, min_periods=1).mean())
                              for c in ["cards", "fouls"]}, index=rd.index)
        rpost["n"] = gr.cumcount() + 1
        rpre = rpost.groupby(rk, sort=False).shift()
        feats["ref_cards"], feats["ref_fouls"], feats["ref_n"] = np.nan, np.nan, 0.0
        feats.loc[rd.index, ["ref_cards", "ref_fouls", "ref_n"]] = rpre[["cards", "fouls", "n"]].to_numpy()
        feats["ref_n"] = feats["ref_n"].fillna(0).clip(upper=200)
        self.ref_state = rpost.assign(rk=rk).groupby("rk").last()

        feats = self._finish(feats, df)
        self.feature_cols = [c for c in feats.columns if not c.startswith("mkt_")]
        out = pd.concat([df, feats], axis=1)
        out["y_res"] = np.select([out["fthg"] > out["ftag"], out["fthg"] == out["ftag"]], [0, 1], 2)
        out["y_o25"] = ((out["fthg"] + out["ftag"]) > 2.5).astype(int)
        out["y_btts"] = ((out["fthg"] > 0) & (out["ftag"] > 0)).astype(int)
        return out

    def referees(self, country: str) -> list[str]:
        if getattr(self, "ref_state", None) is None:
            return []
        idx = [k.split("|", 1)[1] for k in self.ref_state.index if k.startswith(country + "|")]
        return sorted(idx)

    @staticmethod
    def _finish(feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        feats["lg_tier"] = df["tier"].to_numpy()
        feats["d_pts_l"] = feats["h_pts_l"] - feats["a_pts_l"]
        feats["d_gd_l"] = (feats["h_gf_l"] - feats["h_ga_l"]) - (feats["a_gf_l"] - feats["a_ga_l"])
        feats["d_sot_l"] = (feats["h_sotf_l"] - feats["h_sota_l"]) - (feats["a_sotf_l"] - feats["a_sota_l"])
        feats["exp_hg"] = (feats["h_gf_l"] + feats["a_ga_l"]) / 2
        feats["exp_ag"] = (feats["a_gf_l"] + feats["h_ga_l"]) / 2
        mk = market_probs(df)
        feats["mkt_h"], feats["mkt_d"], feats["mkt_a"] = mk[:, 0], mk[:, 1], mk[:, 2]
        feats["mkt_o25"] = market_o25(df)
        return feats

    def _elo_pre(self, key, tier, season) -> float:
        base = elo_base(int(tier))
        r = self.elo.get(key, base)
        last = self.team_season.get(key)
        if last is not None and season != last:
            r = (1 - ELO_REGRESS) * r + ELO_REGRESS * base
        self.team_season[key] = season
        self.elo[key] = r
        return r

    @staticmethod
    def _long(df: pd.DataFrame) -> pd.DataFrame:
        pts_h = np.select([df["fthg"] > df["ftag"], df["fthg"] == df["ftag"]], [3, 1], 0)
        home = pd.DataFrame({
            "mid": df["mid"], "date": df["date"], "tkey": df["country"] + "|" + df["home"], "is_home": 1,
            "gf": df["fthg"], "ga": df["ftag"], "sf": df["h_shots"], "sa": df["a_shots"], "sotf": df["h_sot"],
            "sota": df["a_sot"], "cf": df["h_corners"], "ca": df["a_corners"], "kf": df["h_cards"],
            "ka": df["a_cards"], "ff": df["h_fouls"], "fa": df["a_fouls"], "pts": pts_h})
        away = pd.DataFrame({
            "mid": df["mid"], "date": df["date"], "tkey": df["country"] + "|" + df["away"], "is_home": 0,
            "gf": df["ftag"], "ga": df["fthg"], "sf": df["a_shots"], "sa": df["h_shots"], "sotf": df["a_sot"],
            "sota": df["h_sot"], "cf": df["a_corners"], "ca": df["h_corners"], "kf": df["a_cards"],
            "ka": df["h_cards"], "ff": df["a_fouls"], "fa": df["h_fouls"], "pts": np.select(
                [pts_h == 0, pts_h == 1], [3, 1], 0)})
        return pd.concat([home, away]).sort_values(["date", "mid", "is_home"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------ partidos futuros
    def features_for(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Variables para partidos sin jugar (columnas: league, country, tier, season, date, home, away + cuotas)."""
        rows = []
        for r in fixtures.itertuples(index=False):
            f = {}
            for side, team, is_home in (("h", r.home, 1), ("a", r.away, 0)):
                key = (r.country, team)
                base = elo_base(int(r.tier))
                elo = self.elo.get(key, base)
                if self.team_season.get(key) not in (None, r.season):
                    elo = (1 - ELO_REGRESS) * elo + ELO_REGRESS * base
                f[f"{side}_elo"] = elo
                tkey = f"{r.country}|{team}"
                if tkey in self.team_state.index:
                    st = self.team_state.loc[tkey]
                    for c in self.team_state.columns:
                        if c not in ("last_date",):
                            f[f"{side}_{c}"] = st[c]
                    f[f"{side}_rest"] = min((pd.Timestamp(r.date) - st["last_date"]).days, 30)
                    f[f"{side}_n"] = min(st["n"], 100)
                else:
                    f[f"{side}_n"] = 0
                if (tkey, is_home) in self.venue_state.index:
                    vs = self.venue_state.loc[(tkey, is_home)]
                    for c in self.venue_state.columns:
                        f[f"{side}_{c}"] = vs[c]
            f["elo_diff"] = f["h_elo"] + ELO_HOME - f["a_elo"]
            f["elo_p_home"] = 1.0 / (1.0 + 10 ** (-f["elo_diff"] / 400.0))
            hist = self.h2h.get((r.country, *sorted((r.home, r.away))), [])
            f["h2h_n"] = len(hist)
            f["h2h_gd"] = (1 if r.home < r.away else -1) * np.mean(hist) if hist else np.nan
            if r.league in self.league_state.index:
                f.update(self.league_state.loc[r.league].to_dict())
            rk = f"{r.country}|{str(getattr(r, 'referee', '') or '').strip()}"
            rs = getattr(self, "ref_state", None)
            if rs is not None and rk in rs.index:
                f["ref_cards"], f["ref_fouls"], f["ref_n"] = rs.loc[rk, "cards"], rs.loc[rk, "fouls"], min(rs.loc[rk, "n"], 200)
            else:
                f["ref_n"] = 0
            rows.append(f)
        feats = pd.DataFrame(rows, index=fixtures.index)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = np.nan
        df = fixtures.copy()
        for c in ["odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25"]:
            if c not in df.columns:
                df[c] = np.nan
        feats = self._finish(feats, df)
        return pd.concat([fixtures.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

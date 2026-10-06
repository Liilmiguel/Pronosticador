"""Mercados de estadísticas: córners y tarjetas.

Para cada equipo se predice el número esperado (Gradient Boosting con pérdida Poisson usando la forma
de córners/tarjetas/faltas a favor y en contra, Elo, contexto de liga y el árbitro). Como estos conteos
tienen más varianza que una Poisson, se usa una binomial negativa con dispersión estimada fuera de muestra.
Tarjetas: amarilla = 1, roja = 2 (convención habitual de las casas).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS = {"cor_h": "h_corners", "cor_a": "a_corners", "crd_h": "h_cards", "crd_a": "a_cards"}
MAX_N = 40


def _reg() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=24,
                                         min_samples_leaf=200, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=0)


def nb_pmf(mu: float, alpha: float) -> np.ndarray:
    k = np.arange(MAX_N + 1)
    if alpha <= 1e-6:
        from scipy.stats import poisson
        p = poisson.pmf(k, mu)
    else:
        n = 1.0 / alpha
        p = nbinom.pmf(k, n, n / (n + mu))
    return p / p.sum()


@dataclass
class StatsModel:
    features: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    alpha: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)

    def fit(self, F: pd.DataFrame, feature_cols: list[str], holdout_days: int = 365) -> "StatsModel":
        self.features = feature_cols
        cutoff = F["date"].max() - pd.Timedelta(days=holdout_days)
        for name, col in TARGETS.items():
            d = F.dropna(subset=[col])
            tr, te = d[d["date"] < cutoff], d[d["date"] >= cutoff]
            m = self._fit_one(tr, col)
            mu = m.predict(te[self.features].to_numpy(np.float32))
            y = te[col].to_numpy(float)
            # dispersión NB por momentos: Var = mu + alpha * mu^2
            self.alpha[name] = float(max(0.0, np.mean((y - mu) ** 2 - mu) / np.mean(mu ** 2)))
            naive = te.groupby("league")[col].transform("mean").to_numpy()  # promedio de la liga
            self.validation[name] = {"MAE modelo": float(np.mean(np.abs(y - mu))),
                                     "MAE promedio liga": float(np.mean(np.abs(y - naive))),
                                     "media real": float(y.mean()), "media predicha": float(mu.mean()),
                                     "n": int(len(te))}
            self.models[name] = self._fit_one(d, col)  # modelo final con todos los datos
        # validación de mercados de totales
        self._validate_totals(F, cutoff)
        return self

    def _fit_one(self, d: pd.DataFrame, col: str):
        age = (d["date"].max() - d["date"]).dt.days / 365.25
        w = 0.5 ** (age / 6.0)
        return _reg().fit(d[self.features].to_numpy(np.float32), d[col].to_numpy(float), sample_weight=w.to_numpy())

    def _validate_totals(self, F, cutoff):
        te = F[(F["date"] >= cutoff)].dropna(subset=list(TARGETS.values()))
        if te.empty:
            return
        e = self.expected(te)
        for kind, line in (("cor", 9.5), ("crd", 4.5)):
            tot = te[f"h_{'corners' if kind == 'cor' else 'cards'}"] + te[f"a_{'corners' if kind == 'cor' else 'cards'}"]
            y = (tot > line).to_numpy(float)
            p = np.array([1 - total_dist(a, b, self.alpha[f"{kind}_h"], self.alpha[f"{kind}_a"])[: int(line) + 1].sum()
                          for a, b in zip(e[f"{kind}_h"], e[f"{kind}_a"])])
            base = te.assign(y=y).groupby("league")["y"].transform("mean").to_numpy()
            self.validation[f"más de {line} {'córners' if kind == 'cor' else 'tarjetas'}"] = {
                "Brier modelo": float(np.mean((p - y) ** 2)), "Brier promedio liga": float(np.mean((base - y) ** 2)),
                "n": int(len(y))}

    def expected(self, X: pd.DataFrame) -> pd.DataFrame:
        A = X.reindex(columns=self.features).to_numpy(np.float32)
        return pd.DataFrame({k: m.predict(A) for k, m in self.models.items()}, index=X.index)


def total_dist(mu_h: float, mu_a: float, a_h: float, a_a: float) -> np.ndarray:
    return np.convolve(nb_pmf(mu_h, a_h), nb_pmf(mu_a, a_a))[: MAX_N + 1]


def stat_markets(mu_h: float, mu_a: float, a_h: float, a_a: float, lines_total, lines_team) -> dict:
    """Más/menos del total y por equipo, y quién tiene más (1X2 del conteo)."""
    ph, pa = nb_pmf(mu_h, a_h), nb_pmf(mu_a, a_a)
    tot = np.convolve(ph, pa)
    k = np.arange(len(tot))
    joint = np.outer(ph, pa)
    i, j = np.indices(joint.shape)
    out = {"total": {l: float(tot[k > l].sum()) for l in lines_total},
           "local": {l: float(ph[np.arange(len(ph)) > l].sum()) for l in lines_team},
           "visita": {l: float(pa[np.arange(len(pa)) > l].sum()) for l in lines_team},
           "1X2": (float(joint[i > j].sum()), float(joint[i == j].sum()), float(joint[i < j].sum())),
           "dist_total": tot}
    return out

"""Modelo Dixon-Coles (1997) con ponderación temporal.

Es la base de cómo las casas de apuestas generan cuotas de goles: cada equipo tiene una fuerza de
ataque y de defensa, hay una ventaja de local, y una corrección (rho) para los marcadores bajos
(0-0, 1-0, 0-1, 1-1). Los partidos antiguos pesan menos (decaimiento exponencial xi).
De la matriz de marcadores se derivan todos los mercados: 1X2, más/menos, ambos marcan,
hándicap asiático, marcador exacto, doble oportunidad…
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass
class DixonColes:
    xi: float = 0.0019          # decaimiento por día (vida media ≈ 1 año)
    l2: float = 0.01            # regularización (equipos con pocos partidos tienden a la media)
    teams: list[str] = field(default_factory=list)
    attack: np.ndarray | None = None
    defence: np.ndarray | None = None
    home_adv: float = 0.25
    rho: float = -0.05
    ref_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------ ajuste
    def fit(self, df: pd.DataFrame, ref_date=None, max_days: int = 3 * 365) -> "DixonColes":
        """Si `df` trae columna `neutral` (selecciones), la localía solo aplica a partidos no neutrales."""
        df = df.dropna(subset=["fthg", "ftag"])
        ref_date = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max() + pd.Timedelta(days=1)
        df = df[(df["date"] < ref_date) & (df["date"] >= ref_date - pd.Timedelta(days=max_days))]
        if len(df) < 30:
            raise ValueError("Muy pocos partidos para ajustar Dixon-Coles")
        self.ref_date = ref_date
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        hi = df["home"].map(idx).to_numpy()
        ai = df["away"].map(idx).to_numpy()
        x = df["fthg"].to_numpy(float)
        y = df["ftag"].to_numpy(float)
        days = (ref_date - df["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days)
        hm = 1.0 - df["neutral"].astype(float).to_numpy() if "neutral" in df.columns else np.ones(len(df))
        n = len(self.teams)

        m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)

        def nll(theta):
            att, dfn, home, rho = theta[:n], theta[n:2 * n], theta[2 * n], theta[2 * n + 1]
            ll_lam = home * hm + att[hi] + dfn[ai]
            ll_mu = att[ai] + dfn[hi]
            lam, mu = np.exp(ll_lam), np.exp(ll_mu)
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            ll = np.log(tau) + x * ll_lam - lam + y * ll_mu - mu
            # gradientes respecto de log(lambda), log(mu) y rho
            g_lam = x - lam
            g_mu = y - mu
            g_rho = np.zeros_like(lam)
            t = tau[m00]
            g_lam[m00] += -lam[m00] * mu[m00] * rho / t
            g_mu[m00] += -lam[m00] * mu[m00] * rho / t
            g_rho[m00] = -lam[m00] * mu[m00] / t
            g_lam[m01] += lam[m01] * rho / tau[m01]
            g_rho[m01] = lam[m01] / tau[m01]
            g_mu[m10] += mu[m10] * rho / tau[m10]
            g_rho[m10] = mu[m10] / tau[m10]
            g_rho[m11] = -1.0 / tau[m11]
            g_lam *= w
            g_mu *= w
            grad = np.zeros_like(theta)
            grad[:n] = np.bincount(hi, g_lam, n) + np.bincount(ai, g_mu, n)
            grad[n:2 * n] = np.bincount(ai, g_lam, n) + np.bincount(hi, g_mu, n)
            grad[2 * n] = (g_lam * hm).sum()
            grad[2 * n + 1] = (w * g_rho).sum()
            wsum = w.sum()
            # objetivo: -loglik/peso + penalizaciones (sum(att)=0 y L2)
            f = -(w * ll).sum() / wsum
            g = -grad / wsum
            s = att.sum()
            f += 10 * s ** 2 + self.l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
            g[:n] += 20 * s + 2 * self.l2 * att
            g[n:2 * n] += 2 * self.l2 * dfn
            return f, g

        theta0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.3, 0.3)]
        res = minimize(nll, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        th = res.x
        self.attack, self.defence = th[:n], th[n:2 * n]
        self.home_adv, self.rho = float(th[2 * n]), float(th[2 * n + 1])
        return self

    # ------------------------------------------------------------------ predicción
    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        idx = {t: i for i, t in enumerate(self.teams)}
        # equipo desconocido (ascendido sin historial): fuerza ligeramente bajo la media
        ah, dh = (self.attack[idx[home]], self.defence[idx[home]]) if home in idx else (-0.15, 0.15)
        aa, da = (self.attack[idx[away]], self.defence[idx[away]]) if away in idx else (-0.15, 0.15)
        ha = 0.0 if neutral else self.home_adv
        return float(np.exp(ha + ah + da)), float(np.exp(aa + dh))

    def knows(self, team: str) -> bool:
        return team in self.teams

    def score_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        lam, mu = self.expected_goals(home, away, neutral)
        return score_matrix(lam, mu, self.rho)

    def ratings(self) -> pd.DataFrame:
        return pd.DataFrame({"equipo": self.teams, "ataque": np.exp(self.attack), "defensa": np.exp(self.defence)}) \
            .assign(fuerza=lambda d: d["ataque"] / d["defensa"]).sort_values("fuerza", ascending=False)


def score_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def markets_from_matrix(m: np.ndarray) -> dict:
    """Todos los mercados derivados de la matriz de marcadores P(goles local = i, goles visita = j)."""
    n = m.shape[0]
    i, j = np.indices(m.shape)
    total, diff = i + j, i - j
    out = {
        "1": m[diff > 0].sum(), "X": m[diff == 0].sum(), "2": m[diff < 0].sum(),
        "BTTS sí": m[(i > 0) & (j > 0)].sum(),
        "xG local": (i * m).sum(), "xG visita": (j * m).sum(),
    }
    out["BTTS no"] = 1 - out["BTTS sí"]
    out["1X"], out["12"], out["X2"] = out["1"] + out["X"], out["1"] + out["2"], out["X"] + out["2"]
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        out[f"Más de {line}"] = m[total > line].sum()
        out[f"Menos de {line}"] = m[total < line].sum()
    # hándicap asiático (local), líneas enteras y medias; las de cuarto se dividen en dos
    ah = {}
    for line in np.arange(-3, 3.25, 0.25):
        ah[round(float(line), 2)] = _asian(m, diff, line)
    out["asian"] = ah
    flat = [(f"{a}-{b}", m[a, b]) for a in range(min(n, 7)) for b in range(min(n, 7))]
    out["marcadores"] = sorted(flat, key=lambda t: -t[1])
    return out


def quick_markets(m: np.ndarray) -> dict:
    """Solo 1X2, más de 2.5 y ambos marcan (rápido, para backtests)."""
    i, j = np.indices(m.shape)
    return {"1": m[i > j].sum(), "X": np.trace(m), "2": m[i < j].sum(),
            "Más de 2.5": m[i + j > 2].sum(), "BTTS sí": m[1:, 1:].sum()}


def _asian(m, diff, line) -> dict:
    """P(gana), P(devolución), P(pierde) apostando al local con hándicap `line`."""
    frac = round(line * 4) % 2
    if frac:  # línea de cuarto (-0.25, -0.75…) → mitad en cada línea vecina
        a, b = _asian(m, diff, line - 0.25), _asian(m, diff, line + 0.25)
        return {k: (a[k] + b[k]) / 2 for k in a}
    adj = diff + line
    return {"gana": m[adj > 0].sum(), "push": m[np.isclose(adj, 0)].sum(), "pierde": m[adj < 0].sum()}


def asian_fair_odds(r: dict) -> float:
    """Cuota justa de hándicap asiático considerando devoluciones: p_gana*(o-1) = p_pierde."""
    if r["gana"] <= 0:
        return np.inf
    return 1 + r["pierde"] / r["gana"]

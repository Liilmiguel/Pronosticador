"""Buscador de la mejor apuesta: probabilidades del modelo para todos los mercados de un partido, comparadas
con las cuotas que el usuario copia de su casa (p. ej. Betano).

Cada mercado se describe con P(gana), P(devolución) y P(pierde), así el valor esperado y Kelly son correctos
también en hándicap asiático (líneas enteras devuelven, las de cuarto se reparten en dos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats_markets import stat_markets

GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
SCORES = ("1-0", "2-0", "2-1", "3-0", "3-1", "3-2", "0-0", "1-1", "2-2", "3-3", "0-1", "0-2", "1-2", "0-3", "1-3", "2-3")
AH_LINES = (-2.5, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)


def _m(group, name, win, push=0.0):
    return {"Grupo": group, "Mercado": name, "p_gana": float(win), "p_push": float(push),
            "p_pierde": float(max(0.0, 1 - win - push))}


def goal_markets(p1x2, mk: dict, home: str, away: str, p_o25=None, p_btts=None) -> list[dict]:
    p1, px, p2 = p1x2
    out = [_m("Resultado", f"1 · {home}", p1), _m("Resultado", "X · Empate", px), _m("Resultado", f"2 · {away}", p2),
           _m("Doble oportunidad", "1X", p1 + px), _m("Doble oportunidad", "12", p1 + p2),
           _m("Doble oportunidad", "X2", px + p2),
           _m("Empate no válido", f"{home} (DNB)", p1, px), _m("Empate no válido", f"{away} (DNB)", p2, px)]
    for line in GOAL_LINES:
        po = p_o25 if (line == 2.5 and p_o25 is not None) else mk[f"Más de {line}"]
        out += [_m("Goles", f"Más de {line} goles", po), _m("Goles", f"Menos de {line} goles", 1 - po)]
    pb = p_btts if p_btts is not None else mk["BTTS sí"]
    out += [_m("Ambos marcan", "Ambos marcan: sí", pb), _m("Ambos marcan", "Ambos marcan: no", 1 - pb)]
    ah = mk["asian"]
    for line in AH_LINES:
        h = ah.get(round(line, 2))
        a = ah.get(round(-line, 2))
        if h:
            out.append(_m("Hándicap asiático", f"{home} {line:+.2f}".replace("+0.00", "0"), h["gana"], h["push"]))
        if a:  # visita con hándicap L = local con −L visto desde el otro lado
            out.append(_m("Hándicap asiático", f"{away} {line:+.2f}".replace("+0.00", "0"), a["pierde"], a["push"]))
    scores = dict(mk["marcadores"])
    for sc in SCORES:  # lista fija para que la tabla no cambie de orden entre partidos
        out.append(_m("Marcador exacto", f"Marcador {sc}", scores.get(sc, 0.0)))
    return out


def count_markets(group: str, label: str, mu_h, mu_a, a_h, a_a, home, away, lines_total, lines_team) -> list[dict]:
    sm = stat_markets(mu_h, mu_a, a_h, a_a, lines_total, lines_team)
    out = []
    for line, p in sm["total"].items():
        out += [_m(group, f"Más de {line} {label}", p), _m(group, f"Menos de {line} {label}", 1 - p)]
    for who, key in ((home, "local"), (away, "visita")):
        for line, p in sm[key].items():
            out += [_m(group, f"{who}: más de {line} {label}", p), _m(group, f"{who}: menos de {line} {label}", 1 - p)]
    h, d, a = sm["1X2"]
    out += [_m(group, f"Más {label}: {home}", h), _m(group, f"Más {label}: igual", d),
            _m(group, f"Más {label}: {away}", a)]
    return out


def evaluate(markets: pd.DataFrame, odds_col: str = "Cuota", kelly_fraction: float = 0.25,
             cap: float = 0.05) -> pd.DataFrame:
    """Agrega valor esperado, cuota justa y stake Kelly (con devoluciones) a los mercados con cuota."""
    d = markets.copy()
    o = pd.to_numeric(d[odds_col], errors="coerce")
    d = d[o > 1].copy()
    o = o[o > 1]
    b = o - 1
    d["EV"] = d["p_gana"] * b - d["p_pierde"]
    # cuota justa: la que hace EV = 0
    with np.errstate(divide="ignore"):
        d["Cuota justa"] = np.where(d["p_gana"] > 0, 1 + d["p_pierde"] / d["p_gana"], np.inf)
    d["Kelly"] = np.clip((b * d["p_gana"] - d["p_pierde"]) / b * kelly_fraction, 0, cap)
    d["Prob. ganar"] = d["p_gana"]
    return d.sort_values("EV", ascending=False)


def pick_best(ev: pd.DataFrame, min_prob: float = 0.25, min_ev: float = 0.0) -> dict | None:
    """Mejor apuesta = mayor valor esperado entre las que tienen probabilidad razonable de acertar."""
    c = ev[(ev["p_gana"] >= min_prob) & (ev["EV"] > min_ev)]
    if c.empty:
        return None
    return c.iloc[0].to_dict()


def parlay(legs: list[dict]) -> dict:
    """Combinada de partidos distintos (se asumen independientes). Las devoluciones se tratan como cuota 1."""
    odds = float(np.prod([l["Cuota"] for l in legs]))
    p_all = float(np.prod([l["p_gana"] for l in legs]))
    # valor esperado exacto con devoluciones: cada pata devuelve cuota o 1 (push) o 0
    exp = float(np.prod([l["p_gana"] * l["Cuota"] + l["p_push"] for l in legs]))
    return {"Cuota combinada": odds, "Prob. acertar todo": p_all, "EV": exp - 1,
            "Cuota justa": 1 / p_all if p_all > 0 else np.inf}

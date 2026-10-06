"""Matemática de apuestas: margen de la casa, eliminación del margen (de-vig), valor esperado y Kelly."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def overround(odds) -> float:
    """Suma de probabilidades implícitas. 1.05 = margen de 5 %."""
    return float(np.sum(1.0 / np.asarray(odds, dtype=float)))


def devig(odds, method: str = "power") -> np.ndarray:
    """Probabilidades 'justas' a partir de cuotas con margen.

    - multiplicative: normaliza 1/cuota (asume margen proporcional).
    - power: p_i = (1/o_i)^k con k tal que sum = 1 (corrige el sesgo favorito-sorpresa, el que usan las casas).
    - shin: modelo de Shin (información privilegiada), muy usado en literatura académica.
    """
    o = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        return np.full(o.shape, np.nan)
    inv = 1.0 / o
    if method == "multiplicative":
        return inv / inv.sum()
    if method == "power":
        if abs(inv.sum() - 1.0) < 1e-9:
            return inv
        k = brentq(lambda k: np.sum(inv ** k) - 1.0, 0.2, 5.0)
        return inv ** k
    if method == "shin":
        s = inv.sum()
        def f(z):
            p = (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / s) - z) / (2 * (1 - z))
            return p.sum() - 1.0
        try:
            z = brentq(f, 0.0, 0.4)
        except ValueError:
            return inv / s
        return (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / s) - z) / (2 * (1 - z))
    raise ValueError(method)


def devig_matrix(odds: np.ndarray) -> np.ndarray:
    """De-vig multiplicativo vectorizado para una matriz (n, k) de cuotas."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def fair_odds(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(p > 0, 1.0 / p, np.inf)


def bookmaker_odds(p, margin: float = 0.05) -> np.ndarray:
    """Cuotas como las publicaría una casa: aplica margen con el método 'power' (más margen a las sorpresas)."""
    p = np.asarray(p, dtype=float)
    target = 1.0 + margin
    k = brentq(lambda k: np.sum(p ** k) - target, 0.3, 1.0)
    return 1.0 / p ** k


def expected_value(p, odds) -> np.ndarray:
    """Retorno esperado por unidad apostada. > 0 = apuesta de valor."""
    return np.asarray(p, dtype=float) * np.asarray(odds, dtype=float) - 1.0


def kelly(p, odds, fraction: float = 0.25, cap: float = 0.05) -> np.ndarray:
    """Fracción del bankroll según Kelly fraccional (con tope)."""
    p = np.asarray(p, dtype=float)
    b = np.asarray(odds, dtype=float) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (b * p - (1 - p)) / b
    return np.clip(np.nan_to_num(f) * fraction, 0.0, cap)

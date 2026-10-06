"""Backtest walk-forward: para cada temporada de prueba se entrena solo con datos anteriores,
Dixon-Coles se reajusta cada `dc_refit_days` días, y se simulan apuestas de valor con las cuotas reales."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dixon_coles import DixonColes, quick_markets
from .features import market_probs
from .models import GBMModel, blend


@dataclass
class BetRules:
    min_ev: float = 0.05          # valor esperado mínimo (5 %)
    min_prob: float = 0.15        # evita sorpresas extremas (alta varianza)
    max_odds: float = 8.0
    odds_source: str = "odds"     # 'odds' (promedio), 'max' (mejor cuota), 'b365'
    staking: str = "flat"         # 'flat' o 'kelly'
    kelly_fraction: float = 0.25
    markets: tuple = ("1X2",)     # '1X2' y/o 'O/U 2.5'


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))


def run_backtest(F: pd.DataFrame, feature_cols: list[str], leagues: list[str], seasons: list[int],
                 use_market: bool = True, w_gbm: float = 0.65, dc_refit_days: int = 14,
                 progress=None) -> pd.DataFrame:
    """Devuelve una fila por partido de prueba con probabilidades de cada modelo."""
    out = []
    for si, season in enumerate(seasons):
        test_all = F[F["league"].isin(leagues) & (F["season"] == season)].sort_values("date")
        if use_market:
            test_all = test_all.dropna(subset=["mkt_h"])
        if test_all.empty:
            continue
        if progress:
            progress(si / len(seasons), f"Temporada {season}/{season + 1}: entrenando Gradient Boosting…")
        gbm = GBMModel(use_market=use_market).fit(F, feature_cols, before=test_all["date"].min())
        for league, test in test_all.groupby("league", sort=False):
            g = gbm.predict(test)
            lg = F[F["league"] == league]
            dc_mk, dc, next_fit = [], None, None
            for r in test.itertuples(index=False):
                if dc is None or r.date >= next_fit:
                    dc = DixonColes().fit(lg, ref_date=r.date)
                    next_fit = r.date + pd.Timedelta(days=dc_refit_days)
                dc_mk.append(quick_markets(dc.score_matrix(r.home, r.away)))
            p = blend(g, dc_mk, w_gbm)

            res = test[["date", "season", "league", "home", "away", "fthg", "ftag", "y_res", "y_o25",
                        "odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a", "b365_h", "b365_d", "b365_a",
                        "close_h", "close_d", "close_a", "odds_o25", "odds_u25", "max_o25", "max_u25"]].copy()
            res[["gbm_h", "gbm_d", "gbm_a"]] = g["res"]
            res[["dc_h", "dc_d", "dc_a"]] = [[m["1"], m["X"], m["2"]] for m in dc_mk]
            res["gbm_o25"], res["dc_o25"] = g["o25"], [m["Más de 2.5"] for m in dc_mk]
            res[["p_h", "p_d", "p_a"]] = p["res"]
            res["p_o25"] = p["o25"]
            res[["mkt_h", "mkt_d", "mkt_a"]] = market_probs(test, "odds")
            res[["cls_h", "cls_d", "cls_a"]] = market_probs(test, "close")
            res["mkt_o25"] = test["mkt_o25"].to_numpy()
            out.append(res)
    if progress:
        progress(1.0, "Listo")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def stack_walkforward(oos: pd.DataFrame, leagues: list[str], seasons: list[int], w_gbm: float = 0.65) -> pd.DataFrame:
    """Aplica el stacker de forma honesta: para cada temporada se entrena solo con predicciones anteriores."""
    from .models import Stacker
    out = []
    for s in sorted(seasons):
        te = oos[oos["league"].isin(leagues) & (oos["season"] == s)].copy()
        if te.empty:
            continue
        tr = oos[oos["date"] < te["date"].min()]
        ens = w_gbm * te[["gbm_h", "gbm_d", "gbm_a"]].to_numpy() + (1 - w_gbm) * te[["dc_h", "dc_d", "dc_a"]].to_numpy()
        ens_o25 = w_gbm * te["gbm_o25"].to_numpy() + (1 - w_gbm) * te["dc_o25"].to_numpy()
        te[["ens_h", "ens_d", "ens_a"]] = ens
        if tr["mkt_h"].notna().sum() > 1000:
            res, o25 = Stacker().fit(tr).predict(te, ens, ens_o25)
        else:
            res, o25 = ens, ens_o25
        te[["p_h", "p_d", "p_a"]] = res
        te["p_o25"] = o25
        out.append(te)
    return pd.concat(out).sort_values("date", kind="stable").reset_index(drop=True) if out else pd.DataFrame()


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    y = bt["y_res"].to_numpy(int)
    rows = []
    for name, cols in [("Final (apilado con mercado)", "p"), ("Modelo propio (GBM + DC)", "ens"),
                       ("Gradient Boosting", "gbm"), ("Dixon-Coles", "dc"),
                       ("Mercado (apertura)", "mkt"), ("Mercado (cierre Pinnacle)", "cls")]:
        if f"{cols}_h" not in bt.columns:
            continue
        P = bt[[f"{cols}_h", f"{cols}_d", f"{cols}_a"]].to_numpy(float)
        ok = np.all(np.isfinite(P), axis=1)
        P, yy = P[ok], y[ok]
        rows.append({"modelo": name, "log loss": _log_loss(P, yy), "Brier": _brier(P, yy),
                     "acierto": float(np.mean(P.argmax(1) == yy)), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def simulate_bets(bt: pd.DataFrame, rules: BetRules, bankroll: float = 100.0) -> pd.DataFrame:
    """Simula apostar cuando el modelo encuentra valor. Devuelve el registro de apuestas."""
    src = rules.odds_source
    bets = []
    for r in bt.itertuples(index=False):
        cands = []
        if "1X2" in rules.markets:
            for k, lab, yv in (("h", "1", 0), ("d", "X", 1), ("a", "2", 2)):
                cands.append((lab, getattr(r, f"p_{k}"), getattr(r, f"{src}_{k}"), r.y_res == yv))
        if "O/U 2.5" in rules.markets and src in ("odds", "max"):
            cands.append(("Más 2.5", r.p_o25, getattr(r, f"{src}_o25"), r.y_o25 == 1))
            cands.append(("Menos 2.5", 1 - r.p_o25, getattr(r, f"{src}_u25"), r.y_o25 == 0))
        best = None
        for lab, p, o, won in cands:
            if not (np.isfinite(o) and o > 1) or p < rules.min_prob or o > rules.max_odds:
                continue
            ev = p * o - 1
            if ev >= rules.min_ev and (best is None or ev > best[3]):
                best = (lab, p, o, ev, won)
        if best:
            lab, p, o, ev, won = best
            bets.append({"fecha": r.date, "partido": f"{r.home} - {r.away}", "marcador": f"{int(r.fthg)}-{int(r.ftag)}",
                         "apuesta": lab, "prob": p, "cuota": o, "EV": ev, "ganada": bool(won)})
    b = pd.DataFrame(bets)
    if b.empty:
        return b
    bank, stakes, pnl = bankroll, [], []
    for r in b.itertuples(index=False):
        if rules.staking == "kelly":
            f = max(0.0, (r.prob * r.cuota - 1) / (r.cuota - 1)) * rules.kelly_fraction
            stake = bank * min(f, 0.05)
        else:
            stake = 1.0
        gain = stake * (r.cuota - 1) if r.ganada else -stake
        bank += gain
        stakes.append(stake)
        pnl.append(gain)
    b["stake"], b["ganancia"] = stakes, pnl
    b["acumulado"] = np.cumsum(pnl)
    return b


def bet_summary(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"apuestas": 0}
    return {"apuestas": len(b), "aciertos": float(b["ganada"].mean()), "cuota media": float(b["cuota"].mean()),
            "apostado": float(b["stake"].sum()), "ganancia": float(b["ganancia"].sum()),
            "ROI": float(b["ganancia"].sum() / b["stake"].sum()),
            "máx. caída": float((b["acumulado"].cummax() - b["acumulado"]).max())}

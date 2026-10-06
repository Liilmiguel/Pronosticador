"""Modelos predictivos: Gradient Boosting (sklearn) + Dixon-Coles, combinados en un ensamble."""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import DB_PATH, MODELS_DIR
from .dixon_coles import DixonColes, markets_from_matrix
from .features import FeatureBuilder
from .stats_markets import StatsModel

TARGETS = {"res": "y_res", "o25": "y_o25", "btts": "y_btts"}
MARKET_COLS = ["mkt_h", "mkt_d", "mkt_a", "mkt_o25"]
DEFAULT_W_GBM = 0.65  # peso del Gradient Boosting en el ensamble (el resto: Dixon-Coles)


def _gbm(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=400, max_leaf_nodes=24, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=seed)


@dataclass
class GBMModel:
    """Tres clasificadores (1X2, más de 2.5, ambos marcan) con o sin cuotas del mercado como variable."""
    use_market: bool = True
    half_life_years: float = 6.0
    features: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)

    def fit(self, F: pd.DataFrame, feature_cols: list[str], before=None) -> "GBMModel":
        d = F if before is None else F[F["date"] < pd.Timestamp(before)]
        self.features = feature_cols + (MARKET_COLS if self.use_market else [])
        if self.use_market:
            d = d.dropna(subset=["mkt_h"])
        age = (d["date"].max() - d["date"]).dt.days / 365.25
        w = 0.5 ** (age / self.half_life_years)  # los partidos recientes pesan más
        X = d[self.features].to_numpy(np.float32)
        for name, y in TARGETS.items():
            self.models[name] = _gbm().fit(X, d[y].to_numpy(), sample_weight=w.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        A = X.reindex(columns=self.features).to_numpy(np.float32)
        return {
            "res": self.models["res"].predict_proba(A),
            "o25": self.models["o25"].predict_proba(A)[:, 1],
            "btts": self.models["btts"].predict_proba(A)[:, 1],
        }

    def importances(self, F: pd.DataFrame, n: int = 4000) -> pd.Series:
        from sklearn.inspection import permutation_importance
        d = F.dropna(subset=["mkt_h"]).tail(n) if self.use_market else F.tail(n)
        r = permutation_importance(self.models["res"], d[self.features].to_numpy(np.float32), d["y_res"],
                                   scoring="neg_log_loss", n_repeats=3, random_state=0)
        return pd.Series(r.importances_mean, index=self.features).sort_values(ascending=False)


def blend(gbm_p: dict, dc_markets: list[dict], w_gbm: float = DEFAULT_W_GBM) -> dict[str, np.ndarray]:
    dc_res = np.array([[m["1"], m["X"], m["2"]] for m in dc_markets])
    dc_o25 = np.array([m["Más de 2.5"] for m in dc_markets])
    dc_btts = np.array([m["BTTS sí"] for m in dc_markets])
    return {
        "res": w_gbm * gbm_p["res"] + (1 - w_gbm) * dc_res,
        "o25": w_gbm * gbm_p["o25"] + (1 - w_gbm) * dc_o25,
        "btts": w_gbm * gbm_p["btts"] + (1 - w_gbm) * dc_btts,
    }


# --------------------------------------------------------------------------- caché en disco
def _data_key() -> str:
    st = DB_PATH.stat()
    return hashlib.md5(f"{st.st_mtime}-{st.st_size}".encode()).hexdigest()[:10]


def load_or_build(matches: pd.DataFrame, progress=None):
    """Devuelve (FeatureBuilder, F, GBMModel, StatsModel); reentrena solo si cambió la base de datos."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"bundle_v3_{_data_key()}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    for old in MODELS_DIR.glob("bundle_*.pkl"):
        old.unlink()
    if progress:
        progress(0.1, "Calculando variables (Elo, forma, H2H, liga)…")
    fb = FeatureBuilder()
    F = fb.build(matches)
    if progress:
        progress(0.5, "Entrenando Gradient Boosting (1X2, más/menos 2.5, ambos marcan)…")
    gbm = GBMModel(use_market=False).fit(F, fb.feature_cols)
    if progress:
        progress(0.75, "Entrenando modelos de córners y tarjetas…")
    stats = StatsModel().fit(F, fb.feature_cols)
    fb.h2h = dict(fb.h2h)  # defaultdict con lambda no se puede serializar
    F = F.astype({c: np.float32 for c in F.select_dtypes("float64").columns})
    bundle = (fb, F, gbm, stats)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return bundle


def load_or_build_national(progress=None):
    """Modelo de selecciones (caché en disco ligada a la base de datos)."""
    from .national import NationalModel, load
    from .national_stats import data_version
    stats_key = hashlib.md5(data_version().encode()).hexdigest()[:6]
    path = MODELS_DIR / f"national_{_data_key()}_{stats_key}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    df = load()
    if df.empty:
        return None
    for old in MODELS_DIR.glob("national_*.pkl"):
        old.unlink()
    model = NationalModel().build(df, progress=progress)
    from . import national_stats
    sb = national_stats.load()
    if sb.empty:
        try:
            sb = national_stats.download(progress=progress)
        except Exception:
            sb = pd.DataFrame()
    model.stats = national_stats.NationalStatsModel().build(sb, model) if not sb.empty else None
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return model


# --------------------------------------------------------------------------- stacking anclado al mercado
def _lr(p, q):
    return np.log(np.clip(np.asarray(p, float), 1e-6, 1) / np.clip(np.asarray(q, float), 1e-6, 1))


class Stacker:
    """Regresión logística que parte de la probabilidad del mercado (sin margen) y la corrige con las
    señales independientes del modelo (GBM sin cuotas + Dixon-Coles). Se entrena solo con predicciones
    fuera de muestra (walk-forward), así que aprende cuánto confiar en el modelo frente al mercado."""

    def __init__(self, C: float = 0.1):
        self.C = C
        self.res = None
        self.o25 = None

    @staticmethod
    def x_res(d) -> np.ndarray:
        return np.column_stack([_lr(d["mkt_h"], d["mkt_d"]), _lr(d["mkt_a"], d["mkt_d"]),
                                _lr(d["gbm_h"], d["gbm_d"]), _lr(d["gbm_a"], d["gbm_d"]),
                                _lr(d["dc_h"], d["dc_d"]), _lr(d["dc_a"], d["dc_d"])])

    @staticmethod
    def x_o25(d) -> np.ndarray:
        return np.column_stack([_lr(d["mkt_o25"], 1 - np.asarray(d["mkt_o25"], float)),
                                _lr(d["gbm_o25"], 1 - np.asarray(d["gbm_o25"], float)),
                                _lr(d["dc_o25"], 1 - np.asarray(d["dc_o25"], float))])

    def fit(self, oos: pd.DataFrame) -> "Stacker":
        from sklearn.linear_model import LogisticRegression
        d = oos.dropna(subset=["mkt_h", "mkt_d", "mkt_a"])
        self.res = LogisticRegression(C=self.C, max_iter=2000).fit(self.x_res(d), d["y_res"])
        d = oos.dropna(subset=["mkt_o25"])
        if len(d) > 500:
            self.o25 = LogisticRegression(C=self.C, max_iter=2000).fit(self.x_o25(d), d["y_o25"])
        return self

    def predict(self, d: pd.DataFrame, fallback_res: np.ndarray, fallback_o25: np.ndarray):
        """Probabilidades apiladas donde hay cuotas; el ensamble del modelo donde no."""
        res, o25 = np.array(fallback_res, float), np.array(fallback_o25, float)
        ok = d[["mkt_h", "mkt_d", "mkt_a"]].notna().all(axis=1).to_numpy()
        if self.res is not None and ok.any():
            res[ok] = self.res.predict_proba(self.x_res(d[ok]))
        ok = d["mkt_o25"].notna().to_numpy()
        if self.o25 is not None and ok.any():
            o25[ok] = self.o25.predict_proba(self.x_o25(d[ok]))[:, 1]
        return res, o25


OOS_PATH = MODELS_DIR / "oos.pkl"


def build_oos(F: pd.DataFrame, feature_cols: list[str], seasons: list[int], progress=None) -> pd.DataFrame:
    """Predicciones fuera de muestra (walk-forward) de todas las ligas: base del stacker y del backtest."""
    from .backtest import run_backtest
    oos = run_backtest(F, feature_cols, sorted(F["league"].unique()), seasons, use_market=False, progress=progress)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stacker = Stacker().fit(oos)
    with open(OOS_PATH, "wb") as fh:
        pickle.dump((oos, stacker), fh)
    return oos


def load_oos():
    if not OOS_PATH.exists():
        return None, None
    with open(OOS_PATH, "rb") as fh:
        return pickle.load(fh)


def fit_dc(F: pd.DataFrame, league: str, ref_date=None) -> DixonColes:
    return DixonColes().fit(F[F["league"] == league], ref_date=ref_date)


def predict_fixtures(fb: FeatureBuilder, F: pd.DataFrame, gbm: GBMModel, fixtures: pd.DataFrame,
                     w_gbm: float = DEFAULT_W_GBM, dc_cache: dict | None = None,
                     stacker: Stacker | None = None) -> pd.DataFrame:
    """Predicciones completas para partidos futuros.

    ens_* = modelo propio (GBM + Dixon-Coles, sin mirar cuotas); p_* = final (apilado con el mercado si hay cuotas).
    """
    if fixtures.empty:
        return fixtures
    X = fb.features_for(fixtures)
    g = gbm.predict(X)
    dc_cache = {} if dc_cache is None else dc_cache
    dc_mk = []
    for r in X.itertuples(index=False):
        if r.league not in dc_cache:
            dc_cache[r.league] = fit_dc(F, r.league)
        dc_mk.append(markets_from_matrix(dc_cache[r.league].score_matrix(r.home, r.away)))
    ens = blend(g, dc_mk, w_gbm)
    out = X.copy()
    out[["gbm_h", "gbm_d", "gbm_a"]] = g["res"]
    out[["dc_h", "dc_d", "dc_a"]] = [[m["1"], m["X"], m["2"]] for m in dc_mk]
    out["gbm_o25"], out["dc_o25"] = g["o25"], [m["Más de 2.5"] for m in dc_mk]
    out[["ens_h", "ens_d", "ens_a"]] = ens["res"]
    out["ens_o25"] = ens["o25"]
    if stacker is not None:
        res, o25 = stacker.predict(out, ens["res"], ens["o25"])
    else:
        res, o25 = ens["res"], ens["o25"]
    out[["p_h", "p_d", "p_a"]] = res
    out["p_o25"], out["p_btts"] = o25, ens["btts"]
    out["xg_h"] = [m["xG local"] for m in dc_mk]
    out["xg_a"] = [m["xG visita"] for m in dc_mk]
    out["dc_markets"] = dc_mk
    return out

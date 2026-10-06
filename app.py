"""Predictor de apuestas deportivas (fútbol) — interfaz Streamlit.

Ejecutar:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predictor import betting
from predictor.backtest import BetRules, bet_summary, metrics, simulate_bets, stack_walkforward
from predictor.config import ALL_LEAGUES, CURRENT_SEASON, DB_PATH, league_label
from predictor.dixon_coles import asian_fair_odds
from predictor.models import (DEFAULT_W_GBM, build_oos, fit_dc, load_oos, load_or_build, load_or_build_national,
                              predict_fixtures)
from predictor import apifootball
from predictor import best_bet
from predictor.national import LEVEL_NAMES
from predictor.national_stats import STATS as NAT_STATS
from predictor.stats_markets import stat_markets
from predictor.scraper import last_update, load_table, scrape_all

st.set_page_config(page_title="Predictor de Apuestas", page_icon="⚽", layout="wide")

OUT_LABELS = {"h": "1 (local)", "d": "X (empate)", "a": "2 (visita)"}


def ev_color(v):
    """Verde si hay valor, rojo si no; vacío si no hay cuota."""
    if not isinstance(v, (int, float)) or not np.isfinite(v):
        return ""
    a = min(abs(v) / 0.15, 1.0) * 0.55
    return f"background-color: rgba({'34,160,90' if v > 0 else '210,60,60'},{a:.2f})"


# =========================================================================== datos y modelos
@st.cache_data(show_spinner=False)
def get_tables(_key):
    return load_table("matches"), load_table("fixtures")


@st.cache_resource(show_spinner=False)
def get_bundle(_key):
    matches, _ = get_tables(_key)
    bar = st.progress(0.0, "Preparando modelos (solo la primera vez tras actualizar datos)…")
    bundle = load_or_build(matches, progress=lambda p, m: bar.progress(p, m))
    bar.empty()
    return bundle


@st.cache_resource(show_spinner=False)
def get_dc(_key, league):
    _, F, _, _ = get_bundle(_key)
    return fit_dc(F, league)


@st.cache_resource(show_spinner=False)
def get_national(_key):
    bar = st.progress(0.0, "Preparando modelo de selecciones…")
    m = load_or_build_national(progress=lambda p, msg: bar.progress(p, msg))
    bar.empty()
    return m


@st.cache_resource(show_spinner=False)
def get_oos(_mtime):
    return load_oos()


def oos_mtime():
    from predictor.models import OOS_PATH
    return OOS_PATH.stat().st_mtime if OOS_PATH.exists() else None


def data_key():
    return DB_PATH.stat().st_mtime if DB_PATH.exists() else None


def run_scrape(first_season):
    bar = st.progress(0.0, "Iniciando descarga…")
    summary = scrape_all(first_season=first_season, progress=lambda p, m: bar.progress(min(p, 1.0), m))
    bar.empty()
    st.cache_data.clear()
    st.cache_resource.clear()
    return summary


# =========================================================================== barra lateral
with st.sidebar:
    st.title("⚽ Predictor de Apuestas")
    upd = last_update()
    st.caption(f"Datos: football-data.co.uk · última actualización: **{upd or 'nunca'}**")
    with st.expander("🔄 Actualizar datos (scraping)", expanded=upd is None):
        first = st.number_input("Desde temporada", 2000, CURRENT_SEASON, 2000, help="Año de inicio (2000 = 2000/01)")
        st.caption("Descarga ~600 archivos (38 ligas) + resultados de selecciones nacionales. Las temporadas "
                   "cerradas quedan en caché local, así que actualizar después tarda segundos.")
        if st.button("Descargar / actualizar", type="primary", width="stretch"):
            s = run_scrape(int(first))
            st.success(f"{s['partidos']:,} partidos · {s['ligas']} ligas · {s['proximos']} próximos · "
                       f"{s.get('selecciones', 0):,} partidos de selecciones")
            st.rerun()
        st.caption("El **histórico fuera de muestra** alimenta el modelo apilado con el mercado y el backtest. "
                   "No es necesario regenerarlo en cada actualización (~15 min).")
        if st.button("Regenerar histórico fuera de muestra", width="stretch"):
            _fb, _F, _, _ = get_bundle(data_key())
            bar = st.progress(0.0)
            build_oos(_F, _fb.feature_cols, list(range(2019, CURRENT_SEASON + 1)),
                      progress=lambda p, m: bar.progress(p, m))
            bar.empty()
            st.cache_resource.clear()
            st.rerun()

    with st.expander("🔑 Stats de selecciones (API-Football)"):
        st.caption("Agrega córners, tarjetas y remates de **eliminatorias, amistosos, Nations League, Mundial 2026** "
                   "y las últimas fechas FIFA. Necesita una clave gratuita de "
                   "[API-Football](https://dashboard.api-football.com/register) (100 solicitudes/día): cada "
                   "descarga avanza ~95 partidos y retoma donde quedó.")
        has_key = apifootball.get_key() is not None
        new_key = st.text_input("Clave API", type="password", placeholder="guardada ✓" if has_key else "pega tu clave")
        if st.button("Guardar clave") and new_key:
            apifootball.save_key(new_key)
            st.rerun()
        if has_key and st.button("Descargar estadísticas", type="primary", width="stretch"):
            from predictor.national import load as load_intl
            intl = load_intl()
            bar = st.progress(0.0, "Conectando con API-Football…")
            try:
                res = apifootball.sync(set(intl["home"]) | set(intl["away"]),
                                       progress=lambda p, m: bar.progress(p, m))
                st.session_state["apif_res"] = res
            except apifootball.ApiFootballError as e:
                st.session_state["apif_res"] = {"mensaje": str(e)}
            bar.empty()
            st.cache_resource.clear()
            st.rerun()
        res = st.session_state.get("apif_res")
        if res:
            msg = res.get("mensaje", "ok")
            if msg not in ("ok", "Presupuesto de solicitudes de hoy agotado"):
                st.error(f"API-Football respondió: {msg}")
            if res.get("partidos_nuevos") or msg in ("ok", "Presupuesto de solicitudes de hoy agotado"):
                st.success(f"{res.get('partidos_nuevos', 0)} partidos nuevos · {res.get('pendientes', 0)} pendientes · "
                           f"{res.get('solicitudes', 0)} solicitudes usadas")
            if msg == "Presupuesto de solicitudes de hoy agotado":
                st.caption("Límite diario alcanzado: vuelve a pulsar mañana para seguir.")
            if st.button("Borrar clave"):
                apifootball.KEY_FILE.unlink(missing_ok=True)
                st.session_state.pop("apif_res", None)
                st.rerun()
        n_api = len(apifootball.load()) // 2
        st.caption(f"Partidos con stats desde API-Football: **{n_api}**")

    st.subheader("Modelo")
    w_gbm = st.slider("Peso Gradient Boosting vs Dixon-Coles", 0.0, 1.0, DEFAULT_W_GBM, 0.05,
                      help="El resto del peso va al modelo Dixon-Coles (Poisson con fuerzas de ataque/defensa).")
    use_stack = st.toggle("Anclar al mercado cuando hay cuotas", value=True,
                          help="Modelo apilado (regresión logística entrenada fuera de muestra) que parte de la "
                               "probabilidad sin margen del mercado y la corrige con el modelo propio. "
                               "Desactivado: solo el modelo propio (GBM + Dixon-Coles), que ignora las cuotas.")
    st.subheader("Apuestas")
    odds_src = st.selectbox("Cuotas para buscar valor", ["max", "odds", "b365"], index=0,
                            format_func={"max": "Mejor cuota del mercado", "odds": "Promedio del mercado",
                                         "b365": "Bet365"}.get)
    min_ev = st.slider("Valor esperado mínimo", 0.0, 0.30, 0.05, 0.01, format="%.2f")
    bankroll = st.number_input("Bankroll", 10.0, 1e7, 100000.0, 1000.0)
    kfrac = st.slider("Fracción de Kelly", 0.05, 1.0, 0.25, 0.05)
    devig_method = st.selectbox("Método para quitar el margen", ["power", "shin", "multiplicative"])

if not DB_PATH.exists():
    st.info("👈 Primero descarga los datos históricos desde la barra lateral.")
    st.stop()

KEY = data_key()
matches, fixtures = get_tables(KEY)
fb, F, gbm, stats = get_bundle(KEY)
oos, stacker = get_oos(oos_mtime())
if stacker is None:
    st.sidebar.warning("Sin histórico fuera de muestra: se usa solo el modelo propio. Genéralo con "
                       "`entrenar.py` o desde 'Actualizar datos'.")


def predict(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    dc_cache = {lg: get_dc(KEY, lg) for lg in rows["league"].unique()}
    df = predict_fixtures(fb, F, gbm, rows, w_gbm=w_gbm, dc_cache=dc_cache,
                          stacker=stacker if use_stack else None)
    df[["f_h", "f_d", "f_a"]] = df[["p_h", "p_d", "p_a"]].to_numpy()
    df["f_o25"] = df["p_o25"]
    df[list(stats.models)] = stats.expected(df).to_numpy()
    return df


def stat_table(mk: dict, kind: str) -> pd.DataFrame:
    """Tabla de más/menos (total y por equipo) con probabilidades y cuotas justas."""
    rows = []
    for scope, lab in (("total", "Total"), ("local", "Local"), ("visita", "Visita")):
        for line, p in mk[scope].items():
            rows += [{"Mercado": f"{lab} más de {line} {kind}", "Probabilidad": p, "Cuota justa": 1 / p if p > 0 else np.inf},
                     {"Mercado": f"{lab} menos de {line} {kind}", "Probabilidad": 1 - p,
                      "Cuota justa": 1 / (1 - p) if p < 1 else np.inf}]
    for lab, p in zip(("Local", "Igual", "Visita"), mk["1X2"]):
        rows.append({"Mercado": f"Más {kind}: {lab}", "Probabilidad": p, "Cuota justa": 1 / p if p > 0 else np.inf})
    return pd.DataFrame(rows)


def editable_ev(df: pd.DataFrame, key: str):
    """Tabla donde el usuario escribe sus cuotas; devuelve EV y stake Kelly."""
    df = df.assign(**{"Tu cuota": np.nan})
    ed = st.data_editor(df, key=key, hide_index=True, width="stretch", height=420,
                        disabled=["Mercado", "Probabilidad", "Cuota justa"],
                        column_config={"Probabilidad": st.column_config.NumberColumn(format="%.3f"),
                                       "Cuota justa": st.column_config.NumberColumn(format="%.2f"),
                                       "Tu cuota": st.column_config.NumberColumn(min_value=1.0, step=0.01, format="%.2f")})
    v = ed.dropna(subset=["Tu cuota"])
    v = v[v["Tu cuota"] > 1]
    if not v.empty:
        v = v.assign(EV=v["Probabilidad"] * v["Tu cuota"] - 1,
                     **{"Stake Kelly": [float(betting.kelly(p, o, kfrac) * bankroll)
                                        for p, o in zip(v["Probabilidad"], v["Tu cuota"])]})
        st.dataframe(v.style.format({"Probabilidad": "{:.1%}", "Cuota justa": "{:.2f}", "Tu cuota": "{:.2f}",
                                     "EV": "{:+.1%}", "Stake Kelly": "{:,.0f}"}).map(ev_color, subset=["EV"]),
                     hide_index=True, width="stretch")


def league_teams(league: str) -> list[str]:
    d = matches[matches["league"] == league]
    last = d["season"].max()
    return sorted(set(d.loc[d["season"] == last, "home"]) | set(d.loc[d["season"] == last, "away"]))


def team_last_matches(league, team, n=6):
    country = ALL_LEAGUES[league][1]
    d = matches[(matches["country"] == country) & ((matches["home"] == team) | (matches["away"] == team))]
    d = d.dropna(subset=["fthg"]).tail(n)
    res = []
    for r in d.itertuples():
        gf, ga = (r.fthg, r.ftag) if r.home == team else (r.ftag, r.fthg)
        res.append("🟢" if gf > ga else "⚪" if gf == ga else "🔴")
    return d[["date", "league", "home", "fthg", "ftag", "away"]], "".join(res)


tabs = st.tabs(["📅 Próximos partidos", "🎯 Analizar partido", "💰 Mejor apuesta", "🌍 Selecciones",
                "📊 Equipos y ligas", "🧪 Backtest", "🗄️ Datos y método"])

# =========================================================================== 1. próximos partidos
with tabs[0]:
    fx = fixtures.copy()
    if not fx.empty:
        played = matches.set_index(["league", "home", "away"]).index
        fx = fx[~fx.set_index(["league", "home", "away"]).index.isin(played)]
        fx = fx[fx["date"] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
    if fx.empty:
        st.info("No hay partidos próximos publicados con cuotas en este momento (football-data publica la "
                "jornada siguiente normalmente entre martes y viernes). Usa **Analizar partido** para cualquier "
                "enfrentamiento, o actualiza los datos más tarde.")
    else:
        ligas = sorted(fx["league"].unique(), key=lambda c: league_label(c))
        sel = st.multiselect("Ligas", ligas, default=ligas, format_func=league_label)
        fx = fx[fx["league"].isin(sel)]
        if fx.empty:
            st.info("Selecciona al menos una liga.")
    if not fx.empty:
        P = predict(fx)
        rows = []
        for r in P.itertuples(index=False):
            best = None
            for k in "hda":
                p, o = getattr(r, f"f_{k}"), getattr(r, f"{odds_src}_{k}", np.nan)
                if np.isfinite(o) and o > 1:
                    ev = p * o - 1
                    if best is None or ev > best[2]:
                        best = (k, o, ev, p)
            pick = OUT_LABELS[best[0]] if best and best[2] >= min_ev else "—"
            stake = float(betting.kelly(best[3], best[1], kfrac) * bankroll) if best and best[2] >= min_ev else 0.0
            rows.append({
                "Fecha": r.date.strftime("%d/%m"), "Hora": r.time if isinstance(r.time, str) else "",
                "Liga": r.league, "Partido": f"{r.home} vs {r.away}",
                "P(1)": r.f_h, "P(X)": r.f_d, "P(2)": r.f_a,
                "Cuota 1": getattr(r, f"{odds_src}_h"), "Cuota X": getattr(r, f"{odds_src}_d"),
                "Cuota 2": getattr(r, f"{odds_src}_a"),
                "Justa 1": 1 / r.f_h, "Justa X": 1 / r.f_d, "Justa 2": 1 / r.f_a,
                "xG": f"{r.xg_h:.2f} - {r.xg_a:.2f}", "P(+2.5)": r.f_o25, "P(BTTS)": r.p_btts,
                "Córners esp.": r.cor_h + r.cor_a, "Tarjetas esp.": r.crd_h + r.crd_a,
                "Valor": pick, "EV": best[2] if best else np.nan, "Stake Kelly": stake,
            })
        tbl = pd.DataFrame(rows)
        only_value = st.toggle("Mostrar solo apuestas con valor", value=False)
        if only_value:
            tbl = tbl[tbl["Valor"] != "—"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos", len(P))
        c2.metric("Con valor (EV ≥ {:.0%})".format(min_ev), int((pd.DataFrame(rows)["Valor"] != "—").sum()))
        c3.metric("Ligas", len(sel))
        st.dataframe(
            tbl.style.format({c: "{:.1%}" for c in ["P(1)", "P(X)", "P(2)", "P(+2.5)", "P(BTTS)", "EV"]}
                             | {c: "{:.2f}" for c in ["Cuota 1", "Cuota X", "Cuota 2", "Justa 1", "Justa X", "Justa 2"]}
                             | {"Stake Kelly": "{:,.0f}", "Córners esp.": "{:.1f}", "Tarjetas esp.": "{:.1f}"}, na_rep="—")
            .map(ev_color, subset=["EV"]),
            width="stretch", hide_index=True, height=min(700, 40 + 35 * len(tbl)))
        st.caption("P = probabilidad final del modelo · Justa = cuota sin margen según el modelo · "
                   "EV = valor esperado de la mejor opción con las cuotas elegidas · Stake = Kelly fraccional.")

# =========================================================================== 2. analizar partido
with tabs[1]:
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    leagues_avail = sorted(matches["league"].unique(), key=league_label)
    league = c1.selectbox("Liga", leagues_avail, index=leagues_avail.index("E0") if "E0" in leagues_avail else 0,
                          format_func=league_label)
    teams = league_teams(league)
    home = c2.selectbox("Local", teams, index=0)
    away = c3.selectbox("Visita", teams, index=1 if len(teams) > 1 else 0)
    refs = fb.referees(ALL_LEAGUES[league][1])
    referee = c4.selectbox("Árbitro (opcional)", ["(desconocido)"] + refs,
                           help="Influye en la predicción de tarjetas. Solo ligas cuyo CSV trae árbitro.")

    st.markdown("**Cuotas de tu casa de apuestas** (opcional; se usan como variable del modelo y para calcular el valor)")
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    q_h = o1.number_input("1", 0.0, 100.0, 0.0, 0.05, format="%.2f")
    q_d = o2.number_input("X", 0.0, 100.0, 0.0, 0.05, format="%.2f")
    q_a = o3.number_input("2", 0.0, 100.0, 0.0, 0.05, format="%.2f")
    q_o = o4.number_input("Más de 2.5", 0.0, 100.0, 0.0, 0.05, format="%.2f")
    q_u = o5.number_input("Menos de 2.5", 0.0, 100.0, 0.0, 0.05, format="%.2f")
    margin = o6.number_input("Margen casa simulada", 0.0, 0.2, 0.05, 0.01, format="%.2f",
                             help="Para mostrar qué cuotas publicaría una casa con este margen")

    if home == away:
        st.warning("Elige dos equipos distintos.")
    else:
        name, country, tier = ALL_LEAGUES[league]
        nz = lambda v: v if v > 1 else np.nan
        row = pd.DataFrame([{
            "league": league, "country": country, "tier": tier,
            "season": int(matches.loc[matches["league"] == league, "season"].max()),
            "date": pd.Timestamp.today().normalize(), "time": "", "home": home, "away": away,
            "referee": "" if referee == "(desconocido)" else referee,
            "odds_h": nz(q_h), "odds_d": nz(q_d), "odds_a": nz(q_a), "odds_o25": nz(q_o), "odds_u25": nz(q_u),
        }])
        if np.isfinite(row[["odds_h", "odds_d", "odds_a"]].to_numpy(float)).all():
            mp = betting.devig([q_h, q_d, q_a], devig_method)
        else:
            mp = None
        r = predict(row).iloc[0]
        mk = r["dc_markets"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"Gana {home}", f"{r.f_h:.1%}", f"cuota justa {1 / r.f_h:.2f}", delta_color="off")
        m2.metric("Empate", f"{r.f_d:.1%}", f"cuota justa {1 / r.f_d:.2f}", delta_color="off")
        m3.metric(f"Gana {away}", f"{r.f_a:.1%}", f"cuota justa {1 / r.f_a:.2f}", delta_color="off")
        m4.metric("Goles esperados", f"{r.xg_h:.2f} – {r.xg_a:.2f}")
        m5.metric("Elo", f"{r.h_elo:.0f} – {r.a_elo:.0f}", f"{r.elo_diff:+.0f} con localía", delta_color="off")
        if mp is not None:
            st.caption(f"Margen de tu casa: **{betting.overround([q_h, q_d, q_a]) - 1:.1%}** · "
                       + ("probabilidad final = modelo apilado con el mercado" if use_stack and stacker else
                          "probabilidad final = modelo propio"))
        else:
            st.caption("Sin cuotas ingresadas: probabilidad final = modelo propio (GBM + Dixon-Coles).")

        left, right = st.columns([3, 2])
        with left:
            comp = pd.DataFrame({
                "Resultado": ["1", "X", "2"] * 5,
                "Probabilidad": [r.f_h, r.f_d, r.f_a, r.ens_h, r.ens_d, r.ens_a, r.gbm_h, r.gbm_d, r.gbm_a,
                                 mk["1"], mk["X"], mk["2"], *(mp if mp is not None else [np.nan] * 3)],
                "Fuente": ["Final"] * 3 + ["Modelo propio"] * 3 + ["Gradient Boosting"] * 3 + ["Dixon-Coles"] * 3
                          + ["Mercado (sin margen)"] * 3,
            }).dropna()
            fig = px.bar(comp, x="Resultado", y="Probabilidad", color="Fuente", barmode="group", text_auto=".1%",
                         height=330)
            fig.update_layout(yaxis_tickformat=".0%", margin=dict(t=10, b=10), legend_title=None,
                              xaxis_type="category", legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, width="stretch")

            # tabla de mercados
            book = betting.bookmaker_odds([r.f_h, r.f_d, r.f_a], margin)
            user = {"1": q_h, "X": q_d, "2": q_a, "Más de 2.5": q_o, "Menos de 2.5": q_u}
            mkts = [("1", r.f_h, book[0]), ("X", r.f_d, book[1]), ("2", r.f_a, book[2]),
                    ("1X", r.f_h + r.f_d, None), ("12", r.f_h + r.f_a, None), ("X2", r.f_d + r.f_a, None)]
            for line in (0.5, 1.5, 2.5, 3.5, 4.5):
                po = r.f_o25 if line == 2.5 else mk[f"Más de {line}"]
                bo = betting.bookmaker_odds([po, 1 - po], margin)
                mkts += [(f"Más de {line}", po, bo[0]), (f"Menos de {line}", 1 - po, bo[1])]
            pb = r.p_btts
            bo = betting.bookmaker_odds([pb, 1 - pb], margin)
            mkts += [("Ambos marcan: sí", pb, bo[0]), ("Ambos marcan: no", 1 - pb, bo[1])]
            t = pd.DataFrame([{
                "Mercado": n, "Probabilidad": p, "Cuota justa": 1 / p if p > 0 else np.inf,
                f"Cuota casa ({margin:.0%} margen)": b, "Tu cuota": user.get(n) or np.nan,
                "EV": p * user[n] - 1 if user.get(n, 0) > 1 else np.nan,
                "Stake Kelly": float(betting.kelly(p, user[n], kfrac) * bankroll) if user.get(n, 0) > 1 else np.nan,
            } for n, p, b in mkts])
            t = t.astype({c: float for c in t.columns[1:]})
            fmts = {"Probabilidad": "{:.1%}", "Cuota justa": "{:.2f}", f"Cuota casa ({margin:.0%} margen)": "{:.2f}",
                    "Tu cuota": "{:.2f}", "EV": "{:+.1%}", "Stake Kelly": "{:,.0f}"}
            txt = t.copy()
            for c, f in fmts.items():  # texto ya formateado: Streamlit muestra 'None' en celdas vacías
                txt[c] = [f.format(v) if np.isfinite(v) else "" for v in t[c]]
            colors = pd.DataFrame("", index=t.index, columns=t.columns)
            colors["EV"] = t["EV"].map(ev_color)
            st.dataframe(txt.style.apply(lambda _: colors, axis=None),
                         width="stretch", hide_index=True, height=600)
        with right:
            st.markdown("**Marcador exacto (Dixon-Coles)**")
            dcm = get_dc(KEY, league).score_matrix(home, away)[:6, :6]
            hm = go.Figure(go.Heatmap(z=dcm, x=[str(i) for i in range(6)], y=[str(i) for i in range(6)],
                                      text=[[f"{v:.1%}" for v in row_] for row_ in dcm], texttemplate="%{text}",
                                      colorscale="Greens", showscale=False))
            hm.update_layout(xaxis_title=f"Goles {away}", yaxis_title=f"Goles {home}", height=330,
                             margin=dict(t=10, b=10), yaxis_autorange="reversed", xaxis_type="category",
                             yaxis_type="category")
            st.plotly_chart(hm, width="stretch")
            st.markdown("**Hándicap asiático (local)**")
            ah = pd.DataFrame([{"Línea": f"{k:+.2f}", "Gana": v["gana"], "Devolución": v["push"],
                                "Cuota justa": asian_fair_odds(v)} for k, v in mk["asian"].items()
                               if -2.5 <= k <= 2.5 and (k * 4) % 2 == 0])
            st.dataframe(ah.style.format({"Gana": "{:.1%}", "Devolución": "{:.1%}", "Cuota justa": "{:.2f}"}),
                         hide_index=True, width="stretch", height=250)

        # ---------------------------------------------------------------- córners y tarjetas
        st.subheader("🚩 Córners y 🟨 tarjetas")
        has_stats = np.isfinite([r.h_cf_l, r.a_cf_l]).all()
        if not has_stats:
            st.info("Esta liga no publica córners/tarjetas por partido: las cifras se basan solo en el contexto "
                    "general y son poco fiables.")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Córners esperados", f"{r.cor_h + r.cor_a:.1f}", f"{r.cor_h:.1f} – {r.cor_a:.1f}", delta_color="off")
        s2.metric("Tarjetas esperadas", f"{r.crd_h + r.crd_a:.1f}", f"{r.crd_h:.1f} – {r.crd_a:.1f}", delta_color="off")
        s3.metric("Árbitro: tarjetas/partido", f"{r.ref_cards:.1f}" if np.isfinite(r.ref_cards) else "—",
                  f"{int(r.ref_n)} partidos en la base" if r.ref_n else "sin historial", delta_color="off")
        s4.metric("Promedio de la liga", f"{r.lg_cor:.1f} córn. · {r.lg_crd:.1f} tarj." if np.isfinite(r.lg_cor) else "—")
        cor_mk = stat_markets(r.cor_h, r.cor_a, stats.alpha["cor_h"], stats.alpha["cor_a"],
                              (7.5, 8.5, 9.5, 10.5, 11.5, 12.5), (2.5, 3.5, 4.5, 5.5, 6.5))
        crd_mk = stat_markets(r.crd_h, r.crd_a, stats.alpha["crd_h"], stats.alpha["crd_a"],
                              (2.5, 3.5, 4.5, 5.5, 6.5), (0.5, 1.5, 2.5, 3.5))
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Córners** · escribe tu cuota para ver el valor")
            editable_ev(stat_table(cor_mk, "córners"), "ev_cor")
        with k2:
            st.markdown("**Tarjetas** (amarilla = 1, roja = 2) · escribe tu cuota")
            editable_ev(stat_table(crd_mk, "tarjetas"), "ev_crd")
        dist = pd.DataFrame({"n": np.arange(25), "Córners": cor_mk["dist_total"][:25],
                             "Tarjetas": crd_mk["dist_total"][:25]}).melt("n", var_name="Mercado", value_name="Prob")
        fig = px.bar(dist, x="n", y="Prob", color="Mercado", barmode="group", height=280,
                     labels={"n": "Total en el partido", "Prob": "Probabilidad"})
        fig.update_layout(yaxis_tickformat=".0%", margin=dict(t=10, b=10), legend_title=None)
        st.plotly_chart(fig, width="stretch")
        stat_rows = {"Córners a favor": "cf", "Córners en contra": "ca", "Tarjetas propias": "kf",
                     "Tarjetas del rival": "ka", "Faltas cometidas": "ff", "Faltas recibidas": "fa",
                     "Tiros": "sf", "Tiros a puerta": "sotf", "Goles a favor": "gf", "Goles en contra": "ga"}
        st.markdown("**Promedios recientes** (media exponencial, últimos ~10 partidos)")
        st.dataframe(pd.DataFrame({home: [r[f"h_{v}_l"] for v in stat_rows.values()],
                                   away: [r[f"a_{v}_l"] for v in stat_rows.values()]}, index=list(stat_rows))
                     .style.format("{:.2f}", na_rep="—"), width="stretch")

        f1, f2, f3 = st.columns(3)
        for col, team in ((f1, home), (f2, away)):
            lm, form = team_last_matches(league, team)
            col.markdown(f"**{team}** · forma {form}")
            col.dataframe(lm.assign(date=lm["date"].dt.strftime("%d/%m/%y")).rename(
                columns={"date": "Fecha", "league": "Liga", "home": "Local", "fthg": "", "ftag": " ", "away": "Visita"}),
                hide_index=True, width="stretch")
        h2h = matches[(matches["country"] == country) & (((matches["home"] == home) & (matches["away"] == away)) |
                                                         ((matches["home"] == away) & (matches["away"] == home)))].tail(8)
        f3.markdown(f"**Enfrentamientos directos** ({len(h2h)})")
        f3.dataframe(h2h[["date", "home", "fthg", "ftag", "away"]].assign(date=h2h["date"].dt.strftime("%d/%m/%y")),
                     hide_index=True, width="stretch")
        with st.expander("Variables usadas por el modelo para este partido"):
            show = {c: r[c] for c in fb.feature_cols + ["mkt_h", "mkt_d", "mkt_a", "mkt_o25"] if c in r.index}
            st.dataframe(pd.Series(show, name="valor").to_frame(), width="stretch")

# =========================================================================== 2b. mejor apuesta (cuotas de Betano)
with tabs[2]:
    st.markdown("Copia en la tabla las cuotas que ves en **Betano** (o en cualquier casa) para este partido — "
                "todas las que quieras, el resto déjalas vacías. El modelo calcula la probabilidad de cada mercado, "
                "el **valor esperado** y te recomienda la mejor apuesta y cuánto apostar.")
    kind = st.radio("Tipo de partido", ["Clubes", "Selecciones"], horizontal=True, key="bb_kind")
    if kind == "Clubes":
        b1, b2, b3, b4 = st.columns([3, 2, 2, 2])
        bl = b1.selectbox("Liga", leagues_avail, index=leagues_avail.index("E0") if "E0" in leagues_avail else 0,
                          format_func=league_label, key="bb_lg")
        bteams = league_teams(bl)
        bh = b2.selectbox("Local", bteams, index=0, key="bb_h")
        ba = b3.selectbox("Visita", bteams, index=1 if len(bteams) > 1 else 0, key="bb_a")
        bref = b4.selectbox("Árbitro (opcional)", ["(desconocido)"] + fb.referees(ALL_LEAGUES[bl][1]), key="bb_ref")
    else:
        nat_b = get_national(KEY)
        nteams_b = nat_b.teams()
        b1, b2, b3, b4 = st.columns([2, 2, 2, 1])
        bh = b1.selectbox("Selección local", nteams_b, index=nteams_b.index("Chile") if "Chile" in nteams_b else 0,
                          key="bb_nh")
        ba = b2.selectbox("Selección visita", nteams_b,
                          index=nteams_b.index("Argentina") if "Argentina" in nteams_b else 1, key="bb_na")
        blvl = b3.selectbox("Tipo", list(LEVEL_NAMES), index=2, format_func=LEVEL_NAMES.get, key="bb_lvl")
        bneu = b4.toggle("Neutral", value=False, key="bb_neu")
    min_prob_b = st.slider("Probabilidad mínima de acertar para recomendar", 0.05, 0.8, 0.30, 0.05, key="bb_minp",
                           help="Filtra apuestas con mucho valor teórico pero muy improbables (más varianza).")

    if bh == ba:
        st.warning("Elige dos equipos distintos.")
    else:
        # 1) lista de mercados (sin probabilidades todavía) para el editor de cuotas
        def market_names():
            dummy_mk = {f"Más de {l}": 0.5 for l in best_bet.GOAL_LINES} | {"BTTS sí": 0.5, "marcadores": [],
                         "asian": {round(float(x), 2): {"gana": 0, "push": 0, "pierde": 0}
                                   for x in np.arange(-3, 3.25, 0.25)}}
            ms = best_bet.goal_markets((0, 0, 0), dummy_mk, bh, ba)
            ms += best_bet.count_markets("Córners", "córners", 1, 1, 0, 0, bh, ba, (7.5, 8.5, 9.5, 10.5, 11.5, 12.5),
                                         (3.5, 4.5, 5.5))
            ms += best_bet.count_markets("Tarjetas", "tarjetas", 1, 1, 0, 0, bh, ba, (2.5, 3.5, 4.5, 5.5, 6.5),
                                         (0.5, 1.5, 2.5))
            if kind == "Selecciones":
                ms += best_bet.count_markets("Remates", "remates", 1, 1, 0, 0, bh, ba, (18.5, 20.5, 22.5, 24.5, 26.5),
                                             (8.5, 10.5, 12.5, 14.5))
                ms += best_bet.count_markets("Remates al arco", "remates al arco", 1, 1, 0, 0, bh, ba,
                                             (5.5, 6.5, 7.5, 8.5, 9.5), (2.5, 3.5, 4.5))
            return pd.DataFrame(ms)[["Grupo", "Mercado"]]

        names = market_names()
        groups = st.multiselect("Mercados a mostrar", list(names["Grupo"].unique()),
                                default=["Resultado", "Doble oportunidad", "Goles", "Ambos marcan", "Córners", "Tarjetas"],
                                key="bb_groups")
        shown = names[names["Grupo"].isin(groups)].reset_index(drop=True).assign(Cuota=np.nan)
        ek = f"bb_ed_{kind}_{bh}_{ba}_{'-'.join(groups)}"
        left_b, right_b = st.columns([2, 3])
        with left_b:
            st.markdown("**Cuotas de Betano**")
            edited = st.data_editor(shown, key=ek, hide_index=True, width="stretch", height=560,
                                    disabled=["Grupo", "Mercado"],
                                    column_config={"Cuota": st.column_config.NumberColumn(min_value=1.0, step=0.01,
                                                                                          format="%.2f")})
        odds_map = dict(zip(edited["Mercado"], pd.to_numeric(edited["Cuota"], errors="coerce")))
        q1 = odds_map.get(f"1 · {bh}", np.nan)
        qx = odds_map.get("X · Empate", np.nan)
        q2 = odds_map.get(f"2 · {ba}", np.nan)
        has_1x2 = all(np.isfinite(v) and v > 1 for v in (q1, qx, q2))

        # 2) probabilidades del modelo para todos los mercados
        if kind == "Clubes":
            name_, country_, tier_ = ALL_LEAGUES[bl]
            row_b = pd.DataFrame([{
                "league": bl, "country": country_, "tier": tier_,
                "season": int(matches.loc[matches["league"] == bl, "season"].max()),
                "date": pd.Timestamp.today().normalize(), "time": "", "home": bh, "away": ba,
                "referee": "" if bref == "(desconocido)" else bref,
                "odds_h": q1 if has_1x2 else np.nan, "odds_d": qx if has_1x2 else np.nan,
                "odds_a": q2 if has_1x2 else np.nan,
                "odds_o25": odds_map.get("Más de 2.5 goles", np.nan), "odds_u25": odds_map.get("Menos de 2.5 goles", np.nan)}])
            rb = predict(row_b).iloc[0]
            mkb = rb["dc_markets"]
            probs = best_bet.goal_markets((rb.f_h, rb.f_d, rb.f_a), mkb, bh, ba, rb.f_o25, rb.p_btts)
            probs += best_bet.count_markets("Córners", "córners", rb.cor_h, rb.cor_a, stats.alpha["cor_h"],
                                            stats.alpha["cor_a"], bh, ba, (7.5, 8.5, 9.5, 10.5, 11.5, 12.5), (3.5, 4.5, 5.5))
            probs += best_bet.count_markets("Tarjetas", "tarjetas", rb.crd_h, rb.crd_a, stats.alpha["crd_h"],
                                            stats.alpha["crd_a"], bh, ba, (2.5, 3.5, 4.5, 5.5, 6.5), (0.5, 1.5, 2.5))
            note = "1X2 apilado con tus cuotas de Betano" if has_1x2 and use_stack and stacker else "modelo propio"
        else:
            resb = nat_b.predict(bh, ba, bneu, blvl)
            pb_ = resb["p"].copy()
            if has_1x2:
                pb_ = 0.5 * pb_ + 0.5 * betting.devig([q1, qx, q2], devig_method)
            probs = best_bet.goal_markets(tuple(pb_), resb["markets"], bh, ba)
            ns_b = getattr(nat_b, "stats", None)
            if ns_b is not None:
                fzb = resb["features"]
                eh_b = ns_b.expected(bh, ba, fzb["h_elo"], fzb["a_elo"], bneu, 1)
                ea_b = ns_b.expected(ba, bh, fzb["a_elo"], fzb["h_elo"], bneu, 0)
                for grp, lab, k, lt, lteam in (
                        ("Córners", "córners", "corners", (7.5, 8.5, 9.5, 10.5, 11.5, 12.5), (3.5, 4.5, 5.5)),
                        ("Tarjetas", "tarjetas", "cards", (2.5, 3.5, 4.5, 5.5, 6.5), (0.5, 1.5, 2.5)),
                        ("Remates", "remates", "shots", (18.5, 20.5, 22.5, 24.5, 26.5), (8.5, 10.5, 12.5, 14.5)),
                        ("Remates al arco", "remates al arco", "sot", (5.5, 6.5, 7.5, 8.5, 9.5), (2.5, 3.5, 4.5))):
                    probs += best_bet.count_markets(grp, lab, eh_b[k], ea_b[k], ns_b.alpha[k], ns_b.alpha[k], bh, ba,
                                                    lt, lteam)
            note = "1X2: 50 % modelo + 50 % Betano sin margen" if has_1x2 else "modelo propio"

        P = pd.DataFrame(probs)
        P["Cuota"] = P["Mercado"].map(odds_map)
        ev = best_bet.evaluate(P, kelly_fraction=kfrac)

        with right_b:
            if has_1x2:
                st.caption(f"Margen de Betano en 1X2: **{betting.overround([q1, qx, q2]) - 1:.1%}** · probabilidades: {note}")
            if ev.empty:
                st.info("Escribe al menos una cuota en la tabla de la izquierda. Mientras tanto, esta es la cuota "
                        "**mínima** que necesitarías en Betano para que cada apuesta tenga valor:")
                ref = P[P["Grupo"].isin(groups)].assign(**{
                    "Prob. ganar": P["p_gana"],
                    "Cuota mínima con valor": (1 + P["p_pierde"] / P["p_gana"].where(P["p_gana"] > 0)) * (1 + min_ev)})
                ref = ref[ref["p_gana"] >= min_prob_b].sort_values("p_gana", ascending=False)
                st.dataframe(ref[["Grupo", "Mercado", "Prob. ganar", "Cuota mínima con valor"]].style.format(
                    {"Prob. ganar": "{:.1%}", "Cuota mínima con valor": "{:.2f}"}), hide_index=True, width="stretch",
                    height=480)
            else:
                best = best_bet.pick_best(ev, min_prob=min_prob_b, min_ev=min_ev)
                if best:
                    stake = best["Kelly"] * bankroll
                    st.success(f"### ✅ Mejor apuesta: {best['Mercado']} @ {best['Cuota']:.2f}\n"
                               f"Prob. de ganar **{best['p_gana']:.1%}** · cuota justa **{best['Cuota justa']:.2f}** · "
                               f"valor esperado **{best['EV']:+.1%}** · apuesta sugerida **{stake:,.0f}** "
                               f"({best['Kelly']:.1%} del bankroll, Kelly {kfrac:.0%})")
                    if st.button("➕ Añadir a la combinada", key="bb_add"):
                        tk = st.session_state.setdefault("ticket", [])
                        match_id = f"{bh} vs {ba}"
                        tk[:] = [l for l in tk if l["Partido"] != match_id]  # una selección por partido
                        tk.append({"Partido": match_id, "Mercado": best["Mercado"], "Cuota": best["Cuota"],
                                   "p_gana": best["p_gana"], "p_push": best["p_push"]})
                else:
                    st.warning(f"Ninguna cuota tiene valor esperado ≥ {min_ev:.0%} con probabilidad ≥ {min_prob_b:.0%}. "
                               "Con estas cuotas lo mejor es **no apostar** en este partido.")
                show = ev[["Grupo", "Mercado", "Cuota", "Prob. ganar", "Cuota justa", "EV", "Kelly"]].copy()
                show["Stake"] = show["Kelly"] * bankroll
                st.dataframe(show.drop(columns="Kelly").style.format(
                    {"Cuota": "{:.2f}", "Prob. ganar": "{:.1%}", "Cuota justa": "{:.2f}", "EV": "{:+.1%}",
                     "Stake": "{:,.0f}"}).map(ev_color, subset=["EV"]), hide_index=True, width="stretch",
                    height=min(520, 40 + 35 * len(show)))
                st.caption("EV = ganancia esperada por cada $1 apostado. En hándicap asiático y 'empate no válido' "
                           "se consideran las devoluciones.")

    # 3) combinada
    tk = st.session_state.get("ticket", [])
    if tk:
        st.subheader("🧾 Combinada")
        tdf = pd.DataFrame(tk)
        st.dataframe(tdf.rename(columns={"p_gana": "Prob."}).drop(columns="p_push").style.format(
            {"Cuota": "{:.2f}", "Prob.": "{:.1%}"}), hide_index=True, width="stretch")
        pr = best_bet.parlay(tk)
        b_ = pr["Cuota combinada"] - 1
        k_ = max(0.0, (b_ * pr["Prob. acertar todo"] - (1 - pr["Prob. acertar todo"])) / b_) * kfrac if b_ > 0 else 0
        c1_, c2_, c3_, c4_ = st.columns(4)
        c1_.metric("Cuota combinada", f"{pr['Cuota combinada']:.2f}")
        c2_.metric("Prob. de acertar todo", f"{pr['Prob. acertar todo']:.1%}")
        c3_.metric("Valor esperado", f"{pr['EV']:+.1%}")
        c4_.metric("Apuesta sugerida", f"{min(k_, 0.02) * bankroll:,.0f}")
        st.caption("Se asume independencia entre partidos distintos. Las combinadas acumulan el margen de la casa en "
                   "cada selección: solo tienen sentido si cada pata tiene valor por sí sola. Stake limitado a 2 % del bankroll.")
        if st.button("🗑️ Vaciar combinada"):
            st.session_state["ticket"] = []
            st.rerun()

# =========================================================================== 3. selecciones
with tabs[3]:
    nat = get_national(KEY)
    if nat is None:
        st.info("Sin datos de selecciones. Pulsa **Descargar / actualizar** en la barra lateral.")
    else:
        nteams = nat.teams()
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        nh = c1.selectbox("Selección 1 (local)", nteams, index=nteams.index("Chile") if "Chile" in nteams else 0)
        na = c2.selectbox("Selección 2 (visita)", nteams,
                          index=nteams.index("Argentina") if "Argentina" in nteams else 1)
        lvl = c3.selectbox("Tipo de partido", list(LEVEL_NAMES), index=2, format_func=LEVEL_NAMES.get)
        neutral = c4.toggle("Cancha neutral", value=False)
        o1, o2, o3, o4 = st.columns(4)
        nq = [o1.number_input("Cuota 1", 0.0, 200.0, 0.0, 0.05, format="%.2f", key="nq1"),
              o2.number_input("Cuota X", 0.0, 200.0, 0.0, 0.05, format="%.2f", key="nqx"),
              o3.number_input("Cuota 2", 0.0, 200.0, 0.0, 0.05, format="%.2f", key="nq2")]
        w_nm = o4.slider("Peso del mercado", 0.0, 1.0, 0.5, 0.05,
                         help="No hay cuotas históricas de selecciones para entrenar un modelo apilado: "
                              "mezcla simple entre modelo y mercado sin margen.")
        if nh == na:
            st.warning("Elige dos selecciones distintas.")
        else:
            res = nat.predict(nh, na, neutral, lvl)
            p = res["p"].copy()
            mp = betting.devig(nq, devig_method) if all(q > 1 for q in nq) else None
            if mp is not None:
                p = (1 - w_nm) * p + w_nm * mp
            fz = res["features"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(f"Gana {nh}", f"{p[0]:.1%}", f"cuota justa {1 / p[0]:.2f}", delta_color="off")
            m2.metric("Empate", f"{p[1]:.1%}", f"cuota justa {1 / p[1]:.2f}", delta_color="off")
            m3.metric(f"Gana {na}", f"{p[2]:.1%}", f"cuota justa {1 / p[2]:.2f}", delta_color="off")
            m4.metric("Goles esperados", f"{res['markets']['xG local']:.2f} – {res['markets']['xG visita']:.2f}")
            m5.metric("Elo", f"{fz['h_elo']:.0f} – {fz['a_elo']:.0f}",
                      f"{fz['elo_diff']:+.0f}" + ("" if neutral else " con localía"), delta_color="off")
            if mp is not None:
                st.caption(f"Margen de la casa: **{betting.overround(nq) - 1:.1%}** · final = "
                           f"{1 - w_nm:.0%} modelo + {w_nm:.0%} mercado")
            left, right = st.columns([3, 2])
            with left:
                mk = res["markets"]
                comp = pd.DataFrame({"Resultado": ["1", "X", "2"] * 4,
                                     "Probabilidad": [*p, *res["gbm"], *res["dc"], *(mp if mp is not None else [np.nan] * 3)],
                                     "Fuente": ["Final"] * 3 + ["Gradient Boosting"] * 3 + ["Poisson de goles"] * 3
                                               + ["Mercado (sin margen)"] * 3}).dropna()
                fig = px.bar(comp, x="Resultado", y="Probabilidad", color="Fuente", barmode="group",
                             text_auto=".1%", height=300)
                fig.update_layout(yaxis_tickformat=".0%", margin=dict(t=10, b=10), legend_title=None,
                                  xaxis_type="category", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, width="stretch")
                gm = [("1X", p[0] + p[1]), ("12", p[0] + p[2]), ("X2", p[1] + p[2])]
                gm += [(n, mk[n]) for n in ["Más de 1.5", "Menos de 1.5", "Más de 2.5", "Menos de 2.5", "Más de 3.5",
                                            "Menos de 3.5", "BTTS sí", "BTTS no"]]
                editable_ev(pd.DataFrame([{"Mercado": n.replace("BTTS", "Ambos marcan:"), "Probabilidad": v,
                                           "Cuota justa": 1 / v} for n, v in gm]), "ev_nat")
            with right:
                st.markdown("**Marcador exacto**")
                mat = res["matrix"][:6, :6]
                hm = go.Figure(go.Heatmap(z=mat, x=[str(i) for i in range(6)], y=[str(i) for i in range(6)],
                                          text=[[f"{v:.1%}" for v in row_] for row_ in mat], texttemplate="%{text}",
                                          colorscale="Greens", showscale=False))
                hm.update_layout(xaxis_title=f"Goles {na}", yaxis_title=f"Goles {nh}", height=330,
                                 margin=dict(t=10, b=10), yaxis_autorange="reversed", xaxis_type="category",
                                 yaxis_type="category")
                st.plotly_chart(hm, width="stretch")
            # ------------------------------------------------ córners, tarjetas y remates
            ns = getattr(nat, "stats", None)
            if ns is not None:
                st.subheader("🚩 Córners, 🟨 tarjetas y 🎯 remates")
                e_h = ns.expected(nh, na, fz["h_elo"], fz["a_elo"], neutral, 1)
                e_a = ns.expected(na, nh, fz["a_elo"], fz["h_elo"], neutral, 0)
                for c, (k, lab) in zip(st.columns(len(NAT_STATS)), NAT_STATS.items()):
                    c.metric(f"{lab} · total esperado", f"{e_h[k] + e_a[k]:.1f}", f"{e_h[k]:.1f} – {e_a[k]:.1f}",
                             delta_color="off")
                cov_h, cov_a = ns.coverage(nh), ns.coverage(na)
                src = " · ".join(f"{k}: {v}" for k, v in ns.sources().items())
                msg = (f"Partidos con estadísticas detalladas: **{nh} {cov_h}** · **{na} {cov_a}** (fuentes → {src}).")
                if min(cov_h, cov_a) == 0:
                    st.warning(msg + " Una selección no tiene datos: su estimación se basa en el Elo y la media global."
                               + ("" if apifootball.get_key() else " Con una clave de API-Football (barra lateral) "
                                  "se suman eliminatorias y amistosos."))
                else:
                    st.caption(msg)
                lines = {"corners": ((7.5, 8.5, 9.5, 10.5, 11.5), (2.5, 3.5, 4.5, 5.5, 6.5), "córners"),
                         "cards": ((2.5, 3.5, 4.5, 5.5), (0.5, 1.5, 2.5), "tarjetas"),
                         "shots": ((18.5, 20.5, 22.5, 24.5, 26.5), (8.5, 10.5, 12.5, 14.5), "remates"),
                         "sot": ((5.5, 6.5, 7.5, 8.5, 9.5), (2.5, 3.5, 4.5, 5.5), "remates al arco"),
                         "fouls": ((22.5, 24.5, 26.5, 28.5, 30.5), (11.5, 13.5, 15.5), "faltas")}
                for t_, (k, (lt, lteam, lab)) in zip(st.tabs([NAT_STATS[k] for k in lines]), lines.items()):
                    with t_:
                        mk_s = stat_markets(e_h[k], e_a[k], ns.alpha[k], ns.alpha[k], lt, lteam)
                        editable_ev(stat_table(mk_s, lab), f"ev_nat_{k}")
                tt = pd.DataFrame({nh: ns.team_table(nh), na: ns.team_table(na)})
                if not tt.empty:
                    st.markdown("**Promedios por partido en grandes torneos**")
                    st.dataframe(tt.style.format("{:.2f}", na_rep="—"), width="stretch")
                with st.expander("Validación (dejando fuera cada torneo completo)"):
                    st.dataframe(ns.validation.style.format({"MAE modelo": "{:.3f}", "MAE media global": "{:.3f}",
                                                             "media por equipo": "{:.2f}"}),
                                 hide_index=True, width="stretch")
                    st.caption("Tarjetas: amarilla = 1, roja = 2 (doble amarilla = 3 en StatsBomb). StatsBomb: solo "
                               "tiempo reglamentario; API-Football: partido completo.")

            f1, f2, f3 = st.columns(3)
            for col, t in ((f1, nh), (f2, na)):
                lm = nat.last_matches(t)
                col.markdown(f"**{t}** · últimos partidos")
                col.dataframe(lm.assign(date=lm["date"].dt.strftime("%d/%m/%y")), hide_index=True,
                              width="stretch")
            NF = nat.F
            h2h = NF[((NF["home"] == nh) & (NF["away"] == na)) | ((NF["home"] == na) & (NF["away"] == nh))]
            f3.markdown(f"**Enfrentamientos directos** ({len(h2h)})")
            f3.dataframe(h2h.tail(10)[["date", "home", "fthg", "ftag", "away", "tournament"]]
                         .assign(date=lambda x: x["date"].dt.strftime("%d/%m/%y")), hide_index=True,
                         width="stretch")
            hw = ((h2h["home"] == nh) & (h2h["fthg"] > h2h["ftag"])) | ((h2h["away"] == nh) & (h2h["ftag"] > h2h["fthg"]))
            dr = h2h["fthg"] == h2h["ftag"]
            f3.caption(f"{nh}: {int(hw.sum())} victorias · {int(dr.sum())} empates · {int(len(h2h) - hw.sum() - dr.sum())} derrotas")

        a1, a2 = st.columns([3, 2])
        a1.markdown("**Ranking Elo de selecciones** (activas en los últimos 2 años)")
        a1.dataframe(nat.ranking().style.format({"Elo": "{:.0f}", "Goles a favor (forma)": "{:.2f}",
                                                 "Goles en contra (forma)": "{:.2f}",
                                                 "Último partido": lambda d: d.strftime("%d/%m/%Y")}),
                     width="stretch", height=420)
        a2.markdown("**Validación** (últimos 2 años fuera de muestra)")
        a2.dataframe(nat.validation.style.format({"log loss": "{:.4f}", "acierto": "{:.1%}"}), hide_index=True,
                     width="stretch")
        a2.caption(f"{len(nat.F):,} partidos desde {nat.F['date'].min():%Y} hasta {nat.F['date'].max():%d/%m/%Y}. "
                   "Fuentes: github.com/martj42/international_results (resultados) y StatsBomb Open Data "
                   "(córners, tarjetas, remates).")

# =========================================================================== 4. equipos y ligas
with tabs[4]:
    lg = st.selectbox("Liga ", leagues_avail, index=leagues_avail.index("E0") if "E0" in leagues_avail else 0,
                      format_func=league_label, key="lg_teams")
    d = matches[matches["league"] == lg]
    season = int(d["season"].max())
    cur = d[(d["season"] == season)].dropna(subset=["fthg"])
    table = {}
    for r in cur.itertuples():
        for t, gf, ga in ((r.home, r.fthg, r.ftag), (r.away, r.ftag, r.fthg)):
            s = table.setdefault(t, dict(PJ=0, G=0, E=0, P=0, GF=0, GC=0))
            s["PJ"] += 1; s["GF"] += gf; s["GC"] += ga
            s["G" if gf > ga else "E" if gf == ga else "P"] += 1
    stand = pd.DataFrame(table).T
    if not stand.empty:
        stand["DG"] = stand["GF"] - stand["GC"]
        stand["Pts"] = 3 * stand["G"] + stand["E"]
        country = ALL_LEAGUES[lg][1]
        stand["Elo"] = [fb.elo.get((country, t), np.nan) for t in stand.index]
        dcr = get_dc(KEY, lg).ratings().set_index("equipo")
        stand = stand.join(dcr[["ataque", "defensa"]]).sort_values(["Pts", "DG", "GF"], ascending=False)
        stand.index.name = "Equipo"
        a, b = st.columns([3, 2])
        a.markdown(f"**Tabla {season}/{season + 1}** + ratings")
        a.dataframe(stand.astype({c: int for c in ["PJ", "G", "E", "P", "GF", "GC", "DG", "Pts"]}).style.format(
            {"Elo": "{:.0f}", "ataque": "{:.2f}", "defensa": "{:.2f}"}).background_gradient(subset=["Elo"], cmap="Blues"),
            width="stretch", height=min(800, 40 + 35 * len(stand)))
        sc = px.scatter(stand.reset_index(), x="ataque", y="defensa", text="Equipo", height=500,
                        labels={"ataque": "Ataque (× goles promedio)", "defensa": "Defensa (goles concedidos, menor = mejor)"})
        sc.update_traces(textposition="top center")
        sc.update_yaxes(autorange="reversed")
        b.markdown("**Fuerza Dixon-Coles**")
        b.plotly_chart(sc, width="stretch")

    st.markdown("**Evolución del Elo**")
    sel_t = st.multiselect("Equipos", league_teams(lg), default=list(stand.index[:4]) if not stand.empty else [])
    if sel_t:
        country = ALL_LEAGUES[lg][1]
        Fc = F[F["country"] == country]
        hist = pd.concat([Fc[["date", "home", "h_elo"]].set_axis(["date", "team", "elo"], axis=1),
                          Fc[["date", "away", "a_elo"]].set_axis(["date", "team", "elo"], axis=1)])
        hist = hist[hist["team"].isin(sel_t)].sort_values("date")
        st.plotly_chart(px.line(hist, x="date", y="elo", color="team", height=400,
                                labels={"date": "", "elo": "Elo", "team": ""}), width="stretch")

    # estadísticas de la temporada por equipo
    st.markdown(f"**Estadísticas por partido {season}/{season + 1}**")
    cs = d[d["season"] == season].dropna(subset=["fthg"]).copy()
    cs["h_cards"] = cs["h_yellow"] + 2 * cs["h_red"].fillna(0)
    cs["a_cards"] = cs["a_yellow"] + 2 * cs["a_red"].fillna(0)
    long = pd.concat([
        cs[["home", "h_corners", "a_corners", "h_cards", "a_cards", "h_fouls", "a_fouls", "h_shots", "h_sot", "h_yellow", "h_red"]]
        .set_axis(["Equipo", "Córners", "Córners rival", "Tarjetas", "Tarjetas rival", "Faltas", "Faltas rival", "Tiros",
                   "Tiros a puerta", "Amarillas", "Rojas"], axis=1),
        cs[["away", "a_corners", "h_corners", "a_cards", "h_cards", "a_fouls", "h_fouls", "a_shots", "a_sot", "a_yellow", "a_red"]]
        .set_axis(["Equipo", "Córners", "Córners rival", "Tarjetas", "Tarjetas rival", "Faltas", "Faltas rival", "Tiros",
                   "Tiros a puerta", "Amarillas", "Rojas"], axis=1)])
    ts = long.groupby("Equipo").mean(numeric_only=True)
    ts["Rojas"] = long.groupby("Equipo")["Rojas"].sum()
    if ts.drop(columns="Rojas").notna().any().any():
        st.dataframe(ts.sort_values("Córners", ascending=False).style.format(
                     {c: ("{:.0f}" if c == "Rojas" else "{:.2f}") for c in ts.columns}, na_rep="—").background_gradient(subset=["Córners", "Tarjetas"], cmap="Oranges"),
                     width="stretch", height=min(760, 40 + 35 * len(ts)))
    else:
        st.info("Esta liga no publica córners, tarjetas ni tiros por partido en la fuente de datos.")

    # árbitros
    rd = d[d["season"] >= season - 1].dropna(subset=["h_yellow"])
    rd = rd[~rd["referee"].astype(str).isin(["", "None", "nan"])]
    if not rd.empty:
        st.markdown(f"**Árbitros** (temporadas {season - 1}/{season % 100:02d} y {season}/{(season + 1) % 100:02d})")
        rd = rd.assign(tarjetas=rd["h_yellow"] + rd["a_yellow"] + 2 * (rd["h_red"].fillna(0) + rd["a_red"].fillna(0)),
                       faltas=rd["h_fouls"] + rd["a_fouls"], rojas=rd["h_red"].fillna(0) + rd["a_red"].fillna(0),
                       corners=rd["h_corners"] + rd["a_corners"], local_gana=(rd["fthg"] > rd["ftag"]).astype(float))
        rt = rd.groupby("referee").agg(Partidos=("tarjetas", "size"), **{
            "Tarjetas/partido": ("tarjetas", "mean"), "Rojas totales": ("rojas", "sum"),
            "Faltas/partido": ("faltas", "mean"), "Córners/partido": ("corners", "mean"),
            "% gana local": ("local_gana", "mean")})
        rt = rt[rt["Partidos"] >= 3].sort_values("Tarjetas/partido", ascending=False)
        rt.index.name = "Árbitro"
        st.dataframe(rt.style.format({"Tarjetas/partido": "{:.2f}", "Faltas/partido": "{:.1f}", "Córners/partido": "{:.1f}",
                                      "Rojas totales": "{:.0f}", "% gana local": "{:.0%}"})
                     .background_gradient(subset=["Tarjetas/partido"], cmap="Reds"),
                     width="stretch", height=min(600, 40 + 35 * len(rt)))

# =========================================================================== 6. datos y método
with tabs[6]:
    cov = matches.groupby("league").agg(
        partidos=("home", "size"), desde=("season", "min"), hasta=("season", "max"),
        con_tiros=("h_shots", lambda x: x.notna().mean()), con_cuotas=("odds_h", lambda x: x.notna().mean()))
    cov.insert(0, "liga", [league_label(c) for c in cov.index])
    st.markdown(f"**{len(matches):,} partidos** en **{matches['league'].nunique()} ligas** · "
                f"{len(fixtures)} próximos partidos con cuotas")
    st.dataframe(cov.sort_values("liga").style.format({"con_tiros": "{:.0%}", "con_cuotas": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.download_button("Descargar base completa (CSV)", matches.to_csv(index=False).encode(), "partidos.csv")
    st.markdown("**Validación de córners y tarjetas** (último año fuera de muestra; menor = mejor)")
    st.dataframe(pd.DataFrame(stats.validation).T.style.format("{:.3f}", na_rep="—").format({"n": "{:,.0f}"}),
                 width="stretch")
    st.markdown("""
### Cómo funciona
1. **Scraping masivo** de football-data.co.uk: resultados, medio tiempo, tiros, tiros a puerta, córners, faltas,
   tarjetas, árbitro y cuotas de 1X2, más/menos 2.5 y hándicap asiático de ~10 casas (apertura y cierre).
2. **Variables pre-partido** (sin fuga de información): Elo con localía y regresión entre temporadas, forma
   exponencial corta/larga de goles, tiros, tiros a puerta, córners y puntos, forma local/visita, descanso,
   enfrentamientos directos, contexto de la liga y probabilidades del mercado sin margen.
3. **Modelos**
   - *Dixon-Coles*: Poisson bivariado con fuerzas de ataque/defensa, ventaja local, corrección de marcadores bajos
     y ponderación temporal. Es el método clásico con que las casas fijan cuotas de goles. Genera la matriz de
     marcadores → 1X2, doble oportunidad, más/menos, ambos marcan, hándicap asiático, marcador exacto.
   - *Gradient Boosting* (sklearn) entrenado con ~250 mil partidos para 1X2, más de 2.5 y ambos marcan,
     con mayor peso a partidos recientes.
   - *Ensamble* de ambos y contracción final hacia el mercado.
4. **Córners y tarjetas**: Gradient Boosting con pérdida Poisson por equipo (forma de córners, tarjetas,
   faltas y tiros a favor/en contra, Elo, promedios de la liga y del **árbitro**) y distribución binomial
   negativa → más/menos del total y por equipo, y quién tiene más. Tarjetas: amarilla = 1, roja = 2.
5. **Selecciones nacionales**: ~49.000 partidos desde 1872; Elo con K según el torneo, localía solo fuera de
   cancha neutral, forma y H2H; Gradient Boosting + regresión Poisson de goles.
6. **Lógica de casa de apuestas**: margen (overround), eliminación del margen (power / Shin / multiplicativo),
   cuotas justas, cuotas con margen simulado, valor esperado y Kelly fraccional.

### Advertencia
Los mercados de fútbol (sobre todo las ligas grandes) son muy eficientes: la cuota de cierre de Pinnacle es
difícil de superar. Revisa siempre el **Backtest** antes de confiar en una estrategia; un ROI positivo en una
muestra pequeña puede ser suerte. Apuesta solo lo que estés dispuesto a perder.
""")

# =========================================================================== 5. backtest
with tabs[5]:
    st.markdown("Simulación **walk-forward** honesta: cada predicción se hizo entrenando solo con partidos "
                "anteriores (el Gradient Boosting por temporada, Dixon-Coles cada 2 semanas y el modelo apilado "
                "solo con temporadas previas). Las apuestas usan las cuotas reales publicadas en su momento.")
    if oos is None:
        st.warning("Aún no existe el histórico fuera de muestra. Ejecuta `.venv/bin/python entrenar.py` "
                   "o usa el botón en 'Actualizar datos' (barra lateral).")
        st.stop()
    c1, c2, c3 = st.columns([3, 2, 2])
    lg_oos = sorted(oos["league"].unique(), key=league_label)
    bt_leagues = c1.multiselect("Ligas", lg_oos, default=[l for l in ["E0", "SP1", "I1", "D1", "F1"] if l in lg_oos],
                                format_func=league_label)
    seasons_av = sorted(oos["season"].unique())[1:]  # la primera solo sirve para entrenar el stacker
    bt_seasons = c2.multiselect("Temporadas de prueba", seasons_av, default=seasons_av,
                                format_func=lambda s: f"{s}/{(s + 1) % 100:02d}")
    prob_src = c3.radio("Probabilidades para apostar", ["final", "ens", "mkt"], horizontal=False,
                        format_func={"final": "Final (apilado con mercado)", "ens": "Solo modelo propio",
                                     "mkt": "Solo mercado (línea base)"}.get)
    c4, c5, c6, c7 = st.columns(4)
    r_markets = c4.multiselect("Mercados", ["1X2", "O/U 2.5"], default=["1X2"])
    r_minp = c5.slider("Prob. mínima", 0.0, 0.6, 0.2, 0.05)
    r_maxo = c6.slider("Cuota máxima", 1.5, 15.0, 5.0, 0.5)
    r_stake = c7.selectbox("Stake", ["flat", "kelly"], format_func={"flat": "Plano (1 u.)", "kelly": "Kelly (bank 100)"}.get)
    st.caption("Usa la barra lateral para el valor esperado mínimo, la fuente de cuotas y la fracción de Kelly. "
               "Compara siempre contra **Solo mercado**: si el modelo no la supera, la ganancia viene de "
               "buscar la mejor cuota, no de predecir mejor.")

    bt = stack_walkforward(oos, bt_leagues, bt_seasons, w_gbm=w_gbm) if bt_leagues and bt_seasons else None
    if bt is not None and not bt.empty:
        if prob_src != "final":
            for k in "hda":
                bt[f"p_{k}"] = bt[f"{prob_src}_{k}"]
            o25_src = bt["mkt_o25"] if prob_src == "mkt" else w_gbm * bt["gbm_o25"] + (1 - w_gbm) * bt["dc_o25"]
            bt["p_o25"] = o25_src
            bt = bt.dropna(subset=["p_h"])
        st.markdown("**Precisión probabilística** (menor log loss / Brier = mejor; el cierre de Pinnacle es la referencia más dura)")
        st.dataframe(metrics(bt).style.format({"log loss": "{:.4f}", "Brier": "{:.4f}", "acierto": "{:.1%}"}),
                     hide_index=True, width="stretch")
        rules = BetRules(min_ev=min_ev, min_prob=r_minp, max_odds=r_maxo, odds_source=odds_src, staking=r_stake,
                         kelly_fraction=kfrac, markets=tuple(r_markets))
        bets = simulate_bets(bt, rules, bankroll=100.0)
        s = bet_summary(bets)
        k = st.columns(6)
        k[0].metric("Apuestas", s["apuestas"])
        if s["apuestas"]:
            k[1].metric("Acierto", f"{s['aciertos']:.1%}")
            k[2].metric("Cuota media", f"{s['cuota media']:.2f}")
            k[3].metric("Ganancia (u.)", f"{s['ganancia']:+.1f}")
            k[4].metric("ROI", f"{s['ROI']:+.1%}")
            k[5].metric("Máx. caída", f"{s['máx. caída']:.1f}")
            st.plotly_chart(px.line(bets, x="fecha", y="acumulado", height=320,
                                    labels={"acumulado": "Ganancia acumulada (u.)", "fecha": ""}),
                            width="stretch")
            # calibración
            P = bt[["p_h", "p_d", "p_a"]].to_numpy().ravel()
            Y = np.eye(3)[bt["y_res"].to_numpy()].ravel()
            cal = pd.DataFrame({"p": P, "y": Y}).assign(bin=lambda d: pd.cut(d["p"], np.linspace(0, 1, 11)))
            cal = cal.groupby("bin", observed=True).agg(predicha=("p", "mean"), observada=("y", "mean"), n=("y", "size"))
            cf = px.scatter(cal, x="predicha", y="observada", size="n", height=350, title="Calibración (1X2)")
            cf.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
            a, b = st.columns(2)
            a.plotly_chart(cf, width="stretch")
            by = bets.groupby("apuesta").agg(n=("ganada", "size"), acierto=("ganada", "mean"),
                                             ganancia=("ganancia", "sum"))
            by["ROI"] = by["ganancia"] / bets.groupby("apuesta")["stake"].sum()
            b.markdown("**Por tipo de apuesta**")
            b.dataframe(by.style.format({"acierto": "{:.1%}", "ganancia": "{:+.1f}", "ROI": "{:+.1%}"}),
                        width="stretch")
            per = bets.assign(temporada=bets["fecha"].dt.year).groupby("temporada").agg(
                apuestas=("ganada", "size"), ganancia=("ganancia", "sum"), apostado=("stake", "sum"))
            per["ROI"] = per["ganancia"] / per["apostado"]
            b.markdown("**Por año**")
            b.dataframe(per.style.format({"ganancia": "{:+.1f}", "apostado": "{:.1f}", "ROI": "{:+.1%}"}),
                        width="stretch")
            with st.expander("Registro de apuestas"):
                st.dataframe(bets, hide_index=True, width="stretch")

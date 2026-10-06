"""Descarga masiva de resultados históricos, estadísticas y cuotas desde football-data.co.uk.

- Ligas principales: un CSV por liga y temporada (2000/01 → actual), ~600 archivos.
- Ligas extra: un CSV por liga con todas las temporadas.
- Próximos partidos (con cuotas) para ambos grupos.

Los CSV de temporadas cerradas se guardan en data/raw y no se vuelven a descargar.
Todo se normaliza a un esquema común y se guarda en SQLite (data/futbol.db).
"""
from __future__ import annotations

import io
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .config import (
    BASE_URL, CURRENT_SEASON, DB_PATH, EXTRA_COUNTRY_TO_CODE, EXTRA_LEAGUES, FIRST_SEASON,
    MAIN_LEAGUES, ALL_LEAGUES, RAW_DIR, season_code,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (predictor-apuestas; uso personal/académico)"}
BOOKIES = ["B365", "BW", "IW", "LB", "PS", "WH", "VC", "GB", "SB", "SJ", "BV", "CL", "BMGM", "PP", "SKB", "BFD"]

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- descarga
def _fetch(url: str, cache_file=None, force: bool = False, retries: int = 3) -> bytes | None:
    if cache_file is not None and cache_file.exists() and not force:
        return cache_file.read_bytes()
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            content = r.content
            if cache_file is not None and len(content) > 100:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(content)
            return content
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def _read_csv(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


# --------------------------------------------------------------------------- normalización
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _first(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Primera columna disponible (no nula) de la lista, fila a fila."""
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        out = out.fillna(_num(df, c))
    return out


def _bookie_agg(df: pd.DataFrame, suffix: str, how: str) -> pd.Series:
    cols = [f"{b}{suffix}" for b in BOOKIES if f"{b}{suffix}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    block = df[cols].apply(pd.to_numeric, errors="coerce").where(lambda x: x > 1.0)
    return block.mean(axis=1) if how == "mean" else block.max(axis=1)


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d = d.fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))
    return d


def normalize_main(raw: pd.DataFrame, league: str, season: int | None) -> pd.DataFrame:
    if raw.empty or "HomeTeam" not in raw.columns:
        return pd.DataFrame()
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    name, country, tier = ALL_LEAGUES[league]
    df = pd.DataFrame(index=raw.index)
    df["league"] = league
    df["country"] = country
    df["tier"] = tier
    df["date"] = _parse_dates(raw["Date"])
    df["season"] = season if season is not None else np.where(df["date"].dt.month >= 7, df["date"].dt.year, df["date"].dt.year - 1)
    df["time"] = raw["Time"].astype(str) if "Time" in raw.columns else None
    df["home"] = raw["HomeTeam"].astype(str).str.strip()
    df["away"] = raw["AwayTeam"].astype(str).str.strip()
    df["referee"] = raw["Referee"].astype(str).str.strip() if "Referee" in raw.columns else None
    for out, col in [("fthg", "FTHG"), ("ftag", "FTAG"), ("hthg", "HTHG"), ("htag", "HTAG"),
                     ("h_shots", "HS"), ("a_shots", "AS"), ("h_sot", "HST"), ("a_sot", "AST"),
                     ("h_corners", "HC"), ("a_corners", "AC"), ("h_fouls", "HF"), ("a_fouls", "AF"),
                     ("h_yellow", "HY"), ("a_yellow", "AY"), ("h_red", "HR"), ("a_red", "AR"),
                     ("h_xg", "HxG"), ("a_xg", "AxG")]:
        df[out] = _num(raw, col)
    # Cuotas 1X2: promedio del mercado, máxima (mejor precio), Bet365 y cierre "sharp" (Pinnacle)
    for k, s in (("h", "H"), ("d", "D"), ("a", "A")):
        df[f"odds_{k}"] = _first(raw, [f"Avg{s}", f"BbAv{s}"]).fillna(_bookie_agg(raw, s, "mean"))
        df[f"max_{k}"] = _first(raw, [f"Max{s}", f"BbMx{s}"]).fillna(_bookie_agg(raw, s, "max"))
        df[f"b365_{k}"] = _num(raw, f"B365{s}")
        df[f"close_{k}"] = _first(raw, [f"PSC{s}", f"AvgC{s}", f"PS{s}"]).fillna(df[f"odds_{k}"])
    # Más/menos 2.5 goles y hándicap asiático
    df["odds_o25"] = _first(raw, ["Avg>2.5", "BbAv>2.5", "B365>2.5"])
    df["odds_u25"] = _first(raw, ["Avg<2.5", "BbAv<2.5", "B365<2.5"])
    df["max_o25"] = _first(raw, ["Max>2.5", "BbMx>2.5"])
    df["max_u25"] = _first(raw, ["Max<2.5", "BbMx<2.5"])
    df["ah_line"] = _first(raw, ["AHh", "BbAHh"])
    df["odds_ahh"] = _first(raw, ["AvgAHH", "BbAvAHH", "B365AHH"])
    df["odds_aha"] = _first(raw, ["AvgAHA", "BbAvAHA", "B365AHA"])
    return df.dropna(subset=["date"])


def normalize_extra(raw: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    """CSV de 'new/' (todas las temporadas en un archivo) y new_league_fixtures.csv."""
    if raw.empty or "Home" not in raw.columns:
        return pd.DataFrame()
    raw = raw.dropna(subset=["Home", "Away"]).copy()
    if league is None:
        raw["_code"] = raw["Country"].map(EXTRA_COUNTRY_TO_CODE)
        raw = raw.dropna(subset=["_code"])
    else:
        raw["_code"] = league
    df = pd.DataFrame(index=raw.index)
    df["league"] = raw["_code"]
    df["country"] = df["league"].map(lambda c: ALL_LEAGUES[c][1])
    df["tier"] = 1
    df["date"] = _parse_dates(raw["Date"])
    if "Season" in raw.columns:
        df["season"] = pd.to_numeric(raw["Season"].astype(str).str[:4], errors="coerce")
    else:
        df["season"] = df["date"].dt.year
    df["time"] = raw["Time"].astype(str) if "Time" in raw.columns else None
    df["home"] = raw["Home"].astype(str).str.strip()
    df["away"] = raw["Away"].astype(str).str.strip()
    df["referee"] = None
    df["fthg"] = _num(raw, "HG")
    df["ftag"] = _num(raw, "AG")
    for c in ["hthg", "htag", "h_shots", "a_shots", "h_sot", "a_sot", "h_corners", "a_corners", "h_fouls",
              "a_fouls", "h_yellow", "a_yellow", "h_red", "a_red", "h_xg", "a_xg"]:
        df[c] = np.nan
    for k, s in (("h", "H"), ("d", "D"), ("a", "A")):
        df[f"odds_{k}"] = _first(raw, [f"Avg{s}", f"AvgC{s}", f"B365{s}", f"B365C{s}"])
        df[f"max_{k}"] = _first(raw, [f"Max{s}", f"MaxC{s}"])
        df[f"b365_{k}"] = _first(raw, [f"B365{s}", f"B365C{s}"])
        df[f"close_{k}"] = _first(raw, [f"PSC{s}", f"AvgC{s}", f"PS{s}"]).fillna(df[f"odds_{k}"])
    for c in ["odds_o25", "odds_u25", "max_o25", "max_u25", "ah_line", "odds_ahh", "odds_aha"]:
        df[c] = np.nan
    df["season"] = df["season"].astype("Int64")
    return df.dropna(subset=["date", "season"])


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    res = np.select([df["fthg"] > df["ftag"], df["fthg"] == df["ftag"], df["fthg"] < df["ftag"]], ["H", "D", "A"], default="")
    df["ftr"] = np.where(df["fthg"].notna() & df["ftag"].notna(), res, None)
    df["season"] = df["season"].astype(int)
    df = df.drop_duplicates(subset=["league", "date", "home", "away"], keep="last")
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)


# --------------------------------------------------------------------------- orquestación
def scrape_all(leagues: list[str] | None = None, first_season: int = FIRST_SEASON,
               refresh_current: bool = True, workers: int = 6,
               progress: ProgressFn | None = None) -> dict:
    """Descarga todo y reescribe la base de datos. Devuelve un resumen."""
    leagues = leagues or list(ALL_LEAGUES)
    main = [l for l in leagues if l in MAIN_LEAGUES]
    extra = [l for l in leagues if l in EXTRA_LEAGUES]

    jobs = []  # (tipo, liga, temporada, url, cache, force)
    for lg in main:
        for s in range(first_season, CURRENT_SEASON + 1):
            sc = season_code(s)
            jobs.append(("main", lg, s, f"{BASE_URL}/mmz4281/{sc}/{lg}.csv",
                         RAW_DIR / "main" / f"{lg}_{sc}.csv", refresh_current and s >= CURRENT_SEASON - 1))
    for lg in extra:
        jobs.append(("extra", lg, None, f"{BASE_URL}/new/{lg}.csv", RAW_DIR / "extra" / f"{lg}.csv", refresh_current))

    frames, failed, done = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, url, cache, force): (kind, lg, s, url) for kind, lg, s, url, cache, force in jobs}
        for fut in as_completed(futs):
            kind, lg, s, url = futs[fut]
            done += 1
            content = fut.result()
            if content is None:
                failed.append(url)
            else:
                raw = _read_csv(content)
                df = normalize_main(raw, lg, s) if kind == "main" else normalize_extra(raw, lg)
                if not df.empty:
                    frames.append(df)
            if progress:
                progress(done / (len(jobs) + 2), f"{done}/{len(jobs)} archivos · {lg} {s or ''}")

    matches = _finalize(pd.concat(frames, ignore_index=True))
    fixtures = scrape_fixtures()
    if progress:
        progress(1.0, "Guardando en base de datos…")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        _to_sql(matches, "matches", con)
        _to_sql(fixtures, "fixtures", con)
        pd.DataFrame([{"updated_at": pd.Timestamp.now().isoformat(timespec="seconds")}]).to_sql(
            "meta", con, if_exists="replace", index=False)
    try:  # selecciones nacionales (dataset abierto en GitHub)
        from .national import download
        n_intl = len(download())
        from .national_stats import download as download_sb
        download_sb()  # incremental: solo partidos nuevos de StatsBomb
    except Exception:
        n_intl = 0
    return {"partidos": len(matches), "proximos": len(fixtures), "archivos": len(jobs),
            "sin_datos": len(failed), "ligas": matches["league"].nunique(), "selecciones": n_intl}


def scrape_fixtures() -> pd.DataFrame:
    frames = []
    c = _fetch(f"{BASE_URL}/fixtures.csv")
    if c:
        raw = _read_csv(c)
        raw = raw[raw["Div"].isin(MAIN_LEAGUES)] if "Div" in raw.columns else raw
        for lg, g in raw.groupby("Div"):
            frames.append(normalize_main(g, lg, None))
    c = _fetch(f"{BASE_URL}/new_league_fixtures.csv")
    if c:
        frames.append(normalize_extra(_read_csv(c)))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["fthg"] = np.nan
    df["ftag"] = np.nan
    return _finalize(df)


def _to_sql(df: pd.DataFrame, name: str, con) -> None:
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_sql(name, con, if_exists="replace", index=False, chunksize=5000)


# --------------------------------------------------------------------------- lectura
def load_table(name: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        try:
            df = pd.read_sql(f"SELECT * FROM {name}", con)
        except Exception:
            return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def last_update() -> str | None:
    m = load_table("meta")
    return None if m.empty else m["updated_at"].iloc[0]


if __name__ == "__main__":
    t0 = time.time()
    summary = scrape_all(progress=lambda p, msg: print(f"\r[{p:5.1%}] {msg:<60}", end="", flush=True))
    print(f"\n{summary}  ({time.time() - t0:.0f}s)")

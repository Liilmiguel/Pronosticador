# ⚽ Predictor de Apuestas Deportivas (fútbol)

Interfaz local (Streamlit) para predecir partidos de fútbol y detectar apuestas de valor, con:

- **Scraping masivo** de football-data.co.uk: ~255.000 partidos, 38 ligas, temporadas 2000/01 → 2026/27.
  Resultados, medio tiempo, tiros, tiros a puerta, córners, faltas, tarjetas, árbitro, xG (desde 2026/27)
  y cuotas de ~10 casas (1X2, más/menos 2.5, hándicap asiático; apertura y cierre, incl. Pinnacle).
- **Modelos** como los de una casa de apuestas:
  - *Dixon-Coles* (Poisson bivariado con ataque/defensa, localía, corrección de marcadores bajos y
    decaimiento temporal) → matriz de marcadores → 1X2, doble oportunidad, más/menos 0.5–4.5, ambos
    marcan, hándicap asiático y marcador exacto.
  - *Gradient Boosting* con ~70 variables pre-partido: Elo, forma corta/larga (goles, tiros, tiros a puerta,
    córners, puntos), forma local/visita, descanso, H2H, contexto de la liga.
  - *Modelo apilado anclado al mercado*: regresión logística entrenada **fuera de muestra** que parte de
    la probabilidad sin margen de la casa y la corrige con el modelo propio.
- **Córners y tarjetas**: Gradient Boosting Poisson por equipo (forma de córners/tarjetas/faltas, Elo, liga
  y **árbitro**) + binomial negativa → más/menos total y por equipo, y quién tiene más. Tablas por equipo y
  por árbitro. Validado: Brier 0,237 vs 0,247 (más de 9.5 córners) y 0,222 vs 0,235 (más de 4.5 tarjetas)
  frente al promedio de la liga.
- **Selecciones nacionales**: ~49.900 partidos desde 1872 (github.com/martj42/international_results),
  completados con **eloratings.net** para lo más reciente (Mundial 2026, Nations League 2026-27, última fecha FIFA). Elo con
  K según torneo y cancha neutral, forma, H2H; GBM + regresión Poisson de goles. Validación últimos 2 años:
  log loss 0,851 vs 0,861 solo Elo.
  Córners, tarjetas, remates, remates al arco y faltas de selecciones desde **StatsBomb Open Data**
  (Mundiales 2018/2022, Euro 2020/2024, Copa América 2024, Copa Africana 2023; 314 partidos, solo tiempo
  reglamentario) con regresión Poisson contraída + Elo. Validación dejando fuera cada torneo: MAE remates
  3,72 vs 4,17 y córners 2,03 vs 2,21 frente a la media global.
  Opcional: **API-Football** (clave gratuita propia, 100 solicitudes/día) suma stats de eliminatorias,
  amistosos, Nations League y Mundial 2026. Descarga incremental desde la barra lateral o
  `entrenar.py --api-football`. La clave se guarda en `data/.api_football_key` (excluida de git).
  Ojo: el **plan gratuito solo permite las temporadas 2022–2024** de API-Football (en su numeración: incluye las
  eliminatorias europeas al Mundial 2026 jugadas en 2025-26). Mundial 2026, Nations League 2026-27 y
  amistosos 2025-26 piden plan pago. ~3.300 partidos disponibles → ~5 semanas a 95/día.
- **Matemática de apuestas**: margen (overround), quitar margen (power / Shin / multiplicativo), cuotas
  justas, cuotas con margen simulado, valor esperado (EV) y Kelly fraccional.
- **Backtest walk-forward** honesto por liga y temporada, con comparación contra la línea base "solo mercado".

## Instalación

Requiere Python 3.11+.

```bash
git clone https://github.com/Liilmiguel/Pronosticador.git
cd Pronosticador
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python entrenar.py --scrape      # descarga todo + entrena + histórico fuera de muestra (~25 min)
```

## Uso

```bash
.venv/bin/streamlit run app.py
```

Para actualizar resultados y próximos partidos: botón **Actualizar datos** en la barra lateral
(las temporadas cerradas quedan en caché; solo baja las actuales). El modelo se reentrena solo (~5 min).

Los datos descargados, los modelos entrenados y la clave de API-Football viven en `data/`, que no se sube
al repositorio (`.gitignore`).

## Pestañas

| Pestaña | Qué hace |
|---|---|
| Próximos partidos | Partidos publicados con cuotas → probabilidades, cuotas justas, EV, stake Kelly |
| Analizar partido | Cualquier enfrentamiento + tus cuotas → todos los mercados, marcador exacto, hándicap, córners, tarjetas (con árbitro), forma, H2H |
| Mejor apuesta | Copias las cuotas de Betano (o cualquier casa) de un partido → EV de todos los mercados (1X2, doble oportunidad, DNB, goles, ambos marcan, hándicap asiático, marcador exacto, córners, tarjetas, remates), recomendación con stake Kelly y combinada |
| Selecciones | Cualquier par de selecciones, tipo de torneo, cancha neutral, mercados, ranking Elo mundial |
| Equipos y ligas | Tabla actual + Elo + fuerza Dixon-Coles + evolución Elo + córners/tarjetas/faltas por equipo + árbitros |
| Backtest | Precisión (log loss, Brier) vs mercado y simulación de apuestas por liga/temporada |
| Datos y método | Cobertura de datos, descarga CSV y explicación |

## Estructura

```
predictor/
  config.py       ligas, temporadas, rutas
  scraper.py      descarga masiva + normalización + SQLite (data/futbol.db)
  features.py     variables pre-partido (Elo, forma, H2H, liga, mercado)
  dixon_coles.py  modelo de goles y derivación de mercados (con cancha neutral)
  stats_markets.py córners y tarjetas
  national.py     selecciones nacionales
  national_stats.py córners/tarjetas/remates de selecciones (StatsBomb + API-Football)
  best_bet.py     probabilidades por mercado, valor esperado (con devoluciones), mejor apuesta y combinadas
  apifootball.py  conector API-Football (incremental, respeta límites del plan gratuito)
  models.py       Gradient Boosting, ensamble, stacker, caché de modelos
  betting.py      margen, de-vig, EV, Kelly
  backtest.py     walk-forward, métricas y simulación de apuestas
app.py            interfaz Streamlit
entrenar.py       pipeline por línea de comandos
```

## Resultados honestos (backtest walk-forward 2020–2026, 38 ligas, ~77.000 partidos)

| Modelo | Log loss 1X2 |
|---|---|
| Final (apilado con mercado) | **1.0032** |
| Mercado – apertura (promedio casas) | 1.0038 |
| Mercado – cierre Pinnacle (referencia más dura) | 1.0011 |
| Modelo propio (GBM + Dixon-Coles, sin cuotas) | 1.0192 |

- El modelo propio solo no le gana al mercado; **apilado con el mercado sí supera a la apertura**.
- Apuestas simuladas (EV ≥ 5 %, mejor cuota, prob ≥ 20 %, cuota ≤ 5): 7.676 apuestas, ROI **+3,9 %**,
  positivo en cada año 2020–2026. La línea base "solo mercado + mejor cuota" da +4,1 % pero con años negativos.
- Parte de la ganancia depende de conseguir la **mejor cuota** entre casas; con la cuota promedio el margen
  casi desaparece. Las casas limitan a quienes ganan sistemáticamente.

Apuesta con responsabilidad: un ROI positivo en una muestra pequeña puede ser suerte.

## Descarga automática diaria (selecciones)

En macOS se puede programar con un agente de launchd (`~/Library/LaunchAgents/com.predictor-apuestas.selecciones.plist`,
con `ProgramArguments` = `.venv/bin/python actualizar_selecciones.py` y `StartCalendarInterval` 09:15) que ejecuta
`actualizar_selecciones.py` todos los días a las 09:15 (si el Mac está dormido, al despertar): avanza la
descarga de API-Football y reentrena el modelo de selecciones. Registro: `data/logs/selecciones.log`.

```bash
launchctl kickstart gui/$(id -u)/com.predictor-apuestas.selecciones   # ejecutar ahora
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.predictor-apuestas.selecciones.plist   # desactivar
```

## Fuentes de datos

- [football-data.co.uk](https://www.football-data.co.uk) — resultados, estadísticas y cuotas de clubes.
- [martj42/international_results](https://github.com/martj42/international_results) — resultados de selecciones.
- [eloratings.net](https://www.eloratings.net) — resultados recientes de selecciones.
- [StatsBomb Open Data](https://github.com/statsbomb/open-data) — eventos de grandes torneos (córners, tarjetas, remates).
- [API-Football](https://www.api-football.com) — opcional, con clave propia.

Proyecto con fines educativos. Apuesta con responsabilidad.


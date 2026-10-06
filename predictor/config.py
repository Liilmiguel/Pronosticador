"""Configuración general: ligas, temporadas y rutas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "futbol.db"
MODELS_DIR = DATA_DIR / "models"

BASE_URL = "https://www.football-data.co.uk"

# Ligas "principales": resultados + estadísticas (tiros, córners, tarjetas) + cuotas de ~10 casas.
# código -> (nombre, país, nivel)
MAIN_LEAGUES = {
    "E0": ("Premier League", "Inglaterra", 1),
    "E1": ("Championship", "Inglaterra", 2),
    "E2": ("League One", "Inglaterra", 3),
    "E3": ("League Two", "Inglaterra", 4),
    "EC": ("National League", "Inglaterra", 5),
    "SC0": ("Premiership", "Escocia", 1),
    "SC1": ("Championship", "Escocia", 2),
    "SC2": ("League One", "Escocia", 3),
    "SC3": ("League Two", "Escocia", 4),
    "D1": ("Bundesliga", "Alemania", 1),
    "D2": ("2. Bundesliga", "Alemania", 2),
    "I1": ("Serie A", "Italia", 1),
    "I2": ("Serie B", "Italia", 2),
    "SP1": ("La Liga", "España", 1),
    "SP2": ("Segunda División", "España", 2),
    "F1": ("Ligue 1", "Francia", 1),
    "F2": ("Ligue 2", "Francia", 2),
    "N1": ("Eredivisie", "Países Bajos", 1),
    "B1": ("Pro League", "Bélgica", 1),
    "P1": ("Primeira Liga", "Portugal", 1),
    "T1": ("Süper Lig", "Turquía", 1),
    "G1": ("Super League", "Grecia", 1),
}

# Ligas "extra": solo resultados + cuotas de cierre (sin estadísticas de juego).
EXTRA_LEAGUES = {
    "ARG": ("Liga Profesional", "Argentina", 1),
    "BRA": ("Serie A", "Brasil", 1),
    "MEX": ("Liga MX", "México", 1),
    "USA": ("MLS", "Estados Unidos", 1),
    "AUT": ("Bundesliga", "Austria", 1),
    "DNK": ("Superliga", "Dinamarca", 1),
    "SWZ": ("Super League", "Suiza", 1),
    "NOR": ("Eliteserien", "Noruega", 1),
    "SWE": ("Allsvenskan", "Suecia", 1),
    "FIN": ("Veikkausliiga", "Finlandia", 1),
    "POL": ("Ekstraklasa", "Polonia", 1),
    "ROU": ("Liga 1", "Rumania", 1),
    "IRL": ("Premier Division", "Irlanda", 1),
    "JPN": ("J-League", "Japón", 1),
    "CHN": ("Super League", "China", 1),
    "RUS": ("Premier League", "Rusia", 1),
}

ALL_LEAGUES = {**MAIN_LEAGUES, **EXTRA_LEAGUES}

# Nombre de país tal como aparece en los CSV "new/" -> código de liga
EXTRA_COUNTRY_TO_CODE = {
    "Argentina": "ARG", "Brazil": "BRA", "Mexico": "MEX", "USA": "USA", "Austria": "AUT",
    "Denmark": "DNK", "Switzerland": "SWZ", "Norway": "NOR", "Sweden": "SWE", "Finland": "FIN",
    "Poland": "POL", "Romania": "ROU", "Ireland": "IRL", "Japan": "JPN", "China": "CHN",
    "Russia": "RUS",
}

FIRST_SEASON = 2000  # temporada 2000/01
CURRENT_SEASON = 2026  # temporada 2026/27


def season_code(start_year: int) -> str:
    """2026 -> '2627'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def league_label(code: str) -> str:
    name, country, _ = ALL_LEAGUES[code]
    return f"{country} · {name} ({code})"

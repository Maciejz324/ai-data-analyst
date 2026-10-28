"""Oficjalny zbiór UCI 275: pobranie bez podmiany danych i walidacja hour.csv."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from src.analysis_validation import AnalysisError
from src.data_loader import load_csv

DATASET_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_DIR = Path(__file__).parents[1] / "data" / "bike_sharing"
HOUR_PATH = DATA_DIR / "hour.csv"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SCHEMA = ["instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]


def validate_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sprawdź znaczenie pól i zwróć kopię posortowaną po rzeczywistym czasie.

    Braki pogodowe pozostają do raportowania/imputacji; braki kalendarza,
    celu i liczników są błędem. Nie uzupełniamy brakujących godzin w osi czasu.
    """
    if not df.columns.is_unique or set(df.columns) != set(SCHEMA):
        raise AnalysisError("hour.csv musi zawierać dokładnie 17 kolumn oryginalnego schematu UCI.")
    if len(df) < 4:
        raise AnalysisError("Badanie Bike Sharing wymaga co najmniej czterech obserwacji.")
    result = df.copy(deep=True)
    for column in [name for name in SCHEMA if name != "dteday"]:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise AnalysisError(f"Niepoprawne wartości liczbowe w kolumnie {column}.") from exc
        if np.isinf(result[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise AnalysisError(f"Nieskończone wartości w kolumnie {column}.")
    weather = {"temp", "atemp", "hum", "windspeed"}
    required = [name for name in SCHEMA if name not in weather]
    if result[required].isna().any().any():
        raise AnalysisError("Brakujące daty, kody kalendarza, liczniki lub cnt uniemożliwiają badanie.")
    try:
        dates = pd.to_datetime(result["dteday"], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise AnalysisError("Niepoprawna data dteday w hour.csv.") from exc
    if dates.isna().any():
        raise AnalysisError("Brak daty dteday.")
    categories = {"season": range(1, 5), "yr": range(2), "mnth": range(1, 13), "hr": range(24), "holiday": range(2), "weekday": range(7), "workingday": range(2), "weathersit": range(1, 5)}
    for column, allowed in categories.items():
        if not result[column].isin(allowed).all():
            raise AnalysisError(f"Niepoprawne kody kategorii w kolumnie {column}.")
    for column in ["instant", "cnt", "casual", "registered"]:
        values = result[column]
        if (values < 0).any() or (values % 1 != 0).any():
            raise AnalysisError(f"Kolumna {column} wymaga nieujemnych liczb całkowitych.")
    if result["instant"].duplicated().any():
        raise AnalysisError("Identyfikatory instant muszą być unikalne.")
    if not result["cnt"].eq(result["casual"] + result["registered"]).all():
        raise AnalysisError("Naruszona zależność cnt = casual + registered.")
    if (not dates.dt.year.isin([2011, 2012]).all()
            or not result["yr"].eq(dates.dt.year - 2011).all()
            or not result["mnth"].eq(dates.dt.month).all()
            or not result["weekday"].eq((dates.dt.dayofweek + 1) % 7).all()):
        raise AnalysisError("Kody roku, miesiąca lub dnia tygodnia nie odpowiadają datom UCI 2011–2012.")
    expected_workday = ((~dates.dt.dayofweek.isin([5, 6])) & result["holiday"].eq(0)).astype(int)
    if not result["workingday"].eq(expected_workday).all():
        raise AnalysisError("workingday nie odpowiada weekendom i świętom.")
    for column in weather:
        observed = result[column].dropna()
        if not observed.between(0, 1).all():
            raise AnalysisError(f"Kolumna {column} powinna zawierać wartości znormalizowane UCI w [0,1].")
    timestamps = dates + pd.to_timedelta(result["hr"], unit="h")
    if timestamps.duplicated().any():
        raise AnalysisError("Powtórzone znaczniki godzinowe w hour.csv.")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    return result.iloc[order].reset_index(drop=True)


def download_bike_sharing(directory: Path = DATA_DIR) -> Path:
    """Pobierz wyłącznie oficjalne archiwum HTTPS. Zachowaj bajty oryginału."""
    directory = Path(directory)
    target = directory / "hour.csv"
    if target.exists():
        load_bike_sharing(directory)
        return target
    try:
        with urlopen(ARCHIVE_URL, timeout=30) as response:
            archive = response.read(8 * 1024 * 1024 + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise AnalysisError(f"Nie pobrano danych z oficjalnego UCI: {exc}. Nie zastąpiono danych innym zbiorem.") from exc
    if len(archive) > 8 * 1024 * 1024:
        raise AnalysisError("Archiwum UCI przekracza limit 8 MiB.")
    try:
        with ZipFile(BytesIO(archive)) as bundle:
            member = bundle.getinfo("hour.csv")
            if member.file_size > 10 * 1024 * 1024:
                raise AnalysisError("hour.csv przekracza limit 10 MiB.")
            raw = bundle.read(member)
            readme = bundle.read("Readme.txt") if "Readme.txt" in bundle.namelist() else None
    except (BadZipFile, KeyError, RuntimeError) as exc:
        raise AnalysisError("Oficjalne archiwum nie zawiera poprawnego hour.csv.") from exc
    frame = validate_hourly_data(load_csv(raw, separator=",", encoding="utf-8-sig"))
    manifest = {
        "dataset": "UCI Bike Sharing Dataset (275)", "source_url": DATASET_URL,
        "download_url": ARCHIVE_URL, "license": "CC BY 4.0", "license_url": LICENSE_URL,
        "attribution": "Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. DOI: 10.24432/C5W894.",
        "archive_sha256": sha256(archive).hexdigest(), "hour_sha256": sha256(raw).hexdigest(),
        "rows": len(frame), "columns": SCHEMA,
    }
    directory.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    (directory / "source_archive.zip").write_bytes(archive)
    if readme is not None:
        (directory / "Readme_UCI.txt").write_bytes(readme)
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return target


def load_bike_sharing(directory: Path = DATA_DIR) -> pd.DataFrame:
    """Odczyt lokalnego oryginału, kontrola manifestu i integralności SHA-256."""
    directory = Path(directory)
    try:
        raw = (directory / "hour.csv").read_bytes()
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise AnalysisError("Brak oryginalnego zbioru UCI lub manifestu. Uruchom: python -m experiments.bike_sharing.run --download") from exc
    if manifest.get("download_url") != ARCHIVE_URL or manifest.get("hour_sha256") != sha256(raw).hexdigest():
        raise AnalysisError("Pochodzenie lub SHA-256 hour.csv nie zgadza się z manifestem. Oryginał nie może być edytowany.")
    frame = load_csv(raw, separator=",", encoding="utf-8-sig")
    validated = validate_hourly_data(frame)
    if manifest.get("rows") != len(validated):
        raise AnalysisError("Liczba rekordów różni się od manifestu pobrania.")
    # EDA w aplikacji otrzymuje oryginalne wartości; chronologię porządkuje moduł badania.
    return frame

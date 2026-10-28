"""Czyste funkcje analityczne, gotowe do wywołania przez przyszły plan analiz."""

import numpy as np
import pandas as pd

STAT_COLUMNS = ["count", "mean", "median", "min", "max", "std"]


def numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Kolumny liczbowe bez bool; nieskończoności traktowane jak braki."""
    return df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Statystyki per kolumna; std z próby (ddof=1), NaN gdy niewyznaczalne."""
    numeric = numeric_data(df)
    if numeric.shape[1] == 0:
        return pd.DataFrame(columns=STAT_COLUMNS)
    result = pd.DataFrame(index=numeric.columns, columns=STAT_COLUMNS, dtype=float)
    counts = numeric.count()
    result["count"] = counts
    populated = numeric.loc[:, counts > 0]
    if populated.shape[1]:
        result.loc[populated.columns, STAT_COLUMNS] = populated.agg(STAT_COLUMNS).T
    return result


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Braki w oryginalnych danych; dla pustego zbioru procent wynosi 0."""
    counts = df.isna().sum()
    return pd.DataFrame({
        "missing_count": counts,
        "missing_percent": counts / len(df) * 100 if len(df) else counts.astype(float),
    })


def column_information(df: pd.DataFrame) -> pd.DataFrame:
    info = missing_values(df)
    info.insert(0, "dtype", df.dtypes.astype(str))
    info.index = info.index.copy()
    info.index.name = "Kolumna"
    return info


def pearson_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson parami na >=2 wspólnych obserwacjach; pomija stałe kolumny.

    Para bez wystarczających danych ma NaN. Przy <2 odpowiednich kolumnach
    zwraca pustą ramkę. Korelacja nie oznacza zależności przyczynowej.
    """
    numeric = numeric_data(df)
    valid = numeric.loc[:, numeric.nunique(dropna=True) >= 2]
    if valid.shape[1] < 2:
        return pd.DataFrame()
    return valid.corr(method="pearson", min_periods=2)

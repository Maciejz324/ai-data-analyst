"""Niemutujące operacje eksploracyjne; nie dopasowuj ich przed podziałem ML."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.analysis_validation import AnalysisError, numeric_columns, numeric_selection


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Usuń wiersze z brakami w dowolnej kolumnie; inf również oznacza brak."""
    result = df.copy(deep=True)
    for column in numeric_columns(df):
        if np.isinf(df[column].to_numpy(dtype=float, na_value=np.nan)).any():
            result[column] = numeric_selection(df, [column])[column]
    return result.dropna().copy()


def impute_missing(df: pd.DataFrame, columns: Sequence[str], strategy: str = "median") -> pd.DataFrame:
    """Uzupełnij wybrane kolumny średnią/medianą; pozostałe pozostają bez zmian."""
    if strategy not in ("mean", "median"):
        raise AnalysisError("Uzupełnianie braków wymaga metody mean lub median.")
    data = numeric_selection(df, columns)
    if data.empty or data.isna().all().any():
        raise AnalysisError("Nie można uzupełnić kolumny bez skończonych obserwacji.")
    values = data.mean() if strategy == "mean" else data.median()
    result = df.copy(deep=True)
    for column in columns:
        result[column] = data[column].fillna(values[column])
    return result


@dataclass
class IQRResult:
    """Maska per komórka i progi; brak danych nie jest wartością odstającą."""

    mask: pd.DataFrame
    bounds: pd.DataFrame


def detect_iqr_outliers(df: pd.DataFrame, columns: Sequence[str], factor: float = 1.5) -> IQRResult:
    """Wartości poza [Q1 - factor*IQR, Q3 + factor*IQR]; kwartyle liniowe."""
    if isinstance(factor, bool) or not isinstance(factor, Real) or not np.isfinite(factor) or factor <= 0:
        raise AnalysisError("Mnożnik IQR musi być dodatnią, skończoną liczbą.")
    data = numeric_selection(df, columns)
    if data.empty:
        raise AnalysisError("Wykrywanie wartości odstających wymaga wierszy danych.")
    q1 = data.quantile(0.25, interpolation="linear")
    q3 = data.quantile(0.75, interpolation="linear")
    spread = q3 - q1
    lower, upper = q1 - factor * spread, q3 + factor * spread
    mask = (data.lt(lower) | data.gt(upper)).fillna(False)
    bounds = pd.DataFrame({"q1": q1, "q3": q3, "iqr": spread, "lower": lower, "upper": upper})
    bounds["outlier_count"] = mask.sum()
    return IQRResult(mask=mask, bounds=bounds)


def scale_numeric(df: pd.DataFrame, columns: Sequence[str], method: str = "standard") -> pd.DataFrame:
    """Min–max [0,1] lub z-score (ddof=0); braki pozostają brakami.

    Stała kolumna otrzymuje 0. To operacja eksploracyjna dopasowana na całym
    przekazanym zbiorze, nie transformator do oceny modeli na zbiorze testowym.
    """
    if method not in ("standard", "minmax"):
        raise AnalysisError("Skalowanie wymaga metody standard lub minmax.")
    data = numeric_selection(df, columns)
    if data.empty or data.isna().all().any():
        raise AnalysisError("Skalowanie wymaga co najmniej jednej skończonej wartości w każdej kolumnie.")
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    transformed = scaler.fit_transform(data)
    result = df.copy(deep=True)
    for i, column in enumerate(columns):
        result[column] = transformed[:, i]
    return result

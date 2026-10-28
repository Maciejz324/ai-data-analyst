"""Walidacja wspólna dla funkcji przygotowania danych i modeli."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


class AnalysisError(ValueError):
    """Niepoprawne dane lub parametry analizy; komunikat dla użytkownika."""


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Rzeczywiste cechy liczbowe, bez bool i liczb zespolonych."""
    return [name for name, dtype in df.dtypes.items()
            if is_numeric_dtype(dtype) and not is_bool_dtype(dtype) and not is_complex_dtype(dtype)]


def numeric_selection(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Niezależna kopia float; +/-inf są brakami, indeks pozostaje bez zmian."""
    if isinstance(columns, str) or not isinstance(columns, Sequence) or not columns:
        raise AnalysisError("Wybierz co najmniej jedną kolumnę numeryczną.")
    if not df.columns.is_unique:
        raise AnalysisError("Nazwy kolumn muszą być unikalne.")
    if len(set(columns)) != len(columns):
        raise AnalysisError("Wybrane kolumny nie mogą się powtarzać.")
    if any(column not in numeric_columns(df) for column in columns):
        raise AnalysisError("Wybierz istniejące kolumny numeryczne (bez bool i liczb zespolonych).")
    return df[list(columns)].astype(float).replace([np.inf, -np.inf], np.nan).copy()


def validate_random_state(random_state: int) -> None:
    if isinstance(random_state, bool) or not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
        raise AnalysisError("random_state musi być liczbą całkowitą od 0 do 4294967295.")

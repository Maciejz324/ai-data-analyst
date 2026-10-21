"""Wykresy Plotly z walidacją danych niezależną od Streamlit."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_analysis import numeric_data, pearson_correlation


class VisualizationError(ValueError):
    """Nie można przygotować wykresu dla wybranych danych."""


def _numeric_columns(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    numeric = numeric_data(df)
    if any(col not in numeric.columns for col in columns):
        raise VisualizationError("Wybierz istniejące kolumny numeryczne.")
    return numeric


def histogram(df: pd.DataFrame, column: str, bins: int = 20) -> go.Figure:
    data = _numeric_columns(df, column)[[column]].dropna()
    if data.empty:
        raise VisualizationError("Wybrana kolumna nie zawiera skończonych wartości liczbowych.")
    if not isinstance(bins, int) or not 1 <= bins <= 100:
        raise VisualizationError("Liczba przedziałów musi wynosić od 1 do 100.")
    fig = px.histogram(data, x=column, nbins=bins, title=f"Rozkład: {column}")
    fig.update_layout(yaxis_title="Liczba obserwacji", bargap=0.05)
    return fig


def scatter_plot(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    if x == y:
        raise VisualizationError("Wybierz dwie różne kolumny.")
    data = _numeric_columns(df, x, y)[[x, y]].dropna()
    if data.empty:
        raise VisualizationError("Brak wspólnych kompletnych obserwacji dla wybranych kolumn.")
    return px.scatter(data, x=x, y=y, title=f"{y} a {x}", render_mode="webgl")


def bar_chart(df: pd.DataFrame, category: str, value: str | None = None) -> go.Figure:
    """Liczebność kategorii lub średnia wartości numerycznej w kategorii."""
    if category not in df.columns:
        raise VisualizationError("Wybierz istniejącą kolumnę kategorii.")
    if df[category].nunique() > 50:
        raise VisualizationError("Wybierz kolumnę z co najwyżej 50 kategoriami.")
    # Wewnętrzna tabela z neutralnymi nazwami chroni przed kolizjami
    # dowolnych nazw kolumn CSV z nazwami wyników agregacji.
    data = pd.DataFrame({"category": df[category]})
    if value is None:
        grouped = data.dropna().groupby("category").size().rename("result").reset_index()
        label = "Liczba obserwacji"
    else:
        data["value"] = _numeric_columns(df, value)[value]
        grouped = data.dropna().groupby("category")["value"].mean().rename("result").reset_index()
        label = f"Średnia: {value}"
    if grouped.empty:
        raise VisualizationError("Brak kompletnych danych do wykresu słupkowego.")
    fig = px.bar(grouped, x="category", y="result", labels={"category": category, "result": label})
    fig.update_xaxes(type="category")
    return fig


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> go.Figure:
    if columns is not None:
        _numeric_columns(df, *columns)
        df = df[list(dict.fromkeys(columns))]
    corr = pearson_correlation(df)
    if corr.empty:
        raise VisualizationError("Potrzebne są co najmniej dwie niestałe kolumny numeryczne.")
    if not corr.where(~np.eye(len(corr), dtype=bool)).notna().any().any():
        raise VisualizationError("Brak par kolumn z wyznaczalną korelacją.")
    return px.imshow(
        corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
        text_auto=".2f", aspect="auto", title="Korelacja Pearsona",
        labels={"color": "Pearson r"},
    )

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


def regression_prediction_plot(predictions: pd.DataFrame) -> go.Figure:
    """Wartości rzeczywiste i przewidywane wyłącznie dla części testowej."""
    if not {"actual", "predicted"}.issubset(predictions.columns):
        raise VisualizationError("Brak wartości rzeczywistych i przewidywanych.")
    data = predictions[["actual", "predicted"]]
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise VisualizationError("Wykres predykcji wymaga skończonych wyników.")
    fig = px.scatter(data, x="actual", y="predicted", labels={
        "actual": "Wartość rzeczywista", "predicted": "Wartość przewidywana",
    }, title="Regresja — zbiór testowy", render_mode="webgl")
    lower, upper = float(data.min().min()), float(data.max().max())
    fig.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Idealna predykcja"))
    return fig


def cluster_plot(data: pd.DataFrame, labels: pd.Series, x: str, y: str | None = None) -> go.Figure:
    """Rzut skalowanych cech; dla jednej cechy oś Y to pozycja wiersza."""
    if y == x:
        raise VisualizationError("Wybierz dwie różne osie klastrów.")
    columns = [x] if y is None else [x, y]
    values = _numeric_columns(data, *columns)[columns]
    if len(values) != len(labels) or values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise VisualizationError("Wykres klastrów wymaga skończonych cech i etykiety dla każdego wiersza.")
    # Wewnętrzne nazwy nie kolidują z dowolnymi nazwami kolumn użytkownika.
    plot_data = pd.DataFrame({
        "x": values[x].to_numpy(),
        "y": values[y].to_numpy() if y is not None else np.arange(len(values)),
        "cluster": labels.to_numpy().astype(str),
    })
    return px.scatter(plot_data, x="x", y="y", color="cluster", labels={
        "x": f"{x} (po skalowaniu)",
        "y": f"{y} (po skalowaniu)" if y is not None else "Pozycja wiersza",
        "cluster": "Klaster",
    }, title="Klastery K-Means", render_mode="webgl")


def _pca_variance_table(variance: pd.DataFrame) -> pd.DataFrame:
    columns = ["explained_variance_ratio", "cumulative_explained_variance_ratio"]
    if variance.empty or not set(columns).issubset(variance.columns):
        raise VisualizationError("Wykres PCA wymaga udziałów wariancji i wariancji skumulowanej.")
    try:
        values = variance[columns].astype(float)
    except (TypeError, ValueError) as exc:
        raise VisualizationError("Wariancja PCA musi być liczbowa.") from exc
    if (not np.isfinite(values.to_numpy()).all() or (values < 0).any().any()
            or (values > 1 + 1e-12).any().any()
            or not np.allclose(values[columns[0]].cumsum(), values[columns[1]], atol=1e-12, rtol=1e-12)):
        raise VisualizationError("Udziały wariancji PCA muszą być skończone, nieujemne i mieć poprawną sumę skumulowaną.")
    return values


def pca_scree_plot(variance: pd.DataFrame) -> go.Figure:
    values = _pca_variance_table(variance)
    fig = go.Figure(go.Bar(x=values.index, y=values["explained_variance_ratio"] * 100))
    fig.update_layout(title="PCA — wariancja wyjaśniona (scree plot)", xaxis_title="Składowa", yaxis_title="Wyjaśniona wariancja (%)")
    fig.update_yaxes(range=[0, 100])
    return fig


def pca_cumulative_variance_plot(variance: pd.DataFrame) -> go.Figure:
    values = _pca_variance_table(variance)
    fig = go.Figure(go.Scatter(x=values.index, y=values["cumulative_explained_variance_ratio"] * 100, mode="lines+markers"))
    fig.update_layout(title="PCA — skumulowana wariancja wyjaśniona", xaxis_title="Liczba kolejnych składowych", yaxis_title="Skumulowana wariancja (%)")
    fig.update_yaxes(range=[0, 100])
    return fig


def pca_scatter_plot(scores: pd.DataFrame) -> go.Figure:
    if not {"PC1", "PC2"}.issubset(scores.columns):
        raise VisualizationError("Wykres PCA 2D wymaga co najmniej dwóch składowych: PC1 i PC2.")
    data = _numeric_columns(scores, "PC1", "PC2")[["PC1", "PC2"]]
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise VisualizationError("Wykres PCA 2D wymaga skończonych współrzędnych obserwacji.")
    return px.scatter(data, x="PC1", y="PC2", title="PCA — obserwacje w przestrzeni PC1 i PC2", render_mode="webgl")

"""Modele niezależne od UI, z jawnym przygotowaniem danych i metadanymi."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.analysis_validation import AnalysisError, numeric_selection, validate_random_state

SILHOUETTE_SAMPLE_LIMIT = 2000


def _preprocessing(imputation: str, scaling: str) -> list[tuple]:
    if imputation not in ("mean", "median"):
        raise AnalysisError("Imputacja wymaga metody mean lub median.")
    if scaling not in ("standard", "minmax", "none"):
        raise AnalysisError("Skalowanie wymaga metody standard, minmax lub none.")
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": "passthrough"}[scaling]
    return [("imputer", SimpleImputer(strategy=imputation)), ("scaler", scaler)]


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    """MAE, RMSE i R²; R²=None dla <2 wyników lub stałej zmiennej celu."""
    try:
        y, prediction = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Metryki wymagają skończonych wektorów liczbowych.") from exc
    if (y.ndim != 1 or prediction.ndim != 1 or y.shape != prediction.shape or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(prediction).all()):
        raise AnalysisError("Metryki wymagają niepustych, równych i skończonych wektorów.")
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if len(y) >= 2 and np.ptp(y) > 0 else None,
    }


@dataclass
class RegressionResult:
    pipeline: Pipeline
    metrics: dict[str, float | None]
    coefficients: pd.Series
    intercept: float
    predictions: pd.DataFrame
    train_positions: np.ndarray
    test_positions: np.ndarray
    dropped_target_rows: int


def train_linear_regression(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    imputation: str = "median",
    scaling: str = "standard",
) -> RegressionResult:
    """Podział surowych danych, potem fit imputera, skalera i modelu na train.

    Wywołujący musi przekazać dane sprzed transformacji dopasowywanych na
    całym zbiorze. Brakujące cele są odrzucane, nigdy imputowane. Pozycje
    odnoszą się do oryginalnej ramki i są jednoznaczne także przy powtórzonym indeksie.
    Współczynniki odnoszą się do cech po przygotowaniu przez pipeline.
    """
    validate_random_state(random_state)
    if (isinstance(test_size, bool) or not isinstance(test_size, Real)
            or not np.isfinite(test_size) or not 0 < test_size < 1):
        raise AnalysisError("Udział zbioru testowego musi być liczbą pomiędzy 0 a 1.")
    steps = _preprocessing(imputation, scaling)
    data = numeric_selection(df, features)
    y = numeric_selection(df, [target])[target]
    if target in features:
        raise AnalysisError("Zmienna docelowa nie może należeć do zmiennych wejściowych.")
    positions = np.flatnonzero(y.notna().to_numpy())
    n_test = int(np.ceil(len(positions) * test_size))
    if n_test < 2 or len(positions) - n_test < 2:
        raise AnalysisError("Po odrzuceniu brakujących celów potrzeba co najmniej 2 wierszy w train i 2 w test. Zmień podział lub dane.")
    train, test = train_test_split(positions, test_size=float(test_size), random_state=int(random_state), shuffle=True)
    if data.iloc[train].isna().all().any():
        raise AnalysisError("Wybrana cecha nie ma skończonych wartości w części treningowej. Wybierz inne cechy lub podział.")
    pipeline = Pipeline(steps + [("model", LinearRegression())])
    pipeline.fit(data.iloc[train], y.iloc[train])
    prediction = pipeline.predict(data.iloc[test])
    model = pipeline.named_steps["model"]
    return RegressionResult(
        pipeline=pipeline,
        metrics=regression_metrics(y.iloc[test], prediction),
        coefficients=pd.Series(model.coef_, index=list(features), name="coefficient"),
        intercept=float(model.intercept_),
        predictions=pd.DataFrame({"actual": y.iloc[test].to_numpy(), "predicted": prediction}, index=df.index.take(test)),
        train_positions=train, test_positions=test, dropped_target_rows=len(df) - len(positions),
    )


@dataclass
class KMeansResult:
    pipeline: Pipeline
    labels: pd.Series
    inertia: float
    silhouette: float | None
    silhouette_note: str
    scaled_data: pd.DataFrame
    centers: pd.DataFrame


def train_kmeans(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    n_clusters: int = 3,
    random_state: int = 42,
    imputation: str = "median",
    scaling: str = "standard",
) -> KMeansResult:
    """K-Means na imputowanych, skalowanych cechach; dane źródłowe bez zmian.

    Cały przekazany zbiór jest uczący (analiza nienadzorowana). Inertia
    i silhouette są liczone w przestrzeni skalowanej; centra w jednostkach
    wejściowych. Jawne parametry ułatwiają przyszłe porównanie z R.
    """
    validate_random_state(random_state)
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, Integral) or n_clusters < 1:
        raise AnalysisError("Liczba klastrów musi być dodatnią liczbą całkowitą.")
    if scaling not in ("standard", "minmax"):
        raise AnalysisError("K-Means wymaga skalowania standard lub minmax.")
    preprocessing = Pipeline(_preprocessing(imputation, scaling))
    data = numeric_selection(df, features)
    if len(data) < 2 or n_clusters > len(data):
        raise AnalysisError("Potrzeba co najmniej dwóch obserwacji i nie więcej klastrów niż wierszy.")
    if data.isna().all().any():
        raise AnalysisError("Każda cecha musi mieć co najmniej jedną skończoną wartość.")
    transformed = preprocessing.fit_transform(data)
    if n_clusters > len(np.unique(transformed, axis=0)):
        raise AnalysisError("Liczba klastrów przekracza liczbę różnych obserwacji po przygotowaniu danych.")
    model = KMeans(
        n_clusters=int(n_clusters), init="k-means++", n_init=10, max_iter=300,
        tol=1e-4, algorithm="lloyd", random_state=int(random_state),
    )
    labels = model.fit_predict(transformed)
    pipeline = Pipeline(preprocessing.steps + [("model", model)])
    distinct_labels = len(np.unique(labels))
    score = None
    note = "Silhouette wymaga od 2 do n−1 niepustych klastrów."
    if 1 < distinct_labels < len(data):
        sampled = len(data) > SILHOUETTE_SAMPLE_LIMIT
        try:
            score = float(silhouette_score(
                transformed, labels,
                sample_size=SILHOUETTE_SAMPLE_LIMIT if sampled else None,
                random_state=int(random_state),
            ))
            note = f"Silhouette na losowej próbie {SILHOUETTE_SAMPLE_LIMIT} obserwacji." if sampled else "Silhouette na całym zbiorze."
        except ValueError:
            note = "Wylosowana próba nie pozwala obliczyć silhouette (liczba reprezentowanych klastrów)."
    centers = preprocessing.named_steps["scaler"].inverse_transform(model.cluster_centers_)
    return KMeansResult(
        pipeline=pipeline, labels=pd.Series(labels, index=df.index, name="cluster"),
        inertia=float(model.inertia_), silhouette=score, silhouette_note=note,
        scaled_data=pd.DataFrame(transformed, index=df.index, columns=list(features)),
        centers=pd.DataFrame(centers, columns=list(features)).rename_axis("cluster"),
    )

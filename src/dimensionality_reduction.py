"""PCA niezależne od Streamlit: przygotowanie cech, projekcja i diagnostyka."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.analysis_validation import AnalysisError, numeric_selection


@dataclass
class PCAResult:
    """Wagi to kierunki osi w przestrzeni przygotowanych cech, nie procenty.

    Wiersze scores zachowują indeks wejścia, także gdy indeks się powtarza.
    Pipeline jest dopasowany do całej przekazanej kopii (analiza nienadzorowana).
    """

    pipeline: Pipeline
    scores: pd.DataFrame
    components: pd.DataFrame
    variance: pd.DataFrame
    constant_features: tuple[str, ...]


def run_pca(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    n_components: int = 2,
    imputation: str = "median",
    scaling: str = "standard",
) -> PCAResult:
    """SimpleImputer → skaler → PCA(full, whiten=False), bez mutowania df.

    Wymaga >=2 rzeczywistych cech i >=2 obserwacji. +/-inf oznaczają braki.
    Puste cechy i zerowa całkowita wariancja są błędami. Stałe cechy mogą
    pozostać obok zmiennych, ale są jawnie raportowane. Dopuszczalna liczba
    składowych: 1..min(n_samples, n_features); niektóre mogą mieć zerową wariancję.
    Proporcje wariancji odnoszą się do wszystkich przygotowanych cech, również
    gdy wybrano tylko część składowych. Dla późniejszej oceny nadzorowanego ML
    należy ponownie dopasować cały pipeline wyłącznie na części treningowej.
    """
    if imputation not in ("mean", "median"):
        raise AnalysisError("Imputacja PCA wymaga metody mean lub median.")
    if scaling not in ("standard", "minmax"):
        raise AnalysisError("Skalowanie PCA wymaga metody standard lub minmax.")
    data = numeric_selection(df, features)
    if data.shape[1] < 2:
        raise AnalysisError("PCA wymaga co najmniej dwóch kolumn numerycznych.")
    if len(data) < 2:
        raise AnalysisError("PCA wymaga co najmniej dwóch obserwacji.")
    limit = min(data.shape)
    if isinstance(n_components, bool) or not isinstance(n_components, Integral) or not 1 <= n_components <= limit:
        raise AnalysisError(f"Liczba składowych PCA musi być całkowita i wynosić od 1 do {limit}.")
    empty = data.columns[data.isna().all()].tolist()
    if empty:
        raise AnalysisError(f"Nie można uzupełnić cech bez skończonych wartości: {', '.join(map(str, empty))}.")
    constant = tuple(data.columns[data.nunique(dropna=True) <= 1])
    if len(constant) == data.shape[1]:
        raise AnalysisError("Całkowita wariancja jest zerowa: wszystkie wybrane cechy są stałe po imputacji.")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputation)),
        ("scaler", StandardScaler() if scaling == "standard" else MinMaxScaler()),
        ("pca", PCA(n_components=int(n_components), svd_solver="full", whiten=False)),
    ])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scores = pipeline.fit_transform(data)
            model = pipeline.named_steps["pca"]
            cumulative = np.cumsum(model.explained_variance_ratio_)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise AnalysisError("Nie można wyznaczyć skończonych wyników PCA. Sprawdź zakres wartości i wariancję wybranych cech.") from exc
    outputs = (scores, model.components_, model.explained_variance_, model.explained_variance_ratio_, cumulative)
    if not all(np.isfinite(array).all() for array in outputs) or model.explained_variance_.sum() <= 0:
        raise AnalysisError("PCA dało niewyznaczalne wyniki lub zerową wariancję. Wybierz inne cechy.")
    names = [f"PC{i + 1}" for i in range(int(n_components))]
    return PCAResult(
        pipeline=pipeline,
        scores=pd.DataFrame(scores, index=df.index.copy(), columns=names),
        components=pd.DataFrame(model.components_, index=names, columns=list(features)).rename_axis("component"),
        variance=pd.DataFrame({
            "explained_variance": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative,
        }, index=names).rename_axis("component"),
        constant_features=constant,
    )

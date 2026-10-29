"""Chronologiczna regresja i nienadzorowane badanie pogody Bike Sharing."""

from dataclasses import asdict, dataclass
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.analysis_validation import AnalysisError, validate_random_state
from src.bike_sharing_data import validate_hourly_data
from src.data_analysis import column_information
from src.data_preparation import detect_iqr_outliers
from src.dimensionality_reduction import PCAResult, run_pca
from src.ml_models import regression_metrics
from src.visualizations import pca_cumulative_variance_plot, pca_scatter_plot, pca_scree_plot

CONFIG_PATH = Path(__file__).parents[1] / "experiments" / "bike_sharing" / "config.json"
FORBIDDEN_FEATURES = {"cnt", "casual", "registered", "instant", "dteday"}
WEATHER_COLUMNS = {"temp", "atemp", "hum", "windspeed"}
CALENDAR_COLUMNS = {"season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"}


@dataclass(frozen=True)
class StudyConfig:
    test_fraction: float = 0.2
    random_state: int = 42
    weather_features: tuple[str, ...] = ("temp", "hum", "windspeed")
    regression_numeric: tuple[str, ...] = ("temp", "hum", "windspeed")
    regression_categorical: tuple[str, ...] = ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit")
    cluster_counts: tuple[int, ...] = (2, 3, 4, 5, 6)
    n_init: int = 10
    max_iter: int = 300
    silhouette_sample_size: int = 2000
    pca_components: int = 2
    thread_limit: int = 1

    def validate(self) -> None:
        validate_random_state(self.random_state)
        if isinstance(self.test_fraction, bool) or not isinstance(self.test_fraction, Real) or not np.isfinite(self.test_fraction) or not 0 < self.test_fraction < 1:
            raise AnalysisError("Udział testu w badaniu musi być pomiędzy 0 a 1.")
        for columns, allowed in [(self.weather_features, WEATHER_COLUMNS), (self.regression_numeric, WEATHER_COLUMNS), (self.regression_categorical, CALENDAR_COLUMNS)]:
            if not columns or len(set(columns)) != len(columns):
                raise AnalysisError("Listy cech badania muszą być niepuste i unikalne.")
            if set(columns) & FORBIDDEN_FEATURES or not set(columns) <= allowed:
                raise AnalysisError("Niedozwolona cecha: cel, jego składniki, identyfikator lub kolumna spoza ustalonej listy.")
        if len(self.weather_features) < 2:
            raise AnalysisError("PCA i klasteryzacja wymagają co najmniej dwóch cech pogodowych.")
        if (not self.cluster_counts or len(set(self.cluster_counts)) != len(self.cluster_counts)
                or any(isinstance(k, bool) or not isinstance(k, Integral) or not 2 <= k <= 6 for k in self.cluster_counts)):
            raise AnalysisError("Porównywane k muszą być unikalnymi liczbami całkowitymi od 2 do 6.")
        for value in [self.n_init, self.max_iter, self.thread_limit, self.silhouette_sample_size]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise AnalysisError("Parametry iteracji, próby i liczby wątków muszą być dodatnimi liczbami całkowitymi.")
        if self.pca_components != 2 or isinstance(self.pca_components, bool) or not isinstance(self.pca_components, Integral):
            raise AnalysisError("Badanie porównawcze używa dokładnie dwóch składowych PCA.")
        if self.silhouette_sample_size <= max(self.cluster_counts):
            raise AnalysisError("Próba silhouette musi być większa od maksymalnego k.")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "StudyConfig":
        try:
            values = json.loads(Path(path).read_text())
            for name in ["weather_features", "regression_numeric", "regression_categorical", "cluster_counts"]:
                if name in values:
                    values[name] = tuple(values[name])
            config = cls(**values)
            config.validate()
            return config
        except (OSError, ValueError, TypeError) as exc:
            if isinstance(exc, AnalysisError):
                raise
            raise AnalysisError("Niepoprawny plik konfiguracji badania Bike Sharing.") from exc


@dataclass
class StudyResult:
    config: StudyConfig
    metadata: dict
    tables: dict[str, pd.DataFrame]
    figures: dict[str, go.Figure]
    regression_pipeline: Pipeline
    pca: PCAResult
    cluster_models: dict[str, dict[int, KMeans]]


def hourly_timestamps(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["dteday"], format="%Y-%m-%d") + pd.to_timedelta(df["hr"], unit="h")


def chronological_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = validate_hourly_data(df)
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, Real) or not np.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise AnalysisError("Chronologiczny udział testu musi być pomiędzy 0 a 1.")
    cutoff = int(np.floor(len(ordered) * (1 - test_fraction)))
    if cutoff < 2 or len(ordered) - cutoff < 2:
        raise AnalysisError("Chronologiczny podział wymaga co najmniej dwóch wierszy train i dwóch test.")
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def _group_demand(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.groupby(column)["cnt"].agg(["count", "mean", "median", "std", "min", "max"])


def _error_groups(test: pd.DataFrame, residual: np.ndarray, column: str) -> pd.DataFrame:
    data = pd.DataFrame({"group": test[column].to_numpy(), "residual": residual})
    rows = []
    for group, values in data.groupby("group")["residual"]:
        rows.append({column: group, "count": len(values), "mae": values.abs().mean(), "rmse": np.sqrt(np.square(values).mean()), "bias_actual_minus_predicted": values.mean()})
    return pd.DataFrame(rows).set_index(column)


def _cluster_grid(matrix: np.ndarray, config: StudyConfig, sample: np.ndarray) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    models, rows = {}, []
    if len(np.unique(matrix, axis=0)) < max(config.cluster_counts):
        raise AnalysisError("Za mało różnych obserwacji pogodowych do porównania wymaganych k.")
    for k in sorted(config.cluster_counts):
        model = KMeans(n_clusters=k, init="k-means++", n_init=config.n_init, max_iter=config.max_iter,
                       tol=1e-4, algorithm="lloyd", random_state=config.random_state)
        labels = model.fit_predict(matrix)
        sampled_labels = labels[sample]
        score = np.nan
        if 1 < len(np.unique(sampled_labels)) < len(sample):
            score = float(silhouette_score(matrix[sample], sampled_labels))
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": score, "silhouette_n": len(sample)})
        models[k] = model
    metrics = pd.DataFrame(rows).set_index("k")
    if metrics["silhouette"].notna().sum() == 0:
        raise AnalysisError("Nie można porównać k: silhouette jest niewyznaczalne dla całej próby.")
    return metrics, models


def _best_k(metrics: pd.DataFrame) -> int:
    return int(metrics.reset_index().dropna(subset=["silhouette"]).sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])


def run_bike_study(df: pd.DataFrame, config: StudyConfig | None = None) -> StudyResult:
    """Oblicz wyniki; nie pobieraj danych, nie zapisuj plików i nie mutuj wejścia.

    CLI wymaga oficjalnego CSV i manifestu. Niewielkie ramki mogą być przekazywane
    bezpośrednio wyłącznie do testowania logiki; nie są wynikami badania UCI.
    """
    config = config or StudyConfig.load()
    config.validate()
    data = validate_hourly_data(df)
    train, test = chronological_split(data, config.test_fraction)
    if len(data) <= max(config.cluster_counts):
        raise AnalysisError("Za mało obserwacji do klasteryzacji i silhouette dla wymaganych k.")
    tables: dict[str, pd.DataFrame] = {}
    figures: dict[str, go.Figure] = {}
    timestamp = hourly_timestamps(data)
    tables["schema_missing"] = column_information(data)
    tables["descriptive_statistics"] = data.select_dtypes(include="number").describe().T
    tables["demand_quantiles"] = data["cnt"].quantile([0, .25, .5, .75, .9, .95, .99, 1]).to_frame("cnt")
    tables["iqr_outliers"] = detect_iqr_outliers(data, ["cnt", *config.weather_features]).bounds
    figures["demand_distribution"] = px.histogram(data, x="cnt", nbins=60, title="Rozkład godzinowej liczby wypożyczeń", labels={"cnt": "Wypożyczenia (cnt)"})
    for name in ["hr", "weekday", "season", "weathersit"]:
        tables[f"demand_by_{name}"] = _group_demand(data, name)
        plot = tables[f"demand_by_{name}"].reset_index()
        figures[f"demand_by_{name}"] = px.bar(plot, x=name, y="mean", title=f"Średnia cnt według {name} (kody UCI)", labels={"mean": "Średnia liczba wypożyczeń"})
    corr_features = list(dict.fromkeys([*config.weather_features, "atemp", "cnt"]))
    tables["weather_correlation"] = data[corr_features].corr()
    figures["weather_correlation"] = px.imshow(tables["weather_correlation"], zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r", text_auto=".2f", title="Korelacja pogody i cnt — bez identyfikatorów i składników celu")
    for name in config.weather_features:
        figures[f"demand_vs_{name}"] = px.scatter(data, x=name, y="cnt", opacity=.2, render_mode="webgl",
            title=f"{name} a cnt — związek obserwacyjny", labels={name: f"{name} (jednostki znormalizowane UCI)"})

    numeric = list(config.regression_numeric)
    categorical = list(config.regression_categorical)
    feature_names = numeric + categorical
    if train[numeric].isna().all().any():
        raise AnalysisError("Wybrana cecha pogodowa nie ma obserwacji w treningu; nie wolno dopasować imputacji na teście.")
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder="drop")
    regression = Pipeline([("preprocess", preprocessing), ("model", LinearRegression())])
    with threadpool_limits(limits=config.thread_limit):
        regression.fit(train[feature_names], train["cnt"])
        predicted = regression.predict(test[feature_names])
    baseline = DummyRegressor(strategy="mean").fit(np.zeros((len(train), 1)), train["cnt"])
    baseline_pred = baseline.predict(np.zeros((len(test), 1)))
    tables["regression_metrics"] = pd.DataFrame({
        "baseline_train_mean": regression_metrics(test["cnt"], baseline_pred),
        "linear_regression": regression_metrics(test["cnt"], predicted),
    }).T
    residual = test["cnt"].to_numpy() - predicted
    predictions = pd.DataFrame({"timestamp": hourly_timestamps(test).to_numpy(), "actual": test["cnt"].to_numpy(),
        "baseline": baseline_pred, "predicted": predicted, "residual_actual_minus_predicted": residual})
    tables["test_predictions"] = predictions
    figures["regression_actual_predicted"] = px.scatter(predictions, x="actual", y="predicted", opacity=.25,
        render_mode="webgl", title="Regresja — rzeczywiste i przewidywane cnt, wyłącznie test")
    lower = min(predictions["actual"].min(), predictions["predicted"].min())
    upper = max(predictions["actual"].max(), predictions["predicted"].max())
    figures["regression_actual_predicted"].add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Idealna predykcja"))
    figures["regression_over_time"] = px.line(predictions, x="timestamp", y=["actual", "predicted", "baseline"], title="Chronologiczny test — rzeczywiste cnt, regresja i baseline")
    figures["residual_distribution"] = px.histogram(predictions, x="residual_actual_minus_predicted", nbins=60, title="Rozkład błędów: rzeczywiste − przewidywane")
    for name in ["hr", "weekday", "season", "weathersit"]:
        tables[f"errors_by_{name}"] = _error_groups(test, residual, name)
    figures["errors_by_hour"] = px.bar(tables["errors_by_hr"].reset_index(), x="hr", y="mae", title="MAE według godziny — wyłącznie test")
    transformed_names = preprocessing.get_feature_names_out()
    tables["regression_coefficients"] = pd.DataFrame({"coefficient": regression.named_steps["model"].coef_}, index=transformed_names)

    sample = np.sort(np.random.default_rng(config.random_state).choice(len(data), size=min(len(data), config.silhouette_sample_size), replace=False))
    with threadpool_limits(limits=config.thread_limit):
        pca = run_pca(data, config.weather_features, n_components=len(config.weather_features), imputation="median", scaling="standard")
        original_matrix = pca.pipeline[:-1].transform(data[list(config.weather_features)])
        reduced_matrix = pca.scores.iloc[:, :config.pca_components].to_numpy()
        original_metrics, original_models = _cluster_grid(original_matrix, config, sample)
        reduced_metrics, reduced_models = _cluster_grid(reduced_matrix, config, sample)
        comparison = original_metrics.add_suffix("_original").join(reduced_metrics.add_suffix("_pca"))
        for k in sorted(config.cluster_counts):
            raw_labels, pc_labels = original_models[k].labels_, reduced_models[k].labels_
            comparison.loc[k, "adjusted_rand_index"] = adjusted_rand_score(raw_labels, pc_labels)
            sample_labels = pc_labels[sample]
            comparison.loc[k, "silhouette_pca_labels_in_original_space"] = (
                silhouette_score(original_matrix[sample], sample_labels)
                if 1 < len(np.unique(sample_labels)) < len(sample) else np.nan)
    selected = {"original": _best_k(original_metrics), "pca": _best_k(reduced_metrics)}
    tables["kmeans_original"] = original_metrics
    tables["kmeans_pca"] = reduced_metrics
    tables["clustering_comparison"] = comparison
    tables["pca_variance"] = pca.variance
    tables["pca_components"] = pca.components
    tables["pca_scores"] = pca.scores
    tables["silhouette_sample_positions"] = pd.DataFrame({"position": sample})
    figures["pca_scree"] = pca_scree_plot(pca.variance)
    figures["pca_cumulative"] = pca_cumulative_variance_plot(pca.variance)
    figures["pca_projection"] = pca_scatter_plot(pca.scores)
    for space, matrix, metrics, models in [("original", original_matrix, original_metrics, original_models), ("pca", reduced_matrix, reduced_metrics, reduced_models)]:
        labels = models[selected[space]].labels_
        # Cel jest dołączany dopiero po wyborze klastrów na pogodzie i silhouette.
        profile = data[list(config.weather_features)].copy()
        profile["cluster"] = labels
        weather_profile = profile.groupby("cluster").agg(["count", "mean", "std"])
        weather_profile.columns = ["_".join(column) for column in weather_profile.columns]
        tables[f"weather_profile_{space}"] = weather_profile
        demand = pd.DataFrame({"cluster": labels, "cnt": data["cnt"].to_numpy()})
        tables[f"demand_by_cluster_{space}"] = demand.groupby("cluster")["cnt"].describe(percentiles=[.25, .5, .75, .9])
        tables[f"cluster_labels_{space}"] = pd.DataFrame({"timestamp": timestamp, "cluster": labels})
        projection = pca.scores[["PC1", "PC2"]].copy()
        projection["cluster"] = labels.astype(str)
        figures[f"clusters_{space}"] = px.scatter(projection, x="PC1", y="PC2", color="cluster", opacity=.25,
            render_mode="webgl", title=f"K-Means {space}, wybrane k={selected[space]} — rzut na PC1/PC2")
        figures[f"cnt_after_clustering_{space}"] = px.box(demand.assign(cluster=demand["cluster"].astype(str)), x="cluster", y="cnt", points=False,
            title=f"cnt po wyznaczeniu klastrów {space}; cel nie uczestniczył w grupowaniu")
        figures[f"inertia_{space}"] = px.line(metrics.reset_index(), x="k", y="inertia", markers=True, title=f"Inertia — przestrzeń {space}")
        figures[f"silhouette_{space}"] = px.line(metrics.reset_index(), x="k", y="silhouette", markers=True, title=f"Silhouette — przestrzeń {space}")
    metadata = {
        "rows": len(data), "columns": len(data.columns), "missing_cells": int(data.isna().sum().sum()),
        "demand_mean": float(data["cnt"].mean()), "demand_median": float(data["cnt"].median()),
        "demand_skewness": float(data["cnt"].skew()),
        "start": timestamp.min().isoformat(), "end": timestamp.max().isoformat(),
        "missing_hours_in_span": int((timestamp.max() - timestamp.min()).total_seconds() / 3600) + 1 - len(data),
        "train_rows": len(train), "test_rows": len(test),
        "train_start": hourly_timestamps(train).min().isoformat(), "train_end": hourly_timestamps(train).max().isoformat(),
        "test_start": hourly_timestamps(test).min().isoformat(), "test_end": hourly_timestamps(test).max().isoformat(),
        "selected_k": selected, "negative_linear_predictions": int((predicted < 0).sum()),
        "test_bias_actual_minus_predicted": float(residual.mean()),
        "pca_retained_variance": float(pca.variance.iloc[config.pca_components - 1]["cumulative_explained_variance_ratio"]),
        "configuration": asdict(config),
    }
    return StudyResult(config, metadata, tables, figures, regression, pca, {"original": original_models, "pca": reduced_models})

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

from src.analysis_validation import AnalysisError
from src.ml_models import regression_metrics, train_kmeans, train_linear_regression


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.normal(size=(160, 2)), columns=["x1", "x2"])
    data["target"] = 3 * data["x1"] - 2 * data["x2"] + 7
    return data


def test_regression_known_synthetic_coefficients_and_metrics(regression_data):
    before = regression_data.copy(deep=True)
    result = train_linear_regression(regression_data, "target", ["x1", "x2"], scaling="none")
    np.testing.assert_allclose(result.coefficients, [3, -2], atol=1e-12)
    assert result.intercept == pytest.approx(7)
    assert result.metrics["mae"] < 1e-12
    assert result.metrics["rmse"] < 1e-12
    assert result.metrics["r2"] == pytest.approx(1)
    assert len(result.train_positions) == 120
    assert len(result.test_positions) == 40
    assert set(result.train_positions).isdisjoint(result.test_positions)
    assert sorted(np.r_[result.train_positions, result.test_positions]) == list(range(160))
    np.testing.assert_allclose(result.predictions["actual"], regression_data.iloc[result.test_positions]["target"])
    pd.testing.assert_frame_equal(regression_data, before)


def test_regression_metrics_known_values_and_undefined_r2():
    result = regression_metrics([1, 2, 3], [2, 2, 1])
    assert result == pytest.approx({"mae": 1, "rmse": np.sqrt(5 / 3), "r2": -1.5})
    assert regression_metrics([1], [2]) == {"mae": 1, "rmse": 1, "r2": None}
    assert regression_metrics([2, 2], [1, 3])["r2"] is None


def test_test_set_changes_cannot_affect_imputer_scaler_or_model():
    n, seed = 24, 13
    train, test = train_test_split(np.arange(n), test_size=0.25, random_state=seed)
    data = pd.DataFrame({"x": np.arange(n, dtype=float), "target": 2 * np.arange(n) + 1})
    data.loc[train[0], "x"] = np.nan
    data.loc[test, "x"] = 10_000
    data.loc[test[0], "x"] = np.nan
    result = train_linear_regression(data, "target", ["x"], random_state=seed, imputation="mean")
    expected_mean = data.iloc[train]["x"].mean()
    assert result.pipeline.named_steps["imputer"].statistics_[0] == pytest.approx(expected_mean)
    prepared_train = data.iloc[train]["x"].fillna(expected_mean)
    scaler = result.pipeline.named_steps["scaler"]
    assert scaler.mean_[0] == pytest.approx(prepared_train.mean())
    assert scaler.var_[0] == pytest.approx(prepared_train.var(ddof=0))
    assert expected_mean != pytest.approx(data["x"].mean())
    changed = data.copy()
    changed.loc[test, "x"] = -999_999
    changed.loc[test, "target"] = 1_000_000
    second = train_linear_regression(changed, "target", ["x"], random_state=seed, imputation="mean")
    np.testing.assert_array_equal(result.train_positions, second.train_positions)
    for name, attributes in [("imputer", ["statistics_"]), ("scaler", ["mean_", "var_", "scale_"]), ("model", ["coef_", "intercept_"])]:
        for attribute in attributes:
            np.testing.assert_allclose(getattr(result.pipeline.named_steps[name], attribute), getattr(second.pipeline.named_steps[name], attribute))


def test_missing_targets_are_dropped_not_imputed_and_indices_can_repeat(regression_data):
    regression_data.index = [5] * len(regression_data)
    regression_data.iloc[0, 2] = np.nan
    regression_data.iloc[1, 2] = np.inf
    regression_data.iloc[2, 0] = -np.inf
    result = train_linear_regression(regression_data, "target", ["x1"])
    assert result.dropped_target_rows == 2
    assert len(result.train_positions) + len(result.test_positions) == 158
    assert not {0, 1}.intersection(np.r_[result.train_positions, result.test_positions])
    assert result.predictions.index.tolist() == [5] * len(result.test_positions)
    assert np.isfinite(result.predictions.to_numpy()).all()


def test_regression_constant_target_and_feature():
    data = pd.DataFrame({"x": [3] * 8, "y": [7] * 8})
    result = train_linear_regression(data, "y", ["x"])
    assert result.metrics == {"mae": 0, "rmse": 0, "r2": None}
    assert result.coefficients.iloc[0] == 0


def test_feature_empty_only_in_training_is_rejected():
    train, test = train_test_split(np.arange(8), test_size=0.25, random_state=42)
    data = pd.DataFrame({"x": np.full(8, np.nan), "y": np.arange(8)})
    data.loc[test, "x"] = 1
    with pytest.raises(AnalysisError, match="treningowej"):
        train_linear_regression(data, "y", ["x"])


@pytest.mark.parametrize("kwargs", [
    {"features": []}, {"features": ["x1", "x1"]}, {"features": ["target"]},
    {"features": ["unknown"]}, {"target": "unknown"},
    {"test_size": 0}, {"test_size": 1}, {"test_size": True}, {"test_size": np.nan},
    {"test_size": 0.001}, {"test_size": 0.999}, {"scaling": "bad"}, {"imputation": "bad"},
    {"random_state": -1}, {"random_state": True}, {"random_state": 2**32}, {"random_state": 1.5},
])
def test_regression_invalid_parameters(regression_data, kwargs):
    args = {"target": "target", "features": ["x1"]} | kwargs
    with pytest.raises(AnalysisError):
        train_linear_regression(regression_data, **args)


@pytest.mark.parametrize("actual,predicted", [([], []), ([1], [1, 2]), ([np.inf], [1]), ([1], [np.nan]), ([[1]], [[1]]), (["a"], [1])])
def test_invalid_metrics(actual, predicted):
    with pytest.raises(AnalysisError):
        regression_metrics(actual, predicted)


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_insufficient_regression_rows(size):
    with pytest.raises(AnalysisError):
        train_linear_regression(pd.DataFrame({"x": range(size), "y": range(size)}), "y", ["x"])


def test_kmeans_known_clusters_metrics_and_repeatability():
    x, expected = make_blobs(n_samples=120, centers=[[-8, -8], [8, 8]], cluster_std=0.3, random_state=42)
    data = pd.DataFrame(x, columns=["x", "y"])
    original = data.copy(deep=True)
    result = train_kmeans(data, ["x", "y"], n_clusters=2)
    second = train_kmeans(data, ["x", "y"], n_clusters=2)
    assert adjusted_rand_score(expected, result.labels) == 1
    assert result.silhouette > 0.95
    pd.testing.assert_series_equal(result.labels, second.labels)
    pd.testing.assert_frame_equal(result.centers, second.centers)
    model = result.pipeline.named_steps["model"]
    assert model.init == "k-means++"
    assert model.n_init == 10 and model.random_state == 42
    assert model.max_iter == 300 and model.algorithm == "lloyd" and model.tol == 1e-4
    expected_inertia = np.square(result.scaled_data.to_numpy() - model.cluster_centers_[result.labels]).sum()
    assert result.inertia == pytest.approx(expected_inertia)
    for cluster in range(2):
        np.testing.assert_allclose(result.centers.loc[cluster], data.iloc[np.flatnonzero(result.labels.to_numpy() == cluster)].mean())
    np.testing.assert_array_equal(result.pipeline.predict(data), result.labels)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("scaling", ["standard", "minmax"])
def test_kmeans_imputation_and_duplicate_indices(scaling):
    data = pd.DataFrame({"cluster": [1, 2, np.nan, 9, 10], "y": [1, np.inf, 3, 8, 9]}, index=[7] * 5)
    before = data.copy(deep=True)
    result = train_kmeans(data, ["cluster", "y"], n_clusters=2, scaling=scaling)
    assert len(result.labels) == len(data)
    assert result.labels.index.tolist() == [7] * 5
    np.testing.assert_allclose(result.pipeline.named_steps["imputer"].statistics_, [5.5, 5.5])
    assert np.isfinite(result.scaled_data.to_numpy()).all()
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("values,k", [([1, 2], 1), ([1, 2], 2), ([5, 5, 5], 1)])
def test_kmeans_degenerate_silhouette(values, k):
    result = train_kmeans(pd.DataFrame({"x": values}), ["x"], n_clusters=k)
    assert result.silhouette is None
    assert len(result.labels) == len(values)
    if len(set(values)) == 1:
        assert result.inertia == 0


def test_silhouette_sampling_is_bounded_and_repeatable(monkeypatch):
    monkeypatch.setattr("src.ml_models.SILHOUETTE_SAMPLE_LIMIT", 12)
    data = pd.DataFrame({"x": list(range(25)) + list(range(100, 125))})
    first = train_kmeans(data, ["x"], n_clusters=2)
    second = train_kmeans(data, ["x"], n_clusters=2)
    assert first.silhouette is not None
    assert first.silhouette == second.silhouette
    assert "12 obserwacji" in first.silhouette_note


@pytest.mark.parametrize("kwargs", [
    {"n_clusters": 0}, {"n_clusters": -1}, {"n_clusters": True}, {"n_clusters": 2.5}, {"n_clusters": 5},
    {"features": []}, {"features": ["missing"]}, {"features": ["text"]}, {"features": ["x", "x"]},
    {"imputation": "bad"}, {"scaling": "none"}, {"random_state": -1}, {"random_state": np.nan},
])
def test_kmeans_invalid_parameters(kwargs):
    data = pd.DataFrame({"x": [1, 2, 3, 4], "text": list("abcd")})
    args = {"features": ["x"], "n_clusters": 2} | kwargs
    with pytest.raises(AnalysisError):
        train_kmeans(data, **args)


@pytest.mark.parametrize("values,k", [([], 1), ([1], 1), ([1, 1, 1], 2), ([np.nan, np.inf], 1)])
def test_kmeans_unusable_data(values, k):
    with pytest.raises(AnalysisError):
        train_kmeans(pd.DataFrame({"x": values}), ["x"], n_clusters=k)

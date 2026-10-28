import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.analysis_validation import AnalysisError, numeric_selection
from src.dimensionality_reduction import run_pca


def test_known_rank_one_data_and_component_weights():
    x = np.arange(-2, 3, dtype=float)
    data = pd.DataFrame({"x": x, "twice_x": 2 * x})
    result = run_pca(data, ["x", "twice_x"])
    assert isinstance(result.pipeline, Pipeline)
    assert list(result.pipeline.named_steps) == ["imputer", "scaler", "pca"]
    assert result.pipeline.named_steps["pca"].svd_solver == "full"
    assert result.pipeline.named_steps["pca"].whiten is False
    np.testing.assert_allclose(result.components.loc["PC1"].abs(), 1 / np.sqrt(2))
    np.testing.assert_allclose(result.variance["explained_variance"], [2.5, 0], atol=1e-12)
    np.testing.assert_allclose(result.variance["explained_variance_ratio"], [1, 0], atol=1e-12)
    prepared = result.pipeline[:-1].transform(data)
    expected = (prepared - result.pipeline.named_steps["pca"].mean_) @ result.components.to_numpy().T
    np.testing.assert_allclose(result.scores, expected, atol=1e-12)
    np.testing.assert_allclose(result.pipeline.transform(data), result.scores, atol=1e-12)
    reconstructed = result.pipeline.named_steps["pca"].inverse_transform(result.scores)
    np.testing.assert_allclose(reconstructed, prepared, atol=1e-12)


def test_partial_variance_is_not_renormalized_to_one():
    data = pd.DataFrame({"x": [-1, 1, 0, 0], "y": [0, 0, -1, 1]})
    result = run_pca(data, ["x", "y"], n_components=1)
    assert result.scores.shape == (4, 1)
    assert result.components.shape == (1, 2)
    assert result.variance.loc["PC1", "explained_variance_ratio"] == pytest.approx(0.5)
    assert result.variance.loc["PC1", "cumulative_explained_variance_ratio"] == pytest.approx(0.5)


@pytest.mark.parametrize("scaling", ["standard", "minmax"])
def test_covariance_eigenvalues_full_variance_and_repeatability(scaling):
    rng = np.random.default_rng(17)
    data = pd.DataFrame(rng.normal(size=(50, 4)) * [1, 3, 9, 27], columns=list("abcd"))
    first = run_pca(data, list("abcd"), n_components=4, scaling=scaling)
    second = run_pca(data, list("abcd"), n_components=4, scaling=scaling)
    prepared = first.pipeline[:-1].transform(data)
    eigenvalues = np.linalg.eigvalsh(np.cov(prepared, rowvar=False, ddof=1))[::-1]
    np.testing.assert_allclose(first.variance["explained_variance"], eigenvalues, rtol=1e-12)
    ratio = first.variance["explained_variance_ratio"].to_numpy()
    cumulative = first.variance["cumulative_explained_variance_ratio"].to_numpy()
    assert (ratio >= 0).all()
    assert (np.diff(ratio) <= 0).all()
    assert ratio.sum() == pytest.approx(1)
    assert (np.diff(cumulative) >= 0).all()
    assert cumulative[-1] == pytest.approx(1)
    np.testing.assert_allclose(cumulative, np.cumsum(ratio))
    pd.testing.assert_frame_equal(first.scores, second.scores)
    pd.testing.assert_frame_equal(first.components, second.components)
    pd.testing.assert_frame_equal(first.variance, second.variance)


@pytest.mark.parametrize("imputation,expected", [("mean", [4, 14 / 3]), ("median", [2, 5])])
def test_missing_infinite_values_duplicate_indices_and_no_input_mutation(imputation, expected):
    data = pd.DataFrame({"x": [1, 2, 9, np.nan], "y": [3, np.inf, 5, 6], "text": list("abcd")}, index=[7, 7, 8, 8])
    data.attrs["source"] = "synthetic"
    original = data.copy(deep=True)
    result = run_pca(data, ["x", "y"], imputation=imputation)
    np.testing.assert_allclose(result.pipeline.named_steps["imputer"].statistics_, expected)
    assert result.scores.index.tolist() == [7, 7, 8, 8]
    assert np.isfinite(result.scores).all().all()
    np.testing.assert_allclose(result.pipeline.transform(numeric_selection(data, ["x", "y"])), result.scores, atol=1e-12)
    result.scores.iloc[0, 0] = 999
    result.components.iloc[0, 0] = -999
    pd.testing.assert_frame_equal(data, original)
    assert data.attrs == original.attrs


def test_nullable_integers_and_constant_feature_are_supported():
    data = pd.DataFrame({"x": pd.Series([1, None, 4, 5], dtype="Int64"), "constant": [7.0, 7, np.nan, 7]})
    result = run_pca(data, ["x", "constant"])
    assert result.constant_features == ("constant",)
    np.testing.assert_allclose(result.variance["explained_variance_ratio"], [1, 0], atol=1e-12)
    assert np.isfinite(result.components).all().all()


def test_two_observations_and_more_features_than_rows():
    data = pd.DataFrame({"x": [1, 2], "y": [2, 4], "z": [4, 8]})
    result = run_pca(data, ["x", "y", "z"], n_components=2)
    assert result.scores.shape == (2, 2)
    assert result.components.shape == (2, 3)
    assert result.variance["explained_variance_ratio"].sum() == pytest.approx(1)
    with pytest.raises(AnalysisError, match="od 1 do 2"):
        run_pca(data, ["x", "y", "z"], n_components=3)


@pytest.mark.parametrize("count", [0, -1, 3, 2.0, True, "2", None, np.nan])
def test_invalid_component_count(count):
    with pytest.raises(AnalysisError, match="Liczba składowych"):
        run_pca(pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 6]}), ["x", "y"], n_components=count)


@pytest.mark.parametrize("features", [[], ["x"], ["x", "missing"], ["x", "text"], ["x", "flag"], ["x", "complex"], ["x", "x"], "xy", None])
def test_invalid_or_insufficient_features(features):
    data = pd.DataFrame({"x": [1, 2], "y": [2, 4], "text": ["a", "b"], "flag": [True, False], "complex": [1j, 2j]})
    with pytest.raises(AnalysisError):
        run_pca(data, features)


@pytest.mark.parametrize("size", [0, 1])
def test_insufficient_observations(size):
    with pytest.raises(AnalysisError, match="obserwacji"):
        run_pca(pd.DataFrame({"x": range(size), "y": range(size)}), ["x", "y"])


@pytest.mark.parametrize("values", [[np.nan, np.nan, np.nan], [np.inf, -np.inf, np.nan]])
def test_all_missing_feature_is_not_silently_dropped(values):
    with pytest.raises(AnalysisError, match="bez skończonych wartości: empty"):
        run_pca(pd.DataFrame({"x": [1, 2, 3], "empty": values}), ["x", "empty"])


@pytest.mark.parametrize("scaling", ["standard", "minmax"])
def test_zero_total_variance_rejected_without_runtime_warnings(scaling):
    with pytest.raises(AnalysisError, match="wariancja jest zerowa"):
        run_pca(pd.DataFrame({"x": [1, 1, np.nan], "y": [3, 3, 3]}), ["x", "y"], scaling=scaling)


@pytest.mark.parametrize("kwargs", [{"imputation": "mode"}, {"scaling": "none"}, {"scaling": "unknown"}])
def test_invalid_preprocessing_parameters(kwargs):
    with pytest.raises(AnalysisError):
        run_pca(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), ["x", "y"], **kwargs)


def test_duplicate_column_names_rejected():
    with pytest.raises(AnalysisError, match="unikalne"):
        run_pca(pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"]), ["x", "y"])


def test_unrepresentable_numerical_range_is_a_user_error():
    with pytest.raises(AnalysisError, match="skończonych wyników"):
        run_pca(pd.DataFrame({"x": [1e308, -1e308, 0], "y": [1, 2, 3]}), ["x", "y"])

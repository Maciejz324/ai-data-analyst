import numpy as np
import pandas as pd
import pytest

from src.analysis_validation import AnalysisError, numeric_columns
from src.data_preparation import detect_iqr_outliers, drop_missing_rows, impute_missing, scale_numeric


@pytest.mark.parametrize("strategy,expected", [("mean", 103 / 3), ("median", 2)])
def test_imputation_only_selected_features_without_mutation(strategy, expected):
    data = pd.DataFrame({"x": [1, 2, 100, np.nan, np.inf], "label": ["a", None, "c", "d", "e"]}, index=[1, 1, 2, 3, 4])
    original = data.copy(deep=True)
    data.attrs["source"] = "test"
    result = impute_missing(data, ["x"], strategy)
    assert result["x"].iloc[3:].tolist() == pytest.approx([expected, expected])
    pd.testing.assert_series_equal(result["label"], data["label"])
    result.iloc[0, 0] = -999
    pd.testing.assert_frame_equal(data, original)
    assert result.attrs == data.attrs


def test_nullable_integers_and_booleans():
    data = pd.DataFrame({"x": pd.Series([1, None, 4], dtype="Int64"), "flag": [True, False, True]})
    assert numeric_columns(data) == ["x"]
    assert impute_missing(data, ["x"], "mean")["x"].tolist() == [1, 2.5, 4]


def test_drop_missing_any_column_and_infinities_keeps_original_index():
    data = pd.DataFrame({"x": [1, np.inf, 3, np.nan, 5], "text": ["a", "b", None, "d", "e"]}, index=[7, 7, 8, 9, 10])
    before = data.copy(deep=True)
    result = drop_missing_rows(data)
    assert result.index.tolist() == [7, 10]
    assert result["x"].tolist() == [1, 5]
    pd.testing.assert_frame_equal(data, before)
    assert drop_missing_rows(pd.DataFrame({"x": [np.nan]})).empty
    assert drop_missing_rows(data.iloc[:0]).empty


@pytest.mark.parametrize("method,expected", [("standard", [-1, 1]), ("minmax", [0, 1])])
def test_scaling_missing_constant_and_no_mutation(method, expected):
    data = pd.DataFrame({"x": [2, 4, np.nan, np.inf], "constant": [5, 5, 5, 5], "text": list("abcd")})
    before = data.copy(deep=True)
    result = scale_numeric(data, ["x", "constant"], method)
    np.testing.assert_allclose(result["x"].iloc[:2], expected)
    assert result["x"].iloc[2:].isna().all()
    assert result["constant"].eq(0).all()
    pd.testing.assert_series_equal(result["text"], data["text"])
    pd.testing.assert_frame_equal(data, before)


def test_iqr_bounds_mask_missing_and_degenerate_columns():
    data = pd.DataFrame({"x": [1, 2, 3, 4, 100], "constant": [2] * 5, "empty": [np.nan] * 5}, index=[9, 8, 8, 6, 5])
    before = data.copy(deep=True)
    result = detect_iqr_outliers(data, ["x", "constant", "empty"])
    assert result.bounds.loc["x"].to_dict() == {"q1": 2, "q3": 4, "iqr": 2, "lower": -1, "upper": 7, "outlier_count": 1}
    assert result.mask["x"].tolist() == [False, False, False, False, True]
    assert not result.mask[["constant", "empty"]].any().any()
    assert pd.isna(result.bounds.loc["empty", "lower"])
    pd.testing.assert_frame_equal(data, before)
    assert not detect_iqr_outliers(data.iloc[:1], ["x"]).mask.any().any()
    zero_iqr = detect_iqr_outliers(pd.DataFrame({"x": [2] * 8 + [9]}), ["x"])
    assert zero_iqr.bounds.loc["x", "iqr"] == 0
    assert zero_iqr.mask["x"].sum() == 1


@pytest.mark.parametrize("columns", [[], ["missing"], ["label"], ["flag"], ["complex"], ["x", "x"], "x", None])
@pytest.mark.parametrize("operation", [impute_missing, scale_numeric, detect_iqr_outliers])
def test_invalid_feature_selection(columns, operation):
    data = pd.DataFrame({"x": [1, 2], "label": ["a", "b"], "flag": [True, False], "complex": [1j, 2j]})
    with pytest.raises(AnalysisError):
        operation(data, columns)


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf, True, "1.5"])
def test_invalid_iqr_factor(factor):
    with pytest.raises(AnalysisError):
        detect_iqr_outliers(pd.DataFrame({"x": [1, 2]}), ["x"], factor)


@pytest.mark.parametrize("operation", [impute_missing, scale_numeric])
def test_empty_features_all_missing_and_invalid_methods(operation):
    for data in [pd.DataFrame({"x": []}), pd.DataFrame({"x": [np.nan, np.inf]})]:
        with pytest.raises(AnalysisError):
            operation(data, ["x"])
    with pytest.raises(AnalysisError):
        operation(pd.DataFrame({"x": [1, 2]}), ["x"], "unknown")
    with pytest.raises(AnalysisError):
        operation(pd.DataFrame([[1, 2]], columns=["x", "x"]), ["x"])


def test_empty_iqr_data_rejected():
    with pytest.raises(AnalysisError):
        detect_iqr_outliers(pd.DataFrame({"x": []}), ["x"])

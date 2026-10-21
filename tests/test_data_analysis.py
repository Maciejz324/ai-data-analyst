import numpy as np
import pandas as pd
import pytest

from src.data_analysis import column_information, descriptive_statistics, missing_values, numeric_data, pearson_correlation


def test_statistics_known_values_and_missing_data():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "text": ["a"] * 4})
    stats = descriptive_statistics(df)
    assert stats.index.tolist() == ["x"]
    assert stats.loc["x"].to_dict() == {"count": 3, "mean": 2, "median": 2, "min": 1, "max": 3, "std": 1}
    assert missing_values(df).loc["x"].to_dict() == {"missing_count": 1, "missing_percent": 25}
    assert column_information(df).loc["x", "dtype"] == "float64"


def test_no_numeric_columns_and_empty_frame():
    df = pd.DataFrame({"label": ["a", "b"], "flag": [True, False]})
    assert descriptive_statistics(df).empty
    assert pearson_correlation(df).empty
    assert missing_values(pd.DataFrame({"x": []})).loc["x", "missing_percent"] == 0


def test_undefined_statistics_and_infinity_without_mutation():
    df = pd.DataFrame({"empty": [np.nan] * 3, "single": [2, np.inf, -np.inf]})
    stats = descriptive_statistics(df)
    assert stats.loc["empty", "count"] == 0
    assert pd.isna(stats.loc["empty", "mean"])
    assert stats.loc["single", "mean"] == 2
    assert pd.isna(stats.loc["single", "std"])
    assert np.isinf(df.loc[1, "single"])
    assert numeric_data(df)["single"].isna().sum() == 2
    empty_stats = descriptive_statistics(df[["empty"]])
    assert empty_stats.loc["empty", "count"] == 0
    assert empty_stats.loc["empty", ["mean", "median", "min", "max", "std"]].isna().all()


def test_known_pearson_and_constant_columns():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "z": [4, 3, 2, 1], "constant": [7] * 4})
    corr = pearson_correlation(df)
    assert corr.loc["x", "y"] == pytest.approx(1)
    assert corr.loc["x", "z"] == pytest.approx(-1)
    assert "constant" not in corr


def test_pairwise_missing_values():
    df = pd.DataFrame({"x": [1, 2, 3, np.nan], "y": [2, 4, np.nan, 8], "z": [np.nan, np.nan, 1, 2]})
    corr = pearson_correlation(df)
    assert corr.loc["x", "y"] == pytest.approx(1)
    assert pd.isna(corr.loc["x", "z"])
    assert pearson_correlation(df[["x"]]).empty

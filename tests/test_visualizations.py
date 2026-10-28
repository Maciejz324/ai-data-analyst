import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualizations import VisualizationError, bar_chart, correlation_matrix, histogram, scatter_plot
from src.visualizations import cluster_plot, regression_prediction_plot


@pytest.fixture
def data():
    return pd.DataFrame({"x": [1, 2, 3, np.nan], "y": [2, 4, 6, 8], "group": ["A", "A", "B", "B"]})


def test_all_chart_objects_and_values(data):
    hist = histogram(data, "x")
    scatter = scatter_plot(data, "x", "y")
    count = bar_chart(data, "group")
    average = bar_chart(data, "group", "x")
    corr = correlation_matrix(data, ["x", "y"])
    for fig in [hist, scatter, count, average, corr]:
        assert isinstance(fig, go.Figure)
        assert fig.to_json()
    assert len(hist.data[0].x) == 3
    assert list(scatter.data[0].y) == [2, 4, 6]
    assert list(count.data[0].y) == [2, 2]
    assert list(average.data[0].y) == [1.5, 3]
    assert corr.data[0].z[0][1] == pytest.approx(1)


def test_column_names_do_not_collide_with_aggregation():
    df = pd.DataFrame({"count": ["a", "a", "b"], "result": [1, 3, 5]})
    assert list(bar_chart(df, "count").data[0].y) == [2, 1]
    assert list(bar_chart(df, "result", "result").data[0].y) == [1, 3, 5]


@pytest.mark.parametrize("chart", [
    lambda d: histogram(d, "group"), lambda d: histogram(d, "missing"),
    lambda d: histogram(d, "x", 0), lambda d: scatter_plot(d, "x", "x"),
    lambda d: scatter_plot(d, "x", "group"), lambda d: bar_chart(d, "missing"),
    lambda d: bar_chart(d, "group", "group"), lambda d: correlation_matrix(d, ["x"]),
    lambda d: correlation_matrix(d, ["group", "y"]),
])
def test_invalid_chart_selection(data, chart):
    with pytest.raises(VisualizationError):
        chart(data)


def test_empty_infinite_and_disjoint_data():
    df = pd.DataFrame({"x": [np.inf, np.nan], "y": [1, 2], "group": [None, None]})
    for operation in [lambda: histogram(df, "x"), lambda: scatter_plot(df, "x", "y"),
                      lambda: bar_chart(df, "group"), lambda: correlation_matrix(df)]:
        with pytest.raises(VisualizationError):
            operation()
    disjoint = pd.DataFrame({"x": [1, 2, np.nan, np.nan], "y": [np.nan, np.nan, 3, 4]})
    with pytest.raises(VisualizationError, match="Brak par"):
        correlation_matrix(disjoint)


def test_bar_category_limit():
    with pytest.raises(VisualizationError, match="50"):
        bar_chart(pd.DataFrame({"category": range(51)}), "category")


def test_regression_chart_points_and_identity_line():
    data = pd.DataFrame({"actual": [1, 3, 5], "predicted": [2, 3, 4]})
    fig = regression_prediction_plot(data)
    assert list(fig.data[0].x) == [1, 3, 5]
    assert list(fig.data[0].y) == [2, 3, 4]
    assert list(fig.data[1].x) == list(fig.data[1].y) == [1, 5]
    assert fig.to_json()


def test_cluster_projection_one_feature_and_column_name_collisions():
    data = pd.DataFrame({"cluster": [1, 2, 8, 9], "x": [2, 3, 7, 8]}, index=[5] * 4)
    labels = pd.Series([0, 0, 1, 1], index=data.index)
    fig = cluster_plot(data, labels, "cluster", "x")
    assert list(fig.data[0].x) == [1, 2]
    assert list(fig.data[1].y) == [7, 8]
    one_dimensional = cluster_plot(data, labels, "cluster")
    assert list(one_dimensional.data[0].y) == [0, 1]
    assert one_dimensional.to_json()


@pytest.mark.parametrize("operation", [
    lambda: regression_prediction_plot(pd.DataFrame()),
    lambda: regression_prediction_plot(pd.DataFrame({"actual": [1], "predicted": [np.inf]})),
    lambda: cluster_plot(pd.DataFrame({"x": [1]}), pd.Series([0, 1]), "x"),
    lambda: cluster_plot(pd.DataFrame({"x": [1]}), pd.Series([0]), "x", "x"),
])
def test_invalid_ml_charts(operation):
    with pytest.raises(VisualizationError):
        operation()

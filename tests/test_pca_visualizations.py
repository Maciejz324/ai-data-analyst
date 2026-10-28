import numpy as np
import pandas as pd
import pytest

from src.visualizations import VisualizationError, pca_cumulative_variance_plot, pca_scatter_plot, pca_scree_plot


@pytest.fixture
def variance():
    return pd.DataFrame({
        "explained_variance_ratio": [0.6, 0.3],
        "cumulative_explained_variance_ratio": [0.6, 0.9],
    }, index=["PC1", "PC2"])


def test_variance_plots_use_percentages_and_keep_input_unchanged(variance):
    before = variance.copy(deep=True)
    scree = pca_scree_plot(variance)
    cumulative = pca_cumulative_variance_plot(variance)
    np.testing.assert_allclose(scree.data[0].y, [60, 30])
    np.testing.assert_allclose(cumulative.data[0].y, [60, 90])
    assert list(scree.data[0].x) == ["PC1", "PC2"]
    assert scree.data[0].type == "bar"
    assert cumulative.data[0].mode == "lines+markers"
    assert "%" in scree.layout.yaxis.title.text
    assert scree.to_json() and cumulative.to_json()
    pd.testing.assert_frame_equal(variance, before)


def test_scatter_shows_actual_pc_scores_and_requires_two_components():
    scores = pd.DataFrame({"PC1": [-2, 0, 2], "PC2": [1, -1, 1]}, index=[4, 4, 9])
    before = scores.copy(deep=True)
    fig = pca_scatter_plot(scores)
    assert list(fig.data[0].x) == [-2, 0, 2]
    assert list(fig.data[0].y) == [1, -1, 1]
    assert fig.layout.xaxis.title.text == "PC1"
    assert fig.layout.yaxis.title.text == "PC2"
    assert fig.to_json()
    pd.testing.assert_frame_equal(scores, before)
    with pytest.raises(VisualizationError, match="dwóch składowych"):
        pca_scatter_plot(scores[["PC1"]])


@pytest.mark.parametrize("plot", [pca_scree_plot, pca_cumulative_variance_plot])
@pytest.mark.parametrize("values", [
    pd.DataFrame(),
    pd.DataFrame({"explained_variance_ratio": [0.5]}),
    pd.DataFrame({"explained_variance_ratio": [np.nan], "cumulative_explained_variance_ratio": [0.5]}),
    pd.DataFrame({"explained_variance_ratio": [-0.5], "cumulative_explained_variance_ratio": [-0.5]}),
    pd.DataFrame({"explained_variance_ratio": [1.1], "cumulative_explained_variance_ratio": [1.1]}),
    pd.DataFrame({"explained_variance_ratio": [0.5, 0.4], "cumulative_explained_variance_ratio": [0.5, 0.8]}),
    pd.DataFrame({"explained_variance_ratio": ["text"], "cumulative_explained_variance_ratio": [0.5]}),
])
def test_invalid_variance_tables(plot, values):
    with pytest.raises(VisualizationError):
        plot(values)


@pytest.mark.parametrize("values", [
    pd.DataFrame({"PC1": [], "PC2": []}),
    pd.DataFrame({"PC1": [np.inf], "PC2": [1]}),
    pd.DataFrame({"PC1": [1], "PC2": [np.nan]}),
    pd.DataFrame({"PC1": [1], "PC2": ["text"]}),
])
def test_invalid_scatter_data(values):
    with pytest.raises(VisualizationError):
        pca_scatter_plot(values)

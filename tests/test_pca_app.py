from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "app.py"
FEATURES = ["wiek", "godziny_nauki", "frekwencja", "wynik_egzaminu"]


def sample_app():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.button[0].click().run()
    assert not app.exception
    return app


def state(app):
    return app.session_state["adv_workspace"]


def select_features(app, features=FEATURES, count=2):
    app.multiselect(key="adv_pca_features").set_value(features).run()
    app.number_input(key="adv_pca_count").set_value(count).run()
    assert not app.exception
    return app


def run_sample_pca(count=2):
    app = select_features(sample_app(), count=count)
    app.button(key="adv_train_pca").click().run()
    assert not app.exception
    assert state(app)["pca"] is not None
    return app


def test_pca_tab_requires_explicit_selection_and_preserves_initial_mvp():
    app = sample_app()
    assert any(tab.label == "PCA" for tab in app.tabs)
    assert app.multiselect(key="adv_pca_features").value == []
    assert app.button(key="adv_train_pca").disabled
    assert [metric.value for metric in app.metric] == ["20", "6", "4", "4"]
    assert len(app.get("plotly_chart")) == 1


def test_pca_shows_variance_weights_scores_and_three_plots_without_mutation():
    app = sample_app()
    original = state(app)["original"].copy(deep=True)
    select_features(app)
    app.button(key="adv_train_pca").click().run()
    assert not app.exception
    result = state(app)["pca"][1]
    assert result.scores.shape == (20, 2)
    assert result.components.shape == (2, 4)
    assert result.variance["cumulative_explained_variance_ratio"].iloc[-1] == pytest.approx(0.9744032047073827)
    assert len(app.get("plotly_chart")) == 4
    assert next(metric for metric in app.metric if metric.label == "Wariancja zachowana przez PCA").value == "97.44%"
    assert any(frame.value.equals(result.scores) for frame in app.dataframe)
    assert any(frame.value.equals(result.components.rename_axis("Składowa")) for frame in app.dataframe)
    pd.testing.assert_frame_equal(state(app)["original"], original)
    pd.testing.assert_frame_equal(state(app)["working"], original)


def test_one_component_has_no_two_dimensional_scatter():
    app = run_sample_pca(count=1)
    assert state(app)["pca"][1].scores.shape == (20, 1)
    assert len(app.get("plotly_chart")) == 3
    assert any("Wykres PCA 2D wymaga" in info.value for info in app.info)


@pytest.mark.parametrize("kind,key,new_value,old_value", [
    ("number_input", "adv_pca_count", 1, 2),
    ("multiselect", "adv_pca_features", FEATURES[:2], FEATURES),
    ("selectbox", "adv_pca_imputation", "Średnia", "Mediana"),
    ("selectbox", "adv_pca_scaling", "Normalizacja (min–max)", "Standaryzacja (z-score)"),
    ("selectbox", "adv_pca_source", "Oryginalne dane (kopia)", "Kopia robocza"),
])
def test_parameter_changes_discard_results_instead_of_resurrecting_them(kind, key, new_value, old_value):
    app = run_sample_pca()
    getattr(app, kind)(key=key).set_value(new_value).run()
    assert not app.exception
    assert state(app)["pca"] is None
    assert len(app.get("plotly_chart")) == 1
    getattr(app, kind)(key=key).set_value(old_value).run()
    assert not app.exception
    assert state(app)["pca"] is None


def test_transformations_and_reset_discard_pca_and_working_copy_is_used():
    app = run_sample_pca()
    app.button(key="adv_apply").click().run()
    assert not app.exception
    assert state(app)["pca"] is None
    assert len(state(app)["working"]) == 16
    app.button(key="adv_train_pca").click().run()
    assert not app.exception
    assert len(state(app)["pca"][1].scores) == 16
    app.selectbox(key="adv_pca_source").select("Oryginalne dane (kopia)").run()
    app.button(key="adv_train_pca").click().run()
    assert len(state(app)["pca"][1].scores) == 20
    app.button(key="adv_reset").click().run()
    assert not app.exception
    assert state(app)["pca"] is None
    assert len(state(app)["working"]) == 20


def test_component_limit_adapts_when_selected_features_shrink():
    app = run_sample_pca(count=4)
    app.multiselect(key="adv_pca_features").set_value(FEATURES[:2]).run()
    assert not app.exception
    assert 1 <= app.number_input(key="adv_pca_count").value <= 2
    assert state(app)["pca"] is None
    app.button(key="adv_train_pca").click().run()
    assert not app.exception
    assert state(app)["pca"][1].scores.shape[1] <= 2


def test_new_dataset_resets_pca_even_if_filename_and_schema_are_unchanged():
    first = pd.DataFrame({"x": [1, 2, 3, 4], "y": [3, 1, 6, 7]})
    first.attrs.update(separator=",", encoding="utf-8-sig")
    with patch("src.data_loader.load_csv", return_value=first) as loader:
        app = select_features(sample_app(), ["x", "y"])
        app.button(key="adv_train_pca").click().run()
        assert state(app)["pca"] is not None
        second = first.copy(deep=True)
        second.loc[0, "x"] = 99
        loader.return_value = second
        app.run()
    assert not app.exception
    assert state(app)["pca"] is None
    assert app.multiselect(key="adv_pca_features").value == []
    pd.testing.assert_frame_equal(state(app)["working"], second)


@pytest.mark.parametrize("values,message", [
    ({"x": [1, 1, 1], "y": [2, 2, 2]}, "wariancja jest zerowa"),
    ({"x": [1, 2, 3], "y": [np.nan, np.inf, np.nan]}, "bez skończonych wartości"),
])
def test_invalid_data_show_polish_error_without_traceback(values, message):
    data = pd.DataFrame(values)
    data.attrs.update(separator=",", encoding="utf-8-sig")
    with patch("src.data_loader.load_csv", return_value=data):
        app = select_features(sample_app(), ["x", "y"])
        app.button(key="adv_train_pca").click().run()
    assert not app.exception
    assert state(app)["pca"] is None
    assert any(message in warning.value for warning in app.warning)


@pytest.mark.parametrize("values", [
    {"x": [1], "y": [2]},
    {"label": ["a", "b", "c"]},
])
def test_too_few_observations_or_numeric_features_are_explained(values):
    data = pd.DataFrame(values)
    data.attrs.update(separator=",", encoding="utf-8-sig")
    with patch("src.data_loader.load_csv", return_value=data):
        app = sample_app()
    assert not app.exception
    assert state(app)["pca"] is None
    assert any("PCA wymaga co najmniej" in info.value for info in app.info)


def test_numeric_identifier_is_not_automatically_selected():
    data = pd.DataFrame({"student_id": [101, 102, 103], "wynik": [3, 1, 7]})
    data.attrs.update(separator=",", encoding="utf-8-sig")
    with patch("src.data_loader.load_csv", return_value=data):
        app = sample_app()
    widget = app.multiselect(key="adv_pca_features")
    assert "student_id" in widget.options
    assert widget.value == []
    assert app.button(key="adv_train_pca").disabled

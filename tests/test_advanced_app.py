from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.ml_models import train_linear_regression

APP_PATH = Path(__file__).parents[1] / "app.py"


def sample_app():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.button[0].click().run()
    assert not app.exception
    return app


def state(app):
    return app.session_state["adv_workspace"]


def test_drop_rows_reset_and_existing_mvp_unchanged():
    app = sample_app()
    original = state(app)["original"].copy(deep=True)
    assert any(tab.label == "Zaawansowana analiza danych" for tab in app.tabs)
    app.button(key="adv_apply").click().run()
    assert not app.exception
    assert len(state(app)["working"]) == 16
    assert len(state(app)["history"]) == 1
    pd.testing.assert_frame_equal(state(app)["original"], original)
    assert [metric.value for metric in app.metric] == ["20", "6", "4", "4"]
    app.button(key="adv_reset").click().run()
    assert not app.exception
    pd.testing.assert_frame_equal(state(app)["working"], original)
    assert state(app)["history"] == []


@pytest.mark.parametrize("operation", ["Uzupełnij średnią", "Uzupełnij medianą", "Normalizuj (min–max)", "Standaryzuj (z-score)"])
def test_preparation_operations_from_ui(operation):
    app = sample_app()
    original = state(app)["original"].copy(deep=True)
    app.selectbox(key="adv_operation").select(operation).run()
    app.button(key="adv_apply").click().run()
    assert not app.exception
    current = state(app)["working"]
    pd.testing.assert_frame_equal(state(app)["original"], original)
    if operation.startswith("Uzupełnij"):
        assert current.isna().sum().sum() == 0
    elif operation.startswith("Normalizuj"):
        assert current["wiek"].min() == pytest.approx(0)
        assert current["wiek"].max() == pytest.approx(1)
        assert current.isna().sum().sum() == 4
    else:
        assert current["wiek"].mean() == pytest.approx(0, abs=1e-12)
        assert current["wiek"].std(ddof=0) == pytest.approx(1)
        assert current.isna().sum().sum() == 4


def test_iqr_is_detection_without_modifying_working_data():
    app = sample_app()
    original = state(app)["working"].copy(deep=True)
    app.selectbox(key="adv_operation").select("Wykryj wartości odstające (IQR)").run()
    app.button(key="adv_apply").click().run()
    assert not app.exception
    assert state(app)["iqr"] is not None
    pd.testing.assert_frame_equal(state(app)["working"], original)
    assert state(app)["history"] == []


def test_regression_receives_original_after_exploratory_transformations():
    app = sample_app()
    original = state(app)["original"].copy(deep=True)
    app.selectbox(key="adv_operation").select("Standaryzuj (z-score)").run()
    app.button(key="adv_apply").click().run()
    with patch("src.advanced_ui.train_linear_regression", wraps=train_linear_regression) as train:
        app.button(key="adv_train_regression").click().run()
        pd.testing.assert_frame_equal(train.call_args.args[0], original)
    assert not app.exception
    result = state(app)["regression"][1]
    assert result.dropped_target_rows == 1
    expected = original.iloc[result.train_positions][result.coefficients.index].median()
    np.testing.assert_allclose(result.pipeline.named_steps["imputer"].statistics_, expected)
    assert {"MAE", "RMSE", "R²"}.issubset({metric.label for metric in app.metric})
    assert len(app.get("plotly_chart")) == 2
    app.slider(key="adv_test_size").set_value(0.4).run()
    assert not app.exception
    assert "MAE" not in {metric.label for metric in app.metric}
    assert len(app.get("plotly_chart")) == 1


def test_kmeans_working_data_metrics_one_feature_and_unavailable_silhouette():
    app = sample_app()
    app.button(key="adv_apply").click().run()
    app.button(key="adv_train_kmeans").click().run()
    assert not app.exception
    result = state(app)["kmeans"][1]
    assert len(result.labels) == 16
    assert {"Inertia", "Silhouette score"}.issubset({metric.label for metric in app.metric})
    assert len(app.get("plotly_chart")) == 2
    app.multiselect(key="adv_cluster_features").set_value(["wiek"]).run()
    app.number_input(key="adv_cluster_count").set_value(1).run()
    assert "Inertia" not in {metric.label for metric in app.metric}
    app.button(key="adv_train_kmeans").click().run()
    assert not app.exception
    assert state(app)["kmeans"][1].silhouette is None
    assert next(metric for metric in app.metric if metric.label == "Silhouette score").value == "Nieokreślone"
    app.button(key="adv_reset").click().run()
    assert state(app)["kmeans"] is None


def test_invalid_ml_parameters_are_shown_without_traceback():
    app = sample_app()
    app.multiselect(key="adv_reg_features").set_value([]).run()
    app.button(key="adv_train_regression").click().run()
    assert not app.exception
    assert any("co najmniej jedną kolumnę" in warning.value for warning in app.warning)
    app.number_input(key="adv_cluster_count").set_value(100).run()
    app.button(key="adv_train_kmeans").click().run()
    assert not app.exception
    assert any("nie więcej klastrów" in warning.value for warning in app.warning)
    assert state(app)["regression"] is None and state(app)["kmeans"] is None


def test_changing_target_excludes_it_from_features():
    app = sample_app()
    app.selectbox(key="adv_target").select("wiek").run()
    assert not app.exception
    widget = app.multiselect(key="adv_reg_features")
    assert "wiek" not in widget.value
    assert "wiek" not in widget.options
    assert not any("default value" in warning.value for warning in app.warning)
    app.button(key="adv_train_regression").click().run()
    assert not app.exception
    assert state(app)["regression"] is not None


def test_replacing_dataset_resets_workspace_and_widgets_even_same_filename():
    first = pd.DataFrame({"x": [1.0, 2, np.nan, 4, 5, 6, 7, 8], "y": range(8)})
    first.attrs.update(separator=",", encoding="utf-8-sig")
    second = pd.DataFrame({"label": ["a", "b", "c"]})
    second.attrs.update(first.attrs)
    with patch("src.data_loader.load_csv", return_value=first) as loader:
        app = sample_app()
        app.button(key="adv_apply").click().run()
        app.button(key="adv_train_regression").click().run()
        assert state(app)["regression"] is not None
        old_fingerprint = state(app)["fingerprint"]
        loader.return_value = second
        app.run()
    assert not app.exception
    assert old_fingerprint != state(app)["fingerprint"]
    pd.testing.assert_frame_equal(state(app)["working"], second)
    assert state(app)["history"] == []
    assert state(app)["regression"] is None
    assert state(app)["kmeans"] is None
    assert not any(widget.key == "adv_target" for widget in app.selectbox)


def test_empty_working_dataset_can_be_restored():
    data = pd.DataFrame({"x": [1.0, 2.0], "text": [None, None]})
    data.attrs.update(separator=",", encoding="utf-8-sig")
    with patch("src.data_loader.load_csv", return_value=data):
        app = sample_app()
        app.button(key="adv_apply").click().run()
        assert not app.exception
        assert state(app)["working"].empty
        app.button(key="adv_reset").click().run()
        assert not app.exception
        pd.testing.assert_frame_equal(state(app)["working"], data)

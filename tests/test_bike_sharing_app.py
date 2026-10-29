from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.analysis_validation import AnalysisError
from src.bike_sharing_data import HOUR_PATH
from src.bike_sharing_study import StudyConfig

APP_PATH = Path(__file__).parents[1] / "app.py"


def test_missing_official_data_are_explained_without_substitution(tmp_path):
    with patch("src.bike_sharing_data.HOUR_PATH", tmp_path / "missing.csv"), patch("src.bike_sharing_data.download_bike_sharing", side_effect=AnalysisError("UCI: 403 Forbidden")):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.radio(key="source_type").set_value("Bike Sharing (UCI)").run()
        assert not app.exception
        assert not app.metric
        assert any("Brak lokalnego oryginału" in info.value for info in app.info)
        app.button(key="bike_download").click().run()
        assert not app.exception
        assert any("403 Forbidden" in error.value for error in app.error)
        assert not app.metric
        next(button for button in app.button if button.label == "Załaduj przykładowy zbiór").click().run()
        assert not app.exception
        assert app.metric[0].value == "20"


def test_study_integration_and_chronological_workflow(tmp_path, hourly_bike_fixture):
    # Wyłącznie test UI. Żadne sztuczne dane nie trafiają do data/bike_sharing/.
    path = tmp_path / "hour.csv"
    path.write_text(hourly_bike_fixture.to_csv(index=False))
    config = StudyConfig(cluster_counts=(2, 3), n_init=2, silhouette_sample_size=80)
    with patch("src.bike_sharing_data.HOUR_PATH", path), patch("src.bike_sharing_data.load_bike_sharing", return_value=hourly_bike_fixture), patch("src.bike_sharing_ui.StudyConfig.load", return_value=config):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.radio(key="source_type").set_value("Bike Sharing (UCI)").run()
        assert not app.exception
        assert any(tab.label == "Badanie Bike Sharing" for tab in app.tabs)
        assert not any(button.key == "adv_train_regression" for button in app.button)
        app.button(key="bike_run_study").click().run()
        assert not app.exception
        saved = app.session_state["bike_study_result"]
        assert saved is not None
        result = saved[1]
        assert result.metadata["train_end"] < result.metadata["test_start"]
        assert any(frame.value.equals(result.tables["regression_metrics"]) for frame in app.dataframe)
        assert len(app.get("plotly_chart")) > 10
        assert any("nie są bezpośrednio porównywalne" in item.value for item in app.warning)


def test_changed_dataset_or_configuration_invalidates_saved_study(tmp_path, hourly_bike_fixture):
    path = tmp_path / "hour.csv"
    path.write_text(hourly_bike_fixture.to_csv(index=False))
    config = StudyConfig(cluster_counts=(2, 3), n_init=2, silhouette_sample_size=80)
    with patch("src.bike_sharing_data.HOUR_PATH", path), patch("src.bike_sharing_data.load_bike_sharing", return_value=hourly_bike_fixture) as loader, patch("src.bike_sharing_ui.StudyConfig.load", return_value=config) as configured:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.radio(key="source_type").set_value("Bike Sharing (UCI)").run()
        app.button(key="bike_run_study").click().run()
        assert app.session_state["bike_study_result"] is not None
        configured.return_value = replace(config, random_state=9)
        app.run()
        assert not app.exception
        assert app.session_state["bike_study_result"] is None
        app.button(key="bike_run_study").click().run()
        updated = hourly_bike_fixture.copy(deep=True)
        updated.loc[0, "temp"] = .1234
        loader.return_value = updated
        app.run()
        assert not app.exception
        assert app.session_state["bike_study_result"] is None


@pytest.mark.skipif(not HOUR_PATH.exists(), reason="Pełny test UI wymaga oficjalnego hour.csv z UCI.")
def test_real_bike_data_in_streamlit_and_return_to_students():
    from hashlib import sha256
    import numpy as np
    import pandas as pd

    before = sha256(HOUR_PATH.read_bytes()).hexdigest()
    expected = pd.read_csv(APP_PATH.parent / "experiments/bike_sharing/results/tables/regression_metrics.csv", index_col=0)
    app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
    app.radio(key="source_type").set_value("Bike Sharing (UCI)").run()
    assert not app.exception
    assert app.metric[0].value == "17 379"
    assert not any(button.key == "adv_train_regression" for button in app.button)
    app.button(key="bike_run_study").click().run()
    assert not app.exception and not app.error
    result = app.session_state["bike_study_result"][1]
    assert result.metadata["rows"] == 17379
    assert result.tables["kmeans_original"].index.tolist() == [2, 3, 4, 5, 6]
    np.testing.assert_allclose(result.tables["regression_metrics"], expected, rtol=1e-10, atol=1e-10)
    assert any(frame.value.equals(result.tables["pca_variance"]) for frame in app.dataframe)
    assert sha256(HOUR_PATH.read_bytes()).hexdigest() == before
    next(button for button in app.button if button.label == "Załaduj przykładowy zbiór").click().run()
    assert not app.exception
    assert app.metric[0].value == "20"


def test_switching_to_bike_ignores_retained_user_upload(tmp_path, hourly_bike_fixture):
    from io import BytesIO

    uploaded = BytesIO(b"x,y\n1,2\n3,4\n")
    uploaded.name = "uploaded.csv"
    path = tmp_path / "hour.csv"
    path.write_text(hourly_bike_fixture.to_csv(index=False))
    with patch("streamlit.file_uploader", return_value=uploaded), patch("src.bike_sharing_data.HOUR_PATH", path), patch("src.bike_sharing_data.load_bike_sharing", return_value=hourly_bike_fixture):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        assert not app.exception
        assert app.metric[0].value == "2"
        app.radio(key="source_type").set_value("Bike Sharing (UCI)").run()
        assert not app.exception
        assert app.metric[0].value == "120"
        source_labels = [caption.value for caption in app.caption if caption.value.startswith("Źródło:")]
        assert len(source_labels) == 1
        assert "hour.csv" in source_labels[0]
        assert "uploaded.csv" not in source_labels[0]

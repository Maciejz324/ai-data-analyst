from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_loader import DataLoadError

APP_PATH = Path(__file__).parents[1] / "app.py"


def selectbox(app, label):
    return next(widget for widget in app.selectbox if widget.label == label)


def test_initial_state_and_sample_button():
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert not app.dataframe
    app.button[0].click().run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["20", "6", "4", "4"]
    assert len(app.get("plotly_chart")) == 1
    for chart in ["Wykres punktowy", "Wykres słupkowy", "Macierz korelacji"]:
        selectbox(app, "Typ wykresu").select(chart).run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 1
    app.radio[0].set_value("Własny CSV").run()
    assert not app.exception
    assert not app.dataframe


def test_text_only_dataset():
    df = pd.DataFrame({"label": ["a", "b", None]})
    df.attrs.update(separator=",", encoding="utf-8-sig")
    with patch("src.data_loader.load_csv", return_value=df):
        app = AppTest.from_file(str(APP_PATH)).run()
        app.button[0].click().run()
        assert not app.exception
        selectbox(app, "Typ wykresu").select("Wykres słupkowy").run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 1


def test_loader_error_is_presented_without_traceback():
    with patch("src.data_loader.load_csv", side_effect=DataLoadError("Błędny plik CSV")):
        app = AppTest.from_file(str(APP_PATH)).run()
        app.button[0].click().run()
        assert not app.exception
        assert app.error[0].value == "Błędny plik CSV"

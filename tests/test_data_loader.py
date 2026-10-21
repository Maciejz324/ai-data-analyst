import io
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoadError, load_csv


@pytest.mark.parametrize("separator", [",", ";", "\t"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1250"])
def test_separators_and_polish_encoding(separator, encoding):
    raw = f"imię{separator}wiek\nŁucja{separator}21\nŻaneta{separator}22\n".encode(encoding)
    df = load_csv(raw)
    assert df.shape == (2, 2)
    assert df["imię"].tolist() == ["Łucja", "Żaneta"]
    assert df["wiek"].tolist() == [21, 22]
    assert df.attrs["separator"] == separator


@pytest.mark.parametrize("raw", [
    b"", b" \n\t", b"\xef\xbb\xbf", b"a,b\n", b"a,b\n1,2,3\n",
    b"a,b\n1\n", b'a,b\n1,"unfinished', b"a,a\n1,2", b",b\n1,2",
    b"a,b\n1,\x00", b"\x89PNG\x00\x01",
])
def test_rejects_empty_or_malformed_files(raw):
    with pytest.raises(DataLoadError):
        load_csv(raw)


def test_limit_is_checked_before_parsing():
    with pytest.raises(DataLoadError, match="limit"):
        load_csv(b"a,b\n1,2\n", max_bytes=7)
    assert load_csv(b"a,b\n1,2\n", max_bytes=8).shape == (1, 2)


def test_missing_values_and_quoted_separator():
    df = load_csv(b'city,value\n"A,B",1\nC,\nD,   \n')
    assert df.loc[0, "city"] == "A,B"
    assert df["value"].isna().sum() == 2


def test_decimal_comma_and_explicit_settings():
    df = load_csv("wiek;wynik\n20;3,5\n".encode(), separator=";", encoding="utf-8-sig", decimal=",")
    assert df.loc[0, "wynik"] == 3.5


def test_one_column_and_multiline_quoted_values():
    assert load_csv(b"name\nAlice\nBob").shape == (2, 1)
    df = load_csv(b'name,value\n"A\nB",2\nC,3')
    assert df.loc[0, "name"] == "A\nB"


def test_stream_position_restored():
    stream = io.BytesIO(b"a,b\n1,2")
    stream.seek(3)
    assert load_csv(stream).shape == (1, 2)
    assert stream.tell() == 3


def test_local_paths_and_sample():
    path = Path(__file__).parents[1] / "data" / "sample_students.csv"
    df = load_csv(path)
    assert df.shape == (20, 6)
    assert df.isna().sum().sum() == 4
    assert pd.api.types.is_numeric_dtype(df["wynik_egzaminu"])
    with pytest.raises(DataLoadError, match="odczytać"):
        load_csv(path.parent / "missing.csv")


@pytest.mark.parametrize("settings", [{"separator": "|"}, {"encoding": "unknown"}, {"decimal": ":"}, {"max_bytes": 0}])
def test_invalid_settings(settings):
    with pytest.raises(DataLoadError):
        load_csv(b"a\n1", **settings)

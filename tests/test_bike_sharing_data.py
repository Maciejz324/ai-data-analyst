from hashlib import sha256
from io import BytesIO
import json
from urllib.error import URLError
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.analysis_validation import AnalysisError
from src.bike_sharing_data import ARCHIVE_URL, download_bike_sharing, load_bike_sharing, validate_hourly_data
from src.bike_sharing_study import hourly_timestamps
from src.data_analysis import column_information


def test_structure_sort_and_no_mutation(hourly_bike_fixture):
    data = hourly_bike_fixture.iloc[::-1].copy(deep=True)
    before = data.copy(deep=True)
    result = validate_hourly_data(data)
    assert hourly_timestamps(result).is_monotonic_increasing
    assert result.shape == (120, 17)
    pd.testing.assert_frame_equal(data, before)


def test_schema_table_does_not_mutate_original_column_metadata(hourly_bike_fixture):
    before = hourly_bike_fixture.copy(deep=True)
    info = column_information(hourly_bike_fixture)
    assert info.index.name == "Kolumna"
    pd.testing.assert_frame_equal(hourly_bike_fixture, before)


@pytest.mark.parametrize("mutation", [
    lambda d: d.drop(columns="cnt"),
    lambda d: d.assign(extra=1),
    lambda d: d.assign(hr=25),
    lambda d: d.assign(weathersit=0),
    lambda d: d.assign(cnt=-1),
    lambda d: d.assign(cnt=d["cnt"] + 1),
    lambda d: d.assign(instant=1),
    lambda d: d.assign(dteday="bad-date"),
    lambda d: d.assign(weekday=1),
    lambda d: d.assign(yr=1),
    lambda d: d.assign(holiday=1, workingday=1),
    lambda d: d.assign(temp=np.inf),
    lambda d: d.assign(hum=1.1),
    lambda d: d.assign(registered=np.nan),
])
def test_invalid_hourly_inputs(hourly_bike_fixture, mutation):
    with pytest.raises(AnalysisError):
        validate_hourly_data(mutation(hourly_bike_fixture))


def test_weather_missing_values_are_reported_not_invented(hourly_bike_fixture):
    assert validate_hourly_data(hourly_bike_fixture).isna().sum().sum() == 0
    hourly_bike_fixture.loc[0, "temp"] = np.nan
    assert validate_hourly_data(hourly_bike_fixture)["temp"].isna().sum() == 1


def test_download_retains_exact_bytes_and_checks_integrity(tmp_path, monkeypatch, hourly_bike_fixture):
    # Archiwum testowe jest zapisywane tylko w tmp_path, nigdy jako badanie UCI.
    raw = hourly_bike_fixture.to_csv(index=False, lineterminator="\r\n").encode()
    zipped = BytesIO()
    with ZipFile(zipped, "w") as bundle:
        bundle.writestr("hour.csv", raw)
        bundle.writestr("Readme.txt", "UNIT TEST FIXTURE, NOT UCI DATA")
    archive = zipped.getvalue()
    def mock_download(url, timeout):
        assert url == ARCHIVE_URL and timeout == 30
        return BytesIO(archive)
    monkeypatch.setattr("src.bike_sharing_data.urlopen", mock_download)
    target = download_bike_sharing(tmp_path)
    assert target.read_bytes() == raw
    assert (tmp_path / "source_archive.zip").read_bytes() == archive
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["hour_sha256"] == sha256(raw).hexdigest()
    assert manifest["archive_sha256"] == sha256(archive).hexdigest()
    assert manifest["license"] == "CC BY 4.0"
    assert load_bike_sharing(tmp_path).shape == (120, 17)
    monkeypatch.setattr("src.bike_sharing_data.urlopen", lambda *args, **kwargs: pytest.fail("Must reuse validated original"))
    assert download_bike_sharing(tmp_path) == target
    target.write_bytes(raw + b"\n")
    with pytest.raises(AnalysisError, match="SHA-256"):
        load_bike_sharing(tmp_path)


def test_network_failure_does_not_create_fake_data(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise URLError("403 Forbidden")
    monkeypatch.setattr("src.bike_sharing_data.urlopen", blocked)
    with pytest.raises(AnalysisError, match="Nie zastąpiono"):
        download_bike_sharing(tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_existing_unverified_csv_is_preserved(tmp_path):
    path = tmp_path / "hour.csv"
    path.write_bytes(b"existing user file")
    with pytest.raises(AnalysisError, match="manifestu"):
        download_bike_sharing(tmp_path)
    assert path.read_bytes() == b"existing user file"


@pytest.mark.parametrize("archive", [b"not zip", b""])
def test_bad_archive_is_rejected(tmp_path, monkeypatch, archive):
    monkeypatch.setattr("src.bike_sharing_data.urlopen", lambda *args, **kwargs: BytesIO(archive))
    with pytest.raises(AnalysisError, match="archiwum"):
        download_bike_sharing(tmp_path)
    assert not (tmp_path / "hour.csv").exists()

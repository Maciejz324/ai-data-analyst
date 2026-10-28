"""Małe sztuczne fixture'y wyłącznie do testów; nigdy do raportu badania UCI."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_bike_fixture():
    rng = np.random.default_rng(29)
    times = pd.date_range("2011-01-01", periods=120, freq="h")
    temp, hum, wind = rng.uniform(.05, .95, size=(3, len(times)))
    working = (~times.dayofweek.isin([5, 6])).astype(int)
    cnt = (20 + times.hour * 2 + working * 10 + temp * 30).astype(int)
    casual = (cnt * .3).astype(int)
    frame = pd.DataFrame({
        "instant": np.arange(1, len(times) + 1), "dteday": times.strftime("%Y-%m-%d"),
        "season": 1, "yr": 0, "mnth": times.month, "hr": times.hour, "holiday": 0,
        "weekday": (times.dayofweek + 1) % 7, "workingday": working,
        "weathersit": rng.integers(1, 4, len(times)), "temp": temp, "atemp": temp,
        "hum": hum, "windspeed": wind, "casual": casual, "registered": cnt - casual, "cnt": cnt,
    })
    frame.attrs.update(separator=",", encoding="utf-8-sig")
    return frame

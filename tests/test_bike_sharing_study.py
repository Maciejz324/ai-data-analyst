from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from experiments.bike_sharing.artifacts import export_study
from experiments.bike_sharing.run import main
from src.analysis_validation import AnalysisError
from src.bike_sharing_data import HOUR_PATH, load_bike_sharing
from src.bike_sharing_study import FORBIDDEN_FEATURES, StudyConfig, chronological_split, hourly_timestamps, run_bike_study


@pytest.fixture
def config():
    return StudyConfig(cluster_counts=(2, 3), n_init=2, silhouette_sample_size=80)


def test_canonical_configuration_has_all_requested_k_and_no_leaky_features():
    config = StudyConfig.load()
    assert config.cluster_counts == (2, 3, 4, 5, 6)
    assert not FORBIDDEN_FEATURES.intersection(config.weather_features + config.regression_numeric + config.regression_categorical)


@pytest.mark.parametrize("changes", [
    {"regression_numeric": ("casual",)}, {"regression_categorical": ("registered",)},
    {"weather_features": ("temp", "cnt")}, {"weather_features": ("instant", "temp")},
    {"test_fraction": 0}, {"test_fraction": "0.2"}, {"test_fraction": True},
    {"cluster_counts": (1, 2)}, {"cluster_counts": (2, 2)}, {"cluster_counts": (2, 7)},
    {"weather_features": ()}, {"pca_components": 1}, {"pca_components": 2.0},
    {"n_init": 0}, {"thread_limit": True}, {"silhouette_sample_size": 2},
])
def test_invalid_configuration_and_feature_leakage(changes):
    with pytest.raises(AnalysisError):
        replace(StudyConfig(), **changes).validate()


def test_chronological_split_ignores_input_order(hourly_bike_fixture):
    shuffled = hourly_bike_fixture.sample(frac=1, random_state=4)
    train, test = chronological_split(shuffled, .2)
    assert len(train) == 96 and len(test) == 24
    assert hourly_timestamps(train).max() < hourly_timestamps(test).min()
    assert set(train.instant).isdisjoint(test.instant)
    assert set(train.instant) | set(test.instant) == set(shuffled.instant)


def test_regression_metrics_baseline_and_train_only_preprocessing(hourly_bike_fixture, config):
    data = hourly_bike_fixture.copy(deep=True)
    data.loc[0, "temp"] = np.nan
    result = run_bike_study(data, config)
    train, test = chronological_split(data, config.test_fraction)
    numeric = result.regression_pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, train[list(config.regression_numeric)].median())
    assert not FORBIDDEN_FEATURES.intersection(result.regression_pipeline.feature_names_in_)
    predictions = result.tables["test_predictions"]
    np.testing.assert_allclose(predictions["baseline"], train["cnt"].mean())
    residual = predictions.actual - predictions.predicted
    metrics = result.tables["regression_metrics"].loc["linear_regression"]
    assert metrics["mae"] == pytest.approx(np.abs(residual).mean())
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.square(residual).mean()))
    assert metrics["r2"] == pytest.approx(1 - np.square(residual).sum() / np.square(test.cnt - test.cnt.mean()).sum())
    altered = data.copy(deep=True)
    altered.loc[test.index, ["temp", "hum", "windspeed"]] = [.99, .01, .9]
    altered.loc[test.index, "weathersit"] = 4
    altered.loc[test.index, "cnt"] += 1000
    altered.loc[test.index, "registered"] = altered.loc[test.index, "cnt"] - altered.loc[test.index, "casual"]
    second = run_bike_study(altered, config)
    for name in ["coef_", "intercept_"]:
        np.testing.assert_allclose(getattr(result.regression_pipeline.named_steps["model"], name), getattr(second.regression_pipeline.named_steps["model"], name))
    second_pre = second.regression_pipeline.named_steps["preprocess"]
    np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, second_pre.named_transformers_["numeric"].named_steps["scaler"].mean_)
    weather_category_index = list(config.regression_categorical).index("weathersit")
    assert 4 not in second_pre.named_transformers_["calendar"].categories_[weather_category_index]


def test_clustering_and_pca_do_not_depend_on_cnt(hourly_bike_fixture, config):
    original = hourly_bike_fixture.copy(deep=True)
    first = run_bike_study(original, config)
    altered = original.copy(deep=True)
    altered["cnt"] = altered["cnt"].iloc[::-1].to_numpy() + 200
    altered["casual"] = 0
    altered["registered"] = altered["cnt"]
    second = run_bike_study(altered, config)
    assert first.metadata["selected_k"] == second.metadata["selected_k"]
    for space in ["original", "pca"]:
        for k in config.cluster_counts:
            np.testing.assert_array_equal(first.cluster_models[space][k].labels_, second.cluster_models[space][k].labels_)
            assert first.cluster_models[space][k].init == "k-means++"
    pd.testing.assert_frame_equal(first.pca.scores, second.pca.scores)
    assert first.pca.components.columns.tolist() == list(config.weather_features)
    assert first.pca.variance.explained_variance_ratio.sum() == pytest.approx(1)
    assert first.pca.variance.cumulative_explained_variance_ratio.is_monotonic_increasing
    for k, model in first.cluster_models["pca"].items():
        matrix = first.pca.scores[["PC1", "PC2"]].to_numpy()
        assert np.square(matrix - model.cluster_centers_[model.labels_]).sum() == pytest.approx(first.tables["kmeans_pca"].loc[k, "inertia"])
    pd.testing.assert_frame_equal(hourly_bike_fixture, original)


def test_repeatability_and_export_are_based_on_actual_computations(tmp_path, hourly_bike_fixture, config):
    first = run_bike_study(hourly_bike_fixture, config)
    second = run_bike_study(hourly_bike_fixture, config)
    assert first.metadata == second.metadata
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name], second.tables[name])
    # Wyłącznie tymczasowe artefakty testowe — nie są wynikami badania rzeczywistego UCI.
    provenance = {"hour_sha256": "UNIT_TEST_FIXTURE_NOT_UCI_DATA"}
    export_study(first, provenance, tmp_path)
    metadata = json.loads((tmp_path / "results/metadata.json").read_text())
    retained = metadata["versioned_artifacts_sha256"]
    assert "tables/regression_metrics.csv" in retained
    assert "tables/test_predictions.csv" in retained
    assert "figures/pca_projection.html" in retained
    assert "tables/pca_scores.csv" not in retained
    assert "tables/cluster_labels_original.csv" not in retained
    assert "tables/cluster_labels_pca.csv" not in retained
    assert not any(name.endswith(".json") for name in retained)
    exported = pd.read_csv(tmp_path / "results/tables/regression_metrics.csv", index_col=0)
    np.testing.assert_allclose(exported, first.tables["regression_metrics"])
    assert "UNIT_TEST_FIXTURE_NOT_UCI_DATA" in (tmp_path / "REPORT.md").read_text()
    export_study(second, provenance, tmp_path)
    second_metadata = json.loads((tmp_path / "results/metadata.json").read_text())
    assert metadata == second_metadata


def test_download_error_stops_cli_without_fake_results(tmp_path, monkeypatch):
    def blocked():
        raise AnalysisError("UCI network blocked")
    monkeypatch.setattr("experiments.bike_sharing.run.download_bike_sharing", blocked)
    assert main(["--download", "--output", str(tmp_path)]) == 1
    assert not (tmp_path / "results").exists()


@pytest.mark.skipif(not HOUR_PATH.exists(), reason="Oficjalny hour.csv niepobrany: dostęp do UCI wymaga odblokowania sieci.")
def test_real_official_dataset_and_full_study():
    data = load_bike_sharing()
    result = run_bike_study(data)
    assert result.metadata["rows"] == len(data)
    assert result.tables["kmeans_original"].index.tolist() == [2, 3, 4, 5, 6]
    assert np.isfinite(result.tables["regression_metrics"].to_numpy()).all()

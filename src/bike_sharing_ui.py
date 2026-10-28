"""Odtworzenie kanonicznego badania w Streamlit, bez zapisu danych sesji."""

from dataclasses import asdict
from hashlib import sha256
import json

import pandas as pd
import streamlit as st

from src.analysis_validation import AnalysisError
from src.bike_sharing_study import StudyConfig, run_bike_study


def render_bike_study(data: pd.DataFrame) -> None:
    st.subheader("Badanie Bike Sharing — UCI")
    st.info("Regresja: chronologiczny test na końcu szeregu, preprocessing wyłącznie na treningu. casual i registered sumują się do cnt i są wykluczone z cech, podobnie jak instant i cnt. K-Means i PCA opisują wyłącznie pogodę.")
    try:
        config = StudyConfig.load()
    except AnalysisError as exc:
        st.error(str(exc))
        return
    fingerprint = sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes()
                         + json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    saved = st.session_state.get("bike_study_result")
    if saved is not None and saved[0] != fingerprint:
        st.session_state["bike_study_result"] = None
        saved = None
    st.caption("Jedna konfiguracja dla skryptu i aplikacji: experiments/bike_sharing/config.json. Obliczenia nie zmieniają oryginalnego CSV ani kopii roboczej w zakładce przygotowania danych.")
    if st.button("Uruchom pełne badanie Bike Sharing", key="bike_run_study"):
        st.session_state["bike_study_result"] = None
        saved = None
        try:
            with st.spinner("EDA, chronologiczna regresja, porównanie klastrów oraz PCA…"):
                result = run_bike_study(data, config)
            saved = (fingerprint, result)
            st.session_state["bike_study_result"] = saved
        except AnalysisError as exc:
            st.error(str(exc))
    if saved is None:
        st.code("python -m experiments.bike_sharing.run --download", language="bash")
        st.caption("Komenda oblicza te same wyniki i zapisuje raport, tabele, konfigurację oraz wykresy. Pierwsze uruchomienie wymaga dostępu do oficjalnego UCI.")
        return
    result = saved[1]
    st.write(f"Rekordy: {result.metadata['rows']} · Brakujące pola: {result.metadata['missing_cells']} · Brakujące godziny w zakresie: {result.metadata['missing_hours_in_span']}")
    eda, regression, clustering, pca = st.tabs(["EDA Bike Sharing", "Regresja chronologiczna", "Klastery pogody", "PCA i porównanie klastrów"])
    with eda:
        st.caption("Kody sezonu i pogody pochodzą z UCI. Korelacje i różnice średnich opisują związki, nie przyczynowość. Odstające nie są automatycznie usuwane.")
        for name, label in [("schema_missing", "Typy i braki danych"), ("descriptive_statistics", "Statystyki opisowe"), ("iqr_outliers", "Wartości odstające — IQR")]:
            st.write(label)
            st.dataframe(result.tables[name], use_container_width=True)
        names = ["demand_distribution", "demand_by_hr", "demand_by_weekday", "demand_by_season", "demand_by_weathersit", "weather_correlation"]
        names += [f"demand_vs_{feature}" for feature in config.weather_features]
        for name in names:
            st.plotly_chart(result.figures[name], use_container_width=True)
    with regression:
        st.write(f"Trening do {result.metadata['train_end']}; test od {result.metadata['test_start']}.")
        st.dataframe(result.tables["regression_metrics"], use_container_width=True)
        st.caption("Baseline to średnia cnt z treningu. Metryki dotyczą surowych predykcji, bez obcinania ujemnych wyników. To ocena ex post z pogodą zaobserwowaną w danej godzinie, nie operacyjna prognoza pogody i popytu.")
        for name in ["regression_actual_predicted", "regression_over_time", "residual_distribution", "errors_by_hour"]:
            st.plotly_chart(result.figures[name], use_container_width=True)
        st.dataframe(result.tables["errors_by_hr"], use_container_width=True)
    with clustering:
        st.write("Cechy pogodowe: " + ", ".join(config.weather_features))
        st.dataframe(result.tables["kmeans_original"], use_container_width=True)
        st.caption(f"Najwyższe silhouette wśród kandydatów: k={result.metadata['selected_k']['original']}. To nie jest dowód globalnej optymalności. cnt dołączono dopiero po grupowaniu.")
        for name in ["inertia_original", "silhouette_original", "clusters_original", "cnt_after_clustering_original"]:
            st.plotly_chart(result.figures[name], use_container_width=True)
        st.dataframe(result.tables["weather_profile_original"], use_container_width=True)
        st.dataframe(result.tables["demand_by_cluster_original"], use_container_width=True)
    with pca:
        st.dataframe(result.tables["pca_variance"], use_container_width=True)
        st.dataframe(result.tables["pca_components"], use_container_width=True)
        st.caption("Wagi dotyczą standaryzowanych cech pogodowych, nie są procentami. Znak osi jest umowny. PC1/PC2 nie są ponownie skalowane przed K-Means.")
        for name in ["pca_scree", "pca_cumulative", "pca_projection", "inertia_pca", "silhouette_pca", "clusters_pca"]:
            st.plotly_chart(result.figures[name], use_container_width=True)
        st.dataframe(result.tables["clustering_comparison"], use_container_width=True)
        st.warning("Odległości zmieniają się po redukcji wymiarów. Inertia i silhouette z różnych przestrzeni nie są bezpośrednio porównywalne. ARI porównuje podziały przy tym samym k; dodatkowo oceniamy etykiety PCA przez silhouette w oryginalnej przestrzeni standaryzowanej.")

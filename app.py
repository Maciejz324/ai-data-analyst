"""Interfejs MVP AI Data Analyst — uruchom: streamlit run app.py."""

from pathlib import Path

import numpy as np
import streamlit as st

from src.advanced_ui import render_advanced_analysis
from src.analysis_validation import AnalysisError
from src.bike_sharing_data import HOUR_PATH, download_bike_sharing, load_bike_sharing
from src.bike_sharing_ui import render_bike_study
from src.data_analysis import column_information, descriptive_statistics, numeric_data, pearson_correlation
from src.data_loader import DataLoadError, load_csv
from src.visualizations import VisualizationError, bar_chart, correlation_matrix, histogram, scatter_plot

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_students.csv"


def select_sample() -> None:
    st.session_state["source_type"] = "Dane przykładowe"


st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("AI Data Analyst")
st.write("Poznaj swoje dane: wczytaj CSV, sprawdź statystyki i odkrywaj zależności na wykresach.")
st.caption("Projekt akademicki · Techniki Analizy Danych · Bez integracji LLM")

with st.sidebar:
    st.header("Źródło danych")
    source_type = st.radio("Wybierz źródło", ["Własny CSV", "Dane przykładowe", "Bike Sharing (UCI)"], key="source_type")
    uploaded = st.file_uploader("Prześlij plik CSV", type=["csv"], disabled=source_type != "Własny CSV")
    st.button("Załaduj przykładowy zbiór", use_container_width=True, on_click=select_sample)
    with st.expander("Ustawienia odczytu CSV"):
        separator_label = st.selectbox("Separator", ["Automatycznie", "Przecinek", "Średnik", "Tabulator"])
        encoding_label = st.selectbox("Kodowanie", ["Automatycznie", "UTF-8", "CP1250", "CP1252"])
        decimal = st.selectbox("Separator dziesiętny", [".", ","])
    st.caption("Limit: 10 MiB. CSV musi zawierać nagłówek i wiersze danych.")
    st.info("Dane są przetwarzane w pamięci serwera aplikacji. Nie są wysyłane do zewnętrznych usług.")

use_sample = source_type == "Dane przykładowe"
use_bike = source_type == "Bike Sharing (UCI)"
if uploaded is not None and source_type == "Własny CSV":
    source = uploaded
    source_name = uploaded.name
elif use_sample:
    source = SAMPLE_PATH
    source_name = "sample_students.csv (dane syntetyczne)"
elif use_bike:
    source = HOUR_PATH
    source_name = "hour.csv — Bike Sharing, UCI 275 (CC BY 4.0)"
    if not HOUR_PATH.exists():
        st.info("Brak lokalnego oryginału Bike Sharing. Pobierz go z oficjalnego UCI lub uruchom komendę poniżej.")
        if st.button("Pobierz oryginalny zbiór Bike Sharing z UCI", key="bike_download"):
            try:
                download_bike_sharing()
                st.rerun()
            except AnalysisError as exc:
                st.error(str(exc))
        st.code("python -m experiments.bike_sharing.run --download", language="bash")
        st.stop()
else:
    st.info("Prześlij plik CSV w panelu bocznym lub załaduj przykładowy zbiór studentów.")
    st.stop()

try:
    if use_bike:
        df = load_bike_sharing()
    elif use_sample and source == SAMPLE_PATH:
        df = load_csv(source)
    else:
        df = load_csv(
            source,
            separator={"Przecinek": ",", "Średnik": ";", "Tabulator": "\t"}.get(separator_label),
            encoding={"UTF-8": "utf-8-sig", "CP1250": "cp1250", "CP1252": "cp1252"}.get(encoding_label),
            decimal=decimal,
        )
except (DataLoadError, AnalysisError) as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Informacje o zbiorze")
st.caption(f"Źródło: {source_name}")
cols = st.columns(4)
cols[0].metric("Wiersze", f"{len(df):,}".replace(",", " "))
cols[1].metric("Kolumny", len(df.columns))
cols[2].metric("Kolumny numeryczne", len(numeric_data(df).columns))
cols[3].metric("Brakujące wartości", int(df.isna().sum().sum()))
sep_name = {",": "przecinek", ";": "średnik", "\t": "tabulator"}[df.attrs["separator"]]
st.caption(f"Odczyt: {df.attrs['encoding']} · separator: {sep_name}")

tab_names = [
    "Podgląd i jakość danych", "Statystyki", "Wizualizacje", "Zaawansowana analiza danych",
]
tabs = st.tabs(tab_names + (["Badanie Bike Sharing"] if use_bike else []))
preview_tab, stats_tab, charts_tab, advanced_tab = tabs[:4]
with preview_tab:
    st.subheader("Podgląd danych")
    preview_count = st.slider("Liczba wierszy w podglądzie", 5, 100, 20, 5)
    st.dataframe(df.head(preview_count), use_container_width=True)
    st.caption("Podgląd pokazuje początek zbioru. Analizy obejmują wszystkie wiersze.")
    st.subheader("Kolumny i brakujące wartości")
    st.dataframe(column_information(df).rename(columns={
        "dtype": "Typ danych", "missing_count": "Liczba braków", "missing_percent": "Braki (%)",
    }), use_container_width=True)
    if df.isna().any().any():
        st.warning("W zbiorze są braki danych. Statystyki pomijają je w danej kolumnie, a wykresy w wybranych kolumnach.")
    else:
        st.success("Nie wykryto brakujących wartości.")

with stats_tab:
    st.subheader("Statystyki opisowe")
    stats = descriptive_statistics(df)
    if stats.empty:
        st.info("Brak kolumn numerycznych. Dostępna jest analiza braków i wykres liczebności kategorii.")
    else:
        st.dataframe(stats.rename(columns={
            "count": "Liczebność", "mean": "Średnia", "median": "Mediana",
            "min": "Minimum", "max": "Maksimum", "std": "Odchylenie standardowe",
        }), use_container_width=True)
    st.caption("Odchylenie standardowe z próby (ddof=1) wymaga co najmniej dwóch obserwacji. Niewyznaczalne wyniki są puste.")
    st.subheader("Korelacja Pearsona")
    corr = pearson_correlation(df)
    if corr.empty:
        st.info("Korelacja wymaga co najmniej dwóch niestałych kolumn numerycznych.")
    else:
        st.dataframe(corr, use_container_width=True)
    st.caption("Korelacje liczymy parami na co najmniej dwóch wspólnych obserwacjach. Korelacja nie oznacza przyczynowości.")

with charts_tab:
    st.subheader("Wizualizacje")
    numeric_columns = numeric_data(df).columns.tolist()
    chart = st.selectbox("Typ wykresu", ["Histogram", "Wykres punktowy", "Wykres słupkowy", "Macierz korelacji"])
    st.caption("Wykresy pomijają brakujące i nieskończone wartości. Wykres słupkowy pokazuje liczebność lub średnią według kategorii.")
    fig = None
    try:
        if chart == "Histogram":
            if not numeric_columns:
                st.info("Histogram wymaga kolumny numerycznej.")
            else:
                column = st.selectbox("Kolumna", numeric_columns, key="hist_column")
                bins = st.slider("Liczba przedziałów (orientacyjna)", 1, 100, 20)
                fig = histogram(df, column, bins)
        elif chart == "Wykres punktowy":
            if len(numeric_columns) < 2:
                st.info("Wykres punktowy wymaga dwóch kolumn numerycznych.")
            else:
                x = st.selectbox("Oś X", numeric_columns)
                y = st.selectbox("Oś Y", [col for col in numeric_columns if col != x])
                fig = scatter_plot(df, x, y)
        elif chart == "Wykres słupkowy":
            category = st.selectbox("Kolumna kategorii", df.columns.tolist())
            aggregation = st.radio("Agregacja", ["Liczebność", "Średnia"], horizontal=True)
            if aggregation == "Średnia" and not numeric_columns:
                st.info("Średnia wymaga kolumny numerycznej.")
            else:
                value = st.selectbox("Wartość", numeric_columns) if aggregation == "Średnia" else None
                fig = bar_chart(df, category, value)
        else:
            selected = st.multiselect("Kolumny do korelacji", numeric_columns, default=numeric_columns[:10])
            fig = correlation_matrix(df, selected)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    except VisualizationError as exc:
        st.warning(str(exc))

with advanced_tab:
    render_advanced_analysis(df, source_name, enable_regression=not use_bike)

if use_bike:
    with tabs[4]:
        render_bike_study(df)

if numeric_data(df).shape[1] and np.isinf(df.select_dtypes(include="number")).any().any():
    st.warning("Wykryto wartości nieskończone. Są pomijane w statystykach i wykresach; tabela braków opisuje oryginalne dane.")

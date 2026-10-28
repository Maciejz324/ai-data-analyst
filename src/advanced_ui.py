"""Polski interfejs zaawansowanej analizy; obliczenia znajdują się poza UI."""

from hashlib import sha256

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis_validation import AnalysisError, numeric_columns
from src.data_analysis import column_information
from src.data_preparation import detect_iqr_outliers, drop_missing_rows, impute_missing, scale_numeric
from src.dimensionality_reduction import run_pca
from src.ml_models import train_kmeans, train_linear_regression
from src.visualizations import (
    cluster_plot, pca_cumulative_variance_plot, pca_scatter_plot, pca_scree_plot,
    regression_prediction_plot,
)

SCALING = {"Standaryzacja (z-score)": "standard", "Normalizacja (min–max)": "minmax", "Bez skalowania": "none"}
IMPUTATION = {"Mediana": "median", "Średnia": "mean"}


def _workspace(df: pd.DataFrame, source_name: str) -> dict:
    digest = sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update(repr((source_name, df.columns.tolist(), df.dtypes.astype(str).tolist(), df.attrs)).encode())
    fingerprint = digest.hexdigest()
    state = st.session_state.get("adv_workspace")
    if state is None or state["fingerprint"] != fingerprint:
        # Zmiana pliku lub interpretacji CSV unieważnia również wybór kolumn.
        for key in list(st.session_state):
            if key.startswith("adv_"):
                del st.session_state[key]
        state = {
            "fingerprint": fingerprint, "original": df.copy(deep=True), "working": df.copy(deep=True),
            "history": [], "iqr": None, "regression": None, "kmeans": None, "pca": None,
        }
        st.session_state["adv_workspace"] = state
    state.setdefault("pca", None)
    return state


def _cleaning(state: dict) -> None:
    st.subheader("Czyszczenie i przygotowanie danych")
    st.caption("Operacje zmieniają wyłącznie kopię roboczą tej sesji. Dotychczasowe statystyki i wykresy MVP nadal opisują oryginał.")
    data = state["working"]
    columns = numeric_columns(data)
    operation = st.selectbox("Operacja", [
        "Usuń wiersze z brakami", "Uzupełnij średnią", "Uzupełnij medianą",
        "Wykryj wartości odstające (IQR)", "Normalizuj (min–max)", "Standaryzuj (z-score)",
    ], key="adv_operation")
    selected = columns
    if operation != "Usuń wiersze z brakami":
        selected = st.multiselect("Kolumny do przygotowania", columns, default=columns, key="adv_clean_columns")
    factor = 1.5
    if operation == "Wykryj wartości odstające (IQR)":
        factor = st.number_input("Mnożnik IQR", min_value=0.1, value=1.5, step=0.1, key="adv_iqr_factor")
        st.caption("Wykrywanie oznacza wartości poza Q1 − k·IQR i Q3 + k·IQR; nie usuwa wierszy.")
    apply_col, reset_col = st.columns(2)
    if apply_col.button("Zastosuj operację", key="adv_apply"):
        try:
            if operation == "Wykryj wartości odstające (IQR)":
                state["iqr"] = ((tuple(selected), factor), detect_iqr_outliers(data, selected, factor))
            else:
                if operation == "Usuń wiersze z brakami":
                    transformed = drop_missing_rows(data)
                elif operation in ("Uzupełnij średnią", "Uzupełnij medianą"):
                    transformed = impute_missing(data, selected, "mean" if operation == "Uzupełnij średnią" else "median")
                else:
                    transformed = scale_numeric(data, selected, "minmax" if operation == "Normalizuj (min–max)" else "standard")
                state["working"] = transformed
                state["history"].append(f"{operation}: {', '.join(selected) if operation != 'Usuń wiersze z brakami' else 'wszystkie kolumny'}")
                state["iqr"] = state["kmeans"] = state["pca"] = None
                st.rerun()
        except AnalysisError as exc:
            st.warning(str(exc))
    if reset_col.button("Przywróć oryginalne dane", key="adv_reset"):
        state["working"] = state["original"].copy(deep=True)
        state["history"] = []
        state["iqr"] = state["kmeans"] = state["regression"] = state["pca"] = None
        st.rerun()
    current = state["working"]
    st.caption(f"Oryginał: {len(state['original'])} wierszy · Kopia robocza: {len(current)} wierszy · Operacje: {len(state['history'])}")
    if state["history"]:
        with st.expander("Historia transformacji"):
            for item in state["history"]:
                st.write(item)
    left, right = st.columns(2)
    with left:
        st.write("Oryginalne dane — pierwsze 20 wierszy")
        st.dataframe(state["original"].head(20), use_container_width=True)
    with right:
        st.write("Po transformacjach — pierwsze 20 wierszy")
        st.dataframe(current.head(20), use_container_width=True)
    st.write("Brakujące wartości w kopii roboczej")
    st.dataframe(column_information(current).rename(columns={
        "dtype": "Typ danych", "missing_count": "Liczba braków", "missing_percent": "Braki (%)",
    }), use_container_width=True)
    st.caption("Nowe operacje traktują ±∞ jako braki. Średnia i mediana dotyczą wyłącznie wybranych cech numerycznych; braki tekstowe pozostają. Skalowanie zachowuje braki.")
    numeric = current[numeric_columns(current)].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric).any():
        st.warning(f"W kopii roboczej są wartości nieskończone: {int(np.isinf(numeric).sum())}. Tabela braków liczy tylko puste pola.")
    if current.empty:
        st.info("Kopia robocza nie zawiera wierszy. Przywróć oryginał, aby kontynuować analizę.")
    if operation == "Wykryj wartości odstające (IQR)" and state["iqr"] is not None:
        config, result = state["iqr"]
        if config == (tuple(selected), factor):
            st.dataframe(result.bounds.rename(columns={
                "q1": "Q1", "q3": "Q3", "iqr": "IQR", "lower": "Dolny próg",
                "upper": "Górny próg", "outlier_count": "Liczba odstających",
            }), use_container_width=True)
            st.write("Wiersze z co najmniej jedną wartością odstającą")
            st.dataframe(current.loc[result.mask.any(axis=1)].head(100), use_container_width=True)
            st.caption("Podgląd pokazuje do 100 wierszy. Braki nie są odstającymi; dla pustych kolumn progi są niewyznaczalne.")


def _regression(state: dict) -> None:
    st.subheader("Regresja liniowa")
    st.info("Regresja używa oryginalnych danych. Transformacje kopii roboczej nie wpływają na model. Najpierw dzielimy dane, potem dopasowujemy imputację i skalowanie wyłącznie na części treningowej.")
    original = state["original"]
    columns = numeric_columns(original)
    if len(columns) < 2:
        st.info("Regresja wymaga co najmniej dwóch kolumn numerycznych: celu i cechy.")
        return
    target = st.selectbox("Zmienna docelowa", columns, index=len(columns) - 1, key="adv_target")
    available = [column for column in columns if column != target]
    # Po zmianie celu usuń go z poprzednio wybranych cech przed utworzeniem widgetu.
    if "adv_reg_features" in st.session_state:
        previous = st.session_state["adv_reg_features"]
        filtered = [col for col in previous if col in available]
        if filtered != previous:
            st.session_state["adv_reg_features"] = filtered
    features = st.multiselect(
        "Zmienne wejściowe", available,
        default=available if "adv_reg_features" not in st.session_state else None,
        key="adv_reg_features",
    )
    test_size = st.slider("Udział zbioru testowego", 0.1, 0.5, 0.25, 0.05, key="adv_test_size")
    imputation = IMPUTATION[st.selectbox("Uzupełnianie cech w regresji", list(IMPUTATION), key="adv_reg_imputation")]
    scaling = SCALING[st.selectbox("Skalowanie cech w regresji", list(SCALING), key="adv_reg_scaling")]
    seed = int(st.number_input("random_state regresji", min_value=0, max_value=2**32 - 1, value=42, step=1, key="adv_reg_seed"))
    config = (target, tuple(features), test_size, imputation, scaling, seed)
    if st.button("Trenuj regresję", key="adv_train_regression"):
        state["regression"] = None
        try:
            with st.spinner("Trenowanie regresji…"):
                result = train_linear_regression(original, target, features, test_size=test_size, random_state=seed, imputation=imputation, scaling=scaling)
            state["regression"] = (config, result)
        except AnalysisError as exc:
            st.warning(str(exc))
    if state["regression"] is None:
        return
    previous_config, result = state["regression"]
    if config != previous_config:
        st.info("Parametry regresji zmieniły się. Wytrenuj model ponownie, aby zobaczyć aktualne wyniki.")
        return
    metrics = st.columns(3)
    for column, name, key in zip(metrics, ["MAE", "RMSE", "R²"], ["mae", "rmse", "r2"]):
        value = result.metrics[key]
        column.metric(name, f"{value:.6g}" if value is not None else "Nieokreślone")
    st.caption(f"Trening: {len(result.train_positions)} wierszy · Test: {len(result.test_positions)} · Pominięte brakujące/nieskończone cele: {result.dropped_target_rows}")
    if result.metrics["r2"] is None:
        st.info("R² jest niewyznaczalne dla stałych wartości celu w części testowej.")
    if result.pipeline.named_steps["model"].rank_ < len(features):
        st.warning("Cechy treningowe są liniowo zależne lub stałe. Współczynniki nie mają jednoznacznej interpretacji.")
    st.dataframe(result.coefficients.rename("Współczynnik").rename_axis("Cecha").to_frame(), use_container_width=True)
    st.caption(f"Wyraz wolny: {result.intercept:.6g}. Współczynniki dotyczą cech po wybranym skalowaniu; cel pozostaje w oryginalnych jednostkach.")
    st.plotly_chart(regression_prediction_plot(result.predictions), use_container_width=True)
    st.dataframe(result.predictions.rename(columns={"actual": "Wartość rzeczywista", "predicted": "Wartość przewidywana"}), use_container_width=True)


def _kmeans(state: dict) -> None:
    st.subheader("Klasteryzacja K-Means")
    st.caption("K-Means korzysta z aktualnej kopii roboczej. Imputację i skalowanie dopasowuje do całego analizowanego zbioru; nie ma tu podziału train/test.")
    data = state["working"]
    columns = numeric_columns(data)
    if not columns or data.empty:
        st.info("K-Means wymaga danych i co najmniej jednej kolumny numerycznej.")
        return
    features = st.multiselect("Cechy do klasteryzacji", columns, default=columns[:2], key="adv_cluster_features")
    count = int(st.number_input("Liczba klastrów", min_value=1, value=3, step=1, key="adv_cluster_count"))
    imputation = IMPUTATION[st.selectbox("Uzupełnianie cech w K-Means", list(IMPUTATION), key="adv_cluster_imputation")]
    scaling = SCALING[st.selectbox("Skalowanie cech w K-Means", list(SCALING)[:2], key="adv_cluster_scaling")]
    seed = int(st.number_input("random_state K-Means", min_value=0, max_value=2**32 - 1, value=42, step=1, key="adv_cluster_seed"))
    st.caption("Inicjalizacja: k-means++, n_init=10, max_iter=300, tol=0.0001, algorithm=lloyd. Numery klastrów są umowne.")
    config = (tuple(features), count, imputation, scaling, seed)
    if st.button("Uruchom K-Means", key="adv_train_kmeans"):
        state["kmeans"] = None
        try:
            with st.spinner("Wyznaczanie klastrów…"):
                result = train_kmeans(data, features, n_clusters=count, random_state=seed, imputation=imputation, scaling=scaling)
            state["kmeans"] = (config, result)
        except AnalysisError as exc:
            st.warning(str(exc))
    if state["kmeans"] is None:
        return
    previous_config, result = state["kmeans"]
    if config != previous_config:
        st.info("Parametry K-Means zmieniły się. Uruchom algorytm ponownie, aby zobaczyć aktualne wyniki.")
        return
    left, right = st.columns(2)
    left.metric("Inertia", f"{result.inertia:.6g}")
    right.metric("Silhouette score", f"{result.silhouette:.6g}" if result.silhouette is not None else "Nieokreślone")
    st.caption(f"Metryki dotyczą cech po skalowaniu. {result.silhouette_note}")
    x = st.selectbox("Oś X klastrów", features, key="adv_cluster_x")
    others = [feature for feature in features if feature != x]
    y = st.selectbox("Oś Y klastrów", others, key="adv_cluster_y") if others else None
    st.plotly_chart(cluster_plot(result.scaled_data, result.labels, x, y), use_container_width=True)
    st.caption("Wykres pokazuje rzut na wybrane cechy, nie pełną przestrzeń klasteryzacji. Dla jednej cechy oś Y oznacza pozycję wiersza.")
    st.write("Centra klastrów w jednostkach kopii roboczej (przed skalowaniem modelu)")
    st.dataframe(result.centers.rename_axis("Klaster"), use_container_width=True)
    st.write("Przypisanie obserwacji do klastrów")
    st.dataframe(result.labels.rename("Klaster").to_frame(), use_container_width=True)


def _pca(state: dict) -> None:
    st.subheader("Analiza głównych składowych (PCA)")
    st.info("PCA jest metodą nienadzorowaną: szuka kierunków największej zmienności, bez zmiennej docelowej. Redukcja wymiarów może powodować utratę informacji; skalowanie cech wpływa na wynik.")
    source = st.selectbox("Dane do PCA", ["Kopia robocza", "Oryginalne dane (kopia)"], key="adv_pca_source")
    data = state["working"] if source == "Kopia robocza" else state["original"]
    columns = numeric_columns(data)
    if len(columns) < 2 or len(data) < 2:
        state["pca"] = None
        st.info("PCA wymaga co najmniej dwóch kolumn numerycznych i dwóch obserwacji. Wybierz inne dane lub przywróć oryginał.")
        return
    st.caption("Wybierz cechy świadomie. Nie wybieramy ich automatycznie — identyfikatory i numery porządkowe zwykle nie są wartościowymi cechami analitycznymi.")
    features = st.multiselect("Cechy do PCA", columns, default=[], key="adv_pca_features")
    limit = min(len(data), len(features))
    count = int(st.number_input(
        "Liczba składowych PCA", min_value=1, max_value=max(1, limit),
        value=min(2, max(1, limit)), step=1, key="adv_pca_count",
    ))
    imputation = IMPUTATION[st.selectbox("Uzupełnianie cech w PCA", list(IMPUTATION), key="adv_pca_imputation")]
    scaling = SCALING[st.selectbox("Skalowanie cech w PCA", list(SCALING)[:2], key="adv_pca_scaling")]
    st.caption("Imputację i skalowanie dopasowujemy na całej wybranej kopii. Oryginał pozostaje bez zmian. PCA używa deterministycznego pełnego SVD, bez wybielania.")
    if len(features) < 2:
        st.info("Wskaż co najmniej dwie cechy numeryczne, aby uruchomić PCA.")
    else:
        st.caption(f"Dopuszczalna liczba składowych: 1–{limit} (minimum liczby obserwacji i wybranych cech).")
    config = (source, tuple(features), count, imputation, scaling)
    if state["pca"] is not None and config != state["pca"][0]:
        state["pca"] = None
        st.info("Parametry PCA zmieniły się. Uruchom analizę ponownie, aby zobaczyć aktualne wyniki.")
    if st.button("Uruchom PCA", key="adv_train_pca", disabled=len(features) < 2):
        state["pca"] = None
        try:
            with st.spinner("Wyznaczanie głównych składowych…"):
                result = run_pca(data, features, n_components=count, imputation=imputation, scaling=scaling)
            state["pca"] = (config, result)
        except AnalysisError as exc:
            st.warning(str(exc))
    if state["pca"] is None:
        return
    result = state["pca"][1]
    if result.constant_features:
        st.warning(f"Stałe cechy nie wnoszą wariancji: {', '.join(result.constant_features)}. Pozostawiono je w analizie z wagą zależną od wyznaczonej osi.")
    eigenvalues = result.variance["explained_variance"]
    if (eigenvalues <= np.finfo(float).eps * max(1, eigenvalues.max())).any():
        st.info("Niektóre składowe mają zerową lub numerycznie znikomą wariancję; nie wnoszą dodatkowej informacji.")
    total = float(result.variance["cumulative_explained_variance_ratio"].iloc[-1])
    st.metric("Wariancja zachowana przez PCA", f"{total:.2%}")
    st.caption(f"Obserwacje: {len(result.scores)} · Wybrane cechy: {len(features)} · Składowe: {count}. Udziały odnoszą się do całkowitej wariancji wszystkich wybranych cech po przygotowaniu.")
    summary = result.variance.copy()
    for column in ["explained_variance_ratio", "cumulative_explained_variance_ratio"]:
        summary[column] *= 100
    st.dataframe(summary.rename(columns={
        "explained_variance": "Wariancja składowej",
        "explained_variance_ratio": "Wyjaśniona wariancja (%)",
        "cumulative_explained_variance_ratio": "Skumulowana wariancja (%)",
    }).rename_axis("Składowa"), use_container_width=True)
    st.plotly_chart(pca_scree_plot(result.variance), use_container_width=True)
    st.plotly_chart(pca_cumulative_variance_plot(result.variance), use_container_width=True)
    if result.scores.shape[1] >= 2:
        st.plotly_chart(pca_scatter_plot(result.scores), use_container_width=True)
    else:
        st.info("Wykres PCA 2D wymaga co najmniej dwóch składowych. Dla jednej składowej dostępne są tabela wyników i wykresy wariancji.")
    st.write("Współczynniki składowych (wagi cech po imputacji i skalowaniu)")
    st.dataframe(result.components.rename_axis("Składowa"), use_container_width=True)
    st.caption("Wiersz to składowa, kolumna to cecha. Wagi definiują kierunek osi w przygotowanej przestrzeni; nie są procentami udziału ani korelacjami. PCA dodatkowo centruje przygotowane cechy. Znak składowej jest umowny — odwrócenie całej osi nie zmienia interpretacji wariancji.")
    st.write("Dane po transformacji PCA")
    st.dataframe(result.scores, use_container_width=True)
    st.caption("Każdy wiersz odpowiada obserwacji z wybranej kopii; zachowano jej indeks. To projekcja danych, bez automatycznej oceny jakości klas lub predykcji.")


def render_advanced_analysis(df: pd.DataFrame, source_name: str) -> None:
    state = _workspace(df, source_name)
    st.subheader("Zaawansowana analiza danych")
    cleaning, regression, kmeans, pca = st.tabs(["Czyszczenie danych", "Regresja liniowa", "K-Means", "PCA"])
    with cleaning:
        _cleaning(state)
    with regression:
        _regression(state)
    with kmeans:
        _kmeans(state)
    with pca:
        _pca(state)

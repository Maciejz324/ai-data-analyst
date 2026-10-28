# Powtarzalne badanie Bike Sharing

Po instalacji zależności projektu uruchom z katalogu repozytorium:

```bash
python -m experiments.bike_sharing.run --download
```

To jedna komenda do pobrania oficjalnego archiwum, walidacji hour.csv,
przeprowadzenia EDA, chronologicznej regresji i baseline, porównania K-Means
dla k=2…6 oraz PCA i zapisania rzeczywiście obliczonych wyników. Kolejne
uruchomienia wykorzystują lokalny oryginał po sprawdzeniu SHA-256, bez ponownego
pobierania. Sam eksperyment po pobraniu danych działa bez sieci.

W chmurze wymagany jest dostęp HTTPS do **archive.ics.uci.edu**. Nie wyłączaj
weryfikacji TLS. Jeśli pobranie się nie powiedzie, skrypt kończy się kodem 1,
wyjaśnia problem i nie generuje zastępczych danych ani fikcyjnych metryk.

Oficjalny zbiór został pobrany i badanie wykonano. Oryginał ma 17 379 rekordów
i 17 kolumn, bez pustych pól. Plik jest bajtowo zgodny z członkiem hour.csv
oficjalnego archiwum UCI. Wyniki, interpretacje i ograniczenia są w REPORT.md.
Wcześniejsza blokada 403 nie dotyczy już aktualnego uruchomienia.

## Konfiguracja i artefakty

`config.json` jest wspólny dla skryptu i aplikacji. Zawiera podział czasowy,
listy dozwolonych cech, siatkę k, seed, n_init, limit próbki silhouette,
liczbę składowych oraz limit wątków. Wersje bibliotek i kopia konfiguracji
są zapisywane razem z wynikami. Powtarzalność liczb jest sprawdzana w ramach
tej samej wersji bibliotek; różne BLAS i wersje mogą zmienić ostatnie cyfry.

Po poprawnym uruchomieniu powstają:

- `data/bike_sharing/hour.csv` — niezmienione bajty oryginału;
- `data/bike_sharing/source_archive.zip`, `Readme_UCI.txt`, `manifest.json`;
- `experiments/bike_sharing/REPORT.md` — raport z obliczonymi wynikami i wnioskami;
- `results/tables/*.csv` — statystyki, braki, IQR, regresja, predykcje testowe,
  błędy w grupach, porównanie k, PCA, etykiety i charakterystyki klastrów;
- `results/figures/*.html` — interaktywne wykresy Plotly; ich JavaScript jest
  ładowany z CDN Plotly, więc otwarcie HTML wymaga Internetu;
- `results/figures/*.json` — figury możliwe do odczytania lokalnie przez Plotly;
- `results/metadata.json` — pochodzenie, SHA-256 danych, rzeczywisty podział
  czasowy, konfiguracja, wersje i SHA-256 wszystkich artefaktów tego uruchomienia.

## Co wersjonujemy w GitHubie

Do repozytorium należą oryginalny `hour.csv` (około 1,1 MiB), `Readme_UCI.txt`,
`SOURCE.md` i manifest pochodzenia, kod, konfiguracja i REPORT.md. Z wyników
wersjonujemy **26 tabel CSV** (w tym metryki i predykcje testowe) oraz **24 wykresy
HTML**. Cały zestaw danych i artefaktów przeznaczonych do Git zajmuje około 4,5 MiB.

`.gitignore` pomija redundantne archiwum `source_archive.zip`, 24 duplikaty figur
JSON oraz duże tabele `pca_scores.csv`, `cluster_labels_original.csv` i
`cluster_labels_pca.csv`. Pliki pozostają lokalnie i są ponownie generowane tą
samą komendą. Po świeżym klonowaniu ZIP nie jest wymagany: istniejący oryginalny
CSV jest weryfikowany manifestem, a eksperyment działa bez pobierania.

`metadata.json` zawiera pełny manifest `artifacts_sha256` dla wszystkich 77
generowanych plików oraz `versioned_artifacts_sha256` dla 50 plików wynikowych
wersjonowanych w Git. Brak lokalnych, regenerowanych plików po klonowaniu nie
oznacza niekompletności wersjonowanego zestawu. Odtwarza je:

```bash
python -m experiments.bike_sharing.run --download
```

CSV zapisują 12 cyfr znaczących. Wyniki w pamięci i obliczenia używają float64.
SHA-256 figur dotyczy konkretnych wersji bibliotek. Stare, dodatkowe pliki z
innych konfiguracji nie są częścią bieżącego manifestu; do osobnych porównań
używaj osobnych katalogów wyjściowych:

```bash
python -m experiments.bike_sharing.run --config experiments/bike_sharing/config.json --output /tmp/bike-study-run
```

Pełny opis metod: [METHODOLOGY.md](METHODOLOGY.md). Pochodzenie i słownik:
[SOURCE.md](../../data/bike_sharing/SOURCE.md).

## Odtworzenie w Streamlit

```bash
python -m streamlit run app.py
```

Wybierz **Bike Sharing (UCI)** w panelu źródła. Jeśli oryginału brakuje, aplikacja
oferuje pobranie z UCI i wyświetla błąd, jeśli sieć go blokuje. Gdy dane są
dostępne, otwórz **Badanie Bike Sharing → Uruchom pełne badanie Bike Sharing**.
Obliczenia korzystają z tego samego kodu i konfiguracji co CLI. Nie zapisują
danych sesji na dysku. Obecny przykład studentów jest nadal dostępny.

Regresja z ogólnej zakładki zaawansowanej jest dla tego źródła zastąpiona
komunikatem kierującym do oceny chronologicznej, aby nie sugerować losowej
oceny prognoz ani użycia casual/registered. Eksploracja i pozostałe narzędzia
pozostają dostępne, a oficjalny CSV nie jest modyfikowany.

## Testy

```bash
python -m pytest -q -W error
```

Testy jednostkowe używają małych jawnie sztucznych fixture'ów wyłącznie do
sprawdzania algorytmów, błędów, eksportu i UI w katalogach tymczasowych.
**Fixture'y nie są zamiennikiem hour.csv i nie stanowią wyników badania.**
Test rzeczywistego pełnego zbioru jest pomijany tylko wtedy, gdy nie pobrano
oryginału; powód pominięcia jest jawny. Po pobraniu danych test wykonuje pełne
badanie na oryginale. Zaliczony zestaw testów jednostkowych nie dowodzi jeszcze
ukończenia analizy rzeczywistego zbioru.

Pełny test oryginalnego zbioru jest teraz wykonywany. Dodatkowy test AppTest
uruchamia całe badanie na rzeczywistych danych, porównuje metryki z zapisanymi
CSV i sprawdza niezmienność oryginału oraz powrót do przykładu studentów.

Aktualna weryfikacja: **288 passed**, zero błędów i testów pominiętych dla
`python -m pytest -q -W error`. Zaliczone są test pełnego oryginału i pełny
test UI na 17 379 wierszach. Serwer Streamlit zwrócił HTTP 200 dla HTML i
`ok` dla `/_stcore/health`; proces walidacyjny został zatrzymany.

Eksperyment uruchomiono ponownie z tej samej konfiguracji. Wszystkie **77**
sum SHA-256 artefaktów oraz metadane były identyczne: **29 tabel CSV**, **24
wykresy HTML** i ich **24 odpowiedniki JSON**. Oryginalny CSV pozostał niezmieniony.
`pip check` i `git diff --check` zakończyły się poprawnie.

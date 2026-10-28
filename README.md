# AI Data Analyst

Projekt akademicki na przedmiot **Prompt Engineering**, temat 16:
**Konwersacyjny asystent analizy danych**. Docelowo użytkownik prześle CSV
i zada pytania po polsku, a LLM zaproponuje plan analizy oraz objaśni wyniki.
Obecna wersja to MVP rozszerzone o przygotowanie danych i podstawowe modele ML,
bez integracji LLM i bez symulowania odpowiedzi AI.

## Technologie

Python 3.12, Streamlit (interfejs), pandas i NumPy (analiza), scikit-learn (modele ML), Plotly
(interaktywne wykresy), pytest i Streamlit AppTest (testy).
Zakresy kompatybilnych wersji znajdują się w `requirements.txt`.

## Instalacja i uruchomienie

W katalogu repozytorium, na Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Jeśli `.venv` już istnieje, pomiń pierwszą komendę. Na Windows aktywuj
środowisko poleceniem `.venv\Scripts\Activate.ps1` w PowerShell.
Otwórz http://localhost:8501 przy pracy lokalnej. W środowisku chmurowym
korzystaj z udostępniania portów, jeśli dany dostawca je obsługuje.

Uruchomienie bez aktywacji istniejącego środowiska (Linux/macOS):

```bash
.venv/bin/python -m streamlit run app.py --server.headless true
```

Kliknij **Załaduj przykładowy zbiór** lub prześlij własny CSV w panelu bocznym.
Źródła można przełączać radiowym wyborem. W ustawieniach odczytu CSV można
wybrać separator, kodowanie i separator dziesiętny. Ustawienia te dotyczą
własnego pliku; przykład ma stały format UTF-8 z przecinkiem i kropką dziesiętną.

## Funkcjonalności

- Wczytywanie CSV do 10 MiB: przecinek, średnik lub tabulator; UTF-8 z/bez
  BOM, CP1250 i CP1252. Automatyczne rozpoznawanie oraz ręczne ustawienia.
- Walidacja pustych plików, nagłówka, liczby pól, cudzysłowów i znaków binarnych.
  CSV musi mieć unikalne, niepuste nazwy kolumn i co najmniej jeden rekord.
  Jednokolumnowe CSV są obsługiwane.
- Interaktywna tabela podglądu (5–100 początkowych wierszy), liczba wierszy
  i kolumn, typy kolumn, liczebność i procent braków w każdej kolumnie.
- Liczebność niepustych skończonych obserwacji, średnia, mediana, minimum,
  maksimum i odchylenie standardowe z próby (`ddof=1`) dla kolumn numerycznych.
- Korelacja Pearsona parami oraz macierz korelacji wybranych kolumn.
- Histogram, wykres punktowy, wykres słupkowy liczebności kategorii lub średniej
  według kategorii. Użytkownik wybiera typ wykresu i kolumny.
- Polski interfejs, komunikaty walidacji i syntetyczny przykład studentów
  (20 rekordów, 6 kolumn, 4 brakujące wartości).
- Sekcja **Zaawansowana analiza danych**: czyszczenie danych, regresja liniowa
  i klasteryzacja K-Means, bez integracji LLM ani kodu R.

## Zaawansowana analiza danych

Po wczytaniu pliku lub przykładu otwórz zakładkę **Zaawansowana analiza danych**.
Oryginał i kopia robocza są przechowywane oddzielnie w pamięci sesji. Podgląd
pokazuje je obok siebie, wraz z historią operacji i tabelą braków kopii roboczej.
Przycisk **Przywróć oryginalne dane** usuwa transformacje i wyniki modeli.
Zmiana zawartości pliku, kolumn lub sposobu jego odczytu resetuje kopię i wyniki.
Dotychczasowe zakładki MVP nadal analizują oryginał. Wyniki modeli są ukrywane
po zmianie ich parametrów do czasu ponownego treningu.

### Czyszczenie i przygotowanie

- **Usuń wiersze z brakami** odrzuca wiersze z brakami w dowolnej kolumnie.
- **Uzupełnij średnią / medianą** dotyczy wybranych kolumn numerycznych.
  Braki w kolumnach tekstowych pozostają. Kolumny bez żadnej skończonej
  obserwacji nie mogą być imputowane; aplikacja pokazuje komunikat.
- **Wykryj wartości odstające (IQR)** wyznacza kwartyle z interpolacją liniową,
  `IQR = Q3 − Q1` i granice `Q1 − k·IQR`, `Q3 + k·IQR` (domyślnie `k=1.5`).
  Zwraca progi, liczby odstających i podgląd oznaczonych wierszy. Niczego nie usuwa.
  Braki nie są odstającymi; puste kolumny mają niewyznaczalne progi.
  Przy `IQR=0` odstające są wartości różne od wspólnego kwartyla.
- **Normalizuj (min–max)** przeskalowuje każdą wybraną cechę do zakresu `[0,1]`
  z dokładnością zmiennoprzecinkową. **Standaryzuj (z-score)** odejmuje średnią
  i dzieli przez odchylenie populacyjne (`ddof=0`, jak `StandardScaler`).
  Stałe kolumny otrzymują zera; braki pozostają brakami.

Nowe operacje traktują `+∞` i `−∞` w cechach numerycznych jak braki. Tabela braków
nadal liczy puste pola; nieskończoności są sygnalizowane osobno. Kolumny bool
i liczby zespolone nie są cechami ML. Transformacje eksploracyjne dopasowują
parametry do całej kopii roboczej — nie wolno używać takiej kopii do niezależnej
oceny regresji na zbiorze testowym.

### Regresja liniowa

Wybierz cel numeryczny, jedną lub kilka innych cech, udział części testowej,
imputację, skalowanie i `random_state`. Przycisk **Trenuj regresję** uruchamia
`LinearRegression` i pokazuje MAE, RMSE, R², współczynniki, wyraz wolny oraz
wartości rzeczywiste i przewidywane na części testowej z linią idealnej predykcji.

Regresja **zawsze otrzymuje oryginalne dane**, niezależnie od zmian w kopii roboczej:

1. Wiersze bez skończonego celu są pomijane; cel nigdy nie jest imputowany.
2. `train_test_split` losuje pozycje wierszy z zadanym `random_state` (domyślnie 42).
   W obu częściach muszą pozostać co najmniej dwie obserwacje.
3. `Pipeline(SimpleImputer → StandardScaler/MinMaxScaler/passthrough → LinearRegression)`
   dopasowuje imputer, skaler i model **wyłącznie na treningu**. Test jest tylko
   transformowany i oceniany. Trening z cechą całkowicie pustą w części treningowej
   kończy się komunikatem, nawet jeśli ma ona wartości w części testowej.

`MAE = mean(|y − ŷ|)`, `RMSE = sqrt(mean((y − ŷ)²))`,
`R² = 1 − sum((y − ŷ)²) / sum((y − mean(y))²)`. R² może być ujemne.
Dla stałego celu testowego R² jest wyświetlane jako **Nieokreślone**.
Współczynniki odnoszą się do cech **po wybranym skalowaniu**, cel pozostaje
w swoich jednostkach. Stałe i współliniowe cechy powodują ostrzeżenie o interpretacji.

To model liniowy z jednym losowym podziałem, bez walidacji krzyżowej, regularyzacji,
testów istotności i przedziałów predykcji. Małe próby dają niestabilną ocenę.
Dla szeregów czasowych, powtarzanych osób/grup i silnie zależnych obserwacji
potrzebny jest inny schemat podziału. Wielokrotne dobieranie parametrów na
podstawie tych samych metryk testowych również może prowadzić do optymistycznej oceny.

### K-Means

Wybierz numeryczne cechy **aktualnej kopii roboczej**, liczbę klastrów, imputację,
skalowanie i `random_state`. Algorytm dopasowuje imputer i skaler na całym
analizowanym zbiorze (analiza nienadzorowana), potem uruchamia scikit-learn `KMeans`:
`init="k-means++"`, `n_init=10`, `max_iter=300`, `tol=1e-4`, `algorithm="lloyd"`,
domyślnie `random_state=42`. Liczba klastrów nie może przekraczać liczby różnych
obserwacji po przygotowaniu cech; minimum to dwie obserwacje i jeden klaster.

- **Inertia** jest sumą kwadratów odległości od centrów w przestrzeni skalowanej.
  Maleje przy zwiększaniu liczby klastrów; sama nie wybiera właściwego `k`.
- **Silhouette score** ocenia separację w zakresie `[-1,1]`; wyższy wynik oznacza
  lepiej rozdzielone grupy. Wymaga od 2 do `n−1` reprezentowanych klastrów.
  Dla ponad 2000 wierszy używa deterministycznej losowej próby 2000 obserwacji,
  aby ograniczyć koszt obliczeń kwadratowych. Jeśli próba nie spełnia warunków
  metryki, wynik pozostaje **Nieokreślony**, z objaśnieniem.
- Wykres to rzut na dwie wybrane cechy skalowane, bez redukcji PCA. Dla jednej
  cechy druga oś oznacza pozycję wiersza. Tabele pokazują przypisania i centra
  w jednostkach kopii roboczej sprzed skalowania modelu.

K-Means preferuje zwarte, zbliżone do kulistych grupy, wymaga wyboru `k`, jest
wrażliwy na skalę, wartości odstające i inicjalizację; może znaleźć lokalne minimum.
Numery klastrów są umowne. Przy przyszłym porównaniu z R należy zachować kolejność
cech, imputację, `ddof=0` (R `scale()` zwykle używa `ddof=1`), parametry i tolerancję.
Ten sam seed nie zapewnia identycznej inicjalizacji w różnych bibliotekach ani
identycznych etykiet; należy porównywać podziały z uwzględnieniem permutacji etykiet
(np. ARI) i wartości funkcji celu. W tej wersji nie ma implementacji R.

### Wywołania poza interfejsem

Funkcje nie zależą od Streamlit i nie modyfikują argumentów. Wyniki modeli
zawierają pipeline i ramki/serie pandas; regresja udostępnia także pozycje train/test,
jednoznaczne również przy powtórzonym indeksie. Błędne dane i parametry zgłaszają
`AnalysisError`. Przykład uruchamiany z katalogu repozytorium:

```python
from src.data_loader import load_csv
from src.data_preparation import impute_missing
from src.ml_models import train_linear_regression, train_kmeans

original = load_csv("data/sample_students.csv")
working = impute_missing(original, ["wiek", "godziny_nauki"], strategy="median")
regression = train_linear_regression(
    original, "wynik_egzaminu", ["godziny_nauki", "frekwencja"], random_state=42,
)
clusters = train_kmeans(working, ["wiek", "godziny_nauki"], n_clusters=3, random_state=42)
print(regression.metrics)
print(clusters.inertia, clusters.silhouette)
```

API stanowi podstawę przyszłych adapterów LLM i porównań z R, ale nie dodaje
żadnej z tych integracji.

## Interpretacja i ograniczenia MVP

Analizy obejmują cały zbiór, niezależnie od liczby wierszy w podglądzie.
W dotychczasowych zakładkach MVP braki są pomijane, bez imputacji. Puste pola, same białe znaki i standardowe
znaczniki pandas (np. `NA`, `NaN`, `NULL`) są traktowane jako braki.
Liczby rozpoznaje pandas; kolumny mieszane z tekstem pozostają nienumeryczne.
Identyfikatory zapisane wyłącznie cyframi mogą zostać uznane za liczby;
przy wyborze kolumn do analizy należy uwzględnić ich znaczenie.

Wartości nieskończone są pomijane w analizach i wykresach. Tabela braków
opisuje dane oryginalne i nie zalicza nieskończoności do pustych pól.
Odchylenie standardowe wymaga dwóch skończonych obserwacji. Korelacja pomija
kolumny stałe lub z mniej niż dwiema różnymi wartościami, a dla par wymaga
co najmniej dwóch wspólnych obserwacji. Wyniki niewyznaczalne mają `NaN`.
Korelacja z dwóch punktów ma małą wartość interpretacyjną; korelacja nie oznacza
przyczynowości. Kolumny logiczne nie są analizowane jak liczby.

Automatyczne rozpoznanie separatora opiera się na nagłówku, a kodowania są
próbowane w kolejności UTF-8, CP1250, CP1252. Kodowania jednobajtowe mogą być
niejednoznaczne. Jeśli tekst wygląda niepoprawnie, wskaż kodowanie ręcznie.
Nie każdy plik tekstowy da się odróżnić od jednokolumnowego CSV.
Nie obsługujemy XLSX, archiwów, UTF-16, automatycznej konwersji dat ani ręcznej edycji komórek.
Wykres słupkowy ma limit 50 kategorii; początkowy wybór macierzy obejmuje
do 10 kolumn. Współczynniki i pełne wykresy dużych zbiorów mogą obciążać pamięć;
limit bajtów CSV nie jest limitem pamięci po wczytaniu. Plotly dobiera granice
histogramu, więc zadana liczba przedziałów jest orientacyjna.

## Struktura

```text
ai-data-analyst/
├── app.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_analysis.py
│   ├── analysis_validation.py
│   ├── data_preparation.py
│   ├── ml_models.py
│   ├── advanced_ui.py
│   └── visualizations.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_analysis.py
│   ├── test_visualizations.py
│   ├── test_data_preparation.py
│   ├── test_ml_models.py
│   ├── test_advanced_app.py
│   └── test_app.py
├── prompts/
│   └── README.md
├── data/
│   └── sample_students.csv
├── .streamlit/
│   └── config.toml
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Testy i sprawdzenie serwera

```bash
python -m pytest -q
python -m pytest -q -W error
python -m streamlit run app.py --server.headless true
```

Przy działającym serwerze, w drugim terminalu:

```bash
curl --fail http://localhost:8501/_stcore/health
```

Oczekiwany wynik kontroli zdrowia: `ok`. Testy obejmują separatory, polskie
kodowania, braki, błędne pliki, limit rozmiaru, statystyki na znanych danych,
korelacje, przypadki niewyznaczalne, dane tekstowe, obiekty i wartości wykresów
oraz działanie przykładu i przełączanie wykresów w Streamlit AppTest.
AppTest sprawdza logikę interfejsu; nie zastępuje oceny wyglądu w przeglądarce.

Testy rozszerzenia sprawdzają również: niezmienność oryginału, imputację,
usuwanie wierszy, IQR, oba skalowania, znane współczynniki i metryki regresji,
brak wpływu zmiany części testowej na dopasowane transformatory i model,
znane klastry, powtarzalność, brakujące cele, puste/stałe cechy, powtórzone indeksy,
niewyznaczalne metryki, błędne parametry oraz operacje, trening i reset danych w AppTest.

Weryfikacja rozszerzenia (28.10.2026, Python 3.12.14, pandas 2.3.3,
NumPy 2.5.4, scikit-learn 1.9.1, Streamlit 1.66.0, Plotly 6.9.0):
`python -m pytest -q -W error` — **166 zaliczonych testów** (51 poprzednich
i 115 nowych), bez testów pominiętych. `pip check` nie wykazał konfliktów.
Rzeczywisty serwer `app.py` zwrócił HTTP 200 dla strony głównej i `ok` dla
`/_stcore/health`; proces testowy został następnie zatrzymany.
Przykład API powyżej: MAE 3,71895, RMSE 6,38476, R² 0,77026;
K-Means: inertia 13,55362 i silhouette 0,41429.

Weryfikacja MVP (21.10.2026, Python 3.12.14): `python -m pytest -q -W error`
— **51 testów przeszło, bez ostrzeżeń**. `pip check` nie wykazał konfliktów.
Serwer uruchomił się na porcie 8501, a `/_stcore/health` zwrócił `ok`.
Dla przykładu potwierdzono średnią wyniku egzaminu 70,0526 i korelację
godzin nauki z wynikiem 0,990793 oraz wygenerowanie czterech typów wykresów.

## Prywatność i bezpieczeństwo

Dane użytkownika są przetwarzane w pamięci procesu serwera Streamlit,
bez zapisywania uploadów na dysku i bez przesyłania do zewnętrznych API.
Przy uruchomieniu w chmurze przeglądarka przesyła plik do tego serwera.
Nie ma współdzielonego cache danych użytkownika. Telemetria Streamlit jest
wyłączona w `.streamlit/config.toml`. Aplikacja nie wykonuje kodu z CSV,
nie używa `eval`/`exec` ani nie wymaga kluczy API. Zachowano `.gitignore`,
który wyklucza m.in. `.venv`, `.env` i `.streamlit/secrets.toml`.
Przykładowy zbiór jest całkowicie syntetyczny i nie zawiera rzeczywistych danych osobowych.

## Kolejny etap: integracja LLM

Funkcje analityczne i modele w `src` działają niezależnie od Streamlit;
`advanced_ui.py` jest osobną warstwą interfejsu. W kolejnym etapie warto dodać:

1. Osobny adapter API LLM i interpretację pytań po polsku; klucze wyłącznie
   w zmiennych środowiskowych lub lokalnych sekretach Streamlit.
2. Walidowany plan JSON z nazwą dozwolonej operacji i argumentami (kolumny,
   agregacja, typ wykresu), bez kodu Python generowanego do wykonania.
3. Rejestr dozwolonych funkcji z `src`, walidację argumentów względem schematu
   danych, limity zasobów i zatwierdzanie planu przez użytkownika.
4. Minimalizację danych przekazywanych do modelu i świadomą zgodę użytkownika
   przed wysłaniem informacji do zewnętrznego API.
5. Wersjonowane prompty zero-shot, role prompting i few-shot w `prompts/`.
6. Zbiór pytań z oczekiwanymi planami i wynikami oraz automatyczną ewaluację
   poprawności operacji, wyników, odporności na prompt injection, kosztu i czasu.

Obecne MVP nie wymaga sieci do analizy ani konta u dostawcy LLM.

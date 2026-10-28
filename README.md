# AI Data Analyst

Projekt akademicki przygotowywany do zaliczenia przedmiotu **Techniki Analizy Danych**.
Aplikacja służy do eksploracji CSV, przygotowania danych, regresji, klasteryzacji
i analizy głównych składowych. Docelowy **konwersacyjny asystent analizy danych**
ma również umożliwiać zadawanie pytań po polsku i objaśnianie wyników przez LLM.
Obecna wersja zawiera metody analityczne i podstawowe modele ML,
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
- Sekcja **Zaawansowana analiza danych**: czyszczenie danych, regresja liniowa,
  klasteryzacja K-Means i PCA, bez integracji LLM ani kodu R.

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

### PCA — analiza głównych składowych

PCA (**Principal Component Analysis**) to nienadzorowana metoda liniowego
rzutowania cech na ortogonalne osie największej wariancji. **PC1** wyjaśnia
największą część zmienności danych po przygotowaniu; **PC2** wyjaśnia największą
część pozostałej zmienności w kierunku prostopadłym do PC1. Kolejne składowe
kontynuują ten proces. Nie ma tu zmiennej docelowej ani oceny jakości predykcji.

Obsługa w aplikacji:

1. Wczytaj własny CSV lub kliknij **Załaduj przykładowy zbiór**.
2. Otwórz **Zaawansowana analiza danych → PCA**. Wybierz **Kopię roboczą**
   (po transformacjach z zakładki czyszczenia) lub **Oryginalne dane (kopia)**.
3. Wskaż co najmniej dwie cechy numeryczne. Początkowo żadna nie jest wybrana:
   identyfikator, numer porządkowy czy kod kategorii zwykle nie jest wartościową
   cechą do PCA, nawet jeśli pandas rozpoznaje go jako liczbę.
4. Ustaw liczbę składowych, imputację średnią lub medianą oraz standaryzację
   albo normalizację min–max. Kliknij **Uruchom PCA**.
5. Odczytaj tabelę wariancji, scree plot (udział każdej składowej), wykres
   skumulowanej wariancji, tabelę wag i przekształcone dane. Wykres punktowy
   PC1–PC2 pojawia się tylko dla co najmniej dwóch wyznaczonych składowych.

Przygotowanie i walidacja:

- `src/dimensionality_reduction.py` udostępnia niezależną od Streamlit funkcję
  `run_pca(df, features, n_components=2, imputation="median", scaling="standard")`.
  `features` to jawna lista/tuple nazw kolumn; pozostałe parametry są nazwane.
- Kopia wybranych cech zamienia `±∞` na braki, a następnie dopasowany jest
  **Pipeline: SimpleImputer → StandardScaler/MinMaxScaler → PCA**.
  Nie usuwamy obserwacji ani nie modyfikujemy źródłowego DataFrame.
- `StandardScaler` stosuje odchylenie populacyjne (`ddof=0`), a `MinMaxScaler`
  skaluje każdą cechę do `[0,1]`. PCA następnie centruje przygotowane cechy.
  Imputacja i skalowanie są dopasowane na całej wybranej kopii, ponieważ to
  analiza eksploracyjna. Użycie PCA przed oceną modelu nadzorowanego wymaga
  ponownego dopasowania **całego pipeline tylko na treningu**.
- Wymagane są minimum dwie rzeczywiste cechy numeryczne i dwie obserwacje.
  Liczba składowych musi być całkowita w zakresie `1..min(n_obserwacji, n_cech)`.
  Kolumny bool, tekstowe i zespolone nie są dopuszczane. Cecha bez żadnej
  skończonej wartości powoduje komunikat, zamiast automatycznego usunięcia.
- Stałe cechy pozostają w analizie z ostrzeżeniem i nie wnoszą wariancji.
  Zerowa całkowita wariancja uniemożliwia PCA. Niewyznaczalne wyniki lub zakresy
  liczb powodujące przepełnienie są zgłaszane jako `AnalysisError`.
- Algorytm to `sklearn.decomposition.PCA(svd_solver="full", whiten=False)`:
  pełny, deterministyczny SVD bez losowania. Zmiana pliku, kopii danych,
  zastosowanie transformacji, reset lub zmiana parametrów unieważniają wynik;
  poprzedni wynik nie wraca po ponownym wybraniu dawnych parametrów.

Interpretacja wyników:

- **Wyjaśniona wariancja** to wariancja wyników składowej z dzielnikiem `n−1`.
  **Udział wyjaśnionej wariancji** dzieli ją przez całkowitą wariancję wszystkich
  przygotowanych cech. Suma udziałów przy wyborze tylko części składowych może
  być mniejsza niż 100%; nie przeskalowujemy jej sztucznie do 100%.
- **Skumulowana wariancja** sumuje udziały od PC1 do danej składowej i pokazuje,
  jaką część wariancji zachowuje redukcja. Tabela i wykresy w UI pokazują procenty;
  w wyniku API proporcje są zapisane jako liczby od 0 do 1 (z dokładnością float).
- **Współczynniki składowych** (`components_`) definiują kierunki osi: wiersz
  oznacza składową, kolumna cechę po imputacji i skalowaniu. Dla przygotowanej
  obserwacji `z`: `PCk = suma_j(w[k,j] * (z[j] − średnia_PCA[j]))`.
  To podpisane wagi wektorów jednostkowych, **nie procenty udziału, korelacje
  ani miary przyczynowej ważności**. Stała cecha może mieć wagę na osi zerowej
  wariancji, ale nie wnosi do wyników zmienności.
- Przekształcone dane (`scores`) mają po jednym wierszu na obserwację i kolumny
  PC1, PC2, itd.; zachowują indeks wejściowy, również powtórzony.

Ograniczenia: redukcja zwykle powoduje utratę informacji, a duża zachowana
wariancja nie gwarantuje przydatności dla klasyfikacji, klasteryzacji czy regresji.
PCA jest liniowe, wrażliwe na odstające, dobór cech, skalowanie oraz sposób
uzupełniania braków. Znak osi jest umowny: odwrócenie wag i wyników składowej
nie zmienia wariancji ani geometrii. Przy równych wartościach własnych kierunki
w odpowiadającej im podprzestrzeni również nie są jednoznaczne. Rząd danych
po centrowaniu nie przekracza `n_obserwacji−1`; niektóre dopuszczalne składowe
mogą mieć zerową lub numerycznie znikomą wariancję. Pełny SVD może być kosztowny
dla bardzo szerokich zbiorów, a wykres dwóch osi nie pokazuje pozostałych wymiarów.

Przykład dla syntetycznych studentów: wybierz `wiek`, `godziny_nauki`,
`frekwencja`, `wynik_egzaminu`, dwie składowe, medianę i standaryzację.
PC1 wyjaśnia około **75,04%**, PC2 **22,40%**, razem **97,44%** wariancji;
tabela wynikowa ma 20 wierszy i dwie kolumny. Te wartości opisują tę konkretną
kopię i ustawienia. Zmień liczbę składowych na 1, aby zobaczyć brak wykresu 2D,
lub na 4, aby zachować pełną wariancję (z dokładnością numeryczną).

```python
from src.data_loader import load_csv
from src.dimensionality_reduction import run_pca

data = load_csv("data/sample_students.csv")
result = run_pca(
    data, ["wiek", "godziny_nauki", "frekwencja", "wynik_egzaminu"],
    n_components=2, imputation="median", scaling="standard",
)
print(result.variance)
print(result.components)
print(result.scores.head())
```

`PCAResult` udostępnia dopasowany `pipeline`, `scores`, `components`, `variance`
i `constant_features`. Pozwala to w przyszłości wywoływać PCA z modułu LLM,
porównywać K-Means na cechach pierwotnych i składowych oraz odtworzyć analizę w R.
Przy nowych obserwacjach należy wybrać te same cechy w tej samej kolejności i
zamienić nieskończoności na braki przez istniejące `numeric_selection` przed
`result.pipeline.transform(...)`. Do porównań z R trzeba uzgodnić imputację,
skalowanie (`ddof=0`), centrowanie, wybielanie, liczbę składowych i tolerancję;
porównywać wariancję oraz osie z uwzględnieniem umownych znaków i degeneracji.
Ten etap nie dodaje automatycznego porównania K-Means ani integracji R/LLM.

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
│   ├── dimensionality_reduction.py
│   ├── advanced_ui.py
│   └── visualizations.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_analysis.py
│   ├── test_visualizations.py
│   ├── test_data_preparation.py
│   ├── test_ml_models.py
│   ├── test_advanced_app.py
│   ├── test_dimensionality_reduction.py
│   ├── test_pca_visualizations.py
│   ├── test_pca_app.py
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

Weryfikacja etapu PCA (28.10.2026): `python -m pytest -q -W error` —
**238 zaliczonych testów** (166 dotychczasowych i 72 nowe), zero błędów i testów
pominiętych. Testy PCA obejmują znane kierunki i wartości własne, zgodność
projekcji z wagami, wariancję częściową i pełną, imputację, skalowanie, dane
stałe/puste/nieskończone, błędne parametry, powtarzalność, niezmienność oryginału,
wartości na wykresach oraz unieważnianie wyników i wariant jednej składowej w AppTest.
Na syntetycznych cechach `x` i `2*x` PC1 wyjaśnia 100% wariancji (z dokładnością
numeryczną); na studentach dwie składowe wyjaśniają 97,44%, cztery — 100%.
Serwer `app.py` zwrócił HTTP 200 dla HTML i `ok` dla `/_stcore/health`, po czym
został zatrzymany. `pip check` i `git diff --check` zakończyły się poprawnie.
PCA korzysta z już wymaganej biblioteki scikit-learn; nie dodano zależności.

Weryfikacja etapu Advanced Data Analysis (28.10.2026, Python 3.12.14, pandas 2.3.3,
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

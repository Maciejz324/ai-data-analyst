# AI Data Analyst

Projekt akademicki na przedmiot **Prompt Engineering**, temat 16:
**Konwersacyjny asystent analizy danych**. Docelowo użytkownik prześle CSV
i zada pytania po polsku, a LLM zaproponuje plan analizy oraz objaśni wyniki.
Obecna wersja to działające MVP do samodzielnej eksploracji danych,
bez integracji LLM i bez symulowania odpowiedzi AI.

## Technologie

Python 3.12, Streamlit (interfejs), pandas i NumPy (analiza), Plotly
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
Otwórz http://localhost:8501. W chmurze udostępnij port 8501 mechanizmem
podglądu/forwardowania portów swojego środowiska.

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

## Interpretacja i ograniczenia MVP

Analizy obejmują cały zbiór, niezależnie od liczby wierszy w podglądzie.
Braki są pomijane, bez imputacji. Puste pola, same białe znaki i standardowe
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
Nie obsługujemy XLSX, archiwów, UTF-16, automatycznej konwersji dat ani edycji danych.
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
│   └── visualizations.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_analysis.py
│   ├── test_visualizations.py
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

Moduły `src` działają niezależnie od Streamlit. W kolejnym etapie warto dodać:

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

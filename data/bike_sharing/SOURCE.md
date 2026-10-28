# Bike Sharing Dataset — pochodzenie i słownik

- Oficjalna strona: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Oficjalne archiwum: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- Wykorzystujemy **hour.csv**, nie day.csv ani kopię z innego portalu.
- Atrybucja: Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository.
  DOI: https://doi.org/10.24432/C5W894.
- Licencja: **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/.
  Przy dalszym udostępnianiu należy zachować atrybucję, link do licencji i wskazać zmiany.
  Oryginalny hour.csv nie jest zmieniany; tabele i wykresy są analizami pochodnymi.
- Dane opisują godzinowe wypożyczenia Capital Bikeshare w latach 2011–2012.
- Publikacja wymagana przez oryginalne Readme: Fanaee-T, H. i Gama, J. (2013),
  *Event labeling combining ensemble detectors and background knowledge*,
  https://doi.org/10.1007/s13748-013-0040-3.

Skrypt pobierania zachowuje bajty hour.csv, source_archive.zip i dokumentację
Readme_UCI.txt, jeśli znajduje się w archiwum. Manifest zawiera adres pobrania,
SHA-256 archiwum i CSV oraz rzeczywiście odczytaną liczbę wierszy. SHA-256 służy
kontroli późniejszych zmian; pierwsze pochodzenie zapewnia oficjalny HTTPS z
włączoną weryfikacją TLS. Nie zastępuj pliku wygenerowanymi obserwacjami.

## Znaczenie kolumn

| Kolumna | Znaczenie | Rola w badaniu |
| --- | --- | --- |
| instant | Identyfikator rekordu | Kontrola unikalności, nigdy predyktor |
| dteday | Data kalendarzowa | Porządek czasowy i walidacja; nie jest cechą regresji |
| season | Kod sezonu 1–4 według dokumentacji UCI | Kategoria one-hot i EDA; na wykresach zachowano kody |
| yr | 0: 2011, 1: 2012 | Kategoria one-hot |
| mnth | Miesiąc 1–12 | Kategoria one-hot |
| hr | Godzina 0–23 | Kategoria one-hot i EDA |
| holiday | Znacznik święta 0/1 | Kategoria one-hot |
| weekday | Dzień tygodnia 0–6, 0=niedziela | Kategoria one-hot; zgodność weryfikowana z datą |
| workingday | 1: dzień poza weekendem i świętem | Kategoria one-hot |
| weathersit | Kategorie pogody 1–4: od lepszej widoczności do silnych opadów | Kategoria one-hot, nie liczba ciągła |
| temp | Temperatura w oryginalnej skali znormalizowanej UCI | Regresja, K-Means, PCA |
| atemp | Temperatura odczuwalna w skali znormalizowanej UCI | EDA/korelacje; pominięta w modelach, aby nie powielać temp |
| hum | Wilgotność w skali znormalizowanej UCI | Regresja, K-Means, PCA |
| windspeed | Prędkość wiatru w skali znormalizowanej UCI | Regresja, K-Means, PCA |
| casual | Wypożyczenia użytkowników niezarejestrowanych | Składnik cnt; zakaz użycia jako predyktor i w klastrach |
| registered | Wypożyczenia użytkowników zarejestrowanych | Składnik cnt; zakaz użycia jako predyktor i w klastrach |
| cnt | Łączna liczba wypożyczeń, casual+registered | Cel regresji; opis klastrów dopiero po ich utworzeniu |

Zachowujemy oryginalne wartości i kody UCI. Readme.txt określa skalowanie jako
dzielenie temp przez 41, atemp przez 50, hum przez 100 i windspeed przez 67.
Analiza używa wartości z CSV, bez dodatkowego przeliczania na jednostki fizyczne.
Preprocessing standaryzuje wybrane cechy dopiero w odpowiednim pipeline.
Szczegółowe opisy kategorii należy czytać razem z oryginalnym Readme_UCI.txt.

## Weryfikacja pobranego oryginału

Pobrano oficjalne archiwum przez HTTPS z weryfikacją TLS. Zachowany CSV ma
1 156 736 bajtów i jest bajtowo identyczny z `hour.csv` w archiwum. CRC ZIP-a
jest poprawne, a SHA-256 CSV wynosi:
`e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.
Manifest zawiera także SHA-256 archiwum i adres źródłowy.

Odczytano **17 379 wierszy, 17 kolumn i zero pustych pól**; liczba rekordów
zgadza się z oryginalnym Readme.txt. Sprawdzono unikalność godzin i identyfikatorów,
zgodność kodów kalendarza oraz cnt=casual+registered dla wszystkich rekordów.
Zakres to 2011-01-01 00:00–2012-12-31 23:00, z 165 nieobecnymi godzinami.
Oficjalna strona UCI zwróciła HTTP 200 i zawiera odnośnik do CC BY 4.0.
Dokumentację z archiwum zachowano bez zmian w Readme_UCI.txt.
W Git wersjonujemy CSV, ten opis, Readme_UCI.txt i manifest. Redundantny ZIP
pozostaje lokalnie jako plik ignorowany; nie jest potrzebny do ponownego
uruchomienia eksperymentu na zachowanym i zweryfikowanym CSV.
`.gitattributes` wyłącza normalizację tekstu i kontrolę odstępów wyłącznie
dla dwóch oryginalnych plików UCI: CSV i Readme_UCI.txt. Zachowuje to ich
oryginalne zakończenia CRLF i bajty, sprawdzane przez SHA-256.

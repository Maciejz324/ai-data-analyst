# Metodologia

## Pytania badawcze

1. Jak rozkłada się godzinowy popyt i z jakimi porami dnia, dniami tygodnia,
   kodami sezonu oraz warunkami pogodowymi współwystępuje?
2. Czy addytywna regresja liniowa przewiduje późniejsze cnt lepiej od średniej
   cnt z wcześniejszego treningu?
3. Jakie grupy warunków pogodowych wyznacza K-Means dla k od 2 do 6?
4. Ile wariancji pogody zachowują dwie składowe PCA i jak zmieniają się podziały?

## Dane i EDA

Źródło to wyłącznie oficjalne UCI 275, plik hour.csv. Bajty są przechowywane
bez zmian, a analiza pracuje na kopii. Sprawdzamy schemat 17 kolumn, typy,
zakresy kodów i pogody, unikalność instant i znaczników czasu, zgodność kodów
kalendarza z datą oraz tożsamość cnt=casual+registered. Nie dopisujemy brakujących
godzin i nie utożsamiamy ich z zerową liczbą wypożyczeń.

Liczba i procent pustych pól są obliczane. Nie zakładamy braków; jeśli ich
nie ma, raport to stwierdza. Kalendarz i cel muszą być poprawne, brakujące
wartości pogody można imputować. Nieskończoności i wartości poza zakresem
znormalizowanych zmiennych UCI są błędami tego konkretnego zbioru.

EDA obejmuje statystyki, kwantyle i histogram cnt, średnie/mediany/liczebności
według godziny, dnia tygodnia, sezonu i kategorii pogody, rozrzuty cnt względem
pogody i korelacje Pearsona pogody oraz cnt. Kody sezonu i pogody pozostają
kodami UCI; identyfikatorów i składników cnt nie używamy w macierzy korelacji.
Przedziały IQR = [Q1−1,5·IQR, Q3+1,5·IQR] oznaczają odstające, ale ich nie
usuwają. Duży poprawny popyt nie musi oznaczać błędu pomiaru.

## Regresja i baseline

- Najpierw sortujemy po dteday+hr. Ostatnie 20% dostępnych godzin stanowi test,
  pierwsze 80% trening; dokładne granice i liczebności są w metadanych.
- Cel: cnt. **Zakazane cechy: cnt, casual, registered, instant, surowe dteday.**
  casual i registered bezpośrednio odtwarzałyby cel. instant jest identyfikatorem.
- Ciągłe predyktory: temp, hum, windspeed. atemp pominięto, aby nie powielać
  temperatury odczuwalnej i mierzonej w tym podstawowym modelu.
- Kategorie: season, yr, mnth, hr, holiday, weekday, workingday, weathersit.
  OneHotEncoder(handle_unknown="ignore", sparse_output=False) nie traktuje
  arbitralnych kodów jako odległości liczbowych. Nie używamy drop="first",
  więc nieznane przyszłe kategorie są bezpiecznie kodowane jako zera.
- Pipeline dopasowuje medianę, StandardScaler i kodowanie kategorii tylko na
  treningu. LinearRegression korzysta wyłącznie z wybranych predyktorów.
  Współliniowe wskaźniki kalendarza ograniczają interpretację współczynników;
  celem tej części jest ocena predykcji, nie testy przyczynowe współczynników.
- Baseline: DummyRegressor(strategy="mean"), dopasowany do cnt treningowego.
- MAE, RMSE, R² dotyczą identycznego testu. Ujemne predykcje regresji są
  raportowane, bez obcinania. Błędy to rzeczywiste−przewidywane; dodatni znak
  oznacza niedoszacowanie. Analizujemy histogram, przebieg czasowy oraz MAE,
  RMSE i bias według godziny, dnia tygodnia, sezonu i pogody.
- Jest to pojedyncza ocena czasowa ex post, z pogodą zaobserwowaną w godzinie
  prognozy. Nie jest to system operacyjny przewidujący również przyszłą pogodę.
  Nie stosujemy losowego podziału, tuningu na teście ani walidacji kroczącej.

## Grupowanie pogody

Domyślne cechy: temp, hum, windspeed. Nie wykorzystujemy cnt ani jego składników,
również przy wyborze k. Cały zbiór jest analizowany nienadzorowanie: mediana,
StandardScaler(ddof=0), potem K-Means. Kandydaci k=2…6, k-means++, n_init=10,
max_iter=300, tol=1e-4, algorithm=lloyd, random_state=42. Limit jednego wątku
ogranicza nadmierne równoległe obliczenia i ułatwia powtarzalność.

Inertia dotyczy wszystkich obserwacji. Silhouette używa tej samej próby do 2000
pozycji wybranych bez zwracania przez np.random.default_rng(42), dla wszystkich
k i przestrzeni. Pozycje są zapisane. Jeśli próba nie pozwala wyznaczyć metryki,
wynik jest pusty; jeśli żaden kandydat nie ma wyniku, analiza kończy się błędem.
Wybieramy najwyższe silhouette wśród kandydatów, a przy remisie mniejsze k.
Nie zakładamy k=3 ani nie ogłaszamy globalnej optymalności.

Po ustaleniu etykiet opisujemy liczebność i statystyki pogody w klastrach.
Dopiero wtedy dołączamy cnt i porównujemy jego rozkłady, bez zmieniania modelu
na podstawie popytu. Numery klastrów są umowne, różnice popytu mogą być
powiązane z sezonem i kalendarzem.

## PCA i porównanie K-Means

PCA używa tych samych trzech imputowanych i standaryzowanych cech pogodowych,
svd_solver="full", whiten=False. Pełne trzy składowe pozwalają pokazać całkowitą
i skumulowaną wariancję, a pierwsze dwie dają projekcję i zredukowane wejście
K-Means. Współczynniki są kierunkami w standaryzowanej przestrzeni, nie procentami;
znaki osi są umowne. Nie dopasowujemy tego PCA przed oceną regresji na teście.

**Nie standaryzujemy ponownie PC1/PC2**: zmieniłoby to względną wagę zachowanej
wariancji. Dla każdego k=2…6 wyznaczamy K-Means w oryginalnej przestrzeni
standaryzowanej i na dwóch PC. W każdej przestrzeni odrębnie obliczamy inertia
i silhouette oraz roboczo wybieramy k według tej samej reguły.

Inertia i silhouette zależą od geometrii, więc wartości z różnych przestrzeni
nie dowodzą bezpośrednio, że jeden podział jest lepszy. Dodatkowo porównujemy
podziały przy **tym samym k** przez ARI (odporny na zamianę numerów klastrów)
i obliczamy silhouette etykiet PCA w oryginalnej przestrzeni standaryzowanej.
To ostatnie daje porównanie odległości w jednej wspólnej geometrii.

## Ograniczenia interpretacji

Jeden system, dwa lata, jeden podział czasowy, zależność kolejnych godzin,
brak kontroli czynników zakłócających i brak walidacji kroczącej ograniczają
wnioski. Korelacja i różnice grup nie dowodzą przyczynowości. Model liniowy
nie uwzględnia automatycznie interakcji i nieliniowości. K-Means preferuje zwarte
grupy i jest wrażliwy na odstające. Silhouette na próbie jest przybliżeniem.
PCA może tracić informację użyteczną dla celu nawet przy wysokiej zachowanej wariancji.

Wyniki rzeczywistego zbioru są w REPORT.md i results/; pochodzenie oryginalnego
hour.csv potwierdzają manifest SHA-256 i porównanie z członkiem oficjalnego ZIP-a.
Wartości z fixture'ów jednostkowych nie są wynikami tego badania.

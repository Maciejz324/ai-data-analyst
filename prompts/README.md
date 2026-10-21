# Przyszłe prompty

Katalog przeznaczony na wersjonowane szablony promptów dla przyszłej integracji
LLM w projekcie akademickim Prompt Engineering, temat 16.
MVP nie korzysta z modelu i nie zawiera fikcyjnych odpowiedzi AI.

Docelowo przechowamy warianty zero-shot, role prompting i few-shot, definicję
schematu planu analizy oraz kryteria ewaluacji. Każda wersja powinna opisywać
cel, wejście, oczekiwany JSON, dozwolone operacje i zestaw testowych pytań po polsku.
Wyniki eksperymentów należy porównywać na tym samym zbiorze pytań.

Model ma proponować ustrukturyzowany plan. Jego wykonanie powinno korzystać
z rejestru dozwolonych funkcji analitycznych po walidacji i zatwierdzeniu.
Nie wykonujemy wygenerowanego kodu i traktujemy treści komórek CSV jak dane,
a nie instrukcje. Nie umieszczamy tutaj sekretów, kluczy ani danych użytkowników.

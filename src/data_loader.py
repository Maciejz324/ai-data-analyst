"""Odczyt lokalnego CSV bez zapisywania plików użytkownika na dysku."""

import csv
import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

MAX_FILE_SIZE = 10 * 1024 * 1024
ENCODINGS = ("utf-8-sig", "cp1250", "cp1252")
SEPARATORS = (",", ";", "\t")


class DataLoadError(ValueError):
    """Błąd danych wejściowych, który można pokazać użytkownikowi."""


def load_csv(
    source: bytes | BinaryIO | str | Path,
    *,
    separator: str | None = None,
    encoding: str | None = None,
    decimal: str = ".",
    max_bytes: int = MAX_FILE_SIZE,
) -> pd.DataFrame:
    """Wczytaj CSV z nagłówkiem i jednakową liczbą pól w rekordach.

    Automatyczne kodowania: UTF-8 (także BOM), CP1250, CP1252.
    Ścieżki są lokalne; adresy URL nie są pobierane. Strumienie są binarne.
    W przypadku strumienia pozycja odczytu zostaje przywrócona.
    """
    if separator is not None and separator not in SEPARATORS:
        raise DataLoadError("Wybierz separator: przecinek, średnik lub tabulator.")
    if encoding is not None and encoding not in ENCODINGS:
        raise DataLoadError("Nieobsługiwane kodowanie pliku.")
    if decimal not in (".", ","):
        raise DataLoadError("Separatorem dziesiętnym może być kropka lub przecinek.")
    if max_bytes <= 0:
        raise DataLoadError("Limit rozmiaru pliku musi być dodatni.")
    try:
        if isinstance(source, bytes):
            raw = source
        elif isinstance(source, (str, Path)):
            with Path(source).open("rb") as stream:
                raw = stream.read(max_bytes + 1)
        else:
            position = source.tell()
            try:
                source.seek(0)
                raw = source.read(max_bytes + 1)
            finally:
                source.seek(position)
    except (OSError, AttributeError, ValueError) as exc:
        raise DataLoadError("Nie można odczytać pliku CSV.") from exc
    if not isinstance(raw, bytes):
        raise DataLoadError("Prześlij plik CSV jako dane binarne.")
    if len(raw) > max_bytes:
        raise DataLoadError(f"Plik przekracza limit {max_bytes / 1024 / 1024:g} MiB.")

    text = None
    selected_encoding = None
    for candidate in (encoding,) if encoding else ENCODINGS:
        try:
            text = raw.decode(candidate)
            selected_encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DataLoadError("Nie można rozpoznać kodowania. Wybierz je ręcznie.")
    if not text.strip():
        raise DataLoadError("Plik CSV jest pusty.")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        raise DataLoadError("Plik zawiera dane binarne lub niedozwolone znaki sterujące.")

    if separator is None:
        # Nagłówek pozwala rozpoznać separator również w uszkodzonym pliku,
        # którego Sniffer nie rozpoznaje z powodu nierównych rekordów.
        widths = {}
        try:
            for candidate in SEPARATORS:
                reader = csv.reader(io.StringIO(text), delimiter=candidate, strict=True, skipinitialspace=True)
                header = next(row for row in reader if row and any(v.strip() for v in row))
                widths[candidate] = len(header)
        except (csv.Error, StopIteration) as exc:
            raise DataLoadError("Niepoprawna składnia CSV lub brak nagłówka.") from exc
        separator = max(widths, key=widths.get)
        # Jednokolumnowy CSV jest poprawny; dla niejednoznacznych danych
        # użytkownik może wskazać separator ręcznie.

    try:
        reader = csv.reader(io.StringIO(text), delimiter=separator, strict=True, skipinitialspace=True)
        header = None
        record_count = 0
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if header is None:
                header = [name.strip() for name in row]
                if any(not name for name in header) or len(set(header)) != len(header):
                    raise DataLoadError("Nazwy kolumn muszą być niepuste i unikalne.")
            elif len(row) != len(header):
                raise DataLoadError(
                    f"Nierówna liczba pól w rekordzie kończącym się w linii {reader.line_num}. "
                    "Sprawdź separator i cudzysłowy."
                )
            else:
                record_count += 1
        if not record_count:
            raise DataLoadError("Plik musi zawierać nagłówek i co najmniej jeden wiersz danych.")
        df = pd.read_csv(
            io.StringIO(text), sep=separator, decimal=decimal,
            index_col=False, skipinitialspace=True,
        )
        df.columns = header
        # Pola zawierające jedynie białe znaki traktujemy jako braki.
        df = df.replace(r"^\s*$", pd.NA, regex=True)
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataLoadError("Niepoprawny CSV. Sprawdź separator i cudzysłowy.") from exc
    df.attrs.update(encoding=selected_encoding, separator=separator, size_bytes=len(raw))
    return df

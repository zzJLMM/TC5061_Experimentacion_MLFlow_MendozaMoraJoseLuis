"""Carga el corpus SMS Spam Collection sin alterar el archivo crudo."""

import csv
import io
from pathlib import Path
import zipfile

import pandas as pd

from src.config import (
    CORPUS_FILE_NAME,
    CORPUS_ZIP_NAME,
    INTERIM_DATA_DIR,
    LOADED_MESSAGES_NAME,
    RAW_DATA_DIR,
)

MESSAGE_COLUMNS = ["label", "message"]


def find_corpus_path(raw_directory: Path = RAW_DATA_DIR) -> Path:
    """Devuelve el TSV extraido o, si no esta, el zip original."""
    extracted_path = raw_directory / CORPUS_FILE_NAME
    if extracted_path.exists():
        return extracted_path

    archive_path = raw_directory / CORPUS_ZIP_NAME
    if archive_path.exists():
        return archive_path

    raise FileNotFoundError(
        "No esta SMSSpamCollection ni sms+spam+collection.zip "
        f"en {raw_directory}."
    )


def load_messages(corpus_path: Path | None = None) -> pd.DataFrame:
    """Lee etiqueta y texto. Una comilla dentro del SMS no parte la fila."""
    path = find_corpus_path() if corpus_path is None else corpus_path
    if path.suffix == ".zip":
        return _read_table(io.StringIO(_read_archive_text(path)))
    return _read_table(path)


def save_loaded_messages(
    messages: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Guarda la tabla cargada, antes de limpiar, en data/interim."""
    destination = output_path or (INTERIM_DATA_DIR / LOADED_MESSAGES_NAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    messages.to_csv(destination, index=False)
    return destination


def _read_archive_text(archive_path: Path) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        return archive.read(CORPUS_FILE_NAME).decode("latin-1")


def _read_table(source) -> pd.DataFrame:
    messages = pd.read_csv(
        source,
        sep="\t",
        header=None,
        names=MESSAGE_COLUMNS,
        encoding="latin-1" if not isinstance(source, io.StringIO) else None,
        quoting=csv.QUOTE_NONE,
    )
    return messages


def main() -> None:
    messages = load_messages()
    destination = save_loaded_messages(messages)
    print(f"Registros: {len(messages)}")
    print(messages["label"].value_counts().to_string())
    print(f"Tabla intermedia: {destination}")


if __name__ == "__main__":
    main()

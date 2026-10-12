"""Limpia los SMS y deja la tabla lista para entrenar."""

from pathlib import Path

import pandas as pd

from src.config import (
    EXPECTED_CLEAN_ROWS,
    PROCESSED_DATA_DIR,
    PROCESSED_MESSAGES_NAME,
)
from src.load_data import load_messages

VALID_LABELS = {"ham", "spam"}


def clean_messages(messages: pd.DataFrame) -> pd.DataFrame:
    """Recorta texto, normaliza etiquetas y quita duplicados.

    Los SMS muy largos se conservan: en el analisis exploratorio son
    mensajes legitimos y la longitud distingue ham de spam.
    """
    cleaned = messages.copy()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned["message"] = cleaned["message"].astype(str).str.strip()

    invalid_row = cleaned["message"].eq("") | ~cleaned["label"].isin(
        VALID_LABELS
    )
    cleaned = cleaned.loc[~invalid_row]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    _validate_messages(cleaned)
    return cleaned


def save_messages(
    messages: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Escribe la tabla limpia en data/processed."""
    destination = output_path or (PROCESSED_DATA_DIR / PROCESSED_MESSAGES_NAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    messages.to_csv(destination, index=False)
    return destination


def _validate_messages(messages: pd.DataFrame) -> None:
    if len(messages) != EXPECTED_CLEAN_ROWS:
        raise ValueError(
            f"Se esperaban {EXPECTED_CLEAN_ROWS} filas y hay {len(messages)}."
        )
    if messages["message"].eq("").any():
        raise ValueError("Hay mensajes vacios despues de limpiar.")
    labels = set(messages["label"].unique())
    if labels != VALID_LABELS:
        raise ValueError(f"Etiquetas inesperadas: {labels}.")


def main() -> None:
    cleaned = clean_messages(load_messages())
    destination = save_messages(cleaned)
    print(f"Filas limpias: {len(cleaned)}")
    print(cleaned["label"].value_counts().to_string())
    print(f"Tabla procesada: {destination}")


if __name__ == "__main__":
    main()

"""Скачивание и распаковка kindle_reviews.csv с Kaggle."""

import contextlib
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from scripts.settings import CSV_NAME, FORCE_DOWNLOAD, KAGGLE_DATASET, RAW_DATA_DIR

from .logging_config import setup_auto_logging
from .retry_utils import retry_with_backoff

ZIP_FILENAME = RAW_DATA_DIR / "kindle-reviews.zip"
CSV_PATH = RAW_DATA_DIR / CSV_NAME


def remove_leading_index_column(csv_path: Path = CSV_PATH) -> None:
    """Удаляет лишний первый столбец-индекс в CSV, если он присутствует."""
    df = pd.read_csv(str(csv_path))
    first_col = str(df.columns[0])

    # Удаляем index-колонку с распространёнными именами
    if first_col in ("", "Unnamed: 0", "_c0") or first_col.startswith("Unnamed"):
        df = df.iloc[:, 1:]  # Удаляем первую колонку
        df.to_csv(csv_path, index=False)
        log.info(f"Удалён индекс-столбец '{first_col}' из {csv_path}")


# Используем централизованное логирование
log = setup_auto_logging()


@retry_with_backoff(
    max_attempts=3, initial_delay=5.0, exceptions=(subprocess.CalledProcessError,)
)
def _download_with_retry() -> None:
    """Скачивание датасета с retry-логикой при ошибках сети."""
    log.info(f"Скачивание датасета '{KAGGLE_DATASET}'...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(RAW_DATA_DIR),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    log.info("Датасет успешно скачан")


def main() -> Path:
    """Основная процедура скачивания датасета Kaggle с логированием.

    Возвращает абсолютный путь к CSV.
    """
    if CSV_PATH.exists() and not FORCE_DOWNLOAD:
        log.warning(
            "%s уже существует в %s, пропуск скачивания. Для форсированного скачивания установите флаг FORCE_DOWNLOAD = True.",
            CSV_NAME,
            RAW_DATA_DIR,
        )
        log.info("Абсолютный путь к CSV: %s", str(CSV_PATH.resolve()))
        return CSV_PATH.resolve()

    # Гарантируем директорию
    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        log.error(
            "Нет прав на создание каталога '%s': %s. Проверьте mонтирование volume и права на запись.",
            str(RAW_DATA_DIR),
            e,
        )
        raise

    log.info("Скачивание датасета '%s' в %s...", KAGGLE_DATASET, str(RAW_DATA_DIR))
    try:
        _download_with_retry()
    except subprocess.CalledProcessError as e:
        log.error(
            "Ошибка при скачивании датасета через kaggle CLI: %s\nПроверьте наличие файла kaggle.json и доступ к API.\n"
            "Если его нет, получите токен на https://www.kaggle.com/settings и поместите ~/.kaggle/kaggle.json.",
            e,
        )
        raise

    log.info("Распаковывание архива %s...", str(ZIP_FILENAME))
    with ZipFile(str(ZIP_FILENAME), "r") as zip_ref:
        zip_ref.extract(CSV_NAME, str(RAW_DATA_DIR))

    with contextlib.suppress(FileNotFoundError):
        ZIP_FILENAME.unlink()

    # Удаление первого столбца (индекс)
    remove_leading_index_column()

    resolved = CSV_PATH.resolve()
    log.info("Готово. Абсолютный путь к CSV: %s", str(resolved))
    return resolved


if __name__ == "__main__":
    main()

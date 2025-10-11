"""Простой скрипт для скачивания и распаковки kindle_reviews.csv с удалением первого столбца, чтобы избежать предупреждений.

Модуль вызывает kaggle CLI через subprocess, скачивает zip-архив датасета
в папку RAW_DATA_DIR, извлекает файл CSV и удаляет zip.

Параметры задаются в `config.py` (FORCE_DOWNLOAD, KAGGLE_DATASET, CSV_NAME,
RAW_DATA_DIR, ZIP_FILENAME).
"""

import contextlib
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from scripts.settings import CSV_NAME, FORCE_DOWNLOAD, KAGGLE_DATASET, RAW_DATA_DIR

from .logging_config import setup_auto_logging

ZIP_FILENAME = RAW_DATA_DIR / "kindle-reviews.zip"
CSV_PATH = RAW_DATA_DIR / CSV_NAME


def remove_leading_index_column(csv_path: Path = CSV_PATH) -> None:
    """
    Удаляет лишний первый столбец-индекс в CSV.

    Функция проверяет имя первого столбца и, если он похож на индекс, удаляет этот столбец
    и перезаписывает CSV без индекса.

    Args:
        csv_path (str): путь к CSV-файлу (по умолчанию используется CSV_PATH из config.py).
    """
    df = pd.read_csv(str(csv_path))
    first_col = str(df.columns[0])
    if first_col in ("", "_c0"):
        df.drop(df.columns[0], axis=1).to_csv(csv_path, index=False)


# Используем централизованное логирование
log = setup_auto_logging()


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

    log.info("Скачиваание датасета '%s' в %s...", KAGGLE_DATASET, str(RAW_DATA_DIR))
    try:
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
        )
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

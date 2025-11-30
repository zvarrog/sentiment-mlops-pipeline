"""Скачивание и распаковка kindle_reviews.csv с Kaggle."""

import contextlib
import subprocess
from pathlib import Path
from zipfile import ZipFile

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scripts.config import (
    CSV_NAME,
    FORCE_DOWNLOAD,
    KAGGLE_DATASET,
    RAW_DATA_DIR,
)
from scripts.logging_config import get_logger

ZIP_FILENAME = RAW_DATA_DIR / "kindle-reviews.zip"
CSV_PATH = RAW_DATA_DIR / CSV_NAME

log = get_logger(__name__)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(subprocess.CalledProcessError),
)
def _download_with_retry() -> None:
    """Скачивание датасета с retry-логикой при ошибках сети."""
    log.info("Скачивание датасета '%s'...", KAGGLE_DATASET)
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


def main(force: bool = False) -> Path:
    """Скачивание датасета Kaggle. Возвращает абсолютный путь к CSV."""
    if force is False:
        force = bool(FORCE_DOWNLOAD)

    if CSV_PATH.exists() and not force:
        log.warning(
            "%s уже существует, пропуск. Используйте force=True для повторной загрузки",
            CSV_NAME,
        )
        log.info("Абсолютный путь к CSV: %s", str(CSV_PATH.resolve()))
        return CSV_PATH.resolve()

    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        log.error(
            "Нет прав на создание '%s': %s. Проверьте volume и права на запись.",
            str(RAW_DATA_DIR),
            e,
        )
        raise

    log.info("Скачивание датасета '%s' в %s...", KAGGLE_DATASET, str(RAW_DATA_DIR))
    try:
        _download_with_retry()
    except subprocess.CalledProcessError as e:
        log.error(
            "Ошибка при скачивании датасета: %s\n"
            "Проверьте наличие файла kaggle.json и доступ к API.\n"
            "Если его нет, получите токен на https://www.kaggle.com/settings "
            "и поместите в ~/.kaggle/kaggle.json",
            e,
        )
        raise

    log.info("Распаковывание архива %s...", str(ZIP_FILENAME))
    try:
        with ZipFile(str(ZIP_FILENAME), "r") as zip_ref:
            zip_ref.extract(CSV_NAME, str(RAW_DATA_DIR))
    except (OSError, KeyError) as e:
        log.error("Ошибка при распаковке архива: %s", e)
        raise

    with contextlib.suppress(FileNotFoundError):
        ZIP_FILENAME.unlink()

    resolved = CSV_PATH.resolve()
    log.info("Готово. Абсолютный путь к CSV: %s", str(resolved))
    return resolved


if __name__ == "__main__":
    main()

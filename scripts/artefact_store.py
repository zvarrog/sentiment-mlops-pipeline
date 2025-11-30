"""Унифицированный интерфейс для работы с артефактами.

Решает проблемы:
* Разные способы сохранения JSON/CSV/Parquet
* Отсутствие atomic writes
* Сложность тестирования I/O операций

Упрощённая версия без Protocol: достаточно конкретной реализации
`LocalArtifactStore`, так как в проекте нет альтернативных хранилищ
(S3, GCS и т.п.) и не требуется структурная проверка типов.
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from scripts.logging_config import get_logger

log = get_logger(__name__)


class LocalArtifactStore:
    """Локальное файловое хранилище с atomic writes."""

    def save_json(self, path: Path, data: dict[str, Any], indent: int = 2, **kwargs: Any) -> None:
        """Атомарная запись JSON через временный файл."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent, **kwargs)
            os.replace(temp_path, path_obj)
            log.debug("JSON артефакт сохранен: %s", path_obj)
        except (OSError, TypeError, ValueError) as e:
            if temp_path.exists():
                temp_path.unlink()
            log.error("Ошибка сохранения JSON в %s: %s", path_obj, e)
            raise

    def load_json(self, path: Path) -> dict[str, Any]:
        """Загружает JSON с обработкой ошибок."""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"JSON файл не найден: {path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Невалидный JSON в {path}: {e}") from e
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Ошибка загрузки JSON из {path}: {e}") from e

    def save_csv(self, path: Path, df: pd.DataFrame, index: bool = False, **kwargs: Any) -> None:
        """Атомарная запись CSV."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")

        try:
            df.to_csv(temp_path, index=index, **kwargs)
            os.replace(temp_path, path_obj)
            log.debug("CSV артефакт сохранен: %s", path_obj)
        except (OSError, ValueError) as e:
            if temp_path.exists():
                temp_path.unlink()
            log.error("Ошибка сохранения CSV в %s: %s", path_obj, e)
            raise

    def save_parquet(self, path: Path, df: pd.DataFrame, **kwargs: Any) -> None:
        """Атомарная запись Parquet."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")

        try:
            df.to_parquet(temp_path, **kwargs)
            os.replace(temp_path, path_obj)
            log.debug("Parquet артефакт сохранен: %s", path_obj)
        except (OSError, ValueError) as e:
            if temp_path.exists():
                temp_path.unlink()
            log.error("Ошибка сохранения Parquet в %s: %s", path_obj, e)
            raise

    def save_text(self, path: Path, content: str) -> None:
        """Атомарная запись текстового файла."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, path_obj)
            log.debug("Текстовый артефакт сохранен: %s", path_obj)
        except OSError as e:
            if temp_path.exists():
                temp_path.unlink()
            log.error("Ошибка сохранения текста в %s: %s", path_obj, e)
            raise

    def load_text(self, path: Path) -> str:
        """Загружает текстовый файл."""
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Текстовый файл не найден: {path}") from e
        except OSError as e:
            raise RuntimeError(f"Ошибка загрузки текста из {path}: {e}") from e

    def save_model(self, path: Path, model: Any) -> None:
        """Сохраняет модель с использованием joblib."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(model, path_obj)
            log.info("Модель сохранена: %s", path_obj)
        except (OSError, ValueError, TypeError) as e:
            log.error("Ошибка сохранения модели в %s: %s", path_obj, e)
            raise

    def load_model(self, path: Path) -> Any:
        """Загружает модель с обработкой ошибок."""
        try:
            model = joblib.load(path)
            log.info("Модель загружена: %s", path)
            return model
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Модель не найдена: {path}") from e
        except (OSError, ValueError, EOFError) as e:
            raise RuntimeError(f"Ошибка загрузки модели из {path}: {e}") from e


# Глобальный экземпляр для использования во всём проекте
artefact_store = LocalArtifactStore()

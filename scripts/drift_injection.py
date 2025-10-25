"""Модуль инъекции синтетического дрейфа для тестирования системы мониторинга.

Этот модуль отвечает исключительно за инъекцию синтетического дрейфа в тестовые данные
для демонстрации работы системы мониторинга дрейфа. Вынесен из spark_process.py
для соблюдения принципа единственной ответственности.
"""

import shutil
from pathlib import Path

import pandas as pd

from scripts.config import INJECT_SYNTHETIC_DRIFT, PROCESSED_DATA_DIR
from scripts.logging_config import get_logger

log = get_logger("drift_injection")


def inject_synthetic_drift(
    test_data_path: Path | str | None = None,
) -> dict[str, str | list[str]]:
    """Инъектирует синтетический дрейф в тестовые данные.

    Args:
        test_data_path: Путь к тестовым данным. Если None, используется стандартный путь.

    Returns:
        dict: Результат операции с ключами 'status', 'message', 'changed_columns'
    """
    if not INJECT_SYNTHETIC_DRIFT:
        log.info("Инъекция дрейфа отключена: INJECT_SYNTHETIC_DRIFT=0")
        return {
            "status": "skipped",
            "message": "INJECT_SYNTHETIC_DRIFT=0",
            "changed_columns": [],
        }

    # Определяем путь к тестовым данным
    if test_data_path is None:
        test_path = Path(PROCESSED_DATA_DIR) / "test.parquet"
    else:
        test_path = Path(test_data_path)

    if not test_path.exists():
        error_msg = f"Тестовые данные не найдены: {test_path}"
        log.error(error_msg)
        return {"status": "error", "message": error_msg, "changed_columns": []}

    try:
        # Загружаем тестовые данные (поддерживаем как файлы, так и директории parquet)
        df = _load_parquet_data(test_path)

        # Применяем синтетический дрейф к числовым колонкам
        changed_columns = _apply_synthetic_drift(df)

        if changed_columns:
            # Сохраняем модифицированные данные
            _save_modified_data(df, test_path)

            success_msg = (
                f"Синтетический дрейф применён к колонкам: {', '.join(changed_columns)}"
            )
            log.warning(success_msg)  # WARNING, чтобы выделить в логах
            return {
                "status": "success",
                "message": success_msg,
                "changed_columns": changed_columns,
            }
        else:
            warning_msg = "Подходящих числовых колонок для инъекции дрейфа не найдено"
            log.warning(warning_msg)
            return {
                "status": "no_changes",
                "message": warning_msg,
                "changed_columns": [],
            }

    except Exception as e:
        error_msg = f"Ошибка при инъекции синтетического дрейфа: {e}"
        log.error(error_msg)
        return {"status": "error", "message": error_msg, "changed_columns": []}


def _load_parquet_data(test_path: Path) -> pd.DataFrame:
    """Загружает данные из parquet файла или директории."""
    try:
        # Пробуем загрузить как dataset (поддерживает директории parquet)
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(test_path))
        table = dataset.to_table()
        return table.to_pandas()
    except Exception:
        # Фолбэк для обычных parquet файлов
        return pd.read_parquet(test_path)


def _apply_synthetic_drift(df: pd.DataFrame) -> list[str]:
    """Применяет синтетический дрейф к числовым колонкам.

    Args:
        df: DataFrame для модификации (изменяется in-place)

    Returns:
        list: Список изменённых колонок
    """
    # Список кандидатов для инъекции дрейфа (будут изменены только существующие)
    candidate_columns = [
        "text_len",
        "word_count",
        "kindle_freq",
        "sentiment",
        # При необходимости добавьте другие числовые фичи, используемые в обучении
    ]

    changed_columns = []

    for col in candidate_columns:
        if col in df.columns:
            try:
                # Преобразуем в числовой тип
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

                # Применяем трансформацию для создания дрейфа
                # Масштабируем и смещаем так, чтобы PSI превысил порог (>0.2)
                df[col] = df[col] * 1.5 + 10.0

                changed_columns.append(col)
                log.debug(f"Применён дрейф к колонке '{col}'")

            except Exception as e:
                log.warning(f"Не удалось применить дрейф к колонке '{col}': {e}")

    return changed_columns


def _save_modified_data(df: pd.DataFrame, original_path: Path) -> None:
    """Сохраняет модифицированные данные, заменяя оригинальный файл.

    Args:
        df: Модифицированный DataFrame
        original_path: Путь к оригинальному файлу/директории
    """
    tmp_path = original_path.parent / "test_drift_tmp.parquet"

    # Сохраняем во временный файл
    df.to_parquet(tmp_path, index=False)

    # Удаляем оригинал (может быть как файлом, так и директорией)
    try:
        if original_path.is_dir():
            shutil.rmtree(original_path)
        else:
            original_path.unlink()
    except Exception as e:
        log.warning(f"Не удалось удалить оригинальный файл: {e}")

    # Переименовываем временный файл
    tmp_path.rename(original_path)


def main() -> dict[str, str | list[str]]:
    """Главная функция для запуска инъекции дрейфа из командной строки или DAG."""
    log.info("Запуск инъекции синтетического дрейфа...")
    result = inject_synthetic_drift()
    log.info(f"Результат инъекции: {result['status']} - {result['message']}")
    return result


if __name__ == "__main__":
    main()

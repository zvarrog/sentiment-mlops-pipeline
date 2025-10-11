"""Быстрый тест валидации на реальных данных."""

import logging
import sys
from pathlib import Path

# Делаем доступным пакет scripts
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_validation import log_validation_results, validate_parquet_dataset

logger = logging.getLogger(__name__)


def test_real_data():
    logger.info("Тестирование валидации на реальных данных...")
    results = validate_parquet_dataset(Path("data/processed"))
    all_valid = log_validation_results(results)
    logger.info(f"\nОбщий результат: {'Успешно' if all_valid else 'Есть проблемы'}")
    return all_valid


if __name__ == "__main__":
    test_real_data()

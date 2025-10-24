# Tests

Набор unit-тестов для ключевых модулей sentiment-mlops-pipeline.

## Покрытие

### `test_core_modules.py`

- **TestDataLoading**: загрузка train/val/test splits из parquet
- **TestFeatureSpace**: DenseTransformer для преобразования sparse → dense
- **TestFeatureContract**: валидация входных данных через FeatureContract
- **TestDataValidation**: проверка схемы и качества данных
- **TestDriftMonitor**: вычисление PSI (Population Stability Index)
- **TestTrainModules**: smoke tests для SimpleMLP

### `test_api_service.py`

- **TestAPIServiceHealthCheck**: проверка `/` эндпоинта
- **TestAPISinglePrediction**: тесты `/predict` с различными входами
- **TestAPIBatchPrediction**: тесты `/batch_predict` с батчами
- **TestAPIMetrics**: проверка Prometheus `/metrics`

Примечание: тесты API service требуют запущенного приложения или mock окружения.

### `test_integration.py`

- **TestSparkProcessing**: проверка создания processed файлов
- **TestTrainPipeline**: проверка создания артефактов после обучения
- **TestDownloadModule**: проверка загрузки raw данных
- **TestDockerServices**: smoke tests для API, MLflow, Prometheus
- **TestAirflowDAG**: валидация структуры DAG и наличия задач

## Запуск

```bash
# Все unit тесты
pytest tests/test_core_modules.py -v

# Интеграционные тесты (требуют запущенных сервисов)
pytest tests/test_integration.py -v -m integration

# Все тесты кроме интеграционных
pytest tests/ -v -m "not integration"

# С покрытием
pytest tests/ --cov=scripts --cov-report=html

# Конкретный тест
pytest tests/test_core_modules.py::TestDriftMonitor::test_psi_identical_distributions_returns_zero -v
```

## Зависимости

Установка дополнительных пакетов для тестов:

```bash
pip install -r tests/requirements.txt
```

Содержимое `tests/requirements.txt`:

```
pytest>=8.0
pytest-cov>=4.0
fastapi[standard]>=0.100
```

## Fixtures

`conftest.py` содержит общие fixtures:

- `setup_test_environment`: настройка переменных окружения
- `sample_dataframe`: минимальный DataFrame для тестов
- `sample_text_corpus`: текстовый корпус для NLP
- `temp_artifact_dir`: временная директория для артефактов

## Расширение

Для добавления новых тестов:

1. Создайте `test_<module_name>.py` в директории `tests/`
2. Используйте fixtures из `conftest.py`
3. Следуйте паттерну: один класс на модуль, описательные имена тестов
4. Добавьте докстринг к каждому тесту с пояснением проверки

# Объединённый DAG: kindle_unified_pipeline

Единый параметризованный DAG, объединяющий три предыдущих режима работы:

- **standard**: базовое обучение с Optuna HPO
- **monitored**: + логирование метрик задач в PostgreSQL
- **parallel**: параллельное обучение моделей (logreg, rf, hist_gb) с выбором лучшей

## Структура пайплайна

```
download → validate_data → inject_drift → process → drift_monitor → [branch]
                                                                        ↓
                                                    ┌───────────────────┴──────────────────┐
                                                    │                                      │
                                              train_standard                  [train_logreg, train_rf, train_gb]
                                                                                            ↓
                                                                                      select_best
```

## Параметры DAG

Все параметры передаются через `dag_run.conf` или `dag.params`:

### Режим выполнения

- **execution_mode**: `"standard"` | `"monitored"` | `"parallel"` (default: `"standard"`)

### Флаги управления

- **force_download**: форсировать скачивание датасета (default: `False`)
- **force_process**: форсировать обработку Spark (default: `False`)
- **force_train**: форсировать обучение модели (default: `False`)
- **run_data_validation**: запускать валидацию данных (default: `True`)
- **inject_synthetic_drift**: инжектировать синтетический дрейф (default: `False`)
- **run_drift_monitor**: запускать мониторинг дрейфа (default: `False`)

### Пути

- **raw_data_dir**: путь к сырым данным (default: `"data/raw"`)
- **processed_data_dir**: путь к обработанным данным (default: `"data/processed"`)
- **model_dir**: корневая директория артефактов (default: `"artefacts"`)
- **model_artefacts_dir**: директория артефактов модели (default: `"artefacts/model_artefacts"`)
- **drift_artefacts_dir**: директория артефактов дрейфа (default: `"artefacts/drift_artefacts"`)

## Примеры запуска

### Standard mode (через UI)

```python
# dag_run.conf:
{
    "execution_mode": "standard",
    "force_train": true
}
```

### Monitored mode (через CLI)

```bash
airflow dags trigger kindle_unified_pipeline \
  --conf '{"execution_mode": "monitored", "run_drift_monitor": true}'
```

### Parallel mode

```python
# dag_run.conf:
{
    "execution_mode": "parallel",
    "force_process": true,
    "force_train": true
}
```

## Отличия режимов

| Функция                      | standard | monitored | parallel              |
| ---------------------------- | -------- | --------- | --------------------- |
| Optuna HPO                   | ✅       | ✅        | ✅ (на каждую модель) |
| Метрики задач в PostgreSQL   | ❌       | ✅        | ❌                    |
| Метрики моделей в PostgreSQL | ❌       | ✅        | ❌                    |
| Параллельное обучение        | ❌       | ❌        | ✅                    |
| Автовыбор лучшей модели      | ❌       | ❌        | ✅                    |

## Требования

### Для режима monitored

PostgreSQL connection `metrics_db` должен быть настроен в Airflow:

```sql
-- Создание таблиц метрик
CREATE TABLE IF NOT EXISTS task_metrics (
    dag_id VARCHAR(255),
    task_id VARCHAR(255),
    execution_date TIMESTAMP,
    duration_sec FLOAT,
    status VARCHAR(50),
    PRIMARY KEY (dag_id, task_id, execution_date)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(255),
    execution_date TIMESTAMP,
    model_name VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    split VARCHAR(20)
);
```

### Для всех режимов

- Airflow variable `OPTUNA_STORAGE` (опционально):
  ```bash
  airflow variables set OPTUNA_STORAGE "postgresql+psycopg2://admin:admin@postgres:5432/optuna"
  ```

## Миграция со старых DAG'ов

### Из kindle_reviews_pipeline → standard mode

```python
# Старый запуск
airflow dags trigger kindle_reviews_pipeline --conf '{"force_train": true}'

# Новый запуск
airflow dags trigger kindle_unified_pipeline --conf '{"execution_mode": "standard", "force_train": true}'
```

### Из kindle_reviews_monitored_pipeline → monitored mode

```python
# Новый запуск с теми же возможностями
airflow dags trigger kindle_unified_pipeline --conf '{"execution_mode": "monitored"}'
```

### Из kindle_reviews_parallel_pipeline → parallel mode

```python
# Новый запуск с параллельным обучением
airflow dags trigger kindle_unified_pipeline --conf '{"execution_mode": "parallel"}'
```

## Статус старых DAG'ов

Старые DAG'ы (`kindle_reviews_pipeline`, `kindle_reviews_monitored_pipeline`, `kindle_reviews_parallel_pipeline`) можно удалить или оставить для обратной совместимости. Новый объединённый DAG полностью покрывает их функциональность.

## Рекомендации

1. **Для разработки**: используйте `standard` режим
2. **Для production**: используйте `monitored` режим для отслеживания производительности
3. **Для экспериментов**: используйте `parallel` режим для быстрого сравнения моделей

## Troubleshooting

### DAG не виден в UI

```bash
# Проверка валидности DAG
python airflow/dags/kindle_unified_pipeline.py

# Обновление списка DAG'ов
airflow dags list
```

### Ошибка подключения к metrics_db

```bash
# Проверка connection
airflow connections get metrics_db

# Создание connection
airflow connections add metrics_db \
  --conn-type postgres \
  --conn-host postgres \
  --conn-login admin \
  --conn-password admin \
  --conn-port 5432 \
  --conn-schema metrics
```

### Parallel mode не запускается

Убедитесь, что в `execution_mode` передана строка `"parallel"`, а не другое значение.

# Новые Airflow DAG

## Обзор

В проекте реализовано три DAG для обучения моделей:

1. **kindle_pipeline.py** — базовый последовательный пайплайн
2. **kindle_pipeline_parallel.py** — параллельное обучение нескольких моделей
3. **kindle_pipeline_monitored.py** — пайплайн с мониторингом производительности

---

## 1. Parallel Pipeline — Параллельное обучение

### Архитектура параллельного пайплайна

```text
download
   ↓
[validate_schema, validate_quality] (параллельно)
   ↓
[train_logreg, train_rf, train_gb] (параллельно)
   ↓
select_best
```

### Особенности параллельного обучения

- **Параллельная валидация**: схема и качество данных проверяются одновременно
- **Параллельное обучение**: 3 модели обучаются независимо:
  - Логистическая регрессия (`logreg`)
  - Random Forest (`rf`)
  - Histogram Gradient Boosting (`hist_gb`)
- **Автоматический выбор**: лучшая модель выбирается по метрике `val_f1_macro`
- **Меньше trials**: каждая модель проходит до 10 trials (вместо полного OPTUNA_N_TRIALS) для ускорения

### Использование

```bash
# Через UI Airflow
http://localhost:8080 → DAGs → kindle_reviews_parallel_pipeline → Trigger

# Через CLI
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow dags trigger kindle_reviews_parallel_pipeline
```

### Результаты

После выполнения создаются файлы:

- `artefacts/model_artefacts/model_logreg.joblib` — модель LogReg
- `artefacts/model_artefacts/model_rf.joblib` — модель RF
- `artefacts/model_artefacts/model_hist_gb.joblib` — модель GradientBoosting
- `artefacts/best_model.joblib` — лучшая модель (скопирована из трёх)
- `artefacts/model_artefacts/best_model_meta.json` — метаданные лучшей модели

---

## 2. Monitored Pipeline — Мониторинг производительности

### Архитектура мониторинга

```text
download → validate → process → drift_monitor → train
```

Каждая задача логирует метрики в PostgreSQL БД `metrics`.

### Особенности мониторинга

- **Логирование длительности**: каждая задача записывает время выполнения в БД
- **Отслеживание статусов**: success/failed/skipped
- **Метрики моделей**: val_f1_macro, test_f1_macro, test_accuracy сохраняются в БД
- **Колбеки**: `on_success_callback` и `on_failure_callback` для каждой задачи

### Схема БД metrics

#### Таблица `task_metrics`

```sql
CREATE TABLE task_metrics (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    duration_sec NUMERIC(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'success',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_task_execution UNIQUE (dag_id, task_id, execution_date)
);
```

#### Таблица `model_metrics`

```sql
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC(10, 6) NOT NULL,
    split VARCHAR(50) NOT NULL, -- train/val/test
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Использование мониторинга

```bash
# Запуск
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow dags trigger kindle_reviews_monitored_pipeline

# Просмотр метрик
docker exec -it sentiment-mlops-pipeline-postgres-1 \
  psql -U admin -d metrics -c "SELECT * FROM task_metrics ORDER BY created_at DESC LIMIT 10;"
```

### Примеры SQL-запросов для аналитики

#### Средняя длительность задач по DAG

```sql
SELECT 
    dag_id,
    task_id,
    AVG(duration_sec) as avg_duration,
    MIN(duration_sec) as min_duration,
    MAX(duration_sec) as max_duration,
    COUNT(*) as runs
FROM task_metrics
WHERE status = 'success'
GROUP BY dag_id, task_id
ORDER BY avg_duration DESC;
```

#### Тренд производительности модели

```sql
SELECT 
    execution_date::date as date,
    model_name,
    metric_name,
    AVG(metric_value) as avg_value
FROM model_metrics
WHERE metric_name = 'val_f1_macro'
GROUP BY date, model_name, metric_name
ORDER BY date DESC;
```

#### Самые медленные запуски

```sql
SELECT 
    dag_id,
    execution_date,
    SUM(duration_sec) as total_duration
FROM task_metrics
GROUP BY dag_id, execution_date
ORDER BY total_duration DESC
LIMIT 10;
```

---

## Настройка Airflow Connection

Для работы monitored pipeline необходимо настроить connection к БД metrics:

### Через UI

1. Перейти в **Admin → Connections**
2. Добавить новое соединение:
   - **Connection Id**: `metrics_db`
   - **Connection Type**: `Postgres`
   - **Host**: `postgres`
   - **Schema**: `metrics`
   - **Login**: `admin`
   - **Password**: `admin`
   - **Port**: `5432`

### Через переменную окружения (уже настроено)

```yaml
environment:
  AIRFLOW_CONN_METRICS_DB: 'postgresql://admin:admin@postgres:5432/metrics'
```

---

## Сравнение DAG'ов

| Характеристика | Base | Parallel | Monitored |
|---|---|---|---|
| Валидация | Последовательно | Параллельно (2 задачи) | Последовательно |
| Обучение | 1 модель с Optuna | 3 модели параллельно | 1 модель с Optuna |
| Trials | ~50 | ~10 на модель | ~50 |
| Время выполнения | ~20-30 мин | ~15-20 мин | ~20-30 мин |
| Мониторинг | Нет | Нет | Да (БД metrics) |
| Лучший для | Production | Эксперименты | Анализ производительности |

---

## Рекомендации

1. **Для разработки**: используйте `kindle_pipeline_parallel.py` для быстрого сравнения моделей
2. **Для production**: используйте `kindle_pipeline_monitored.py` для отслеживания производительности
3. **Для экспериментов**: используйте базовый `kindle_pipeline.py` с настройкой SELECTED_MODEL_KINDS

---

## Устранение проблем

### Ошибка "Connection metrics_db not found"

```bash
# Пересоздать соединение через UI или переменную окружения
docker-compose down
docker-compose up -d
```

### Таблицы metrics не созданы

```bash
# Проверить инициализацию БД
docker exec -it sentiment-mlops-pipeline-postgres-1 \
  psql -U admin -d metrics -c "\dt"

# Если таблиц нет — пересоздать контейнер
docker-compose down -v
docker-compose up -d
```

### Медленное выполнение parallel pipeline

Это нормально — три модели обучаются одновременно. Для ускорения:

```python
# В scripts/settings.py
OPTUNA_N_TRIALS = 5  # Уменьшить количество trials
```

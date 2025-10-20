# Быстрый старт: Новые Airflow DAG

## Предварительные требования

```bash
# Остановить существующие контейнеры
docker-compose down -v

# Пересобрать для применения изменений
docker-compose build --no-cache

# Запустить систему
docker-compose up -d
```

## Запуск Parallel Pipeline

### 1. Через Airflow UI

```text
1. Открыть http://localhost:8080
2. Войти: admin/admin
3. Найти DAG "kindle_reviews_parallel_pipeline"
4. Нажать кнопку "Trigger DAG"
5. Следить за выполнением в Graph View
```

### 2. Через CLI

```bash
# Триггер DAG
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow dags trigger kindle_reviews_parallel_pipeline

# Проверить статус
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow dags list-runs -d kindle_reviews_parallel_pipeline
```

### 3. Ожидаемый результат

После успешного выполнения (~15-20 минут):

```bash
# Проверить созданные модели
ls -lh artefacts/model_artefacts/model_*.joblib

# Вывод:
# model_logreg.joblib    - Логистическая регрессия
# model_rf.joblib        - Random Forest
# model_hist_gb.joblib   - Gradient Boosting

# Лучшая модель
ls -lh artefacts/best_model.joblib

# Метаданные
cat artefacts/model_artefacts/best_model_meta.json
```

---

## Запуск Monitored Pipeline

### 1. Через Airflow UI

```text
1. Открыть http://localhost:8080
2. Войти: admin/admin
3. Найти DAG "kindle_reviews_monitored_pipeline"
4. Нажать кнопку "Trigger DAG"
5. Следить за выполнением в Graph View
```

### 2. Через CLI

```bash
# Триггер DAG
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow dags trigger kindle_reviews_monitored_pipeline

# Проверить статус
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow dags list-runs -d kindle_reviews_monitored_pipeline
```

### 3. Просмотр метрик

```bash
# Подключиться к БД metrics
docker exec -it sentiment-mlops-pipeline-postgres-1 \
  psql -U admin -d metrics

# Выполнить SQL-запросы
```

#### Последние 10 выполнений задач

```sql
SELECT
    task_id,
    execution_date,
    duration_sec,
    status
FROM task_metrics
ORDER BY created_at DESC
LIMIT 10;
```

#### Средняя длительность по задачам

```sql
SELECT
    task_id,
    ROUND(AVG(duration_sec)::numeric, 2) as avg_duration,
    COUNT(*) as runs
FROM task_metrics
WHERE status = 'success'
GROUP BY task_id
ORDER BY avg_duration DESC;
```

#### Метрики моделей

```sql
SELECT
    model_name,
    metric_name,
    ROUND(metric_value::numeric, 4) as value,
    split,
    execution_date
FROM model_metrics
ORDER BY execution_date DESC
LIMIT 20;
```

---

## Проверка работоспособности

### Тест импорта DAG'ов

```bash
# Запустить тесты
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  python -m pytest tests/test_new_dags.py -v
```

### Проверка БД metrics

```bash
# Список таблиц
docker exec -it sentiment-mlops-pipeline-postgres-1 \
  psql -U admin -d metrics -c "\dt"

# Должны быть:
# - task_metrics
# - model_metrics

# Проверка структуры
docker exec -it sentiment-mlops-pipeline-postgres-1 \
  psql -U admin -d metrics -c "\d task_metrics"
```

### Проверка соединений Airflow

```bash
# Список соединений
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow connections list

# Должно быть: metrics_db
```

---

## Устранение проблем

### Ошибка "database metrics does not exist"

```bash
# Пересоздать контейнеры
docker-compose down -v
docker-compose up -d

# Подождать инициализации (~30 сек)
docker logs -f sentiment-mlops-pipeline-postgres-1
```

### Ошибка "Connection metrics_db not found"

```bash
# Проверить переменные окружения
docker exec -it sentiment-mlops-pipeline-airflow-1 env | grep METRICS

# Должно быть:
# AIRFLOW_CONN_METRICS_DB=postgresql://admin:admin@postgres:5432/metrics
```

### DAG не отображается в UI

```bash
# Перезапустить scheduler
docker-compose restart airflow

# Проверить логи
docker logs -f sentiment-mlops-pipeline-airflow-1
```

### Медленное выполнение

Это нормально для первого запуска:

- **Parallel**: ~15-20 минут (3 модели × 10 trials каждая)
- **Monitored**: ~20-30 минут (1 модель × 50 trials)

Для ускорения тестирования:

```python
# В scripts/settings.py
OPTUNA_N_TRIALS = 5  # Вместо 50
```

---

## Дополнительная информация

Полная документация: `airflow/dags/README_NEW_DAGS.md`

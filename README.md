# Kindle Reviews: Система анализа тональности отзывов

MLOps-проект для предсказания рейтингов Kindle-отзывов на основе текста и дополнительных признаков. Включает полный пайплайн от сбора данных до продакшн API с мониторингом.

## 🏗️ Архитектура проекта

```text
├── scripts/                    # Основной код
│   ├── api_service.py         # FastAPI сервис
│   ├── train.py               # Обучение моделей с Optuna
│   ├── spark_process.py       # Обработка данных в Spark
│   ├── download.py            # Загрузка данных из Kaggle
│   ├── drift_monitor.py       # Мониторинг дрифта
│   └── models/                # Модели и утилиты
├── airflow/dags/              # DAG для оркестрации
├── tests/                     # Комплексное тестирование
├── data/                      # Данные (raw/processed)
├── artefacts/                 # Модели и метрики
└── postgres-init/             # Инициализация БД
```

## 🚀 Быстрый старт

### 1. Подготовка окружения

```bash
# Клонирование репозитория
git clone https://github.com/zvarrog/sentiment-mlops-pipeline
cd sentiment-mlops-pipeline

# Виртуальное окружение
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Kaggle API (для загрузки данных)
# Разместите ваш kaggle.json в корне проекта или поменяйте путь к нему в .env
```

### 2. Запуск через Docker Compose

```bash
# Полная инфраструктура (рекомендуется)
docker compose up -d

# Только API
docker compose up -d api postgres

# Только Airflow
docker compose up -d airflow postgres
```

- Состав: Postgres (Optuna storage), Airflow (оркестрация и обучение), API (инференс).
- MLflow используется в file‑режиме внутри Airflow контейнера (`/opt/airflow/mlruns`), вынесен на volume `mlruns_data`.
- UI для MLflow не поднимается по умолчанию, чтобы уменьшить потребление ресурсов. При необходимости можно добавить сервис `mlflow ui` или запустить UI локально, указав `MLFLOW_TRACKING_URI`.
- Обучение выполняется в контейнере Airflow, API — только для инференса (read‑only, монтируется `./artefacts`).

Примечание: в репозитории уже есть baseline‑модель и метаданные в `artefacts/` — API можно запускать сразу (без обучения). Подробности — в разделе «Артефакты модели» ниже.

## 📊 Основные компоненты

### ⚙️ Airflow DAG

В проекте реализовано три Airflow DAG:

1. **kindle_pipeline.py** — базовый последовательный пайплайн
2. **kindle_pipeline_parallel.py** — параллельное обучение нескольких моделей (15-20 мин)
3. **kindle_pipeline_monitored.py** — мониторинг производительности в PostgreSQL

**Запуск:**

```bash
# Через UI: http://localhost:8080 (admin/admin)
# Через CLI:
docker exec -it sentiment-mlops-pipeline-airflow-1 \
  airflow dags trigger kindle_reviews_parallel_pipeline
```

**Подробная документация:** `airflow/dags/README_NEW_DAGS.md`, `QUICK_START_NEW_DAGS.md`

### 🤖 Модели машинного обучения

Поддерживаемые алгоритмы:

- **Random Forest** — базовая модель с TF-IDF и числовыми признаками
- **Logistic Regression** — быстрая линейная модель
- **Histogram Gradient Boosting** — современный градиентный бустинг
- **MLP** — простая нейронная сеть
- **DistilBERT** — трансформер для глубокого понимания текста

### 🔧 Инженерия признаков

**Автоматически извлекаемые:**

- `text_len` — длина текста в символах
- `word_count` — количество слов
- `kindle_freq` — частота упоминания "Kindle"
- `sentiment` — тональность (TextBlob)

**Дополнительные (при наличии):**

- `user_avg_len`, `user_review_count` — профиль пользователя
- `item_avg_len`, `item_review_count` — профиль товара

Агрегации user/item (Spark):

- `user_avg_len`, `user_review_count` — средняя длина отзывов пользователя и их количество
- `item_avg_len`, `item_review_count` — средняя длина отзывов по товару и их количество
- Вычисляются в `scripts/spark_process.py` и сохраняются в parquet; затем могут быть переданы в API через `numeric_features`.

Текстовый конвейер: TF‑IDF → SVD

- Spark‑этап: `Tokenizer` → `CountVectorizer` (vocabSize=HASHING_TF_FEATURES, minDF=MIN_DF, minTF=MIN_TF) → `IDF` формирует столбец `tfidfFeatures` для анализа и агрегатов.
- Тренировка (sklearn): `TfidfVectorizer` (ngram_range=(1,2), max_features подбирается Optuna) → опциональный `TruncatedSVD`.
- Включение SVD управляется автоматически по памяти (`FORCE_SVD_THRESHOLD_MB`) или параметром `use_svd` (Optuna). Компоненты: `svd_components` (20..100, step=20).

### 🎯 Оптимизация гиперпараметров

- **Optuna** для автоматического подбора параметров
- **MLflow** для трекинга экспериментов
- **PostgreSQL** для хранения результатов исследований
- Ранняя остановка при отсутствии прогресса

## 🌐 API

### Эндпоинты

```bash
# Проверка здоровья
GET /health

# Метаданные модели
GET /metadata

# Предсказание для текстов
POST /predict
{
  "texts": ["Great product!", "Not satisfied"],
  "numeric_features": {"user_avg_len": [150, 200]}
}

# Пакетное предсказание
POST /batch_predict
{
  "data": [
    {"reviewText": "Amazing!", "user_review_count": 15},
    {"reviewText": "Poor quality", "item_avg_len": 120}
  ]
}
```

Агрегаты и numeric_features в API:

- API принимает заранее посчитанные числовые признаки в поле `numeric_features` (POST /predict) или в составе объектов (POST /batch_predict).
- Поддерживаются, например: `user_avg_len`, `user_review_count`, `item_avg_len`, `item_review_count`, а также базовые: `text_len`, `word_count`, `kindle_freq`, `sentiment`.
- Если часть признаков не передана, API рассчитает базовые текстовые признаки автоматически; агрегаты user/item — только если вы их передадите.
- См. пример и тест `tests/test_api_aggregates.py`.

### Документация

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Тестирование API

Для быстрого тестирования API используйте готовые скрипты:

```bash
# Простое предсказание
python scripts/request.py

# Пакетное предсказание с реальными данными
python scripts/request_batch.py
```

## 🔄 MLOps Pipeline

### Airflow DAG

Автоматизированный пайплайн включает:

1. **Загрузка данных** из Kaggle
2. **Валидация** входных данных
3. **Обработка** в Apache Spark
4. **Обучение** с гиперпараметрической оптимизацией
5. **Валидация модели** и сохранение артефактов
6. **Мониторинг дрифта** данных

```bash
# Запуск через Airflow UI
# http://localhost:8080
# admin/admin
```

### Локальный запуск полного пайплайна

```bash
# Полный пайплайн
python scripts/download.py
python scripts/spark_process.py
python scripts/train.py

# Мониторинг дрифта
python scripts/drift_monitor.py
```

Валидация данных:

- Базовая валидация csv/parquet выполняется в `scripts/data_validation.py` и в конце Spark‑пайплайна (контроль колонок/типов/диапазонов).
- Флаг `RUN_DATA_VALIDATION=1` (по умолчанию) включает проверку после сохранения parquet.
- Проверяются типы, обязательные колонки (включая Spark‑векторные), наборы значений и диапазоны.

Инъекция и мониторинг дрейфа:

- Инъекция: модуль `scripts/drift_injection.py` (флаг `INJECT_SYNTHETIC_DRIFT=1`) синтетически сдвигает числовые колонки тестовых данных (для демонстрации).
- Мониторинг: `scripts/drift_monitor.py` вычисляет PSI по числовым признакам, порог дрейфа по умолчанию 0.2; отчёты сохраняются в `artefacts/drift_artefacts/` (JSON/CSV и графики).
- В DAG (Airflow): шаги drift_injection → spark_processing → drift_monitoring позволяют отследить эффект инъекции.

## 🔍 Мониторинг и наблюдаемость

### 🎯 Новые возможности

#### Мониторинг в реальном времени

- **Prometheus** собирает метрики API (RPS, latency, errors)
- **Grafana** отображает дашборды с алертами
- Доступ: http://localhost:3000 (admin/admin)

#### Drift Alerting

- Автоматическая отправка в Slack при PSI > 0.2
- Настройка: установите `SLACK_WEBHOOK_URL` в `.env`

#### MLflow UI

- Просмотр всех экспериментов: http://localhost:5000
- Сравнение моделей, параметров и метрик
- Скачивание артефактов (confusion matrix, feature importances)

### 📊 Дашборды Grafana

**API Performance:**

- Request rate (по эндпоинтам)
- Response time (p50, p95, p99)
- Error rate
- Active predictions by model

**Model Health:**

- Drift PSI по фичам
- Prediction distribution
- Feature importance changes over time

**Infrastructure:**

- CPU/Memory usage
- Disk I/O
- Network traffic

### 🚨 Алертинг

Настроены автоматические алерты:

| Alert             | Condition                 | Notification                |
| ----------------- | ------------------------- | --------------------------- |
| High Error Rate   | Errors > 1% for 2min      | Slack + Email               |
| High Latency      | p95 > 500ms for 5min      | Slack                       |
| Significant Drift | PSI > 0.2 for any feature | Slack + retrain recommended |
| Low Disk Space    | Free space < 10%          | Email                       |

### Task Performance Monitoring

Мониторинг производительности задач Airflow в PostgreSQL:

```bash
# Просмотр метрик длительности задач
docker exec -it sentiment-mlops-pipeline-postgres-1 \
  psql -U admin -d metrics -c \
  "SELECT task_id, ROUND(AVG(duration_sec), 2) as avg_sec FROM task_metrics GROUP BY task_id;"
```

**Таблицы:**

- `task_metrics` — длительность и статус каждой задачи
- `model_metrics` — метрики моделей (F1, accuracy) по запускам

**Подробнее:** `IMPLEMENTATION_NEW_DAGS.md`

### Drift Detection

- Автоматическое обнаружение дрифта данных
- Сравнение распределений признаков
- Статистические тесты и визуализация
- Алерты при критических изменениях

### Логирование

Структурированное логирование с trace ID:

```python
from scripts.logging_config import setup_auto_logging
log = setup_auto_logging()
log.info("Модель обучена", extra={"accuracy": 0.85})
```

### MLflow Tracking

- Автоматическое логирование метрик и параметров
- Сравнение экспериментов
- Версионирование моделей

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
python -m pytest tests/ -v

# Smoke тесты API
python -m pytest tests/test_api_smoke.py -v

# Тесты обучения
python -m pytest tests/test_*training*.py -v

# E2E тесты
python -m pytest tests/test_e2e_smoke.py -v
```

### Типы тестов

- **Unit** — отдельные компоненты
- **Integration** — взаимодействие компонентов
- **E2E** — полный пайплайн
- **Performance** — производительность API
- **Data validation** — проверка качества данных

## 📈 Производительность

### Оптимизации

- **Векторизация** операций с NumPy/Pandas
- **Пайплайны** scikit-learn для эффективной обработки
- **Кэширование** промежуточных результатов
- **Параллелизм** в Optuna и Spark

### Метрики

- **Accuracy** — доля правильных предсказаний
- **F1-score** (macro/weighted) — сбалансированная метрика
- **Время обучения** — эффективность пайплайна
- **Время инференса** — скорость API

## 🔒 Безопасность

### Production checklist

- [ ] Enable HTTPS (Let's Encrypt)
- [ ] Set up API key authentication
- [ ] Configure firewall (allow only 80, 443, 22)
- [ ] Enable audit logging
- [ ] Rotate secrets monthly
- [ ] Set up VPN for internal services

### Secrets Management

**Development:** `.env` file (gitignored)

**Production:** Kubernetes secrets или Vault

```bash
kubectl create secret generic kindle-secrets \
  --from-env-file=.env.prod
```

### API Security Features

- **Rate Limiting:** 100 req/min per IP (slowapi)
- **Graceful Shutdown:** корректное завершение текущих запросов при остановке
- **Read-only container:** безопасность через tmpfs и drop capabilities

## ⚙️ Конфигурация

### Переменные окружения

```bash
# Данные
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
KAGGLE_DATASET=gunjanbaid/amazon-product-reviews

# Модели
MODEL_DIR=artefacts
OPTUNA_N_TRIALS=30
OPTUNA_STORAGE=postgresql://admin:admin@postgres:5432/optuna

# API
API_HOST=0.0.0.0
API_PORT=8000

# Spark
SPARK_DRIVER_MEMORY=4g
SPARK_EXECUTOR_MEMORY=4g
```

### Настройка в scripts/settings.py

Централизованное управление всеми параметрами проекта.

## 🐳 Docker

### Образы

```bash
# API сервис
docker build -f Dockerfile.api -t kindle-api .

# Airflow с зависимостями
docker build -f Dockerfile.airflow -t kindle-airflow .
```

### Конфигурация

```yaml
# docker-compose.yml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - '8000:8000'
    volumes:
      - ./artefacts:/app/artefacts

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin
```

## 📊 Данные

### Источник

**Amazon Kindle Reviews** с Kaggle:

- 982,619 отзывов
- Рейтинги 1-5 звёзд
- Текст отзывов
- Метаданные пользователей и товаров

### Обработка

1. **Очистка** — удаление дубликатов и невалидных записей
2. **Балансировка** — лимитирование на класс (35K записей)
3. **Разделение** — train/val/test (70/15/15%)
4. **Признаки** — извлечение текстовых и числовых фичей

### Артефакты модели

Для запуска API необходимы сохранённые артефакты модели. В репозитории уже включён baseline‑набор файлов (см. пути ниже), поэтому можно стартовать без обучения. Минимальный набор:

- `artefacts/best_model.joblib` — сериализованный пайплайн модели
- `artefacts/model_artefacts/best_model_meta.json` — метаданные о лучшей модели и метрики
- `artefacts/model_artefacts/baseline_numeric_stats.json` — статистики числовых признаков для валидации

Как обновить артефакты:

1. Обучить модель локально: `python scripts/train.py` — файлы будут сгенерированы автоматически в директории из `scripts/settings.py`.
2. Скопировать свои артефакты в указанные пути (например, из релиза/артефактов CI).
3. При использовании Docker убедитесь, что `./artefacts` на хосте смонтирована в контейнеры API/Airflow.

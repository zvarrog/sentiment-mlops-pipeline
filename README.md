# Sentiment MLOps Pipeline

Production-ready MLOps пайплайн для sentiment analysis отзывов на Kindle книги. Реализует полный цикл: загрузка данных → валидация → обработка → HPO → обучение → мониторинг дрифта → API inference.

> 📋 **Улучшения**: См. [IMPROVEMENTS.md](IMPROVEMENTS.md) для деталей последних улучшений

## Архитектура

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Airflow    │────▶│   Spark      │────▶│   Optuna     │
│  Orchestr.   │     │  Processing  │     │     HPO      │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                          │
       │                                          ▼
       │                                   ┌──────────────┐
       │                                   │    MLflow    │
       │                                   │   Tracking   │
       │                                   └──────────────┘
       ▼                                          │
┌──────────────┐                                  │
│ PostgreSQL   │◀─────────────────────────────────┘
│   Metrics    │
└──────────────┘     ┌──────────────┐     ┌──────────────┐
       ▲             │   FastAPI    │────▶│ Prometheus   │
       │             │   Serving    │     │   Metrics    │
       │             └──────────────┘     └──────┬───────┘
       │                    │                     │
       └────────────────────┘                     ▼
                                           ┌──────────────┐
                                           │   Grafana    │
                                           │  Dashboards  │
                                           └──────────────┘
```

## Возможности

### ML Pipeline

- **Обучение моделей**: LogisticRegression, RandomForest, HistGradientBoosting, MLP, DistilBERT
- **HPO**: Optuna с многокритериальной оптимизацией (accuracy, latency, complexity)
- **Feature engineering**: TF-IDF + TruncatedSVD + числовые признаки (text_len, word_count, etc.)
- **Валидация**: строгая проверка схемы данных, детекция аномалий

### MLOps Features

- **Orchestration**: Airflow с Dynamic Task Mapping для параллельного обучения
- **Experiment tracking**: MLflow для версионирования моделей и метрик
- **Monitoring**: drift detection через PSI, автоматический ретрейнинг при дрифте
- **API serving**: FastAPI с /predict и /batch_predict эндпоинтами
- **Observability**: Prometheus metrics + Grafana dashboards + structured logging

### Data Processing

- **Spark**: распределённая обработка больших датасетов
- **Validation**: контракты данных, проверка качества
- **Drift injection**: синтетический дрифт для тестирования мониторинга

## Быстрый старт

### Предварительные требования

- Docker + Docker Compose
- Python 3.10+
- 8GB RAM минимум

### Локальный запуск

```bash
# Клонирование репозитория
git clone <repo_url>
cd sentiment-mlops-pipeline

# Создание секретов
mkdir -p secrets
echo "your_secure_password" > secrets/postgres_password.txt

# Запуск всех сервисов
docker-compose up -d

# Проверка статуса
docker-compose ps
```

Сервисы будут доступны:

- **Airflow UI**: http://localhost:8080 (admin/admin)
- **MLflow UI**: http://localhost:5000
- **FastAPI**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Запуск пайплайна

```bash
# Через Airflow UI: выберите DAG "kindle_unified_pipeline"
# Или через CLI:
docker exec airflow-webserver airflow dags trigger kindle_unified_pipeline
```

DAG автоматически:

1. Загружает данные Kindle отзывов
2. Обрабатывает через Spark
3. Обучает модели параллельно (на основе `SELECTED_MODEL_KINDS`)
4. Выбирает лучшую по F1-score
5. Логирует метрики в MLflow + PostgreSQL

Подробности: [airflow/dags/README_UNIFIED_DAG.md](airflow/dags/README_UNIFIED_DAG.md)

### API инференс

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "reviewText": "Great book, highly recommend!",
    "text_len": 30.0,
    "word_count": 5.0
  }'

# Batch prediction
python scripts/request_batch.py --samples 100
```

## Мониторинг и SLO

### Service Level Objectives (SLO)

Целевые показатели для production-системы:

| Метрика               | SLO Target | Метод измерения                                         |
| --------------------- | ---------- | ------------------------------------------------------- |
| **API Latency (p95)** | < 500ms    | Prometheus histogram `api_request_duration_seconds`     |
| **API Latency (p99)** | < 1000ms   | Prometheus histogram `api_request_duration_seconds`     |
| **API Availability**  | > 99.5%    | Uptime = (1 - error_rate) \* 100%                       |
| **Error Rate**        | < 1%       | `api_errors_total / api_request_duration_seconds_count` |
| **Drift Detection**   | PSI < 0.1  | Population Stability Index для числовых признаков       |
| **Model F1 Score**    | > 0.85     | Валидационная метрика при обучении                      |

### Grafana Dashboards

Автоматически настроены дашборды (http://localhost:3000):

**API SLO Dashboard** (`sentiment-api-slo`):

- 📊 Latency перцентили (p50/p95/p99) по эндпоинтам
- 📈 Throughput (запросов/сек)
- ⚠️ Error Rate с детализацией по типам ошибок
- 📦 Размеры запросов/ответов (p95)
- 🎯 Gauge-метрики: p99 latency, error rate, throughput

**Alerting Rules**:

- Error rate > 5% → критичный алерт
- p99 latency > 1s → предупреждение
- Drift PSI > 0.1 → автоматический ретрейнинг (см. `drift_retraining_dag`)

### Метрики API

Prometheus собирает следующие метрики:

```promql
# Latency по перцентилям
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Throughput (запросов/сек)
rate(api_request_duration_seconds_count[1m])

# Error rate
rate(api_errors_total[5m]) / rate(api_request_duration_seconds_count[5m])

# Размер запросов
histogram_quantile(0.95, rate(api_request_size_bytes_bucket[5m]))
```

### Drift Monitoring

Мониторинг дрифта работает автоматически:

1. **drift_monitor.py** вычисляет PSI для каждого признака
2. При PSI > 0.1 записывается `artefacts/drift_artefacts/drift_report.csv`
3. **drift_retraining_dag** отслеживает файл через FileSensor
4. При появлении дрифта триггерится автоматический ретрейнинг

### Логи

Структурированное логирование (JSON format поддерживается):

```bash
# Просмотр логов Airflow
docker logs airflow-webserver --tail 100 -f

# Логи API
docker logs api-service --tail 100 -f

# Включить JSON формат через переменную окружения
LOG_FORMAT=json docker-compose up -d
```

## Структура проекта

```
.
├── airflow/
│   └── dags/
│       ├── kindle_pipeline.py           # Основной unified DAG
│       ├── drift_retraining_dag.py      # Автоматический ретрейнинг
│       └── README_UNIFIED_DAG.md
├── scripts/
│   ├── config.py                        # Централизованная конфигурация (SSoT)
│   ├── settings.py                      # Адаптер для обратной совместимости
│   ├── train.py                         # Основной скрипт обучения
│   ├── api_service.py                   # FastAPI сервис
│   ├── drift_monitor.py                 # Мониторинг дрифта
│   ├── data_validation.py               # Валидация данных
│   ├── feature_contract.py              # Контракт признаков
│   ├── logging_config.py                # Конфигурация логирования
│   └── train_modules/
│       ├── data_loading.py
│       ├── feature_space.py
│       ├── models.py
│       └── text_analyzers.py
├── tests/
│   ├── test_core_modules.py             # Тесты ключевых модулей
│   ├── test_api_service.py              # Тесты API
│   └── conftest.py
├── grafana/
│   ├── provisioning/                    # Автоконфигурация Grafana
│   │   ├── datasources/
│   │   │   └── prometheus.yml
│   │   └── dashboards/
│   │       └── default.yml
│   └── dashboards/
│       └── api_slo.json                 # SLO dashboard
├── artefacts/                           # Модели и метрики
├── postgres-init/                       # SQL инициализация
├── .github/workflows/
│   └── lint.yml                         # CI с Ruff
├── .ruff.toml                           # Конфигурация линтера
├── docker-compose.yml
└── README.md
```

## Разработка

### Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# Зависимости для API
pip install -r requirements.api.txt

# Зависимости для разработки
pip install -r requirements.dev.txt
```

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ --cov=scripts --cov-report=html

# Конкретный модуль
pytest tests/test_core_modules.py -v
```

Подробности: [tests/README.md](tests/README.md)

### Линтинг и форматирование

```bash
# Проверка кода
ruff check .

# Автоисправление
ruff check --fix .

# Проверка форматирования
ruff format --check .

# Форматирование
ruff format .
```

CI автоматически проверяет код через GitHub Actions (`.github/workflows/lint.yml`).

### Переменные окружения

Ключевые переменные (см. `scripts/config.py`):

```bash
# Директории данных
RAW_DATA_DIR=/data/raw
PROCESSED_DATA_DIR=/data/processed
MODEL_DIR=/artefacts

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# PostgreSQL
POSTGRES_USER=admin
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password  # через Docker secrets
POSTGRES_METRICS_URI=postgresql://admin:***@postgres:5432/metrics

# Optuna
OPTUNA_STORAGE=postgresql+psycopg2://admin:***@postgres:5432/optuna

# Логирование
LOG_LEVEL=INFO
LOG_FORMAT=text  # или json
```

## Troubleshooting

### Проблема: Airflow DAG не появляется в UI

```bash
# Проверка синтаксиса DAG
python -m py_compile airflow/dags/kindle_pipeline.py

# Перезапуск scheduler
docker restart airflow-scheduler

# Проверка логов
docker logs airflow-scheduler --tail 50
```

### Проблема: API возвращает 500 при предсказании

Проверка наличия модели:

```bash
docker exec api-service ls -lh /app/artefacts/best_model.joblib
```

Если модель отсутствует — запустите DAG для обучения.

### Проблема: Grafana дашборд не загружается

```bash
# Проверка provisioning
docker exec grafana ls -la /etc/grafana/provisioning/datasources/
docker exec grafana ls -la /etc/grafana/provisioning/dashboards/

# Перезапуск Grafana
docker restart grafana

# Проверка подключения к Prometheus
docker exec grafana wget -O- http://prometheus:9090/-/healthy
```

### Проблема: PostgreSQL метрики не сохраняются

Проверка подключения:

```bash
docker exec airflow-webserver python -c "
from scripts.config import settings
from sqlalchemy import create_engine
engine = create_engine(settings.postgres_metrics_uri)
with engine.connect() as conn:
    print(conn.execute('SELECT version()').fetchone())
"
```

## CI/CD

Настроен GitHub Actions pipeline (`.github/workflows/lint.yml`):

- **Lint**: Ruff проверка кода + форматирования
- **Auto-trigger**: на push в master/main/develop и PR

Локальный запуск CI проверок:

```bash
# Lint (как в CI)
ruff check .
ruff format --check .

# Tests
pytest tests/ -v

# DAG validation
python -m py_compile airflow/dags/kindle_pipeline.py
python -m py_compile airflow/dags/drift_retraining_dag.py
```

## Streamlit Demo

Интерактивное демо для визуализации и инференса:

```bash
# Запуск
streamlit run streamlit_app.py

# Откроется http://localhost:8501
```

Возможности:

- 💬 Real-time inference через API
- 📈 Визуализация метрик модели
- 🔍 Мониторинг дрифта с графиками
- ℹ️ Информация о модели и гиперпараметрах

## Roadmap

- [x] Dynamic Task Mapping для параллельного обучения
- [x] Автоматический ретрейнинг при дрифте
- [x] Grafana dashboards с SLO метриками
- [x] CI с Ruff линтингом
- [ ] MLflow Model Registry интеграция
- [ ] A/B тестирование моделей
- [ ] Advanced drift: multivariate, text embeddings
- [ ] Model serving через Kubernetes

## Лицензия

MIT

## Контакты

Для вопросов и предложений: [ваш email]

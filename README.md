# Sentiment MLOps Pipeline

Production-ready MLOps пайплайн для sentiment analysis отзывов на Kindle книги. Реализует полный цикл: загрузка данных → валидация → обработка → HPO → обучение → мониторинг дрифта → API inference.

## Архитектура

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Airflow    │────▶│   Spark     │────▶│   Optuna     │
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
       │             └──────────────┘     └──────────────┘
       │                    │
       └────────────────────┘
```

## Возможности

### ML Pipeline

- **Обучение моделей**: LogisticRegression, RandomForest, HistGradientBoosting, MLP, DistilBERT
- **HPO**: Optuna с многокритериальной оптимизацией (accuracy, latency, complexity)
- **Feature engineering**: TF-IDF + TruncatedSVD + числовые признаки (text_len, word_count, etc.)
- **Валидация**: строгая проверка схемы данных, детекция аномалий

### MLOps Features

- **Orchestration**: Airflow с тремя режимами работы (см. unified DAG)
- **Experiment tracking**: MLflow для версионирования моделей и метрик
- **Monitoring**: drift detection через PSI, логирование метрик в PostgreSQL
- **API serving**: FastAPI с /predict и /batch_predict эндпоинтами
- **Observability**: Prometheus metrics + structured logging

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

# Запуск всех сервисов
docker-compose up -d

# Проверка статуса
docker-compose ps
```

Сервисы будут доступны:

- Airflow UI: http://localhost:8080 (admin/admin)
- MLflow UI: http://localhost:5000
- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090

### Запуск пайплайна

```bash
# Через Airflow UI: выберите DAG "kindle_unified_pipeline"
# Или через CLI:
docker exec airflow-webserver airflow dags trigger kindle_unified_pipeline \
  --conf '{"execution_mode": "standard", "force_download": false}'
```

Режимы работы unified DAG:

- `standard`: базовое обучение с Optuna HPO
- `monitored`: + логирование метрик задач и моделей в PostgreSQL
- `parallel`: параллельное обучение трёх моделей с автовыбором лучшей

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

## Структура проекта

```
.
├── airflow/
│   └── dags/
│       ├── kindle_unified_pipeline.py    # Объединённый параметризованный DAG
│       └── README_UNIFIED_DAG.md         # Документация DAG
├── scripts/
│   ├── train.py                          # Основной скрипт обучения
│   ├── api_service.py                    # FastAPI сервис
│   ├── drift_monitor.py                  # Мониторинг дрифта
│   ├── data_validation.py                # Валидация данных
│   ├── feature_contract.py               # Контракт признаков
│   └── train_modules/                    # Модули обучения
│       ├── data_loading.py
│       ├── feature_space.py
│       ├── models.py
│       └── text_analyzers.py
├── tests/
│   ├── test_core_modules.py              # Тесты ключевых модулей
│   ├── test_api_service.py               # Тесты API
│   └── conftest.py                       # Shared fixtures
├── artefacts/                            # Модели и метрики
├── postgres-init/                        # SQL инициализация
├── docker-compose.yml                    # Инфраструктура
└── README.md
```

## Разработка

### Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# Зависимости для API
pip install -r requirements.api.txt

# Зависимости для тестов
pip install -r tests/requirements.txt
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

### Переменные окружения

Ключевые переменные (см. `scripts/settings.py`):

```bash
# Директории данных
RAW_DATA_DIR=/data/raw
PROCESSED_DATA_DIR=/data/processed
MODEL_DIR=/models

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# PostgreSQL metrics
POSTGRES_METRICS_URI=postgresql://airflow:airflow@postgres:5432/metrics_db

# Slack (опционально)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## Мониторинг

### Метрики модели

- **Accuracy, Precision, Recall, F1**: сохраняются в MLflow + PostgreSQL
- **Latency**: отслеживается через Prometheus histogram
- **Drift**: PSI для каждого числового признака

### Логи

Структурированное логирование (JSON) в Docker контейнерах:

```bash
# Просмотр логов Airflow
docker logs airflow-webserver --tail 100 -f

# Логи API
docker logs api-service --tail 100 -f
```

### Алерты

При превышении порога дрифта (PSI > 0.1):

- Запись в `drift_report.csv`
- Опционально: Slack webhook

## Troubleshooting

### Проблема: Airflow DAG не появляется в UI

```bash
# Проверка синтаксиса DAG
python -m py_compile airflow/dags/kindle_unified_pipeline.py

# Перезапуск scheduler
docker restart airflow-scheduler
```

### Проблема: API возвращает 500 при предсказании

Проверка наличия модели:

```bash
docker exec api-service ls -lh /app/artefacts/best_model.joblib
```

Если модель отсутствует — запустите DAG для обучения.

### Проблема: PostgreSQL метрики не сохраняются

Проверка подключения:

```bash
docker exec airflow-webserver python -c "
from scripts.settings import POSTGRES_METRICS_URI
from sqlalchemy import create_engine
engine = create_engine(POSTGRES_METRICS_URI)
print(engine.table_names())
"
```

## CI/CD

Настроен GitHub Actions pipeline (`.github/workflows/ci.yml`):

- **Lint**: ruff + black проверка кода
- **Tests**: unit + integration тесты на Python 3.10/3.11
- **DAG validation**: проверка синтаксиса Airflow DAG
- **Docker build**: сборка API и Airflow образов

Запуск локально:

```bash
# Lint
ruff check scripts/ tests/
black --check scripts/ tests/

# Tests
pytest tests/ -v

# DAG validation
python -m py_compile airflow/dags/kindle_unified_pipeline.py
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

- [ ] A/B тестирование моделей
- [ ] Advanced drift: multivariate, text embeddings
- [ ] Model serving через Kubernetes
- [ ] Auto-retraining pipeline

## Лицензия

MIT

## Контакты

Для вопросов и предложений: [ваш email]

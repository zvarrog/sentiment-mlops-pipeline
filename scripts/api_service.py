import asyncio
import json
import signal
import time
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from textblob import TextBlob

from .config import (
    BASELINE_NUMERIC_STATS_PATH,
    BEST_MODEL_META_PATH,
    BEST_MODEL_PATH,
    MODEL_ARTEFACTS_DIR,
)
from .feature_contract import FeatureContract
from .logging_config import (
    clear_trace_id,
    get_logger,
    set_trace_id,
)

log = get_logger("api_service")

MAX_TEXT_LENGTH = 10_000
MAX_BATCH_SIZE = 100


class NumericFeatures(TypedDict, total=False):
    """Числовые признаки для модели."""

    text_len: float
    word_count: float
    kindle_freq: float
    sentiment: float
    user_avg_len: float
    user_review_count: float
    item_avg_len: float
    item_review_count: float
    exclamation_count: float
    caps_ratio: float
    question_count: float
    avg_word_length: float


_artifacts_loaded = False


def signal_handler(signum, frame):
    log.info("Получен сигнал %s, завершаем обработку", signum)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Prometheus метрики
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "API request duration",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_name"],
)

# Бизнес-метрики: распределение уверенности и классов
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Max predicted probability per request item",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0),
)

PREDICTION_LABELS = Counter(
    "prediction_labels_total",
    "Distribution of predicted labels",
    ["label"],
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Total API errors",
    ["method", "endpoint", "error_type"],
)


class PredictRequest(BaseModel):
    """Запрос: список текстов и опциональные числовые признаки."""

    texts: list[str]
    numeric_features: dict[str, list[float]] | None = None


class PredictResponse(BaseModel):
    """Метки, вероятности и опциональные предупреждения."""

    labels: list[int]
    probs: list[list[float]] | None = None
    warnings: dict[str, list[str]] | None = None


class BatchPredictRequest(BaseModel):
    """Пакетный запрос."""

    data: list[dict[str, Any]]


class BatchPredictResponse(BaseModel):
    """Результат пакетного предсказания и предупреждения по объектам."""

    predictions: list[dict[str, Any]]
    # предупреждения по каждому объекту: item_{i} -> {category: [messages]}
    warnings: dict[str, dict[str, list[str]]] | None = None


class MetadataResponse(BaseModel):
    """Ответ с метаданными модели, контрактом признаков и статусом здоровья."""

    model_info: dict[str, Any]
    feature_contract: dict[str, Any]
    health: dict[str, Any]


def create_app(defer_artifacts: bool = False) -> FastAPI:
    """Фабрика приложения FastAPI.

    Args:
        defer_artifacts: Если True, пропускает загрузку артефактов на старте (удобно для тестов).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _artifacts_loaded

        if not defer_artifacts and not _artifacts_loaded:
            _load_artifacts(app)
            _artifacts_loaded = True

        # Инициализация метрик ошибок для корректной работы дашборда
        for endpoint in ["/predict", "/batch_predict"]:
            for error_type in ["model_not_loaded", "empty_input", "validation_error"]:
                ERROR_COUNT.labels(
                    method="POST", endpoint=endpoint, error_type=error_type
                ).inc(0)

        yield

        await asyncio.sleep(0.1)

        for attr in ("MODEL", "META", "NUMERIC_DEFAULTS", "FEATURE_CONTRACT"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)

    application = FastAPI(
        title="Kindle Reviews API", version="1.0.0", lifespan=lifespan
    )

    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Регистрация middleware и маршрутов
    _register_middlewares(application)
    _register_routes(application, limiter)

    return application


def _register_middlewares(application: FastAPI) -> None:
    """Регистрирует middleware для приложения."""

    @application.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()

        response = await call_next(request)
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        return response

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or f"req-{id(request)}"
        set_trace_id(req_id)

        try:
            response = await call_next(request)
        finally:
            clear_trace_id()
        response.headers["X-Request-ID"] = req_id
        return response


def _register_routes(application: FastAPI, limiter: Limiter) -> None:
    """Регистрирует маршруты API."""

    @application.get("/health")
    def health():
        """Liveness probe: API процесс жив."""
        return {"status": "alive"}

    @application.get("/ready")
    def ready(response: Response):
        """Readiness probe: API готов обрабатывать запросы."""
        model_loaded = getattr(application.state, "MODEL", None) is not None

        if not model_loaded:
            response.status_code = 503
            return {
                "status": "not_ready",
                "model_exists": BEST_MODEL_PATH.exists(),
                "message": "Модель не загружена, ожидается обучение",
            }

        return {
            "status": "ready",
            "model_loaded": True,
            "best_model": getattr(application.state, "META", {}).get("best_model"),
        }

    @application.get("/metrics")
    def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.get("/metadata", response_model=MetadataResponse)
    def get_metadata():
        """Возвращает метаданные модели и информацию о признаках."""
        meta = getattr(application.state, "META", {})
        feature_contract = getattr(application.state, "FEATURE_CONTRACT", None)

        model_info = {
            "best_model": meta.get("best_model", "unknown"),
            "best_params": meta.get("best_params", {}),
            "test_metrics": meta.get("test_metrics", {}),
            "training_duration_sec": meta.get("duration_sec", None),
            "dataset_sizes": meta.get("sizes", {}),
        }

        feature_info = feature_contract.get_feature_info() if feature_contract else {}

        health_info = {
            "model_loaded": getattr(application.state, "MODEL", None) is not None,
            "baseline_stats_loaded": bool(
                getattr(application.state, "NUMERIC_DEFAULTS", {})
            ),
            "feature_contract_loaded": feature_contract is not None,
        }

        return MetadataResponse(
            model_info=model_info, feature_contract=feature_info, health=health_info
        )

    @application.post("/predict", response_model=PredictResponse)
    @limiter.limit("100/minute")
    def predict(request: Request, req: PredictRequest):
        try:
            labels, probs, warnings = _validate_and_predict(
                application, "/predict", req.texts, req.numeric_features
            )

            return PredictResponse(
                labels=[int(x) for x in labels], probs=probs, warnings=warnings
            )
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as e:
            ERROR_COUNT.labels(
                method="POST", endpoint="/predict", error_type="bad_request"
            ).inc()
            raise HTTPException(status_code=400, detail=str(e)) from e
        except (RuntimeError, FileNotFoundError, OSError) as e:
            ERROR_COUNT.labels(
                method="POST", endpoint="/predict", error_type="internal_error"
            ).inc()
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.post("/batch_predict", response_model=BatchPredictResponse)
    @limiter.limit("50/minute")
    def batch_predict(request: Request, req: BatchPredictRequest):
        try:
            # Формируем список текстов и числовых признаков из batch данных
            rows: list[dict[str, Any]] = []
            for item in req.data:
                row: dict[str, Any] = {"reviewText": str(item.get("reviewText", ""))}
                for key, value in item.items():
                    if key != "reviewText" and isinstance(value, (int, float)):
                        row[key] = float(value)
                rows.append(row)

            import pandas as _pd

            df_in = _pd.DataFrame(rows)
            texts = df_in["reviewText"].tolist()

            numeric_features: dict[str, list[float]] | None = None
            num_cols = [c for c in df_in.columns if c != "reviewText"]
            if num_cols:
                numeric_features = {c: df_in[c].tolist() for c in num_cols}

            labels, probs, ignored = _validate_and_predict(
                application, "/batch_predict", texts, numeric_features
            )

            all_ignored: dict[str, dict[str, list[str]]] | None = None
            if ignored:
                all_ignored = {"global": {"ignored_features": ignored}}

            predictions = [
                {
                    "index": i,
                    "prediction": int(labels[i]),
                    **({"probabilities": probs[i]} if probs else {}),
                }
                for i in range(len(texts))
            ]

            return BatchPredictResponse(predictions=predictions, warnings=all_ignored)
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as e:
            ERROR_COUNT.labels(
                method="POST", endpoint="/batch_predict", error_type="bad_request"
            ).inc()
            raise HTTPException(status_code=400, detail=str(e)) from e
        except (RuntimeError, FileNotFoundError, OSError) as e:
            ERROR_COUNT.labels(
                method="POST", endpoint="/batch_predict", error_type="internal_error"
            ).inc()
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.get("/")
    def root():
        """Корневой эндпоинт с информацией об API."""
        return {
            "service": "Kindle Reviews Sentiment Analysis API",
            "version": "1.0.0",
            "endpoints": {
                "predict": "/predict (POST)",
                "batch_predict": "/batch_predict (POST)",
                "health": "/health (GET)",
                "metrics": "/metrics (GET)",
            },
            "docs": "/docs",
        }


def _validate_and_predict(
    application: FastAPI,
    endpoint: str,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None = None,
) -> tuple[list[int], list[list[float]] | None, dict[str, list[str]] | None]:
    """Общая логика валидации и предсказания для всех эндпоинтов.

    Args:
        application: FastAPI приложение
        endpoint: Имя эндпоинта для логирования метрик
        texts: Список текстов для предсказания
        numeric_features: Опциональные числовые признаки

    Returns:
        Кортеж (предсказания, вероятности, предупреждения)

    Raises:
        HTTPException: При ошибках валидации или работы модели
    """
    model = getattr(application.state, "MODEL", None)
    if model is None:
        ERROR_COUNT.labels(
            method="POST", endpoint=endpoint, error_type="model_not_loaded"
        ).inc()
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if not texts:
        ERROR_COUNT.labels(
            method="POST", endpoint=endpoint, error_type="empty_input"
        ).inc()
        raise HTTPException(status_code=400, detail="Список texts пуст")

    if len(texts) > MAX_BATCH_SIZE:
        ERROR_COUNT.labels(
            method="POST", endpoint=endpoint, error_type="validation_error"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail=f"Размер батча {len(texts)} превышает максимум {MAX_BATCH_SIZE}",
        )

    for i, text in enumerate(texts):
        if len(text) > MAX_TEXT_LENGTH:
            ERROR_COUNT.labels(
                method="POST", endpoint=endpoint, error_type="validation_error"
            ).inc()
            raise HTTPException(
                status_code=400,
                detail=f"Текст #{i} длиной {len(text)} превышает лимит {MAX_TEXT_LENGTH}",
            )

    meta = getattr(application.state, "META", {})
    model_name = meta.get("best_model", "unknown")
    PREDICTION_COUNT.labels(model_name=model_name).inc(len(texts))

    labels, probs, warnings = _predict_with_model(
        application, model, texts, numeric_features
    )

    # Бизнес-метрики
    if probs is not None:
        for p in probs:
            PREDICTION_CONFIDENCE.observe(max(p))
    for lbl in labels:
        PREDICTION_LABELS.labels(label=str(int(lbl))).inc()

    return labels, probs, warnings


def _load_artifacts(application: FastAPI) -> None:
    """Загружает модель и артефакты (метаданные, baseline статистики, контракт признаков)."""
    log.info("Загрузка артефактов модели...")
    if not BEST_MODEL_PATH.exists():
        log.warning(
            "Модель не найдена: %s — API запустится в режиме ожидания", BEST_MODEL_PATH
        )
        application.state.MODEL = None
        application.state.META = {}
        application.state.NUMERIC_DEFAULTS = {}
        application.state.FEATURE_CONTRACT = None
        return

    application.state.MODEL = joblib.load(BEST_MODEL_PATH)
    log.info("Модель загружена: %s", BEST_MODEL_PATH)

    # Метаданные модели (обязательно)
    if not BEST_MODEL_META_PATH.exists():
        log.warning("Метаданные модели не найдены: %s", BEST_MODEL_META_PATH)
        application.state.META = {}
    else:
        application.state.META = json.loads(
            BEST_MODEL_META_PATH.read_text(encoding="utf-8")
        )

    # Baseline статистики (опционально, используются для мониторинга дрифта)
    if BASELINE_NUMERIC_STATS_PATH.exists():
        application.state.NUMERIC_DEFAULTS = json.loads(
            BASELINE_NUMERIC_STATS_PATH.read_text(encoding="utf-8")
        )
        log.info("Baseline статистики загружены")
    else:
        log.warning(
            "Baseline статистики не найдены: %s (дрифт-мониторинг недоступен)",
            BASELINE_NUMERIC_STATS_PATH,
        )
        application.state.NUMERIC_DEFAULTS = {}

    # Контракт признаков (опционально)
    try:
        application.state.FEATURE_CONTRACT = FeatureContract.from_model_artifacts(
            MODEL_ARTEFACTS_DIR
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        log.warning("Не удалось загрузить контракт признаков: %s", e)
        application.state.FEATURE_CONTRACT = None

    log.info("Артефакты модели успешно загружены")


# (Дубликаты middleware и загрузчика артефактов удалены — логика вынесена в create_app)


def _build_dataframe(
    application: FastAPI,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Собирает DataFrame для предсказания и возвращает список проигнорованных признаков."""
    df = pd.DataFrame({"reviewText": texts})
    ignored_features = []

    # Получаем список ожидаемых числовых колонок
    feature_contract = getattr(application.state, "FEATURE_CONTRACT", None)
    if not feature_contract:
        raise RuntimeError(
            "Feature contract (артефакты) не загружены — невозможно определить список признаков. Проверьте наличие артефактов модели."
        )
    numeric_cols = feature_contract.expected_numeric_columns

    # Если переданы числовые признаки, используем их
    if numeric_features:
        for col, values in numeric_features.items():
            if col not in numeric_cols:
                ignored_features.append(f"{col} (неизвестный признак)")
            elif len(values) != len(texts):
                ignored_features.append(f"{col} (длина {len(values)} != {len(texts)})")
            else:
                df[col] = values

    def _sentiment_textblob(s: pd.Series) -> pd.Series:
        def calculate_sentiment(text):
            if not text or len(str(text).strip()) < 3:
                return 0.0

            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            return float(max(-1.0, min(1.0, round(polarity, 4))))

        return s.apply(calculate_sentiment).astype(float)

    def _text_feature_extractors():
        return {
            "text_len": lambda s: s.str.len().astype(float),
            "word_count": lambda s: s.str.split().str.len().fillna(0).astype(float),
            "kindle_freq": lambda s: s.str.lower().str.count("kindle").astype(float),
            "exclamation_count": lambda s: s.str.count("!").astype(float),
            "caps_ratio": lambda s: (
                s.str.replace(r"[^A-Z]", "", regex=True).str.len().astype(float)
                / s.str.len().clip(lower=1).astype(float)
            ).fillna(0.0),
            "question_count": lambda s: s.str.count(r"\?").astype(float),
            "avg_word_length": lambda s: (
                s.str.len().astype(float)
                / s.str.split().str.len().clip(lower=1).astype(float)
            ),
            "sentiment": _sentiment_textblob,
        }

    s_raw = df["reviewText"].fillna("")
    for col, extractor in _text_feature_extractors().items():
        if col in numeric_cols and col not in df.columns and col != "sentiment":
            df[col] = extractor(s_raw)

    # Очистка текста
    from scripts.text_features import clean_text

    df["reviewText"] = s_raw.apply(clean_text)

    if "sentiment" in numeric_cols and "sentiment" not in df.columns:
        df["sentiment"] = _text_feature_extractors()["sentiment"](
            df["reviewText"].fillna("")
        )

    # Недостающие числовые колонки — заполняем из baseline статистик
    baseline_stats = getattr(application.state, "NUMERIC_DEFAULTS", {})
    missing_features = [col for col in numeric_cols if col not in df.columns]
    if missing_features:
        log.warning(
            "Отсутствуют признаки %s — заполнены из baseline статистик (потенциальная утечка данных)",
            missing_features,
        )
    for col in missing_features:
        default_val = baseline_stats.get(col, {}).get("mean", 0.0)
        df[col] = default_val

    return df, ignored_features


def _predict_with_model(
    application: FastAPI,
    model: Any,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None = None,
) -> tuple[list[int], list[list[float]] | None, list[str] | None]:
    """Единая логика предсказаний.

    Если модель DistilBERT — используем только тексты.
    Иначе собираем DataFrame с числовыми признаками.
    """
    is_text_only = not hasattr(model, "named_steps") or (
        hasattr(model, "named_steps") and "pre" not in getattr(model, "named_steps", {})
    )
    if is_text_only:
        preds = model.predict(pd.Series(texts))
        probs: list[list[float]] | None = None
        if hasattr(model, "predict_proba"):
            try:
                probs_arr = model.predict_proba(pd.Series(texts))
                probs = [row.tolist() for row in probs_arr]
            except (ValueError, TypeError):
                probs = None
        return [int(x) for x in preds], probs, None

    df, ignored = _build_dataframe(application, texts, numeric_features)
    preds = model.predict(df)
    probs: list[list[float]] | None = None
    if hasattr(model, "predict_proba"):
        try:
            probs_arr = model.predict_proba(df)
            probs = [row.tolist() for row in probs_arr]
        except (ValueError, TypeError):
            probs = None
    return [int(x) for x in preds], probs, ignored or None


app = create_app()

# Локальный запуск: uvicorn scripts.api_service:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

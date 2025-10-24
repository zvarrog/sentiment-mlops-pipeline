import asyncio
import json
import signal
import time
from contextlib import asynccontextmanager
from typing import Any

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

from .feature_contract import FeatureContract

# Настраиваем логирование для API
from .logging_config import (
    clear_trace_id,
    get_trace_id,
    set_trace_id,
    setup_auto_logging,
)
from .settings import MODEL_ARTEFACTS_DIR, MODEL_FILE_DIR

log = setup_auto_logging()

# Пути к артефактам модели
BEST_MODEL_PATH = MODEL_FILE_DIR / "best_model.joblib"
META_PATH = MODEL_ARTEFACTS_DIR / "best_model_meta.json"
BASELINE_NUMERIC_PATH = MODEL_ARTEFACTS_DIR / "baseline_numeric_stats.json"


shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Обработчик системных сигналов для корректного завершения приложения."""
    log.info("Получен сигнал %s, инициирую shutdown", signum)
    shutdown_event.set()


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
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_name"],
)


class PredictRequest(BaseModel):
    """Запрос: список текстов и опциональные числовые признаки."""

    texts: list[str]
    # Опциональные числовые признаки для более точного предсказания
    numeric_features: dict[str, list[float]] | None = None


class PredictResponse(BaseModel):
    """Метки, вероятности и опциональные предупреждения."""

    labels: list[int]
    probs: list[list[float]] | None = None
    warnings: dict[str, list[str]] | None = None


class BatchPredictRequest(BaseModel):
    """Пакетный запрос: список объектов с полем reviewText и числовыми признаками."""

    data: list[
        dict[str, Any]
    ]  # Список объектов с полями reviewText и числовыми признаками


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка артефактов при старте, мягкое завершение при shutdown."""
    # Startup
    _load_artifacts()
    log.info("API запущен и готов принимать запросы")

    yield

    # Shutdown
    log.info("Завершаю обработку текущих запросов...")
    await asyncio.sleep(2)

    # Очищаем ресурсы
    for attr in ("MODEL", "META", "NUMERIC_DEFAULTS", "FEATURE_CONTRACT"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)

    log.info("API корректно остановлен")


app = FastAPI(title="Kindle Reviews API", version="1.0.0", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware для сбора метрик Prometheus."""
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


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Middleware: добавляет X-Request-ID в заголовок и устанавливает trace_id."""
    req_id = request.headers.get("X-Request-ID")
    if not req_id:
        # Простой вариант без зависимостей: используем id объекта и время
        req_id = f"req-{id(request)}"
    set_trace_id(req_id)
    log.info("Запрос: %s %s, X-Request-ID=%s", request.method, request.url.path, req_id)
    try:
        response = await call_next(request)
    finally:
        # В логе после обработки
        log.info(
            "Ответ: %s %s -> %s, X-Request-ID=%s",
            request.method,
            request.url.path,
            getattr(request.state, "status_code", "?"),
            get_trace_id(),
        )
        clear_trace_id()
    # Проставляем заголовок в ответ
    response.headers["X-Request-ID"] = req_id
    return response


def _load_artifacts():
    """Загружает модель и артефакты (метаданные, baseline статистики, контракт признаков)."""
    log.info("Загрузка артефактов модели...")
    if not BEST_MODEL_PATH.exists():
        log.error("Модель не найдена: %s", BEST_MODEL_PATH)
        raise FileNotFoundError(f"Модель не найдена: {BEST_MODEL_PATH}")

    app.state.MODEL = joblib.load(BEST_MODEL_PATH)
    log.info("Модель загружена: %s", BEST_MODEL_PATH)

    app.state.META = json.loads(META_PATH.read_text(encoding="utf-8"))
    app.state.NUMERIC_DEFAULTS = json.loads(
        BASELINE_NUMERIC_PATH.read_text(encoding="utf-8")
    )
    app.state.FEATURE_CONTRACT = FeatureContract.from_model_artifacts(
        MODEL_ARTEFACTS_DIR
    )
    log.info("Артефакты модели успешно загружены")


def _build_dataframe(
    texts: list[str], numeric_features: dict[str, list[float]] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Собирает DataFrame для предсказания и возвращает список проигнорованных признаков."""
    df = pd.DataFrame({"reviewText": texts})
    ignored_features = []

    # Получаем список ожидаемых числовых колонок
    feature_contract = getattr(app.state, "FEATURE_CONTRACT", None)
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

    # Автоматическое вычисление текстовых признаков через mapping
    def _text_feature_extractors():
        """Набор извлечения простых текстовых признаков (len, counts, sentiment)."""

        def _sentiment_textblob(s: pd.Series) -> pd.Series:
            from textblob import TextBlob

            def calculate_sentiment(text):
                if not text or len(str(text).strip()) < 3:
                    return 0.0

                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                return float(max(-1.0, min(1.0, round(polarity, 4))))

            return s.apply(calculate_sentiment).astype(float)

        return {
            "text_len": lambda s: s.str.len().astype(float),
            "word_count": lambda s: s.str.split().str.len().fillna(0).astype(float),
            "kindle_freq": lambda s: s.str.lower().str.count("kindle").astype(float),
            "exclamation_count": lambda s: s.str.count("!").astype(float),
            "caps_ratio": lambda s: (
                s.str.replace(r"[^A-Z]", "", regex=True).str.len().astype(float)
                / s.str.len().clip(lower=1).astype(float)
            ).fillna(0.0),
            "question_count": lambda s: s.str.count(r"\\?").astype(float),
            "avg_word_length": lambda s: (
                s.str.len().astype(float)
                / s.str.split().str.len().clip(lower=1).astype(float)
            ),
            "sentiment": _sentiment_textblob,
        }

    s = df["reviewText"].fillna("")
    for col, extractor in _text_feature_extractors().items():
        if col in numeric_cols and col not in df.columns:
            df[col] = extractor(s)

    # Остальные требуемые числовые колонки заполним базовыми значениями или нулями
    baseline_stats = getattr(app.state, "NUMERIC_DEFAULTS", {})
    for col in numeric_cols:
        if col not in df.columns:
            default_val = baseline_stats.get(col, {}).get("mean", 0.0)
            df[col] = default_val

    return df, ignored_features


@app.get("/health")
def health():
    """Проверка состояния API и загруженной модели."""
    return {
        "status": "ok",
        "model_exists": BEST_MODEL_PATH.exists(),
        "best_model": getattr(app.state, "META", {}).get("best_model"),
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    """Возвращает метаданные модели и информацию о признаках."""
    meta = getattr(app.state, "META", {})
    feature_contract = getattr(app.state, "FEATURE_CONTRACT", None)

    model_info = {
        "best_model": meta.get("best_model", "unknown"),
        "best_params": meta.get("best_params", {}),
        "test_metrics": meta.get("test_metrics", {}),
        "training_duration_sec": meta.get("duration_sec", None),
        "dataset_sizes": meta.get("sizes", {}),
    }

    feature_info = feature_contract.get_feature_info() if feature_contract else {}

    health_info = {
        "model_loaded": getattr(app.state, "MODEL", None) is not None,
        "baseline_stats_loaded": bool(getattr(app.state, "NUMERIC_DEFAULTS", {})),
        "feature_contract_loaded": feature_contract is not None,
    }

    return MetadataResponse(
        model_info=model_info, feature_contract=feature_info, health=health_info
    )


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("100/minute")
def predict(request: Request, req: PredictRequest):
    """Предсказание для списка текстов с опциональными числовыми признаками."""
    model = getattr(app.state, "MODEL", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    if not req.texts:
        raise HTTPException(status_code=400, detail="Список texts пуст")

    # Логируем метрику predictions
    meta = getattr(app.state, "META", {})
    model_name = meta.get("best_model", "unknown")
    PREDICTION_COUNT.labels(model_name=model_name).inc(len(req.texts))

    name = model.__class__.__name__.lower()
    if "distil" in name:
        preds = model.predict(pd.Series(req.texts))
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(pd.Series(req.texts)).tolist()
            except (AttributeError, ValueError, TypeError):
                probs = None
        return PredictResponse(
            labels=[int(x) for x in preds],
            probs=probs,
            warnings=None,
        )

    df, ignored = _build_dataframe(req.texts, req.numeric_features)
    preds = model.predict(df)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df).tolist()
        except (AttributeError, ValueError, TypeError):
            probs = None

    warnings = {"ignored_features": ignored} if ignored else None
    return PredictResponse(
        labels=[int(x) for x in preds],
        probs=probs,
        warnings=warnings,
    )


@app.post("/batch_predict", response_model=BatchPredictResponse)
@limiter.limit("50/minute")
def batch_predict(request: Request, req: BatchPredictRequest):
    """Пакетное предсказание для списка объектов с полными данными."""
    model = getattr(app.state, "MODEL", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    if not req.data:
        raise HTTPException(status_code=400, detail="Список data пуст")

    # Логируем метрику predictions
    meta = getattr(app.state, "META", {})
    model_name = meta.get("best_model", "unknown")
    PREDICTION_COUNT.labels(model_name=model_name).inc(len(req.data))

    # Строим DataFrame из всех объектов
    texts = []
    numeric_features = {}
    all_ignored = {}

    for item in req.data:
        texts.append(item.get("reviewText", ""))
        # Собираем числовые признаки
        for key, value in item.items():
            if key != "reviewText" and isinstance(value, (int, float)):
                if key not in numeric_features:
                    numeric_features[key] = []
                numeric_features[key].append(float(value))

    # Выравниваем длины списков числовых признаков
    for key, values in numeric_features.items():
        while len(values) < len(texts):
            values.append(0.0)

    # Предсказание
    name = model.__class__.__name__.lower()
    if "distil" in name:
        preds = model.predict(pd.Series(texts))
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs_array = model.predict_proba(pd.Series(texts))
                probs = [probs_array[i].tolist() for i in range(len(probs_array))]
            except (AttributeError, ValueError, TypeError):
                probs = None
    else:
        df, ignored = _build_dataframe(texts, numeric_features)
        if ignored:
            all_ignored["global"] = ignored
        preds = model.predict(df)
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs_array = model.predict_proba(df)
                probs = [probs_array[i].tolist() for i in range(len(probs_array))]
            except (AttributeError, ValueError, TypeError):
                probs = None

    # Формируем результат
    predictions = []
    for i in range(len(texts)):
        pred_item = {
            "index": i,
            "prediction": int(preds[i]),
        }
        if probs:
            pred_item["probabilities"] = probs[i]
        predictions.append(pred_item)

    return BatchPredictResponse(
        predictions=predictions, warnings=all_ignored if all_ignored else None
    )


@app.get("/health")
def health_check():
    """Health check эндпоинт для мониторинга."""
    model_loaded = hasattr(app.state, "MODEL") and app.state.MODEL is not None
    artifacts_loaded = (
        hasattr(app.state, "META")
        and hasattr(app.state, "FEATURE_CONTRACT")
        and app.state.META is not None
    )

    status = "healthy" if (model_loaded and artifacts_loaded) else "unhealthy"
    return {
        "status": status,
        "model_loaded": model_loaded,
        "artifacts_loaded": artifacts_loaded,
        "model_type": app.state.META.get("best_model", "unknown")
        if artifacts_loaded
        else None,
    }


@app.get("/")
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


# Локальный запуск: uvicorn scripts.api_service:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

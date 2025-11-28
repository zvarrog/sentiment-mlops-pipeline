import asyncio
import signal
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from scripts.config import (
    API_HOST,
    API_PORT,
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
)
from scripts.logging_config import (
    clear_trace_id,
    get_logger,
    set_trace_id,
)
from scripts.model_service import ModelService

log = get_logger(__name__)

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

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Max predicted probability per request item",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0),
)

PREDICTION_DURATION = Histogram(
    "prediction_duration",
    "Prediction handler duration",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
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
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    numeric_features: dict[str, list[float]] | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": ["This Kindle is amazing!", "Battery life is poor."],
                "numeric_features": None,
            }
        }
    )


class BatchItem(BaseModel):
    """Элемент батч-запроса с текстом и опциональными числовыми признаками."""

    reviewText: str
    text_len: float | None = None
    word_count: float | None = None
    kindle_freq: float | None = None
    sentiment: float | None = None
    user_avg_len: float | None = None
    user_review_count: float | None = None
    item_avg_len: float | None = None
    item_review_count: float | None = None
    exclamation_count: float | None = None
    caps_ratio: float | None = None
    question_count: float | None = None
    avg_word_length: float | None = None


class BatchPredictRequest(BaseModel):
    data: list[BatchItem] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class PredictResponse(BaseModel):
    labels: list[int]
    probs: list[list[float]] | None = None
    warnings: dict[str, list[str]] | None = None


class MetadataResponse(BaseModel):
    model_info: dict[str, Any]
    feature_contract: dict[str, Any]
    health: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["alive", "ready", "not_ready"]
    model_loaded: bool = False
    best_model: str | None = None
    message: str | None = None


model_service = ModelService()


def signal_handler(signum, frame):
    log.info("Получен сигнал %s, завершаем обработку", signum)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def create_app(defer_artifacts: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not defer_artifacts:
            model_service.load_artifacts()

        # Инициализация метрик нулями
        for endpoint in ["/predict", "/batch_predict"]:
            for error_type in ["model_not_loaded", "empty_input", "validation_error"]:
                ERROR_COUNT.labels(
                    method="POST", endpoint=endpoint, error_type=error_type
                ).inc(0)

        yield

        await asyncio.sleep(0.1)

    application = FastAPI(
        title="Kindle Reviews API", version="1.0.0", lifespan=lifespan
    )

    # Global Exception Handler
    @application.exception_handler(Exception)
    async def _unhandled_exc_handler(request, exc):
        log.exception("Unhandled exception in API: %s", exc)
        return JSONResponse(
            status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"}
        )

    _register_middlewares(application)
    _register_routes(application)

    return application


def _register_middlewares(application: FastAPI) -> None:
    @application.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        return response

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_trace_id(req_id)
        try:
            response = await call_next(request)
        finally:
            clear_trace_id()
        response.headers["X-Request-ID"] = req_id
        return response


def _register_routes(application: FastAPI) -> None:
    @application.get("/health")
    def health():
        return {"status": "alive"}

    @application.get("/ready", response_model=HealthResponse)
    def ready(response: Response):
        if not model_service.loaded:
            response.status_code = 503
            return {
                "status": "not_ready",
                "message": "Модель не загружена",
                "model_loaded": False,
            }
        return {
            "status": "ready",
            "model_loaded": True,
            "best_model": model_service.meta.get("best_model"),
        }

    @application.get("/metrics")
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.get("/metadata", response_model=MetadataResponse)
    def get_metadata():
        return model_service.get_metadata()

    @application.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        labels, probs, warnings = _handle_prediction(
            "/predict", req.texts, req.numeric_features
        )
        return PredictResponse(labels=labels, probs=probs, warnings=warnings)

    @application.post("/batch_predict", response_model=dict[str, list[int]])
    def batch_predict(req: BatchPredictRequest):
        try:
            texts: list[str] = []
            numeric_map: dict[str, list[float]] = {}

            for item in req.data:
                texts.append(item.reviewText)
                extra_data = item.model_dump(exclude={"reviewText"})
                for k, v in extra_data.items():
                    numeric_map.setdefault(k, []).append(
                        float(v) if isinstance(v, (int, float)) else 0.0
                    )

            n = len(texts)
            for k in numeric_map:
                if len(numeric_map[k]) < n:
                    numeric_map[k].extend([0.0] * (n - len(numeric_map[k])))

            labels, _, _ = _handle_prediction("/batch_predict", texts, numeric_map)
            return {"predictions": labels}

        except ValueError as e:
            ERROR_COUNT.labels(
                method="POST", endpoint="/batch_predict", error_type="validation_error"
            ).inc()
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            log.exception("Error in /batch_predict: %s", e)
            ERROR_COUNT.labels(
                method="POST", endpoint="/batch_predict", error_type="internal_error"
            ).inc()
            raise HTTPException(status_code=500, detail=str(e)) from e

    @application.get("/")
    def root():
        return {
            "service": "Kindle Reviews Sentiment Analysis API",
            "endpoints": {
                "predict": "/predict (POST)",
                "batch_predict": "/batch_predict (POST)",
                "health": "/health (GET)",
                "ready": "/ready (GET)",
                "metrics": "/metrics (GET)",
                "metadata": "/metadata (GET)",
            },
            "docs": "/docs",
        }


def _handle_prediction(
    endpoint: str, texts: list[str], numeric_features: dict[str, list[float]] | None
) -> tuple[list[int], list[list[float]] | None, dict[str, list[str]] | None]:
    try:
        start = time.perf_counter()

        for i, text in enumerate(texts):
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Текст #{i} превышает лимит {MAX_TEXT_LENGTH}")

        labels, probs, warnings = model_service.predict(texts, numeric_features)

        PREDICTION_DURATION.observe(time.perf_counter() - start)

        model_name = model_service.meta.get("best_model", "unknown")
        PREDICTION_COUNT.labels(model_name=model_name).inc(len(texts))

        if probs:
            for p in probs:
                PREDICTION_CONFIDENCE.observe(max(p))
        for lbl in labels:
            PREDICTION_LABELS.labels(label=str(int(lbl))).inc()

        return labels, probs, warnings

    except ValueError as e:
        ERROR_COUNT.labels(
            method="POST", endpoint=endpoint, error_type="validation_error"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        ERROR_COUNT.labels(
            method="POST", endpoint=endpoint, error_type="model_not_loaded"
        ).inc()
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        ERROR_COUNT.labels(
            method="POST", endpoint=endpoint, error_type="internal_error"
        ).inc()
        log.exception("Error in %s: %s", endpoint, e)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)

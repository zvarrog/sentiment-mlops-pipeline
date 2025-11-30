"""Роутеры API: эндпоинты для предсказаний, метрик, health check."""

import time

from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from scripts.api.metrics import (
    ERROR_COUNT,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_DURATION,
    PREDICTION_LABELS,
)
from scripts.api.rate_limiter import RATE_LIMIT_STRING, limiter
from scripts.api.schemas import (
    BatchPredictRequest,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)
from scripts.logging_config import get_logger
from scripts.model_service import ModelService

log = get_logger(__name__)

router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    """Dependency для получения ModelService из lifespan state приложения."""
    return cast(ModelService, request.app.state.model_service)


@router.get("/")
def root():
    """Корневой эндпоинт с информацией о сервисе."""
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


@router.get("/health")
def health():
    """Liveness probe."""
    return {"status": "alive"}


@router.get("/ready", response_model=HealthResponse)
def ready(
    response: Response,
    service: ModelService = Depends(get_model_service),
):
    """Readiness probe — проверяет загружена ли модель."""
    if not service.model:
        response.status_code = 503
        return {
            "status": "not_ready",
            "message": "Модель не загружена",
            "model_loaded": False,
        }
    return {
        "status": "ready",
        "model_loaded": True,
        "best_model": service.model_name,
    }


@router.get("/metrics")
def metrics():
    """Prometheus-метрики."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/metadata", response_model=MetadataResponse)
def get_metadata(service: ModelService = Depends(get_model_service)):
    """Метаданные модели и feature contract."""
    return service.get_metadata()


@router.post("/predict", response_model=PredictResponse)
@limiter.limit(RATE_LIMIT_STRING)
def predict(
    request: Request,  # Требуется slowapi для rate limiting
    req: PredictRequest,
    service: ModelService = Depends(get_model_service),
):
    """Предсказание тональности для списка текстов."""
    labels, probs, warnings = _handle_prediction(
        "/predict", req.texts, req.numeric_features, service
    )
    return PredictResponse(labels=labels, probs=probs, warnings=warnings)


@router.post("/batch_predict", response_model=dict[str, list[int]])
@limiter.limit(RATE_LIMIT_STRING)
def batch_predict(
    request: Request,  # Требуется slowapi для rate limiting
    req: BatchPredictRequest,
    service: ModelService = Depends(get_model_service),
):
    """Батч-предсказание с расширенными числовыми признаками."""
    try:
        texts: list[str] = []
        numeric_map: dict[str, list[float]] = {}

        for item in req.data:
            texts.append(item.reviewText)
            item_features = item.get_numeric_features()
            for k, v in item_features.items():
                numeric_map.setdefault(k, []).append(float(v))

        n = len(texts)
        for k in numeric_map:
            if len(numeric_map[k]) != n:
                raise ValueError(f"Несогласованные длины признаков для '{k}'")

        labels, _, _ = _handle_prediction("/batch_predict", texts, numeric_map, service)
        return {"predictions": labels}

    except ValueError as e:
        ERROR_COUNT.labels(
            method="POST", endpoint="/batch_predict", error_type="validation_error"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (TypeError, KeyError, AttributeError) as e:
        log.exception("Ошибка в /batch_predict: %s", e)
        ERROR_COUNT.labels(
            method="POST", endpoint="/batch_predict", error_type="internal_error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e


def _handle_prediction(
    endpoint: str,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None,
    service: ModelService,
) -> tuple[list[int], list[list[float]] | None, dict[str, list[str]] | None]:
    """Общая логика предсказания с обновлением метрик."""
    try:
        start = time.perf_counter()

        labels, probs, warnings = service.predict(texts, numeric_features)

        PREDICTION_DURATION.observe(time.perf_counter() - start)

        model_name = service.model_name
        PREDICTION_COUNT.labels(model_name=model_name).inc(len(texts))

        if probs:
            for p in probs:
                PREDICTION_CONFIDENCE.observe(max(p))
        for lbl in labels:
            PREDICTION_LABELS.labels(label=str(int(lbl))).inc()

        return labels, probs, warnings

    except ValueError as e:
        ERROR_COUNT.labels(method="POST", endpoint=endpoint, error_type="validation_error").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        ERROR_COUNT.labels(method="POST", endpoint=endpoint, error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (TypeError, KeyError, AttributeError) as e:
        ERROR_COUNT.labels(method="POST", endpoint=endpoint, error_type="internal_error").inc()
        log.exception("Ошибка в %s: %s", endpoint, e)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

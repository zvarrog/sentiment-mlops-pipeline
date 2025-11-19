from __future__ import annotations

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
from pydantic import BaseModel, Field

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

_artifacts_loaded = False


def signal_handler(signum, frame):
    log.info("Получен сигнал %s, завершаем обработку", signum)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

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

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["This Kindle is amazing!", "Battery life is poor."],
                "numeric_features": None,
            }
        }


class PredictResponse(BaseModel):
    labels: list[int]
    probs: list[list[float]] | None = None
    warnings: dict[str, list[str]] | None = None


class MetadataResponse(BaseModel):
    """Ответ с метаданными модели, контрактом признаков и статусом здоровья."""

    model_info: dict[str, Any]
    feature_contract: dict[str, Any]
    health: dict[str, Any]


def create_app(defer_artifacts: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _artifacts_loaded

        if not defer_artifacts and not _artifacts_loaded:
            _load_artifacts(app)
            _artifacts_loaded = True

        for endpoint in ["/predict"]:
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

    try:
        from fastapi.responses import JSONResponse as _JSONResponse

        @application.exception_handler(Exception)
        async def _unhandled_exc_handler(request, exc):
            log.exception("Unhandled exception in API: %s", exc)
            return _JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
    except Exception:
        pass

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
        req_id = request.headers.get("X-Request-ID") or f"req-{id(request)}"
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

    @application.get("/ready")
    def ready(response: Response):
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
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.get("/metadata", response_model=MetadataResponse)
    def get_metadata():
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

    @application.post("/predict")
    def predict(request: Request, req: PredictRequest):
        try:
            start = time.perf_counter()
            log.info("/predict called; MODEL=%s, FEATURE_CONTRACT=%s", type(getattr(application.state, 'MODEL', None)), getattr(application.state, 'FEATURE_CONTRACT', None))
            _ensure_artifacts_loaded(application)
            labels, probs, warnings = _validate_and_predict(
                application, "/predict", req.texts, req.numeric_features
            )
            PREDICTION_DURATION.observe(time.perf_counter() - start)

            return {
                "labels": [int(x) for x in labels],
                "probs": probs,
                "warnings": warnings,
            }
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as e:
            ERROR_COUNT.labels(
                method="POST", endpoint="/predict", error_type="bad_request"
            ).inc()
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            ERROR_COUNT.labels(method="POST", endpoint="/predict", error_type="internal_error").inc()
            log.exception("Error in /predict: %s", e)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @application.post("/batch_predict")
    def batch_predict(request: Request, payload: dict[str, Any]):
        """Пакетное предсказание. Ожидает {"data": [{"reviewText": str, ...}, ...]}"""
        try:
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                raise HTTPException(status_code=400, detail="Поле 'data' должно быть списком")
            if len(data) == 0:
                raise HTTPException(status_code=400, detail="Пустой список 'data'")

            texts: list[str] = []
            numeric_map: dict[str, list[float]] = {}
            for row in data:
                if not isinstance(row, dict) or "reviewText" not in row:
                    raise HTTPException(status_code=400, detail="Каждый элемент 'data' должен содержать 'reviewText'")
                texts.append(str(row.get("reviewText", "")))
                for k, v in row.items():
                    if k == "reviewText":
                        continue
                    # Только числовые признаки
                    if isinstance(v, (int, float)):
                        numeric_map.setdefault(k, []).append(float(v))
                    else:
                        # Выравниваем длину, заполняя нулями для нечисловых значений
                        numeric_map.setdefault(k, []).append(0.0)

            start = time.perf_counter()
            log.info("/batch_predict called; MODEL=%s, FEATURE_CONTRACT=%s", type(getattr(application.state, 'MODEL', None)), getattr(application.state, 'FEATURE_CONTRACT', None))
            _ensure_artifacts_loaded(application)
            labels, _, _ = _validate_and_predict(application, "/batch_predict", texts, numeric_map)
            PREDICTION_DURATION.observe(time.perf_counter() - start)
            return {"predictions": [int(x) for x in labels]}

        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as e:
            ERROR_COUNT.labels(method="POST", endpoint="/batch_predict", error_type="validation_error").inc()
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            ERROR_COUNT.labels(method="POST", endpoint="/batch_predict", error_type="internal_error").inc()
            log.exception("Error in /batch_predict: %s", e)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @application.get("/")
    def root():
        return {
            "service": "Kindle Reviews Sentiment Analysis API",
            "version": "1.0.0",
            "endpoints": {
                "predict": "/predict (POST)",
                "health": "/health (GET)",
                "metrics": "/metrics (GET)",
            },
            "docs": "/docs",
        }


def _validate_input(
    texts: list[str],
    numeric_features: dict[str, list[float]] | None,
    expected_numeric_cols: list[str],
) -> list[str]:
    warnings: list[str] = []
    n = len(texts)
    if numeric_features:
        for col, vals in numeric_features.items():
            if col not in expected_numeric_cols:
                warnings.append(f"Игнорирован {col}: неизвестный признак")
            elif not isinstance(vals, list) or len(vals) != n:
                warnings.append(f"Игнорирован {col}: длина {len(vals) if isinstance(vals, list) else 'n/a'} != {n}")
    return warnings


def _extract_features(
    application: FastAPI,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None,
) -> tuple[pd.DataFrame, list[str]]:
    feature_contract = getattr(application.state, "FEATURE_CONTRACT", None)
    if not feature_contract:
        raise RuntimeError(
            "Feature contract (артефакты) не загружены — невозможно определить список признаков."
        )
    from .feature_engineering import transform_features

    expected_cols = feature_contract.expected_numeric_columns
    df, ignored = transform_features(texts, numeric_features, list(expected_cols))
    return df, ignored


def _fill_missing(df: pd.DataFrame, expected_numeric_cols: list[str]) -> pd.DataFrame:
    for col in expected_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _validate_and_predict(
    application: FastAPI,
    endpoint: str,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None = None,
) -> tuple[list[int], list[list[float]] | None, dict[str, list[str]] | None]:
    model = getattr(application.state, "MODEL", None)
    if model is None:
        ERROR_COUNT.labels(
            method="POST", endpoint=endpoint, error_type="model_not_loaded"
        ).inc()
        raise HTTPException(status_code=500, detail="Модель не загружена")

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

    feature_contract = getattr(application.state, "FEATURE_CONTRACT", None)
    expected_cols = (
        feature_contract.expected_numeric_columns if feature_contract else []
    )
    input_warnings = _validate_input(texts, numeric_features, list(expected_cols))

    labels, probs, ignored = _predict_with_model(
        application, model, texts, numeric_features
    )

    if probs is not None:
        for p in probs:
            PREDICTION_CONFIDENCE.observe(max(p))
    for lbl in labels:
        PREDICTION_LABELS.labels(label=str(int(lbl))).inc()

    warnings_dict: dict[str, list[str]] | None = None
    if input_warnings or ignored:
        warnings_dict = {}
        if input_warnings:
            warnings_dict["input_issues"] = input_warnings
        if ignored:
            warnings_dict["ignored_features"] = ignored

    return labels, probs, warnings_dict


def _load_artifacts(application: FastAPI) -> None:
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


def _ensure_artifacts_loaded(application: FastAPI) -> None:
    if getattr(application.state, "MODEL", None) is None and BEST_MODEL_PATH.exists():
        try:
            _load_artifacts(application)
        except Exception as e:
            log.exception("Не удалось лениво загрузить артефакты: %s", e)
            raise

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


def _build_dataframe(
    application: FastAPI,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.DataFrame({"reviewText": texts})
    ignored_features = []

    feature_contract = getattr(application.state, "FEATURE_CONTRACT", None)
    if not feature_contract:
        raise RuntimeError(
            "Feature contract (артефакты) не загружены — невозможно определить список признаков. Проверьте наличие артефактов модели."
        )
    numeric_cols = feature_contract.expected_numeric_columns

    if numeric_features:
        for col, values in numeric_features.items():
            if col not in numeric_cols:
                ignored_features.append(f"{col} (неизвестный признак)")
            elif len(values) != len(texts):
                ignored_features.append(f"{col} (длина {len(values)} != {len(texts)})")
            else:
                df[col] = values

    from .feature_engineering import transform_features

    df, ignored_calc = transform_features(
        texts=df["reviewText"].tolist(),
        numeric_features=numeric_features,
        expected_numeric_cols=list(numeric_cols),
    )

    missing_features = [col for col in numeric_cols if col not in df.columns]
    for col in missing_features:
        df[col] = 0.0

    ignored_features.extend(ignored_calc)

    return df, ignored_features


def _predict_with_model(
    application: FastAPI,
    model: Any,
    texts: list[str],
    numeric_features: dict[str, list[float]] | None = None,
) -> tuple[list[int], list[list[float]] | None, list[str] | None]:
    is_text_only = bool(getattr(model, "text_only", False))
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

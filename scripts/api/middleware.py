"""Middleware для FastAPI: timeout, request tracing, error handling.

Rate limiting вынесен в отдельный модуль rate_limiter.py и применяется
через декораторы slowapi на эндпоинтах.
"""

import asyncio
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from scripts.api.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_DURATION
from scripts.api.rate_limiter import limiter
from scripts.config import API_REQUEST_TIMEOUT_SEC
from scripts.logging_config import clear_trace_id, get_logger, set_trace_id

log = get_logger(__name__)


def register_middlewares(app: FastAPI) -> None:
    """Регистрирует все middleware и обработчики на приложении."""
    # Регистрация slowapi limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Сбор метрик Prometheus для всех запросов."""
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

    @app.middleware("http")
    async def timeout_middleware(request: Request, call_next):
        """Таймаут для всех запросов."""
        try:
            return await asyncio.wait_for(call_next(request), timeout=API_REQUEST_TIMEOUT_SEC)
        except TimeoutError:
            ERROR_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                error_type="timeout",
            ).inc()
            return JSONResponse(
                status_code=504,
                content={"detail": "Таймаут запроса"},
            )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Добавляет X-Request-ID для трассировки."""
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_trace_id(req_id)
        try:
            response = await call_next(request)
        finally:
            clear_trace_id()
        response.headers["X-Request-ID"] = req_id
        return response

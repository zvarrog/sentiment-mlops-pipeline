"""Фабрика приложения FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from scripts.api.metrics import init_metrics_labels
from scripts.api.middleware import register_middlewares
from scripts.api.routers import router
from scripts.config import API_HOST, API_PORT
from scripts.logging_config import get_logger
from scripts.model_service import ModelService

log = get_logger(__name__)


def create_app(defer_artifacts: bool = False) -> FastAPI:
    """Создаёт и настраивает FastAPI приложение.

    Args:
        defer_artifacts: Отложить загрузку артефактов модели (для тестов).

    Returns:
        Настроенный экземпляр FastAPI.
    """
    model_service = ModelService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Сохраняем ModelService в state приложения для доступа из роутеров
        app.state.model_service = model_service

        if not defer_artifacts:
            model_service.load_artifacts()

        init_metrics_labels()
        yield

    application = FastAPI(
        title="Kindle Reviews API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Глобальный обработчик необработанных исключений."""
        log.exception("Необработанное исключение на %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}"},
        )

    register_middlewares(application)
    application.include_router(router)

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)

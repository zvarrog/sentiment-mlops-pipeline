"""API модуль для Kindle Reviews Sentiment Analysis.

Структура:
- schemas.py: Pydantic модели запросов/ответов
- middleware.py: Middleware (rate limiting, timeout, request_id)
- metrics.py: Prometheus метрики
- routers.py: Эндпоинты API
- app.py: Фабрика приложения FastAPI
"""

from scripts.api.app import create_app

__all__ = ["create_app"]

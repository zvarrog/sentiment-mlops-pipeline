"""Rate limiter с поддержкой Redis для multi-worker deployment.

Использует slowapi с автоматическим выбором backend:
- Redis если задан REDIS_URL (production)
- In-memory если Redis недоступен (development/single-worker)
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from scripts.config import API_RATE_LIMIT_PER_MIN, REDIS_URL
from scripts.logging_config import get_logger

log = get_logger(__name__)


def _create_limiter() -> Limiter:
    """Создаёт Limiter с оптимальным storage backend."""
    storage_uri: str | None = None

    if REDIS_URL:
        try:
            # Проверяем доступность Redis
            import redis

            client = redis.from_url(REDIS_URL, socket_connect_timeout=2)
            client.ping()
            storage_uri = REDIS_URL
            log.info("Rate limiter: используется Redis backend")
        except (ImportError, redis.ConnectionError, redis.TimeoutError) as e:
            log.warning(
                "Redis недоступен (%s), rate limiter работает в in-memory режиме. "
                "Для multi-worker deployment настройте Redis.",
                e,
            )
    else:
        log.info(
            "REDIS_URL не задан, rate limiter работает в in-memory режиме. "
            "Для multi-worker deployment задайте REDIS_URL."
        )

    return Limiter(
        key_func=get_remote_address,
        default_limits=[f"{API_RATE_LIMIT_PER_MIN}/minute"],
        storage_uri=storage_uri,
        strategy="fixed-window",
    )


limiter = _create_limiter()

# Строка лимита для декораторов эндпоинтов
RATE_LIMIT_STRING = f"{API_RATE_LIMIT_PER_MIN}/minute"

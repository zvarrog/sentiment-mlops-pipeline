"""Утилиты для retry logic с экспоненциальным backoff."""

import functools
import time
from typing import Callable, Type

from .logging_config import setup_auto_logging

log = setup_auto_logging()


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """Декоратор для retry с экспоненциальным backoff.

    Args:
        max_attempts: максимальное количество попыток
        initial_delay: начальная задержка в секундах
        backoff_factor: множитель для увеличения задержки
        exceptions: типы исключений для retry
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        log.error(
                            "Все попытки исчерпаны для %s: %s",
                            func.__name__,
                            str(e),
                        )
                        raise

                    log.warning(
                        "Попытка %d/%d для %s не удалась: %s. Retry через %.1f сек",
                        attempt,
                        max_attempts,
                        func.__name__,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

            raise last_exception

        return wrapper

    return decorator

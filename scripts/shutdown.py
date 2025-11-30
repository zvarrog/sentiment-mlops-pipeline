"""Унифицированный модуль для graceful shutdown.

Избегаем дублирования signal handler в разных модулях.
"""

import signal
import sys
import threading
from collections.abc import Callable

from scripts.logging_config import get_logger

log = get_logger(__name__)

# Глобальный флаг прерывания (thread-safe)
_shutdown_event = threading.Event()


def is_shutdown_requested() -> bool:
    """Проверяет, запрошено ли завершение работы."""
    return _shutdown_event.is_set()


def request_shutdown() -> None:
    """Устанавливает флаг завершения работы."""
    _shutdown_event.set()


def _default_handler(signum: int, frame) -> None:
    """Обработчик сигнала по умолчанию."""
    sig_name = signal.Signals(signum).name
    log.warning("Получен сигнал %s (%d), завершаем работу...", sig_name, signum)
    request_shutdown()


def register_shutdown_handlers(
    custom_handler: Callable[[int, object], None] | None = None,
    exit_on_signal: bool = False,
) -> None:
    """Регистрирует обработчики сигналов SIGTERM и SIGINT.

    Args:
        custom_handler: Опциональный кастомный обработчик. Если None, используется дефолтный.
        exit_on_signal: Если True, вызывает sys.exit(0) после обработки сигнала.
    """

    def handler(signum: int, frame) -> None:
        if custom_handler:
            custom_handler(signum, frame)
        else:
            _default_handler(signum, frame)

        if exit_on_signal:
            sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    log.debug("Зарегистрированы обработчики сигналов SIGTERM, SIGINT")

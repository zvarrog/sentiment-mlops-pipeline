"""Централизованная конфигурация логирования для всех компонентов проекта."""

import contextvars
import json
import logging
import logging.config
from contextlib import contextmanager
from pathlib import Path

# Контекстный trace_id для корреляции событий
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)


class TraceIdFilter(logging.Filter):
    """Фильтр, добавляющий trace_id в запись лога."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """JSON-форматтер без внешних зависимостей."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None),
            "module": record.module,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


@contextmanager
def trace_context(trace_id: str):
    """Контекст-менеджер для установки trace_id на время блока."""
    token = _trace_id_var.set(trace_id)
    try:
        yield
    finally:
        _trace_id_var.reset(token)


def get_logging_config(
    level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "text",
) -> dict:
    """Создает конфигурацию логирования."""
    base_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter_name = "default" if log_format != "json" else "json"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": base_format, "datefmt": "%Y-%m-%d %H:%M:%S"},
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "scripts.logging_config.JsonFormatter",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "filters": {
            "trace": {"()": "scripts.logging_config.TraceIdFilter"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": formatter_name,
                "stream": "ext://sys.stdout",
                "filters": ["trace"],
            }
        },
        "root": {"level": level, "handlers": ["console"]},
        "loggers": {
            # Наши компоненты
            "kindle": {"level": level, "handlers": ["console"], "propagate": False},
            "scripts": {"level": level, "handlers": ["console"], "propagate": False},
            # Подавляем избыточные логи внешних библиотек
            "urllib3": {"level": "WARNING"},
            "pyspark": {"level": "WARNING"},
            "py4j": {"level": "WARNING"},
            "mlflow": {"level": "ERROR"},  # Убираем INFO/WARNING от MLflow
            "optuna": {"level": "ERROR"},  # Убираем INFO/WARNING от Optuna
            "git": {"level": "ERROR"},  # Убираем предупреждения git
        },
    }

    # Добавляем файловый хендлер если нужен
    if log_file:
        # Создаем директорию для логов если нужно
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": formatter_name if log_format == "json" else "detailed",
            "filename": str(log_path),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
            "filters": ["trace"],
        }

        # Добавляем файловый хендлер к root и основным логгерам
        config["root"]["handlers"].append("file")
        config["loggers"]["kindle"]["handlers"].append("file")
        config["loggers"]["scripts"]["handlers"].append("file")

    return config


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "text",
) -> logging.Logger:
    """Настраивает логирование для проекта."""
    config = get_logging_config(level=level, log_file=log_file, log_format=log_format)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.config.dictConfig(config)

    logger = logging.getLogger("kindle")
    logger.setLevel(getattr(logging, level.upper()))

    return logger


def get_logger(name: str = "kindle") -> logging.Logger:
    """Получить логгер с автоматической настройкой."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()
    return logging.getLogger(name)

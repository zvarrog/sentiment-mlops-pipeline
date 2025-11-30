"""Централизованная конфигурация логирования и matplotlib backend."""

import contextvars
import json
import logging
import logging.config
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
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
    token = _trace_id_var.set(trace_id)
    try:
        yield
    finally:
        _trace_id_var.reset(token)


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(trace_id)


def get_trace_id() -> str | None:
    return _trace_id_var.get()


def clear_trace_id() -> None:
    _trace_id_var.set(None)


def get_logging_config(
    level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "text",
) -> dict[str, object]:
    base_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter_name = "default" if log_format != "json" else "json"

    handlers: dict[str, dict[str, object]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": formatter_name,
            "stream": "ext://sys.stdout",
            "filters": ["trace"],
        }
    }

    root_handlers: list[str] = ["console"]
    kindle_handlers: list[str] = ["console"]
    scripts_handlers: list[str] = ["console"]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": formatter_name if log_format == "json" else "detailed",
            "filename": str(log_path),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
            "filters": ["trace"],
        }
        root_handlers.append("file")
        kindle_handlers.append("file")
        scripts_handlers.append("file")

    config: dict[str, object] = {
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
        "handlers": handlers,
        "root": {"level": level, "handlers": root_handlers},
        "loggers": {
            "kindle": {"level": level, "handlers": kindle_handlers, "propagate": False},
            "scripts": {"level": level, "handlers": scripts_handlers, "propagate": False},
            "urllib3": {"level": "WARNING"},
            "pyspark": {"level": "WARNING"},
            "py4j": {"level": "WARNING"},
            "mlflow": {"level": "ERROR"},
            "optuna": {"level": "ERROR"},
            "git": {"level": "ERROR"},
        },
    }

    return config


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "text",
) -> logging.Logger:
    config = get_logging_config(level=level, log_file=log_file, log_format=log_format)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.config.dictConfig(config)

    logger = logging.getLogger("kindle")
    logger.setLevel(getattr(logging, level.upper()))

    return logger


def get_logger(name: str = "kindle") -> logging.Logger:
    """Получает logger с автоматической инициализацией."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()
    return logging.getLogger(name)

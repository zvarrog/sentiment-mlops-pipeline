"""Prometheus-метрики для API."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Общее количество API запросов",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "Длительность обработки API запроса",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Общее количество предсказаний",
    ["model_name"],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Максимальная вероятность предсказания на элемент",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0),
)

PREDICTION_DURATION = Histogram(
    "prediction_duration",
    "Длительность выполнения предсказания",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTION_LABELS = Counter(
    "prediction_labels_total",
    "Распределение предсказанных меток",
    ["label"],
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Общее количество ошибок API",
    ["method", "endpoint", "error_type"],
)


def init_metrics_labels() -> None:
    """Инициализация меток метрик нулями для стабильного экспорта."""
    for endpoint in ["/predict", "/batch_predict"]:
        for error_type in ["model_not_loaded", "empty_input", "validation_error"]:
            ERROR_COUNT.labels(
                method="POST", endpoint=endpoint, error_type=error_type
            ).inc(0)

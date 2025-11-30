"""End-to-end тесты для проверки запущенных Docker сервисов."""

import pytest


class TestDockerServices:
    """Smoke tests для Docker сервисов.

    Эти тесты проверяют доступность сервисов, запущенных в Docker контейнерах.
    Требуют запущенного docker-compose окружения.
    """

    @pytest.mark.integration
    def test_api_service_responds(self):
        """Проверка доступности FastAPI service."""
        import os
        import requests

        # В контейнере используем docker network endpoints
        api_url = os.getenv("API_URL") or "http://api:8000"

        try:
            response = requests.get(f"{api_url}/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data or "detail" in data
        except requests.RequestException as e:
            pytest.skip(f"API service недоступен ({api_url}): {e}")

    @pytest.mark.integration
    def test_mlflow_ui_responds(self):
        """Проверка доступности MLflow UI."""
        import os
        import requests

        mlflow_url = os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:5000"

        try:
            response = requests.get(mlflow_url, timeout=5)
            assert response.status_code == 200
        except requests.RequestException as e:
            pytest.skip(f"MLflow UI недоступен ({mlflow_url}): {e}")

    @pytest.mark.integration
    def test_prometheus_metrics_endpoint(self):
        """Проверка доступности Prometheus metrics."""
        import os
        import requests

        api_url = os.getenv("API_URL") or "http://api:8000"

        try:
            response = requests.get(f"{api_url}/metrics", timeout=5)
            assert response.status_code == 200
            # Prometheus metrics должны содержать TYPE/HELP директивы
            assert "# TYPE" in response.text or "# HELP" in response.text
        except requests.RequestException as e:
            pytest.skip(f"API service недоступен ({api_url}): {e}")



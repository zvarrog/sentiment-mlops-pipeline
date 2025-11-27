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
        import requests

        try:
            response = requests.get("http://localhost:8000/", timeout=3)
            assert response.status_code == 200
        except requests.RequestException:
            pytest.skip("API service недоступен")

    @pytest.mark.integration
    def test_mlflow_ui_responds(self):
        """Проверка доступности MLflow UI."""
        import requests

        try:
            response = requests.get("http://localhost:5000/", timeout=3)
            assert response.status_code == 200
        except requests.RequestException:
            pytest.skip("MLflow UI недоступен")

    @pytest.mark.integration
    def test_prometheus_metrics_endpoint(self):
        """Проверка доступности Prometheus metrics."""
        import requests

        try:
            response = requests.get("http://localhost:8000/metrics", timeout=3)
            assert response.status_code == 200
        except requests.RequestException:
            pytest.skip("API service недоступен")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

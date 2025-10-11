"""E2E/интеграционные смоки.

- Проверка API вживую (lifespan) через TestClient
- Проверка drift_monitor генерации отчёта
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_api_smoke_lifespan():
    # Локальный запуск приложения (без Docker) с lifespan
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.api_service import app

    with TestClient(app) as client:
        # health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

        # predict с простым текстом, без агрегатов
        r = client.post(
            "/predict", json={"texts": ["kindle is great", "bad experience"]}
        )
        assert r.status_code == 200
        data = r.json()
        assert "labels" in data and isinstance(data["labels"], list)


def test_drift_monitor_report(tmp_path: Path):
    # Создаём копию тестовых данных и запускаем drift_monitor на них
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.drift_monitor import run_drift_monitor
    from scripts.settings import PROCESSED_DATA_DIR

    test_parquet = Path(PROCESSED_DATA_DIR) / "test.parquet"
    assert test_parquet.exists(), "Ожидается наличие test.parquet в data/processed"

    # Запускаем мониторинг дрейфа и проверяем, что отчёт сохранён в ./drift
    report = run_drift_monitor(str(test_parquet), threshold=0.2, save=True)
    assert isinstance(report, list)
    report_path = Path("drift") / "drift_report.json"
    assert report_path.exists(), "Ожидается drift_report.json после drift_monitor"

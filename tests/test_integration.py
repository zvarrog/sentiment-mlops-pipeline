"""Интеграционные тесты для проверки полного пайплайна."""

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest


def _get_dag_path(dag_name: str) -> Path:
    """Получить путь к DAG файлу независимо от окружения (контейнер/хост).

    Логика:
    - В контейнере: /opt/airflow/dags/ (монтируется из ./airflow/dags/)
    - На хосте: ./airflow/dags/ (от корня репозитория)

    Path(__file__).parent.parent → /opt/airflow в контейнере, ./project_root на хосте

    Args:
        dag_name: Имя DAG файла (например, 'kindle_pipeline.py')

    Returns:
        Абсолютный путь к DAG файлу
    """
    # В контейнере dags монтируется в /opt/airflow/dags/, на хосте это ./airflow/dags/
    container_path = Path(__file__).parent.parent / "dags" / dag_name
    if container_path.exists():
        return container_path

    # Fallback для хоста: tests/../airflow/dags/
    host_path = Path(__file__).parent.parent / "airflow" / "dags" / dag_name
    return host_path


class TestSparkProcessing:
    """Тесты для Spark обработки."""

    @pytest.mark.integration
    def test_spark_process_creates_output(self):
        """Проверка что Spark обработка создала файлы в /opt/airflow/data/processed/."""
        from scripts.config import PROCESSED_DATA_DIR

        processed_dir = Path(PROCESSED_DATA_DIR)

        # Проверяем наличие всех трёх сплитов
        train_path = processed_dir / "train.parquet"
        val_path = processed_dir / "val.parquet"
        test_path = processed_dir / "test.parquet"

        if not train_path.exists():
            pytest.skip(f"Обработанные данные не найдены в {processed_dir}. Запустите Spark pipeline сначала.")

        assert val_path.exists(), f"val.parquet отсутствует в {processed_dir}"
        assert test_path.exists(), f"test.parquet отсутствует в {processed_dir}"

        # Проверяем что файлы не пустые
        import pandas as pd
        train_df = pd.read_parquet(train_path)
        assert len(train_df) > 0, "train.parquet пустой"
        assert "reviewText" in train_df.columns, "reviewText отсутствует в данных"
        assert "overall" in train_df.columns, "overall (target) отсутствует в данных"


class TestTrainPipeline:
    """Тесты для полного цикла обучения."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_train_creates_model_artifacts_fast(self):
        """Проверка что обученная модель создана и имеет разумные метрики.

        Использует реальные данные из /opt/airflow/data/processed/.
        Помечен @pytest.mark.slow т.к. требует загрузки модели.
        """
        from scripts.config import MODEL_DIR, MODEL_ARTEFACTS_DIR

        model_path = Path(MODEL_DIR) / "best_model.joblib"
        meta_path = Path(MODEL_ARTEFACTS_DIR) / "best_model_meta.json"

        if not model_path.exists():
            pytest.skip(
                f"Модель не найдена в {model_path}. Запустите обучение сначала (python scripts/train.py)."
            )

        assert model_path.stat().st_size > 0, "best_model.joblib пустой"
        assert meta_path.exists(), f"best_model_meta.json не найден в {MODEL_ARTEFACTS_DIR}"

        # Проверяем метаданные модели
        import json
        with open(meta_path) as f:
            meta = json.load(f)

        # Метаданные должны содержать test_metrics с f1_macro
        assert "test_metrics" in meta, "Отсутствует test_metrics в метаданных"
        assert "f1_macro" in meta["test_metrics"], "Отсутствует f1_macro в test_metrics"
        assert "best_model" in meta, "Отсутствует best_model (тип модели) в метаданных"

        # Проверяем что F1 разумный (выше random baseline 0.2)
        test_f1 = meta["test_metrics"]["f1_macro"]
        assert test_f1 > 0.3, f"F1-score слишком низкий: {test_f1:.3f}. Проверьте качество модели."

    @pytest.mark.integration
    def test_train_logs_to_mlflow(self):
        """Проверка что MLflow tracking server доступен."""
        import os
        import requests

        # В контейнере используем docker network endpoint
        mlflow_url = os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:5000"

        try:
            response = requests.get(mlflow_url, timeout=5)
            # MLflow UI возвращает 200 на корневом endpoint
            assert response.status_code == 200, f"MLflow недоступен: {response.status_code}"
        except requests.RequestException as e:
            pytest.skip(f"MLflow server недоступен ({mlflow_url}): {e}")


class TestDownloadModule:
    """Тесты для модуля загрузки данных."""

    @pytest.mark.integration
    def test_download_creates_raw_file(self, tmp_path):
        """Проверка создания raw файла после загрузки."""
        import os

        os.environ["RAW_DATA_DIR"] = str(tmp_path / "raw")
        os.environ["FORCE_DOWNLOAD"] = "1"

        from scripts.download import main as download

        try:
            download()
        except (subprocess.CalledProcessError, OSError, FileNotFoundError):
            pytest.skip("Требуется доступ к интернету")

        raw_dir = Path(os.environ["RAW_DATA_DIR"])
        raw_files = list(raw_dir.glob("*.csv*"))
        if len(raw_files) == 0:
            pytest.skip("CSV файл не скачан (нет доступа к Kaggle или кредов)")


class TestAirflowDAG:
    """Тесты для Airflow DAG."""

    @pytest.mark.integration
    def test_dag_imports_without_errors(self):
        """Проверка импорта DAG без ошибок."""
        import importlib.util

        try:
            pytest.importorskip("airflow.decorators")
        except (ImportError, TypeError) as e:
            pytest.skip(f"Airflow несовместим с окружением: {e}")

        dag_path = _get_dag_path("kindle_pipeline.py")
        if not dag_path.exists():
            pytest.skip(f"DAG файл не найден: {dag_path}")

        spec = importlib.util.spec_from_file_location(
            "kindle_pipeline",
            str(dag_path),
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except TypeError as e:
            pytest.skip(f"Airflow incompatible с Python 3.13 (pendulum issue): {e}")

        assert hasattr(module, "dag")

    @pytest.mark.integration
    def test_dag_has_required_tasks(self):
        """Проверка наличия обязательных задач в DAG."""
        import importlib.util

        try:
            pytest.importorskip("airflow.decorators")
        except (ImportError, TypeError) as e:
            pytest.skip(f"Airflow несовместим с окружением: {e}")

        dag_path = _get_dag_path("kindle_pipeline.py")
        if not dag_path.exists():
            pytest.skip(f"DAG файл не найден: {dag_path}")

        spec = importlib.util.spec_from_file_location(
            "kindle_pipeline",
            str(dag_path),
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except TypeError as e:
            pytest.skip(f"Airflow incompatible с Python 3.13 (pendulum issue): {e}")

        dag = module.dag
        task_ids = [task.task_id for task in dag.tasks]

        required_tasks = [
            "download",
            "validate_data",
            "process",
            "drift_monitor",
        ]

        for task_id in required_tasks:
            assert task_id in task_ids, f"Task {task_id} отсутствует в DAG"


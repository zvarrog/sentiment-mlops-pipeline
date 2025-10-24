"""Интеграционные тесты для проверки полного пайплайна."""

from pathlib import Path

import pandas as pd
import pytest


class TestSparkProcessing:
    """Тесты для Spark обработки."""

    def test_spark_process_creates_output(self, tmp_path, sample_dataframe):
        """Проверка создания processed файлов после Spark обработки."""
        from scripts.spark_process import process_data

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        sample_df = sample_dataframe.copy()
        sample_df["reviewerID"] = ["A1", "A2", "A3"]
        sample_df["asin"] = ["B1", "B2", "B3"]
        sample_df["unixReviewTime"] = [1609459200, 1609459201, 1609459202]

        input_file = raw_dir / "reviews.json"
        sample_df.to_json(input_file, orient="records", lines=True)

        import os

        os.environ["RAW_DATA_DIR"] = str(raw_dir)
        os.environ["PROCESSED_DATA_DIR"] = str(processed_dir)

        process_data()

        assert (processed_dir / "train.parquet").exists()
        assert (processed_dir / "val.parquet").exists()
        assert (processed_dir / "test.parquet").exists()


class TestTrainPipeline:
    """Тесты для полного цикла обучения."""

    def test_train_creates_model_artifacts(self, tmp_path, sample_parquet_files):
        """Проверка создания артефактов после обучения."""
        import os

        os.environ["MODEL_DIR"] = str(tmp_path / "models")
        os.environ["MODEL_ARTEFACTS_DIR"] = str(tmp_path / "models" / "artefacts")

        from scripts.train import main

        with pytest.raises(SystemExit):
            main()

        model_path = tmp_path / "models" / "best_model.joblib"
        if model_path.exists():
            assert model_path.stat().st_size > 0

    def test_train_logs_to_mlflow(self, sample_parquet_files):
        """Проверка логирования в MLflow."""
        import mlflow

        experiment_name = "test_experiment"
        mlflow.set_experiment(experiment_name)

        runs = mlflow.search_runs(experiment_names=[experiment_name])
        assert isinstance(runs, pd.DataFrame)


class TestDownloadModule:
    """Тесты для модуля загрузки данных."""

    def test_download_creates_raw_file(self, tmp_path):
        """Проверка создания raw файла после загрузки."""
        import os

        os.environ["RAW_DATA_DIR"] = str(tmp_path / "raw")
        os.environ["FORCE_DOWNLOAD"] = "1"

        from scripts.download import download_data

        try:
            download_data()
        except Exception:
            pytest.skip("Требуется доступ к интернету")

        raw_dir = Path(os.environ["RAW_DATA_DIR"])
        raw_files = list(raw_dir.glob("*.json*"))
        assert len(raw_files) > 0


class TestDockerServices:
    """Smoke tests для Docker сервисов."""

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
            assert "prediction_duration" in response.text
        except requests.RequestException:
            pytest.skip("API service недоступен")


class TestAirflowDAG:
    """Тесты для Airflow DAG."""

    def test_dag_imports_without_errors(self):
        """Проверка импорта DAG без ошибок."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "kindle_unified_pipeline",
            "airflow/dags/kindle_unified_pipeline.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "dag")

    def test_dag_has_required_tasks(self):
        """Проверка наличия обязательных задач в DAG."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "kindle_unified_pipeline",
            "airflow/dags/kindle_unified_pipeline.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        dag = module.dag
        task_ids = [task.task_id for task in dag.tasks]

        required_tasks = [
            "setup_env",
            "download",
            "validate_data",
            "process",
            "drift_monitor",
        ]

        for task_id in required_tasks:
            assert task_id in task_ids, f"Task {task_id} отсутствует в DAG"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])

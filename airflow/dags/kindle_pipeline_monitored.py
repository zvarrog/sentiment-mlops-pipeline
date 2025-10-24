"""DAG с логированием метрик производительности задач в PostgreSQL."""

from datetime import datetime

try:
    from airflow.models import TaskInstance
    from airflow.operators.python import PythonOperator
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    from airflow import DAG
except ImportError:

    class _Dummy:
        def __init__(self, *_, **__):
            self.task_id = "dummy"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other

    class DAG(_Dummy):
        pass

    class PythonOperator(_Dummy):
        pass

    class TaskInstance:
        pass

    class PostgresHook:
        def __init__(self, *_, **__):
            pass


_DOC = "Пайплайн с мониторингом производительности задач через PostgreSQL."

default_args = {
    "start_date": datetime(2025, 1, 1),
}


def log_task_duration(**context):
    ti: TaskInstance = context["task_instance"]

    if not ti.duration:
        return

    duration = ti.duration
    status = ti.state  # success, failed, skipped, etc.

    try:
        pg_hook = PostgresHook(postgres_conn_id="metrics_db")
        pg_hook.run(
            """
            INSERT INTO task_metrics (dag_id, task_id, execution_date, duration_sec, status)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (dag_id, task_id, execution_date)
            DO UPDATE SET duration_sec = EXCLUDED.duration_sec, status = EXCLUDED.status
            """,
            parameters=(ti.dag_id, ti.task_id, ti.execution_date, duration, status),
        )
    except Exception as e:
        # Не прерываем пайплайн если не удалось залогировать метрики
        print(f"Не удалось залогировать метрику задачи: {e}")


def log_task_failure(**context):
    ti: TaskInstance = context["task_instance"]

    try:
        duration = ti.duration if ti.duration else 0

        pg_hook = PostgresHook(postgres_conn_id="metrics_db")
        pg_hook.run(
            """
            INSERT INTO task_metrics (dag_id, task_id, execution_date, duration_sec, status)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (dag_id, task_id, execution_date)
            DO UPDATE SET duration_sec = EXCLUDED.duration_sec, status = EXCLUDED.status
            """,
            parameters=(ti.dag_id, ti.task_id, ti.execution_date, duration, "failed"),
        )
    except Exception as e:
        print(f"Не удалось залогировать ошибку задачи: {e}")


def _setup_env(**context):
    import os
    from pathlib import Path

    base = os.environ.get("AIRFLOW_HOME", "/opt/airflow")

    def _abs(p: str) -> str:
        pp = Path(p)
        return str(pp) if pp.is_absolute() else str((Path(base) / pp).resolve())

    os.environ["RAW_DATA_DIR"] = _abs("data/raw")
    os.environ["PROCESSED_DATA_DIR"] = _abs("data/processed")
    os.environ["MODEL_DIR"] = _abs("artefacts")
    os.environ["MODEL_ARTEFACTS_DIR"] = _abs("artefacts/model_artefacts")
    os.environ["DRIFT_ARTEFACTS_DIR"] = _abs("artefacts/drift_artefacts")
    os.environ["MLFLOW_TRACKING_URI"] = "file:///opt/airflow/mlruns"


def _task_download(**context):
    _setup_env(**context)
    import os

    from scripts.download import CSV_PATH
    from scripts.download import main as download_main
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Начинаем загрузку данных")

    force = os.environ.get("FORCE_DOWNLOAD", "0") == "1"
    if not force and CSV_PATH.exists():
        log.info("CSV уже существует — пропуск загрузки")
        return str(CSV_PATH.resolve())

    csv_abs = download_main()
    log.info("Данные загружены: %s", csv_abs)
    return str(csv_abs)


def _task_validate(**context):
    _setup_env(**context)
    from scripts.data_validation import main as validate_main
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Валидация данных")

    success = validate_main()
    if not success:
        raise ValueError("Ошибки валидации данных")

    log.info("Валидация успешна")
    return "validation_ok"


def _task_process(**context):
    _setup_env(**context)
    from scripts.logging_config import setup_auto_logging
    from scripts.spark_process import TEST_PATH, TRAIN_PATH, VAL_PATH

    log = setup_auto_logging()
    log.info("Обработка данных через Spark")

    from scripts.spark_process import main as process_main

    process_main()

    log.info("Обработка завершена")
    return {"train": str(TRAIN_PATH), "val": str(VAL_PATH), "test": str(TEST_PATH)}


def _task_train(**context):
    _setup_env(**context)
    import json
    import os

    from scripts.logging_config import setup_auto_logging
    from scripts.settings import MODEL_ARTEFACTS_DIR

    log = setup_auto_logging()
    log.info("Обучение модели")

    try:
        from airflow.models import Variable

        optuna_storage = Variable.get(
            "OPTUNA_STORAGE",
            default_var=os.getenv(
                "OPTUNA_STORAGE",
                "postgresql+psycopg2://admin:admin@postgres:5432/optuna",
            ),
        )
    except Exception:
        optuna_storage = os.getenv(
            "OPTUNA_STORAGE", "postgresql+psycopg2://admin:admin@postgres:5432/optuna"
        )

    os.environ["OPTUNA_STORAGE"] = optuna_storage
    os.environ["FORCE_TRAIN"] = "1"

    from scripts.train import run as train_run

    train_run()

    meta_path = MODEL_ARTEFACTS_DIR / "best_model_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        try:
            pg_hook = PostgresHook(postgres_conn_id="metrics_db")
            ti = context["task_instance"]

            model_name = meta.get("best_model", "unknown")
            val_f1 = meta.get("best_val_f1_macro", 0.0)
            test_metrics = meta.get("test_metrics", {})

            metrics_to_log = [
                (model_name, "val_f1_macro", val_f1, "val"),
                (
                    model_name,
                    "test_f1_macro",
                    test_metrics.get("f1_macro", 0.0),
                    "test",
                ),
                (
                    model_name,
                    "test_accuracy",
                    test_metrics.get("accuracy", 0.0),
                    "test",
                ),
            ]

            for model, metric_name, value, split in metrics_to_log:
                pg_hook.run(
                    """
                    INSERT INTO model_metrics (dag_id, execution_date, model_name, metric_name, metric_value, split)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    parameters=(
                        ti.dag_id,
                        ti.execution_date,
                        model,
                        metric_name,
                        value,
                        split,
                    ),
                )

            log.info(f"Метрики модели {model_name} залогированы в БД")

        except Exception as e:
            log.warning(f"Не удалось залогировать метрики модели: {e}")

    log.info("Обучение завершено")
    return "training_complete"


def _task_drift_monitor(**context):
    _setup_env(**context)
    import os
    from pathlib import Path

    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()

    run_dm = os.environ.get("RUN_DRIFT_MONITOR", "0") == "1"
    if not run_dm:
        log.info("Мониторинг дрейфа пропущен")
        return "drift_monitor_skipped"

    from scripts.settings import DRIFT_ARTEFACTS_DIR, PROCESSED_DATA_DIR

    test_parquet = Path(PROCESSED_DATA_DIR) / "test.parquet"
    if not test_parquet.exists():
        log.warning("Файл test.parquet не найден")
        return "no_data"

    try:
        from scripts.drift_monitor import run_drift_monitor

        log.info("Запуск мониторинга дрейфа")
        report = run_drift_monitor(
            str(test_parquet),
            threshold=0.2,
            save=True,
            out_dir=Path(DRIFT_ARTEFACTS_DIR),
        )

        drifted = [r for r in report if r.get("drift")]
        if drifted:
            log.warning(
                f"Обнаружен дрифт по фичам: {', '.join(r.get('feature', '?') for r in drifted)}"
            )
        else:
            log.info("Дрифт не обнаружен")

        return {"count": len(report), "drifted": [r.get("feature") for r in drifted]}

    except Exception as e:
        log.error(f"Ошибка мониторинга дрейфа: {e}")
        raise


with DAG(
    dag_id="kindle_reviews_monitored_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description=_DOC,
    tags=["sentiment", "monitored", "ml"],
) as dag:
    download = PythonOperator(
        task_id="downloading",
        python_callable=_task_download,
        on_success_callback=log_task_duration,
        on_failure_callback=log_task_failure,
    )

    validate = PythonOperator(
        task_id="data_validation",
        python_callable=_task_validate,
        on_success_callback=log_task_duration,
        on_failure_callback=log_task_failure,
    )

    process = PythonOperator(
        task_id="spark_processing",
        python_callable=_task_process,
        on_success_callback=log_task_duration,
        on_failure_callback=log_task_failure,
    )

    drift_monitor = PythonOperator(
        task_id="drift_monitoring",
        python_callable=_task_drift_monitor,
        on_success_callback=log_task_duration,
        on_failure_callback=log_task_failure,
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=_task_train,
        on_success_callback=log_task_duration,
        on_failure_callback=log_task_failure,
    )

    download >> validate >> process >> drift_monitor >> train

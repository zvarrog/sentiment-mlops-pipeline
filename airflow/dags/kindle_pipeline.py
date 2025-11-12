"""Параметризованный DAG для обработки Kindle reviews с поддержкой:
- последовательного обучения с Optuna HPO
- параллельного обучения нескольких моделей
- опционального мониторинга метрик в PostgreSQL
"""

from datetime import datetime

from airflow.decorators import task
from airflow.models import TaskInstance
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from airflow import DAG
from scripts.utils import get_flag, get_value

_DOC = """Пайплайн для обработки и обучения модели на Kindle отзывах.

Управление через параметры DAG:
- parallel: параллельное обучение моделей (по умолчанию False)
- enable_monitoring: логирование метрик в PostgreSQL (по умолчанию False)
- force_download/force_process/force_train: флаги форсирования этапов
- inject_synthetic_drift: инъекция синтетического дрейфа
- run_drift_monitor: запуск мониторинга дрейфа
- run_data_validation: валидация данных
"""

default_args = {
    "start_date": datetime(2025, 1, 1),
}


def log_task_metric(status: str, **context):
    ti: TaskInstance = context["task_instance"]
    duration = ti.duration if ti.duration else 0

    try:
        from scripts.logging_config import setup_auto_logging

        _log = setup_auto_logging()
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
        import contextlib

        with contextlib.suppress(Exception):
            _log.warning(f"Не удалось залогировать метрику задачи: {e}")


def log_task_duration(**context):
    ti: TaskInstance = context["task_instance"]
    status = ti.state if ti.state else "success"
    log_task_metric(status=status, **context)


def log_task_failure(**context):
    log_task_metric(status="failed", **context)


# Функции задачи


def _task_download(**context):
    from scripts.download import main as download_main
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Загрузка данных из Kaggle")

    csv_abs = download_main()
    log.info("Данные загружены: %s", csv_abs)
    return str(csv_abs)


def _task_validate_data(**context):
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()

    if not get_flag(context, "run_data_validation", True):
        log.info("Валидация пропущена: RUN_DATA_VALIDATION=0")
        return

    log.info("Валидация данных")

    from scripts.data_validation import main as validate_main

    success = validate_main()
    if not success:
        raise ValueError("Ошибки валидации данных")

    log.info("Валидация успешна")


def _task_inject_drift(**context):
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Инъекция дрейфа")

    from scripts.drift_injection import main as inject_main

    result = inject_main()
    if result.get("status") in ["skipped", "no_changes"]:
        return result
    if result.get("status") == "error":
        raise RuntimeError(f"Ошибка инъекции дрейфа: {result.get('message')}")

    return result


def _task_process(**context):
    from scripts.logging_config import setup_auto_logging
    from scripts.spark_process import TEST_PATH, TRAIN_PATH, VAL_PATH, process_data

    log = setup_auto_logging()
    log.info("Обработка данных через Spark")

    # Явно запускаем обработку
    process_data()

    return {"train": str(TRAIN_PATH), "val": str(VAL_PATH), "test": str(TEST_PATH)}


def _task_drift_monitor(**context):
    from pathlib import Path

    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()

    if not get_flag(context, "run_drift_monitor", False):
        log.info("Мониторинг дрейфа пропущен: RUN_DRIFT_MONITOR=0")
        return

    # Единый источник путей — scripts.config (SSoT)
    from scripts.config import DRIFT_ARTEFACTS_DIR, PROCESSED_DATA_DIR

    proc_dir_override = get_value(
        context, "processed_data_dir", str(PROCESSED_DATA_DIR)
    )
    test_parquet = Path(proc_dir_override) / "test.parquet"
    if not test_parquet.exists():
        log.warning("Файл test.parquet не найден")
        return

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


def _task_train_standard(**context):
    """Стандартное обучение с Optuna оптимизацией."""
    from scripts.logging_config import setup_auto_logging
    from scripts.train import run

    log = setup_auto_logging()
    log.info("Обучение модели (standard режим)")

    run()

    log.info("Обучение завершено")
    return "training_complete"


def _log_model_metrics_to_db(**context):
    """Логирование метрик модели в PostgreSQL."""
    import json

    # Единый источник путей — scripts.config (SSoT)
    from scripts.config import MODEL_ARTEFACTS_DIR
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()

    meta_path = MODEL_ARTEFACTS_DIR / "best_model_meta.json"
    if not meta_path.exists():
        log.warning("Метаданные модели не найдены")
        return

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
            (model_name, "test_f1_macro", test_metrics.get("f1_macro", 0.0), "test"),
            (model_name, "test_accuracy", test_metrics.get("accuracy", 0.0), "test"),
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


def _train_model_parallel(model_kind: str, **context):
    """Обучение одной модели через прямой вызов run()."""
    import os
    from pathlib import Path

    from scripts.config import MODEL_ARTEFACTS_DIR
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info(f"Обучение модели: {model_kind}")

    # Переопределяем через переменные окружения для изоляции
    old_model_kinds = os.environ.get("SELECTED_MODEL_KINDS")
    old_force_train = os.environ.get("FORCE_TRAIN")

    try:
        os.environ["SELECTED_MODEL_KINDS"] = model_kind
        os.environ["FORCE_TRAIN"] = "1"

        from scripts.train import run

        run()
        log.info(f"Модель {model_kind} успешно обучена")
    finally:
        # Восстанавливаем старые значения
        if old_model_kinds is not None:
            os.environ["SELECTED_MODEL_KINDS"] = old_model_kinds
        else:
            os.environ.pop("SELECTED_MODEL_KINDS", None)

        if old_force_train is not None:
            os.environ["FORCE_TRAIN"] = old_force_train
        else:
            os.environ.pop("FORCE_TRAIN", None)

    import json

    meta_path = Path(MODEL_ARTEFACTS_DIR) / f"meta_{model_kind}.json"
    model_path = Path(MODEL_ARTEFACTS_DIR) / f"model_{model_kind}.joblib"

    if not meta_path.exists():
        raise FileNotFoundError(f"Метаданные модели не найдены: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "model": model_kind,
        "val_f1_macro": meta.get("val_f1_macro", 0.0),
        "test_f1_macro": meta.get("test_f1_macro", 0.0),
        "meta_path": str(meta_path),
        "model_path": str(model_path),
    }


@task
def train_one(model_kind: str, **context):
    """Обучает одну модель указанного вида (динамическое маппинг‑задание)."""
    return _train_model_parallel(model_kind, **context)


@task
def select_best(results: list[dict], **context):
    """Выбирает лучшую модель из результатов динамически обученных моделей."""
    import contextlib
    import json
    import shutil
    from datetime import datetime
    from pathlib import Path

    from scripts.config import MODEL_ARTEFACTS_DIR, MODEL_FILE_DIR
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Выбор лучшей модели из обученных")

    if not results:
        raise ValueError("Ни одна модель не была успешно обучена")

    best = max(results, key=lambda x: x.get("val_f1_macro", 0.0))
    log.info(
        f"Лучшая модель: {best['model']} с val_f1_macro={best['val_f1_macro']:.4f}"
    )

    MODEL_FILE_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_FILE_DIR / "best_model.joblib"

    from scripts.model_registry import load_old_model_metric, should_replace_model

    old_model_metric = load_old_model_metric()
    new_model_metric = best["val_f1_macro"]

    if not should_replace_model(new_model_metric, old_model_metric, best["model"]):
        return {
            "status": "skipped",
            "reason": "new_model_not_better",
            "old_metric": old_model_metric,
            "new_metric": new_model_metric,
        }

    src_model = Path(best["model_path"]) if best.get("model_path") else None
    if src_model and src_model.exists():
        try:
            shutil.copy2(src_model, best_model_path)
        except PermissionError:
            shutil.copyfile(src_model, best_model_path)
        except OSError as e:
            log.warning(f"copy2 не удался: {e}; пробую copyfile")
            shutil.copyfile(src_model, best_model_path)
        log.info(f"Лучшая модель скопирована в {best_model_path}")

    try:
        from scripts.postprocessing import generate_best_bundle
        from scripts.train_modules.data_loading import load_splits

        x_train, x_val, x_test, y_train, y_val, y_test = load_splits()

        src_meta = Path(best.get("meta_path", ""))
        best_params = {}
        if src_meta and src_meta.exists():
            with open(src_meta, encoding="utf-8") as f:
                _meta_raw = json.load(f)
                best_params = _meta_raw.get("best_params", {})

        generate_best_bundle(
            best_model=str(best["model"]),
            best_params=best_params,
            best_val_f1_macro=float(best.get("val_f1_macro", 0.0)),
            pipeline_path=best_model_path,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            artefacts_dir=Path(MODEL_ARTEFACTS_DIR),
        )
    except Exception as e:
        log.warning(f"Постпроцессинг лучшей модели частично пропущен: {e}")

    try:
        import mlflow

        from scripts.config import MLFLOW_TRACKING_URI
        from scripts.model_registry import register_model_in_mlflow
        from scripts.models.kinds import ModelKind

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        best_kind = ModelKind(best["model"])
        test_f1 = best.get("test_f1_macro", 0.0)

        register_model_in_mlflow(
            best_model_path, best_kind, test_f1, mlflow_run_active=False
        )
    except Exception as e:
        log.warning("Не удалось зарегистрировать модель в MLflow Registry: %s", e)

    try:
        keep = get_flag(context, "keep_candidates", False)
        arte_dir = Path(MODEL_ARTEFACTS_DIR)
        best_kind = str(best["model"]).strip()
        model_files = list(arte_dir.glob("model_*.joblib"))
        meta_files = list(arte_dir.glob("meta_*.json"))
        to_affect = []
        for p in model_files + meta_files:
            name = p.name
            if name.startswith("model_") and name != f"model_{best_kind}.joblib":
                to_affect.append(p)
            if name.startswith("meta_") and name != f"meta_{best_kind}.json":
                to_affect.append(p)

        if to_affect:
            if keep:
                dst_root = (
                    arte_dir
                    / "candidates"
                    / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                )
                dst_root.mkdir(parents=True, exist_ok=True)
                for p in to_affect:
                    shutil.move(str(p), str(dst_root / p.name))
                log.info(f"Кандидаты перемещены в {dst_root}")
            else:
                for p in to_affect:
                    with contextlib.suppress(Exception):
                        p.unlink()
                log.info("Лишние кандидаты удалены (KEEP_CANDIDATES=0)")
    except Exception as e:
        log.warning(f"Не удалось обработать кандидатные артефакты: {e}")

    log.info("Выбор лучшей модели завершен")
    return best


# Определение DAG
with DAG(
    dag_id="kindle_unified_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description=_DOC,
    tags=["sentiment", "ml", "unified"],
    params={
        "parallel": False,
        "enable_monitoring": False,
        "keep_candidates": False,
        "force_download": False,
        "force_process": False,
        "force_train": False,
        "run_data_validation": True,
        "inject_synthetic_drift": False,
        "run_drift_monitor": False,
    },
) as dag:
    enable_monitoring = get_flag(dag.params, "enable_monitoring", False)
    callbacks = {}
    if enable_monitoring:
        callbacks = {
            "on_success_callback": log_task_duration,
            "on_failure_callback": log_task_failure,
        }

    download = PythonOperator(
        task_id="download",
        python_callable=_task_download,
        **callbacks,
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=_task_validate_data,
    )

    inject_drift = PythonOperator(
        task_id="inject_drift",
        python_callable=_task_inject_drift,
    )

    process = PythonOperator(
        task_id="process",
        python_callable=_task_process,
    )

    drift_monitor = PythonOperator(
        task_id="drift_monitor",
        python_callable=_task_drift_monitor,
    )

    train_standard = PythonOperator(
        task_id="train_standard",
        python_callable=_task_train_standard,
    )

    from scripts.config import SELECTED_MODEL_KINDS

    _MODEL_KINDS = [mk.value for mk in SELECTED_MODEL_KINDS]
    train_results = train_one.expand(model_kind=_MODEL_KINDS)
    select_best_task = select_best(train_results)
    _parallel_branch_targets = ["train_one"]

    def _branch_by_mode(**context):
        """Ветвление по флагу parallel."""
        parallel_flag = get_flag(context, "parallel", False)
        return _parallel_branch_targets if parallel_flag else ["train_standard"]

    branch = BranchPythonOperator(
        task_id="branch_by_mode",
        python_callable=_branch_by_mode,
    )

    # Общий граф до обучения
    download >> validate_data >> inject_drift >> process >> drift_monitor >> branch

    # Ветви обучения
    branch >> train_standard
    # В параллельном режиме используем динамический маппинг
    branch >> train_results
    train_results >> select_best_task

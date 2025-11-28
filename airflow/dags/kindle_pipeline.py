"""Параметризованный DAG для обработки Kindle reviews.

Поддерживает:
- последовательное обучение с Optuna HPO
- параллельное обучение нескольких моделей
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import task
from airflow.operators.python import BranchPythonOperator, PythonOperator

from airflow import DAG

# Корректный способ импорта модулей проекта в Airflow:
# 1. Установить пакет в editable mode: pip install -e /path/to/project
# 2. Либо добавить PYTHONPATH в airflow.cfg или docker-compose.yml
# 3. Либо использовать переменную окружения AIRFLOW__CORE__DAGS_FOLDER
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (
    DATA_PATHS,
    DRIFT_ARTEFACTS_DIR,
    PROCESSED_DATA_DIR,
    SELECTED_MODEL_KINDS,
)
from scripts.logging_config import setup_auto_logging
from scripts.models.kinds import ModelKind
from scripts.utils import get_flag, get_value

_DOC = """Пайплайн для обработки и обучения модели на Kindle отзывах.

Управление через параметры DAG:
- parallel: параллельное обучение моделей (по умолчанию False)
- force_download/force_process/force_train: флаги форсирования этапов
- inject_synthetic_drift: инъекция синтетического дрейфа
- run_drift_monitor: запуск мониторинга дрейфа
- run_data_validation: валидация данных
"""

default_args = {
    "start_date": datetime(2025, 1, 1),
}


def _task_download(**context):
    from scripts.download import main as download_main

    log = setup_auto_logging()
    force_download = get_flag(context, "force_download", False)
    log.info("Загрузка данных (force=%s)", force_download)
    return str(download_main(force=force_download))


def _task_validate_data(**context):
    from scripts.data_validation import main as validate_main

    log = setup_auto_logging()
    if not get_flag(context, "run_data_validation", True):
        log.info("Валидация пропущена")
        return

    log.info("Валидация данных")
    if not validate_main():
        raise ValueError("Ошибки валидации данных")


def _task_inject_drift(**context):
    from scripts.drift_injection import main as inject_main

    log = setup_auto_logging()
    log.info("Инъекция дрейфа")
    result = inject_main()

    if result.get("status") == "error":
        raise RuntimeError(f"Ошибка инъекции дрейфа: {result.get('message')}")
    return result


def _task_process(**context):
    from scripts.spark_process import process_data

    log = setup_auto_logging()
    force_process = get_flag(context, "force_process", False)
    log.info("Обработка данных (force=%s)", force_process)

    process_data(force=force_process)

    return {
        "train": str(DATA_PATHS.train),
        "val": str(DATA_PATHS.val),
        "test": str(DATA_PATHS.test),
    }


def _task_drift_monitor(**context):
    from scripts.drift_monitor import run_drift_monitor

    log = setup_auto_logging()
    if not get_flag(context, "run_drift_monitor", False):
        log.info("Мониторинг пропущен")
        return

    proc_dir = get_value(context, "processed_data_dir", str(PROCESSED_DATA_DIR))
    test_parquet = Path(proc_dir) / "test.parquet"

    log.info("Запуск мониторинга дрейфа для %s", test_parquet)

    if not test_parquet.exists():
        raise FileNotFoundError(f"Файл {test_parquet} не найден")

    report = run_drift_monitor(
        str(test_parquet),
        threshold=0.2,
        save=True,
        out_dir=Path(DRIFT_ARTEFACTS_DIR),
    )

    drifted = [r.get("feature") for r in report if r.get("drift")]
    if drifted:
        log.warning("Обнаружен дрифт: %s", drifted)

    return {"count": len(report), "drifted": drifted}


def _task_train_standard(**context):
    from scripts.train import run as train_run

    log = setup_auto_logging()
    force_train = get_flag(context, "force_train", False)
    log.info("Обучение (standard, force=%s)", force_train)
    train_run(force=force_train)


def _train_model_parallel(model_kind: str, **context):
    from scripts.train import run as train_run

    log = setup_auto_logging()
    force_train = get_flag(context, "force_train", False)
    log.info("Обучение модели: %s", model_kind)

    train_run(
        force=force_train,
        selected_models=[ModelKind(model_kind)],
    )
    return {"model": model_kind, "status": "completed"}


@task
def train_one(model_kind: str, **context):
    return _train_model_parallel(model_kind, **context)


@task
def select_best(results: list[dict], **context):
    log = setup_auto_logging()
    if not results:
        raise ValueError("Нет обученных моделей")
    log.info("Параллельное обучение завершено, моделей: %d", len(results))
    return {"status": "complete", "count": len(results)}


def _branch_by_mode(**context):
    parallel = get_flag(context, "parallel", False)
    if parallel:
        from scripts.config import SELECTED_MODEL_KINDS
        return [f"train_one.{model.value}" for model in SELECTED_MODEL_KINDS]
    return ["train_standard"]


with DAG(
    dag_id="kindle_unified_pipeline",
    default_args=default_args,
    schedule=None,
    catchup=False,
    description=_DOC,
    tags=["sentiment", "ml"],
    params={
        "parallel": False,
        "force_download": False,
        "force_process": False,
        "force_train": False,
        "run_data_validation": True,
        "inject_synthetic_drift": False,
        "run_drift_monitor": False,
    },
) as dag:
    download = PythonOperator(
        task_id="download",
        python_callable=_task_download,
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

    branch = BranchPythonOperator(
        task_id="branch_by_mode",
        python_callable=_branch_by_mode,
    )

    train_standard = PythonOperator(
        task_id="train_standard",
        python_callable=_task_train_standard,
    )

    # Параллельная ветка
    model_kinds = [mk.value for mk in SELECTED_MODEL_KINDS]
    train_results = train_one.expand(model_kind=model_kinds)
    select_best_task = select_best(train_results)

    # Граф
    download >> validate_data >> inject_drift >> process >> drift_monitor >> branch

    branch >> train_standard
    branch >> train_results >> select_best_task

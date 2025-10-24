"""DAG автоматического переобучения при обнаружении дрейфа.

Логика:
- Ждём появления отчёта о дрейфе (CSV), который генерирует scripts.drift_monitor.
- При наличии файла — триггерим основной unified DAG с конфигурацией force_train=True.

Все пути и параметры берём из единого источника истины (scripts.config).
Комментарии краткие и на русском.
"""

from datetime import datetime
from pathlib import Path

try:
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    from airflow.sensors.filesystem import FileSensor

    from airflow import DAG
except ImportError:  # Заглушки для локального импорта вне Airflow

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

    class FileSensor(_Dummy):
        pass

    class TriggerDagRunOperator(_Dummy):
        pass


# Единый источник путей — SSoT
try:
    from scripts.config import DRIFT_ARTEFACTS_DIR
except Exception:
    DRIFT_ARTEFACTS_DIR = Path("artefacts/drift_artefacts")

_DRIFT_FILE = str(Path(DRIFT_ARTEFACTS_DIR) / "drift_report.csv")

with DAG(
    dag_id="drift_retraining_dag",
    default_args={"start_date": datetime(2025, 1, 1)},
    schedule_interval=None,
    catchup=False,
    description="Автоперезапуск обучения при наличии отчёта о дрейфе",
    tags=["sentiment", "ml", "drift"],
) as dag:
    wait_for_drift_report = FileSensor(
        task_id="wait_for_drift_report",
        filepath=_DRIFT_FILE,
        poke_interval=3600,  # проверяем раз в час
        mode="poke",
        timeout=60 * 60 * 24,  # максимум сутки
        soft_fail=False,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="kindle_unified_pipeline",
        conf={
            "execution_mode": "standard",
            "force_train": True,
        },
        reset_dag_run=True,
        wait_for_completion=False,
    )

    wait_for_drift_report >> trigger_retrain

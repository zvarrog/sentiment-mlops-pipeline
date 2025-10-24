from datetime import datetime

try:  # Локальная среда без установленного Airflow (статический анализ тестов)
    from airflow.operators.python import PythonOperator

    from airflow import DAG
except ImportError:

    class _Dummy:
        def __init__(self, *_, **__):
            self.task_id = "dummy"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class DAG(_Dummy):
        def __init__(self, *_, **__):
            super().__init__()

    class PythonOperator(_Dummy):
        def __init__(self, *_, **__):
            super().__init__()

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other


_DOC = "Пайплайн обработки и обучения модели на Kindle отзывах с поддержкой валидации, drift injection и мониторинга."

default_args = {
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    dag_id="kindle_reviews_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description=_DOC,
    params={
        # Флаги форсирования
        "force_download": False,
        "force_process": False,
        "force_train": False,
        # Флаги валидации и инъекции
        "run_data_validation": True,
        "inject_synthetic_drift": False,
        "run_drift_monitor": False,
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        # Корень артефактов и подкаталоги
        "model_dir": "artefacts",
        "model_artefacts_dir": "artefacts/model_artefacts",
        "drift_artefacts_dir": "artefacts/drift_artefacts",
    },
) as dag:

    def _to_bool(x, default: bool = False) -> bool:
        if x is None:
            return bool(default)
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            return x.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(x)

    def _get_value(context, name: str, default: str | None = None) -> str:
        # dag_run.conf
        try:
            dr = context.get("dag_run") if context else None
            if dr and getattr(dr, "conf", None) is not None and name in dr.conf:
                val = dr.conf.get(name)
                if isinstance(val, (str, int, float)):
                    return str(val)
        except Exception:
            pass
        # dag.params
        if name in dag.params:
            val = dag.params.get(name)
            return str(val)
        return str(default or "")

    def _with_env(context=None, **kwargs):
        import os
        from pathlib import Path as _P

        for k, v in kwargs.items():
            os.environ[k] = str(int(_to_bool(v)))

        base = os.environ.get("AIRFLOW_HOME", "/opt/airflow")

        def _abs(p: str) -> str:
            try:
                pp = _P(p)
                if pp.is_absolute():
                    return str(pp)
                return str((_P(base) / pp).resolve())
            except Exception:
                return p

        raw_p = _get_value(context, "raw_data_dir", "data/raw")
        proc_p = _get_value(context, "processed_data_dir", "data/processed")
        model_dir_p = _get_value(context, "model_dir", "artefacts")
        model_arts_p = _get_value(
            context, "model_artefacts_dir", str(_P(model_dir_p) / "model_artefacts")
        )
        drift_arts_p = _get_value(
            context, "drift_artefacts_dir", str(_P(model_dir_p) / "drift_artefacts")
        )
        os.environ["RAW_DATA_DIR"] = _abs(raw_p)
        os.environ["PROCESSED_DATA_DIR"] = _abs(proc_p)
        os.environ["MODEL_DIR"] = _abs(model_dir_p)
        os.environ["MODEL_ARTEFACTS_DIR"] = _abs(model_arts_p)
        os.environ["DRIFT_ARTEFACTS_DIR"] = _abs(drift_arts_p)

        try:
            from scripts.logging_config import setup_auto_logging as _alog

            _lg = _alog()
            _lg.info(
                "Пути DAG: RAW_DATA_DIR=%s, PROCESSED_DATA_DIR=%s, MODEL_DIR=%s, MODEL_ARTEFACTS_DIR=%s, DRIFT_ARTEFACTS_DIR=%s",
                os.environ.get("RAW_DATA_DIR"),
                os.environ.get("PROCESSED_DATA_DIR"),
                os.environ.get("MODEL_DIR"),
                os.environ.get("MODEL_ARTEFACTS_DIR"),
                os.environ.get("DRIFT_ARTEFACTS_DIR"),
            )
        except Exception:
            pass

    def _get_flag(context, name: str, default: bool = False) -> bool:
        try:
            dr = context.get("dag_run")
            if dr and getattr(dr, "conf", None) is not None and name in dr.conf:
                return _to_bool(dr.conf.get(name), default)
        except Exception:
            pass
        if name in dag.params:
            return _to_bool(dag.params.get(name), default)
        return bool(default)

    def _task_download(**context):
        _with_env(context, FORCE_DOWNLOAD=_get_flag(context, "force_download", False))
        import os

        from scripts.download import CSV_PATH
        from scripts.download import main as download_main
        from scripts.logging_config import setup_auto_logging

        log = setup_auto_logging()
        log.info(
            "downloading: RAW_DATA_DIR=%s, CSV_NAME=%s",
            os.environ.get("RAW_DATA_DIR"),
            os.environ.get("CSV_NAME", "kindle_reviews.csv"),
        )
        force = os.environ.get("FORCE_DOWNLOAD", "0").lower() in {"1", "true", "yes"}
        try:
            if force:
                log.info("FORCE_DOWNLOAD=1: принудительно скачиваю датасет")
                csv_abs = download_main()
            else:
                if not CSV_PATH.exists():
                    log.info("CSV не найден: %s — запускаю скачивание", str(CSV_PATH))
                    csv_abs = download_main()
                else:
                    log.info(
                        "CSV уже существует: %s — пропуск скачивания", str(CSV_PATH)
                    )
                    csv_abs = CSV_PATH.resolve()
        except Exception as e:
            log.error("Ошибка в downloading: %s", e)
            raise

        return str(csv_abs)

    def _task_validate_data(**context):
        flag_value = _get_flag(context, "run_data_validation", True)
        _with_env(context, RUN_DATA_VALIDATION=flag_value)

        import os

        from scripts.logging_config import setup_auto_logging

        log = setup_auto_logging()

        run_validation = os.environ.get("RUN_DATA_VALIDATION", "1").lower() in {
            "1",
            "true",
            "yes",
        }

        if not run_validation:
            log.info(
                "Валидация данных пропущена: RUN_DATA_VALIDATION=%s",
                os.environ.get("RUN_DATA_VALIDATION", "NOT_SET"),
            )
            return "validation_skipped"

        try:
            from scripts.data_validation import main as validate_main

            validate_main()
            return "validation_success"
        except Exception as e:
            log.error("Ошибка валидации данных: %s", e)
            raise

    def _task_inject_drift(**context):
        _with_env(
            context,
            INJECT_SYNTHETIC_DRIFT=_get_flag(context, "inject_synthetic_drift", False),
        )

        from scripts.logging_config import setup_auto_logging

        log = setup_auto_logging()

        try:
            from scripts.drift_injection import main as inject_main

            result = inject_main()
            if result.get("status") in ["skipped", "no_changes"]:
                return result
            elif result.get("status") == "error":
                raise RuntimeError(f"Drift injection failed: {result.get('message')}")

            return result
        except ImportError as e:
            log.warning("Модуль drift_injection недоступен: %s", e)
            return {
                "status": "module_unavailable",
                "message": str(e),
                "changed_columns": [],
            }
        except Exception as e:
            log.error("Неожиданная ошибка инъекции дрейфа: %s", e)
            raise

    def _task_process(**context):
        _with_env(
            context,
            FORCE_PROCESS=_get_flag(context, "force_process", False),
            RUN_DRIFT_MONITOR=_get_flag(context, "run_drift_monitor", False),
            RUN_DATA_VALIDATION=_get_flag(context, "run_data_validation", True),
        )
        from scripts.spark_process import TEST_PATH, TRAIN_PATH, VAL_PATH

        return {"train": str(TRAIN_PATH), "val": str(VAL_PATH), "test": str(TEST_PATH)}

    def _task_train(**context):
        import os

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
                "OPTUNA_STORAGE",
                "postgresql+psycopg2://admin:admin@postgres:5432/optuna",
            )
        os.environ["OPTUNA_STORAGE"] = optuna_storage
        _with_env(context, FORCE_TRAIN=_get_flag(context, "force_train", False))
        from scripts.train import run

        run()

    download = PythonOperator(
        task_id="downloading",
        python_callable=_task_download,
    )

    validate_data = PythonOperator(
        task_id="data_validation",
        python_callable=_task_validate_data,
    )

    inject_drift = PythonOperator(
        task_id="drift_injection",
        python_callable=_task_inject_drift,
    )

    process = PythonOperator(
        task_id="spark_processing",
        python_callable=_task_process,
    )

    def _task_drift_monitor(**context):
        import os
        from pathlib import Path

        from scripts.logging_config import setup_auto_logging

        log = setup_auto_logging()

        run_dm = _get_flag(context, "run_drift_monitor", False)
        _with_env(context, RUN_DRIFT_MONITOR=run_dm)
        if not run_dm:
            log.info("Мониторинг дрейфа пропущен: RUN_DRIFT_MONITOR=0")
            return "drift_monitor_skipped"

        processed_dir = Path(os.environ.get("PROCESSED_DATA_DIR", "data/processed"))
        test_parquet = processed_dir / "test.parquet"
        if not test_parquet.exists():
            log.warning("Файл для мониторинга дрейфа не найден: %s", str(test_parquet))
            return "no_data"

        try:
            from scripts.drift_monitor import run_drift_monitor

            out_dir = Path(
                os.environ.get("DRIFT_ARTEFACTS_DIR", "artefacts/drift_artefacts")
            )

            log.info(
                "Запускаю мониторинг дрейфа: src=%s, out=%s",
                str(test_parquet),
                str(out_dir),
            )
            report = run_drift_monitor(
                str(test_parquet), threshold=0.2, save=True, out_dir=out_dir
            )
            drifted = [r for r in report if r.get("drift")]
            if drifted:
                log.warning(
                    "ДРИФТ обнаружен по фичам: %s",
                    ", ".join(r.get("feature", "?") for r in drifted),
                )
            else:
                log.info("Дрифт не обнаружен")
            return {
                "count": len(report),
                "drifted": [r.get("feature") for r in drifted],
            }
        except Exception as e:
            log.error("Ошибка мониторинга дрейфа: %s", e)
            raise

    drift_monitor = PythonOperator(
        task_id="drift_monitoring",
        python_callable=_task_drift_monitor,
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=_task_train,
    )

    download >> validate_data >> inject_drift >> process >> drift_monitor >> train

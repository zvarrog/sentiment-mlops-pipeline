"""Параметризованный DAG для обработки Kindle reviews с поддержкой режимов:
- standard: последовательная обработка с одной моделью (Optuna HPO)
- monitored: стандартный режим + логирование метрик в PostgreSQL
- parallel: параллельное обучение нескольких моделей и выбор лучшей
"""

from datetime import datetime

try:
    from airflow.decorators import task
    from airflow.models import TaskInstance
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    from airflow import DAG

    AIRFLOW_AVAILABLE = True
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

    class BranchPythonOperator(_Dummy):
        pass

    class TaskInstance:
        pass

    class PostgresHook:
        def __init__(self, *_, **__):
            pass

    # Простейший декоратор-заглушка для @task, чтобы модуль импортировался вне Airflow
    def task(*_args, **_kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    AIRFLOW_AVAILABLE = False


_DOC = """Пайплайн для обработки и обучения модели на Kindle отзывах.

Режимы работы (параметр execution_mode):
- standard: базовый режим с Optuna оптимизацией (по умолчанию)
- monitored: + логирование метрик задач в PostgreSQL
- parallel: параллельное обучение моделей (logreg, rf, hist_gb) и выбор лучшей

Управление через параметры DAG:
- execution_mode: режим выполнения (standard/monitored/parallel)
- force_download/force_process/force_train: флаги форсирования этапов
- inject_synthetic_drift: инъекция синтетического дрейфа
- run_drift_monitor: запуск мониторинга дрейфа
- run_data_validation: валидация данных
- raw_data_dir, processed_data_dir, model_dir: пути к директориям
"""

default_args = {
    "start_date": datetime(2025, 1, 1),
}


# Utility functions
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
    try:
        dr = context.get("dag_run") if context else None
        if dr and getattr(dr, "conf", None) is not None and name in dr.conf:
            val = dr.conf.get(name)
            if isinstance(val, (str, int, float)):
                return str(val)
    except Exception:
        pass

    dag = context.get("dag")
    if dag and name in dag.params:
        val = dag.params.get(name)
        return str(val)
    return str(default or "")


def _get_flag(context, name: str, default: bool = False) -> bool:
    try:
        dr = context.get("dag_run")
        if dr and getattr(dr, "conf", None) is not None and name in dr.conf:
            return _to_bool(dr.conf.get(name), default)
    except Exception:
        pass

    dag = context.get("dag")
    if dag and name in dag.params:
        return _to_bool(dag.params.get(name), default)
    return bool(default)


def _setup_env(**context):
    import os
    from pathlib import Path

    base = os.environ.get("AIRFLOW_HOME", "/opt/airflow")

    def _abs(p: str) -> str:
        pp = Path(p)
        return str(pp) if pp.is_absolute() else str((Path(base) / pp).resolve())

    raw_p = _get_value(context, "raw_data_dir", "data/raw")
    proc_p = _get_value(context, "processed_data_dir", "data/processed")
    model_dir_p = _get_value(context, "model_dir", "artefacts")

    os.environ["RAW_DATA_DIR"] = _abs(raw_p)
    os.environ["PROCESSED_DATA_DIR"] = _abs(proc_p)
    os.environ["MODEL_DIR"] = _abs(model_dir_p)
    os.environ["MODEL_ARTEFACTS_DIR"] = _abs(
        _get_value(
            context,
            "model_artefacts_dir",
            str(Path(model_dir_p) / "model_artefacts"),
        )
    )
    os.environ["DRIFT_ARTEFACTS_DIR"] = _abs(
        _get_value(
            context, "drift_artefacts_dir", str(Path(model_dir_p) / "drift_artefacts")
        )
    )

    os.environ["FORCE_DOWNLOAD"] = str(int(_get_flag(context, "force_download", False)))
    os.environ["FORCE_PROCESS"] = str(int(_get_flag(context, "force_process", False)))
    os.environ["FORCE_TRAIN"] = str(int(_get_flag(context, "force_train", False)))
    os.environ["INJECT_SYNTHETIC_DRIFT"] = str(
        int(_get_flag(context, "inject_synthetic_drift", False))
    )
    os.environ["RUN_DRIFT_MONITOR"] = str(
        int(_get_flag(context, "run_drift_monitor", False))
    )
    os.environ["RUN_DATA_VALIDATION"] = str(
        int(_get_flag(context, "run_data_validation", True))
    )

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


# Monitoring callbacks для режима monitored
def log_task_duration(**context):
    ti: TaskInstance = context["task_instance"]

    if not ti.duration:
        return

    duration = ti.duration
    status = ti.state

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


# Task functions
def _task_download(**context):
    _setup_env(**context)
    import os

    from scripts.download import CSV_PATH
    from scripts.download import main as download_main
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Загрузка данных из Kaggle")

    force = os.environ.get("FORCE_DOWNLOAD", "0") == "1"
    if not force and CSV_PATH.exists():
        log.info("CSV уже существует: %s — пропуск", str(CSV_PATH))
        return str(CSV_PATH.resolve())

    csv_abs = download_main()
    log.info("Данные загружены: %s", csv_abs)
    return str(csv_abs)


def _task_validate_data(**context):
    _setup_env(**context)
    import os

    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()

    if os.environ.get("RUN_DATA_VALIDATION", "1") != "1":
        log.info("Валидация пропущена: RUN_DATA_VALIDATION=0")
        return "validation_skipped"

    log.info("Валидация данных")

    from scripts.data_validation import main as validate_main

    success = validate_main()
    if not success:
        raise ValueError("Ошибки валидации данных")

    log.info("Валидация успешна")
    return "validation_ok"


def _task_inject_drift(**context):
    _setup_env(**context)
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


def _task_process(**context):
    _setup_env(**context)
    from scripts.logging_config import setup_auto_logging
    from scripts.spark_process import TEST_PATH, TRAIN_PATH, VAL_PATH

    log = setup_auto_logging()
    log.info("Обработка данных через Spark")

    return {"train": str(TRAIN_PATH), "val": str(VAL_PATH), "test": str(TEST_PATH)}


def _task_drift_monitor(**context):
    _setup_env(**context)
    import os
    from pathlib import Path

    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()

    if os.environ.get("RUN_DRIFT_MONITOR", "0") != "1":
        log.info("Мониторинг дрейфа пропущен: RUN_DRIFT_MONITOR=0")
        return "drift_monitor_skipped"

    # Единый источник путей — scripts.config (SSoT)
    from scripts.config import DRIFT_ARTEFACTS_DIR, PROCESSED_DATA_DIR

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


def _task_train_standard(**context):
    """Стандартное обучение с Optuna оптимизацией."""
    _setup_env(**context)
    from scripts.logging_config import setup_auto_logging
    from scripts.train import run

    log = setup_auto_logging()
    log.info("Обучение модели (standard режим)")

    run()

    execution_mode = _get_value(context, "execution_mode", "standard")
    if execution_mode == "monitored":
        _log_model_metrics_to_db(**context)

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
    """Обучение одной модели для parallel режима."""
    _setup_env(**context)
    import json
    import time
    from pathlib import Path

    import joblib
    import mlflow
    import optuna

    # Единый источник настроек — scripts.config (SSoT)
    from scripts.config import (
        MODEL_ARTEFACTS_DIR,
        OPTUNA_N_TRIALS,
        OPTUNA_STORAGE,
    )
    from scripts.logging_config import setup_auto_logging
    from scripts.models.kinds import ModelKind
    from scripts.train import build_pipeline, compute_metrics, objective
    from scripts.train_modules.data_loading import load_splits
    from scripts.train_modules.feature_space import NUMERIC_COLS

    log = setup_auto_logging()
    log.info(f"Обучение модели: {model_kind}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    model_enum = ModelKind(model_kind)
    study_name = f"parallel_{model_kind}_{int(time.time())}"

    mlflow.set_experiment("kindle_parallel_experiment")

    with mlflow.start_run(run_name=f"train_{model_kind}"):
        mlflow.log_param("model", model_kind)

        study = optuna.create_study(
            direction="maximize",
            storage=OPTUNA_STORAGE,
            study_name=study_name,
            load_if_exists=False,
        )

        def opt_obj(trial):
            return objective(trial, model_enum, X_train, y_train, X_val, y_val)

        n_trials = min(OPTUNA_N_TRIALS, 10)
        study.optimize(opt_obj, n_trials=n_trials, timeout=300, show_progress_bar=False)

        if not study.best_trial or study.best_trial.value is None:
            raise ValueError(f"Не удалось обучить модель {model_kind}")

        best_params = study.best_trial.params

        fixed_trial = optuna.trial.FixedTrial(best_params)
        fixed_trial.set_user_attr(
            "numeric_cols", [c for c in NUMERIC_COLS if c in X_train.columns]
        )

        pipeline = build_pipeline(fixed_trial, model_enum)
        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds)

        test_preds = pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, test_preds)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        model_path = Path(MODEL_ARTEFACTS_DIR) / f"model_{model_kind}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

        meta = {
            "model": model_kind,
            "best_params": best_params,
            "val_f1_macro": val_metrics["f1_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
        }

        meta_path = Path(MODEL_ARTEFACTS_DIR) / f"meta_{model_kind}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        log.info(
            f"Модель {model_kind} обучена: val_f1={val_metrics['f1_macro']:.4f}, test_f1={test_metrics['f1_macro']:.4f}"
        )

        return {
            "model": model_kind,
            "val_f1_macro": val_metrics["f1_macro"],
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
    _setup_env(**context)
    import shutil
    from pathlib import Path

    # Единый источник путей — scripts.config (SSoT)
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

    src_model = Path(best["model_path"]) if best.get("model_path") else None
    if src_model and src_model.exists():
        shutil.copy2(src_model, best_model_path)
        log.info(f"Лучшая модель скопирована в {best_model_path}")

    best_meta_path = Path(MODEL_ARTEFACTS_DIR) / "best_model_meta.json"
    src_meta = Path(best["meta_path"]) if best.get("meta_path") else None
    if src_meta and src_meta.exists():
        shutil.copy2(src_meta, best_meta_path)

    log.info("Выбор лучшей модели завершен")
    return best


def _task_select_best(**context):
    """Fallback-функция выбора лучшей модели для режима без Airflow mapping."""
    _setup_env(**context)
    import shutil
    from pathlib import Path

    from scripts.config import MODEL_ARTEFACTS_DIR, MODEL_FILE_DIR
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Выбор лучшей модели (fallback)")

    ti = context["ti"]
    results = []
    for task_id in ["train_logreg", "train_rf", "train_gb"]:
        try:
            result = ti.xcom_pull(task_ids=task_id)
            if result and result.get("val_f1_macro"):
                results.append(result)
        except Exception as e:
            log.warning(f"Не удалось получить результат от {task_id}: {e}")

    if not results:
        raise ValueError("Ни одна модель не была успешно обучена")

    best = max(results, key=lambda x: x.get("val_f1_macro", 0.0))
    MODEL_FILE_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_FILE_DIR / "best_model.joblib"

    src_model = Path(best["model_path"]) if best.get("model_path") else None
    if src_model and src_model.exists():
        shutil.copy2(src_model, best_model_path)

    best_meta_path = Path(MODEL_ARTEFACTS_DIR) / "best_model_meta.json"
    src_meta = Path(best["meta_path"]) if best.get("meta_path") else None
    if src_meta and src_meta.exists():
        shutil.copy2(src_meta, best_meta_path)

    log.info("Выбор лучшей модели завершен (fallback)")
    return best


# DAG definition
with DAG(
    dag_id="kindle_unified_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description=_DOC,
    tags=["sentiment", "ml", "unified"],
    params={
        "execution_mode": "standard",
        "force_download": False,
        "force_process": False,
        "force_train": False,
        "run_data_validation": True,
        "inject_synthetic_drift": False,
        "run_drift_monitor": False,
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "model_dir": "artefacts",
        "model_artefacts_dir": "artefacts/model_artefacts",
        "drift_artefacts_dir": "artefacts/drift_artefacts",
    },
) as dag:

    def _get_callbacks(context):
        """Возвращает callbacks в зависимости от режима."""
        mode = _get_value(context, "execution_mode", "standard")
        if mode == "monitored":
            return {
                "on_success_callback": log_task_duration,
                "on_failure_callback": log_task_failure,
            }
        return {}

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

    train_standard = PythonOperator(
        task_id="train_standard",
        python_callable=_task_train_standard,
    )

    # Динамическая генерация заданий обучения на основе SELECTED_MODEL_KINDS (если доступен Airflow)
    if AIRFLOW_AVAILABLE:
        try:
            from scripts.config import SELECTED_MODEL_KINDS

            _MODEL_KINDS = [mk.value for mk in SELECTED_MODEL_KINDS]
        except Exception:
            _MODEL_KINDS = ["logreg", "rf", "hist_gb"]

        train_results = train_one.expand(model_kind=_MODEL_KINDS)  # type: ignore[attr-defined]
        select_best_task = select_best(train_results)  # type: ignore[call-arg]
        _parallel_branch_targets = ["train_one"]
    else:
        # Fallback: три статические задачи, параллельные при наличии исполнителя
        train_logreg = PythonOperator(
            task_id="train_logreg",
            python_callable=lambda **ctx: _train_model_parallel("logreg", **ctx),
        )
        train_rf = PythonOperator(
            task_id="train_rf",
            python_callable=lambda **ctx: _train_model_parallel("rf", **ctx),
        )
        train_gb = PythonOperator(
            task_id="train_gb",
            python_callable=lambda **ctx: _train_model_parallel("hist_gb", **ctx),
        )
        select_best_task = PythonOperator(
            task_id="select_best",
            python_callable=_task_select_best,
        )
        _parallel_branch_targets = ["train_logreg", "train_rf", "train_gb"]

    def _branch_by_mode(**context):
        """Ветвление по режиму выполнения."""
        mode = _get_value(context, "execution_mode", "standard")
        if mode == "parallel":
            # Для Airflow возвращаем id маппинг‑задачи; для fallback — список статических задач
            return _parallel_branch_targets
        return ["train_standard"]

    branch = BranchPythonOperator(
        task_id="branch_by_mode",
        python_callable=_branch_by_mode,
    )

    # Общий граф до обучения
    download >> validate_data >> inject_drift >> process >> drift_monitor >> branch

    # Ветви обучения
    branch >> train_standard
    # В параллельном режиме: либо динамический маппинг, либо fallback из трёх задач
    if AIRFLOW_AVAILABLE:
        branch >> train_results  # type: ignore[operator]
        train_results >> select_best_task  # type: ignore[operator]
    else:
        branch >> [train_logreg, train_rf, train_gb] >> select_best_task

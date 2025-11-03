"""Параметризованный DAG для обработки Kindle reviews с поддержкой режимов:
- standard: последовательная обработка с одной моделью (Optuna HPO)
- monitored: стандартный режим + логирование метрик в PostgreSQL
- parallel: параллельное обучение нескольких моделей и выбор лучшей
"""

from datetime import datetime

from airflow.decorators import task
from airflow.models import TaskInstance
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from airflow import DAG

_DOC = """Пайплайн для обработки и обучения модели на Kindle отзывах.

Режимы работы (параметр execution_mode):
- standard: базовый режим с Optuna оптимизацией (по умолчанию)
- monitored: + логирование метрик задач в PostgreSQL
- parallel: параллельное обучение моделей и выбор лучшей

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
    """Получить строковый параметр запуска.

    Порядок источников:
    1) context['params'] — если доступен (UI/CLI override, где поддерживается)
    2) context['dag_run'].conf — значения запуска из формы UI/CLI
    3) dag.params — дефолтные параметры DAG
    4) default — значение по умолчанию из вызова
    """
    params = context.get("params") or {}
    if name in params:
        return str(params.get(name))
    dag_run = context.get("dag_run")
    if getattr(dag_run, "conf", None) and name in dag_run.conf:
        return str(dag_run.conf.get(name))
    dag = context.get("dag")
    if dag and name in getattr(dag, "params", {}):
        return str(dag.params.get(name))
    return str(default or "")


def _get_flag(context, name: str, default: bool = False) -> bool:
    """Получить булев флаг запуска с тем же порядком источников."""
    params = context.get("params") or {}
    if name in params:
        return _to_bool(params.get(name), default)
    dag_run = context.get("dag_run")
    if getattr(dag_run, "conf", None) and name in dag_run.conf:
        return _to_bool(dag_run.conf.get(name), default)
    dag = context.get("dag")
    if dag and name in getattr(dag, "params", {}):
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
    os.environ["KEEP_CANDIDATES"] = str(
        int(_get_flag(context, "keep_candidates", False))
    )
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


def log_task_failure(**context):
    ti: TaskInstance = context["task_instance"]

    try:
        from scripts.logging_config import setup_auto_logging

        _log = setup_auto_logging()
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
        import contextlib

        with contextlib.suppress(Exception):
            _log.warning(f"Не удалось залогировать ошибку задачи: {e}")


# Функции задачи


def _task_download(**context):
    _setup_env(**context)

    from scripts.download import main as download_main
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Загрузка данных из Kaggle")

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
    log.info("Инъекция дрейфа")

    from scripts.drift_injection import main as inject_main

    result = inject_main()
    if result.get("status") in ["skipped", "no_changes"]:
        return result
    if result.get("status") == "error":
        raise RuntimeError(f"Ошибка инъекции дрейфа: {result.get('message')}")

    return result


def _task_process(**context):
    _setup_env(**context)
    from scripts.logging_config import setup_auto_logging
    from scripts.spark_process import TEST_PATH, TRAIN_PATH, VAL_PATH, process_data

    log = setup_auto_logging()
    log.info("Обработка данных через Spark")

    # Явно запускаем обработку
    process_data()

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
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    # Единый источник настроек — scripts.config (SSoT)
    from scripts.config import (
        MLFLOW_TRACKING_URI,
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

    x_train, x_val, x_test, y_train, y_val, y_test = load_splits()

    model_enum = ModelKind(model_kind)
    study_name = f"parallel_{model_kind}_{int(time.time())}"

    # Безопасная инициализация эксперимента MLflow
    exp_name = "kindle_parallel_experiment"
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        existing = client.get_experiment_by_name(exp_name)
        if existing is None:
            try:
                client.create_experiment(exp_name)
            except MlflowException as e:
                # Два воркера могут создать эксперимент одновременно
                if "already exists" not in str(e).lower():
                    raise
        mlflow.set_experiment(exp_name)
    except Exception as e:
        log.warning(f"Не удалось инициализировать MLflow эксперимент: {e}")
        raise

    with mlflow.start_run(run_name=f"train_{model_kind}"):
        mlflow.log_param("model", model_kind)

        study = optuna.create_study(
            direction="maximize",
            storage=OPTUNA_STORAGE,
            study_name=study_name,
            load_if_exists=False,
        )

        def opt_obj(trial):
            return objective(trial, model_enum, x_train, y_train, x_val, y_val)

        n_trials = min(OPTUNA_N_TRIALS, 10)
        study.optimize(opt_obj, n_trials=n_trials, timeout=300, show_progress_bar=False)

        if not study.best_trial or study.best_trial.value is None:
            raise ValueError(f"Не удалось обучить модель {model_kind}")

        best_params = study.best_trial.params

        fixed_trial = optuna.trial.FixedTrial(best_params)
        fixed_trial.set_user_attr(
            "numeric_cols", [c for c in NUMERIC_COLS if c in x_train.columns]
        )

        pipeline = build_pipeline(fixed_trial, model_enum)

        # DistilBERT работает только с текстом, остальные модели — с полным DataFrame
        if model_enum == ModelKind.distilbert:
            pipeline.fit(x_train["reviewText"], y_train)
            val_preds = pipeline.predict(x_val["reviewText"])
            test_preds = pipeline.predict(x_test["reviewText"])
        else:
            pipeline.fit(x_train, y_train)
            val_preds = pipeline.predict(x_val)
            test_preds = pipeline.predict(x_test)

        val_metrics = compute_metrics(y_val, val_preds)
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
    import contextlib
    import json
    import os
    import shutil
    from datetime import datetime
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
        try:
            shutil.copy2(src_model, best_model_path)
        except PermissionError:
            shutil.copyfile(src_model, best_model_path)
        except OSError as e:
            log.warning(f"copy2 не удался: {e}; пробую copyfile")
            shutil.copyfile(src_model, best_model_path)
        log.info(f"Лучшая модель скопирована в {best_model_path}")

    # Загрузка данных и модели для постпроцессинга
    try:
        from scripts.postprocessing import generate_best_bundle
        from scripts.train_modules.data_loading import load_splits

        x_train, x_val, x_test, y_train, y_val, y_test = load_splits()

        # Читаем best_params из исходного meta_{kind}.json
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

    # Обработка кандидатов по флагу KEEP_CANDIDATES
    try:
        keep = os.environ.get("KEEP_CANDIDATES", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
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
        # Флаг параллельного обучения (UI-параметр)
        "parallel": False,
        "keep_candidates": False,
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

    # Динамическая генерация заданий обучения на основе SELECTED_MODEL_KINDS
    from scripts.config import SELECTED_MODEL_KINDS

    _MODEL_KINDS = [mk.value for mk in SELECTED_MODEL_KINDS]
    train_results = train_one.expand(model_kind=_MODEL_KINDS)
    select_best_task = select_best(train_results)
    _parallel_branch_targets = ["train_one"]

    def _branch_by_mode(**context):
        """Ветвление по режиму выполнения.

        Приоритет выбора параллельности:
        1) UI-параметр DAG: params.parallel (bool)
        2) Переменная окружения PARALLEL (bool: "1"|"true"|...)
        3) Значение по умолчанию: standard
        """
        import os

        from scripts.logging_config import setup_auto_logging

        parallel_flag = _get_flag(context, "parallel", False)
        if not parallel_flag:
            parallel_env = os.getenv("PARALLEL", "0").strip()
            parallel_flag = parallel_env.lower() in {"1", "true", "yes", "on"}

        log = setup_auto_logging()
        log.info("Выбран режим: %s", "parallel" if parallel_flag else "standard")

        if parallel_flag:
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
    # В параллельном режиме используем динамический маппинг
    branch >> train_results
    train_results >> select_best_task

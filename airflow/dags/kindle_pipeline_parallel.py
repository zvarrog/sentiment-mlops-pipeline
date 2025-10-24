"""DAG с параллельным обучением моделей и выбором лучшей по F1-macro."""

from datetime import datetime

try:
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

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other

    class DAG(_Dummy):
        pass

    class PythonOperator(_Dummy):
        pass


_DOC = "Пайплайн с параллельным обучением моделей и автоматическим выбором лучшей."

default_args = {
    "start_date": datetime(2025, 1, 1),
}


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
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "kindle_parallel_experiment"


def _task_download(**context):
    _setup_env(**context)
    import os

    from scripts.download import CSV_PATH
    from scripts.download import main as download_main
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Начинаем загрузку данных из Kaggle")

    force = os.environ.get("FORCE_DOWNLOAD", "0") == "1"
    if not force and CSV_PATH.exists():
        log.info("CSV уже существует: %s — пропуск загрузки", str(CSV_PATH))
        return str(CSV_PATH.resolve())

    csv_abs = download_main()
    log.info("Данные загружены: %s", csv_abs)
    return str(csv_abs)


def _task_validate_schema(**context):
    _setup_env(**context)
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Валидация схемы данных")

    from pathlib import Path

    from scripts.data_validation import KINDLE_REVIEWS_SCHEMA, validate_parquet_file
    from scripts.settings import PROCESSED_DATA_DIR

    train_file = Path(PROCESSED_DATA_DIR) / "train.parquet"
    if not train_file.exists():
        log.warning("Файл train.parquet не найден — создаем обработанные данные")
        from scripts.spark_process import main as process_main

        process_main()

    result = validate_parquet_file(train_file, KINDLE_REVIEWS_SCHEMA)
    if not result.is_valid:
        raise ValueError(f"Ошибка валидации схемы: {result.errors}")

    log.info("Валидация схемы успешна")
    return "schema_ok"


def _task_validate_quality(**context):
    _setup_env(**context)
    from scripts.logging_config import setup_auto_logging

    log = setup_auto_logging()
    log.info("Валидация качества данных")

    from pathlib import Path

    from scripts.data_validation import validate_parquet_dataset
    from scripts.settings import PROCESSED_DATA_DIR

    results = validate_parquet_dataset(Path(PROCESSED_DATA_DIR))

    errors = []
    for name, result in results.items():
        if not result.is_valid:
            errors.extend([f"{name}: {e}" for e in result.errors])

    if errors:
        raise ValueError(f"Ошибки валидации качества: {errors}")

    log.info("Валидация качества успешна")
    return "quality_ok"


def _train_model_wrapper(model_kind: str, **context):
    _setup_env(**context)
    import os

    os.environ["FORCE_TRAIN"] = "1"

    from scripts.logging_config import setup_auto_logging
    from scripts.models.kinds import ModelKind

    log = setup_auto_logging()
    log.info(f"Обучение модели: {model_kind}")

    import json
    import time
    from pathlib import Path

    import joblib
    import mlflow
    import optuna

    from scripts.settings import (
        MODEL_ARTEFACTS_DIR,
        OPTUNA_N_TRIALS,
        OPTUNA_STORAGE,
    )
    from scripts.train import build_pipeline, compute_metrics, objective
    from scripts.train_modules.data_loading import load_splits

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
        from scripts.train_modules.feature_space import NUMERIC_COLS

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


def _task_train_logreg(**context):
    return _train_model_wrapper("logreg", **context)


def _task_train_rf(**context):
    return _train_model_wrapper("rf", **context)


def _task_train_gb(**context):
    return _train_model_wrapper("hist_gb", **context)


def _task_select_best(**context):
    _setup_env(**context)
    import shutil
    from pathlib import Path

    from scripts.logging_config import setup_auto_logging
    from scripts.settings import MODEL_ARTEFACTS_DIR, MODEL_FILE_DIR

    log = setup_auto_logging()
    log.info("Выбор лучшей модели из обученных")

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

    best = max(results, key=lambda x: x["val_f1_macro"])
    log.info(
        f"Лучшая модель: {best['model']} с val_f1_macro={best['val_f1_macro']:.4f}"
    )

    MODEL_FILE_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_FILE_DIR / "best_model.joblib"

    src_model = Path(best["model_path"])
    if src_model.exists():
        shutil.copy2(src_model, best_model_path)
        log.info(f"Лучшая модель скопирована в {best_model_path}")

    best_meta_path = Path(MODEL_ARTEFACTS_DIR) / "best_model_meta.json"
    src_meta = Path(best["meta_path"])
    if src_meta.exists():
        shutil.copy2(src_meta, best_meta_path)

    log.info("Выбор лучшей модели завершен")
    return best


with DAG(
    dag_id="kindle_reviews_parallel_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description=_DOC,
    tags=["sentiment", "parallel", "ml"],
) as dag:
    download = PythonOperator(
        task_id="download",
        python_callable=_task_download,
    )

    validate_schema = PythonOperator(
        task_id="validate_schema",
        python_callable=_task_validate_schema,
    )

    validate_quality = PythonOperator(
        task_id="validate_quality",
        python_callable=_task_validate_quality,
    )

    train_logreg = PythonOperator(
        task_id="train_logreg",
        python_callable=_task_train_logreg,
    )

    train_rf = PythonOperator(
        task_id="train_rf",
        python_callable=_task_train_rf,
    )

    train_gb = PythonOperator(
        task_id="train_gb",
        python_callable=_task_train_gb,
    )

    select_best = PythonOperator(
        task_id="select_best",
        python_callable=_task_select_best,
    )

    download >> [validate_schema, validate_quality]
    [validate_schema, validate_quality] >> [train_logreg, train_rf, train_gb]
    [train_logreg, train_rf, train_gb] >> select_best

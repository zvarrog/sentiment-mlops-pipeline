"""Обучение моделей с Optuna и MLflow без тестовых костылей."""

import argparse
import json
import logging
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from scripts.config import (
    BEST_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_ARTEFACTS_DIR,
    MODEL_DIR,
    N_FOLDS,
    NUMERIC_COLS,
    OPTUNA_STORAGE,
    SEED,
    SELECTED_MODEL_KINDS,
    STUDY_BASE_NAME,
    TRAIN_DEVICE,
)
from scripts.logging_config import get_logger
from scripts.model_registry import (
    load_old_model_metric,
    register_model_in_mlflow,
    should_replace_model,
)
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind
from scripts.train_modules.data_loading import load_splits
from scripts.train_modules.evaluation import compute_metrics, log_confusion_matrix
from scripts.train_modules.optuna_optimizer import optimize_model
from scripts.train_modules.pipeline_builders import ModelBuilderFactory
from scripts.utils import get_baseline_stats

# Настройка matplotlib для работы без дисплея
matplotlib.use("Agg")

log = get_logger("train")


def _configure_logging_levels() -> None:
    """Настройка уровней логирования для внешних библиотек."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("optuna").setLevel(logging.ERROR)
    logging.getLogger("git").setLevel(logging.ERROR)


def log_artifact_safe(path: Path, artifact_name: str) -> None:
    """Безопасное логирование артефакта в MLflow."""
    try:
        mlflow.log_artifact(str(path))
    except Exception as e:
        log.warning("Не удалось залогировать %s: %s", artifact_name, e)


def build_pipeline(
    trial: optuna.Trial, model_name: str | ModelKind, fixed_solver: str | None = None
) -> Pipeline:
    """Создаёт Pipeline для модели через фабрику строителей."""
    model_kind: ModelKind
    if isinstance(model_name, ModelKind):
        model_kind = model_name
    else:
        model_kind = ModelKind(str(model_name))

    builder = ModelBuilderFactory.get_builder(model_kind, trial, fixed_solver)
    return builder.build()


def _get_feature_names(preprocessor: ColumnTransformer, use_svd: bool) -> list[str]:
    """Извлекает имена признаков из препроцессора."""
    feature_names: list[str] = []

    try:
        text_pipe: Pipeline = preprocessor.named_transformers_["text"]
        tfidf: TfidfVectorizer = text_pipe.named_steps["tfidf"]
        vocab_inv = (
            {idx: tok for tok, idx in tfidf.vocabulary_.items()}
            if hasattr(tfidf, "vocabulary_")
            else {}
        )
        text_dim = len(vocab_inv) if vocab_inv else 0
        numeric_cols = preprocessor.transformers_[1][2]

        if not use_svd and vocab_inv:
            feature_names.extend(
                [vocab_inv.get(i, f"tok_{i}") for i in range(text_dim)]
            )
        elif use_svd and "svd" in text_pipe.named_steps:
            svd_model = text_pipe.named_steps["svd"]
            n_components = svd_model.n_components

            for comp_idx in range(n_components):
                component = svd_model.components_[comp_idx]
                top_indices = np.argsort(np.abs(component))[-10:][::-1]
                top_terms = [vocab_inv.get(i, f"tok_{i}") for i in top_indices]
                feature_name = f"svd_{comp_idx}[{','.join(top_terms[:3])}...]"
                feature_names.append(feature_name)

        feature_names.extend(list(numeric_cols))
    except (KeyError, AttributeError, IndexError) as e:
        log.warning("Ошибка при извлечении имен признаков: %s", e)

    return feature_names


def _get_model_coefficients(model: Any) -> np.ndarray | None:
    """Извлекает коэффициенты или feature importances из модели."""
    if hasattr(model, "coef_"):
        return np.mean(np.abs(model.coef_), axis=0)
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None


def _extract_feature_importances(
    pipeline: Pipeline, use_svd: bool
) -> list[dict[str, float]]:
    """Извлекает топ-50 наиболее важных признаков из обученной модели."""
    res: list[dict[str, float]] = []
    try:
        if "pre" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            return res

        model = pipeline.named_steps["model"]
        pre: ColumnTransformer = pipeline.named_steps["pre"]

        feature_names = _get_feature_names(pre, use_svd)
        coefs = _get_model_coefficients(model)

        if coefs is None:
            return res

        top_idx = np.argsort(coefs)[::-1][:50]
        for i in top_idx:
            if i < len(feature_names):
                res.append({"feature": feature_names[i], "importance": float(coefs[i])})
    except (KeyError, AttributeError, ValueError) as e:
        log.warning("Не удалось извлечь feature importances: %s", e)
    return res


def setup_environment() -> None:
    """Настройка окружения: сигналы, директории, MLflow, логирование."""
    _configure_logging_levels()

    def signal_handler(signum, frame):
        log.warning("Получен сигнал %d, завершаю обучение...", signum)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)


def log_initial_params(x_train: pd.DataFrame) -> None:
    """Логирование начальных параметров в MLflow."""
    mlflow.log_params(
        {
            "seed": SEED,
            "numeric_cols": ",".join([c for c in NUMERIC_COLS if c in x_train.columns]),
            "text_clean_stage": "spark_process",
            "cv_n_folds": N_FOLDS,
            "version_sklearn": sklearn.__version__,
            "version_optuna": optuna.__version__,
            "version_mlflow": mlflow.__version__,
            "version_pandas": pd.__version__,
        }
    )


def run_optimization(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    selected_models: list[ModelKind] | None,
) -> dict[ModelKind, dict[str, Any]]:
    """Запуск оптимизации Optuna для выбранных моделей."""
    per_model_results: dict[ModelKind, dict[str, Any]] = {}
    models_to_train = (
        selected_models if selected_models is not None else SELECTED_MODEL_KINDS
    )

    for model_kind in models_to_train:
        with mlflow.start_run(nested=True, run_name=f"model={model_kind.value}"):
            log.info("Начало оптимизации модели: %s", model_kind.value)

            study = optimize_model(
                STUDY_BASE_NAME,
                model_kind,
                x_train,
                y_train,
                x_val,
                y_val,
                OPTUNA_STORAGE,
            )

            if not study.trials or not study.best_trial:
                log.warning("%s: нет успешных trials — пропуск", model_kind.value)
                continue

            best_trial = study.best_trial
            per_model_results[model_kind] = {
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "study_name": study.study_name,
            }

    return per_model_results


def train_final_model(
    best_model: ModelKind,
    best_params: dict[str, Any],
    x_full: pd.DataFrame,
    y_full: np.ndarray,
) -> Any:
    """Обучение финальной модели на полном наборе данных (train + val)."""
    if best_model is ModelKind.distilbert:
        epochs = best_params.get("db_epochs", 2)
        lr = best_params.get("db_lr", 2e-5)
        max_len = best_params.get("db_max_len", 160)
        use_bi = best_params.get("db_use_bigrams", False)
        final_pipeline = DistilBertClassifier(
            epochs=epochs,
            lr=lr,
            max_len=max_len,
            device=TRAIN_DEVICE,
            use_bigrams=use_bi,
        )
        final_pipeline.fit(x_full["reviewText"], y_full)
    else:
        fixed_trial = optuna.trial.FixedTrial(best_params)
        # Прокидываем список доступных числовых колонок через user_attrs
        fixed_trial.set_user_attr(
            "numeric_cols", [c for c in NUMERIC_COLS if c in x_full.columns]
        )
        final_pipeline = build_pipeline(fixed_trial, best_model)
        final_pipeline.fit(x_full, y_full)

    return final_pipeline


def evaluate_model(
    final_pipeline: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    best_model: ModelKind,
) -> dict[str, float]:
    """Оценка финальной модели на тестовой выборке."""
    if best_model is ModelKind.distilbert:
        test_preds = final_pipeline.predict(x_test["reviewText"])
    else:
        test_preds = final_pipeline.predict(x_test)

    test_metrics = compute_metrics(y_test, test_preds)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    cm_path = MODEL_ARTEFACTS_DIR / "confusion_matrix_test.png"
    log_confusion_matrix(y_test, test_preds, cm_path)
    log_artifact_safe(cm_path, "confusion_matrix")

    cr_txt = classification_report(y_test, test_preds)
    cr_path = MODEL_ARTEFACTS_DIR / "classification_report_test.txt"
    cr_path.write_text(cr_txt, encoding="utf-8")
    log_artifact_safe(cr_path, "classification_report")

    # Ошибки классификации
    mis_idx = np.where(test_preds != y_test)[0]
    if len(mis_idx):
        mis_samples = x_test.iloc[mis_idx].copy()
        mis_samples["true"] = (
            y_test.iloc[mis_idx] if isinstance(y_test, pd.Series) else y_test[mis_idx]
        )
        mis_samples["pred"] = test_preds[mis_idx]

        mis_path = MODEL_ARTEFACTS_DIR / "misclassified_samples_test.csv"
        mis_samples.head(200).to_csv(mis_path, index=False)
        log_artifact_safe(mis_path, "misclassified_samples")

    return test_metrics


def save_artifacts(
    final_pipeline: Any,
    best_model: ModelKind,
    best_params: dict[str, Any],
    baseline_stats: dict[str, Any],
    meta_info: dict[str, Any],
    y_test: pd.Series,
    x_test: pd.DataFrame,
) -> None:
    """Сохранение артефактов модели, графиков и метаданных."""
    joblib.dump(final_pipeline, BEST_MODEL_PATH)
    log_artifact_safe(BEST_MODEL_PATH, "best_model")

    bs_path = MODEL_ARTEFACTS_DIR / "baseline_numeric_stats.json"
    bs_path.write_text(
        json.dumps(baseline_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_artifact_safe(bs_path, "baseline_stats")

    # Feature importances
    if best_model is not ModelKind.distilbert:
        _save_feature_importances(final_pipeline, best_params)

    # ROC/PR curves
    _save_curves(final_pipeline, best_model, x_test, y_test)

    # Meta info
    _meta_path = MODEL_ARTEFACTS_DIR / "best_model_meta.json"
    _meta_path.write_text(
        json.dumps(meta_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_feature_importances(final_pipeline: Any, best_params: dict[str, Any]) -> None:
    try:
        use_svd_flag = False
        pre: ColumnTransformer | None = final_pipeline.named_steps.get("pre")
        if pre is not None:
            try:
                text_pipe: Pipeline = pre.named_transformers_["text"]
                use_svd_flag = "svd" in getattr(text_pipe, "named_steps", {})
            except (KeyError, AttributeError):
                use_svd_flag = bool(best_params.get("use_svd", False))

        fi_list = _extract_feature_importances(final_pipeline, use_svd_flag)
        if fi_list:
            fi_path = MODEL_ARTEFACTS_DIR / "feature_importances.json"
            with open(fi_path, "w", encoding="utf-8") as f:
                json.dump(fi_list, f, ensure_ascii=False, indent=2)
            log_artifact_safe(fi_path, "feature_importances")
    except (OSError, ValueError, KeyError, AttributeError) as e:
        log.warning("Не удалось сохранить feature importances: %s", e)


def _save_curves(
    final_pipeline: Any, best_model: ModelKind, x_test: pd.DataFrame, y_test: pd.Series
) -> None:
    try:
        if best_model is ModelKind.distilbert:
            x_for_proba = x_test["reviewText"]
        else:
            x_for_proba = x_test

        if hasattr(final_pipeline, "predict_proba"):
            y_score = final_pipeline.predict_proba(x_for_proba)
            classes = sorted(set(y_test.tolist()))
            y_true_bin = label_binarize(y_test, classes=classes)

            # ROC
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
            ax_roc.plot(fpr, tpr, label=f"micro-avg ROC (AUC={roc_auc:.3f})")
            ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax_roc.set_title("ROC Curve (micro)")
            ax_roc.legend(loc="lower right", fontsize=8)

            roc_path = MODEL_ARTEFACTS_DIR / "roc_curve_test.png"
            fig_roc.tight_layout()
            fig_roc.savefig(roc_path)
            plt.close(fig_roc)
            log_artifact_safe(roc_path, "roc_curve")

            # PR
            precision, recall, _ = precision_recall_curve(
                y_true_bin.ravel(), y_score.ravel()
            )
            ap_micro = average_precision_score(y_true_bin, y_score, average="micro")
            fig_pr, ax_pr = plt.subplots(figsize=(4, 4))
            ax_pr.plot(recall, precision, label=f"micro-avg PR (AP={ap_micro:.3f})")
            ax_pr.set_title("Precision-Recall Curve (micro)")
            ax_pr.legend(loc="lower left", fontsize=8)

            pr_path = MODEL_ARTEFACTS_DIR / "pr_curve_test.png"
            fig_pr.tight_layout()
            fig_pr.savefig(pr_path)
            plt.close(fig_pr)
            log_artifact_safe(pr_path, "pr_curve")
    except (ValueError, OSError, RuntimeError) as e:
        log.warning("Не удалось построить ROC/PR кривые: %s", e)


def run(
    force: bool = False,
    selected_models: list[ModelKind] | None = None,
) -> None:
    """Основная функция запуска обучения."""
    setup_environment()

    log.info("force=%s, наличие best_model=%s", force, BEST_MODEL_PATH.exists())

    old_model_metric = load_old_model_metric()

    if BEST_MODEL_PATH.exists() and not force:
        log.info("Модель уже существует и force=False — пропуск")
        return

    x_train, x_val, x_test, y_train, y_val, y_test = load_splits()
    log.info(
        "Размеры: train=%d, val=%d, test=%d", len(x_train), len(x_val), len(x_test)
    )

    start_time = time.time()

    with mlflow.start_run(run_name="classical_pipeline"):
        log_initial_params(x_train)
        baseline_stats = get_baseline_stats(x_train)

        per_model_results = run_optimization(
            x_train, y_train, x_val, y_val, selected_models
        )

        if not per_model_results:
            log.error("Нет ни одного успешного результата оптимизации — выход")
            raise RuntimeError("Оптимизация не дала успешных результатов")

        # Выбираем лучшую модель
        best_model = max(per_model_results.items(), key=lambda x: x[1]["best_value"])[0]
        best_info = per_model_results[best_model]
        new_model_metric = best_info["best_value"]

        if not force and not should_replace_model(
            new_model_metric, old_model_metric, best_model.value
        ):
            return

        log.info(
            "Лучшая модель: %s (val_f1_macro=%.4f)", best_model.value, new_model_metric
        )
        mlflow.log_param("best_model", best_model.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_info["best_params"].items()})
        mlflow.log_metric("best_val_f1_macro", best_info["best_value"])

        # Retrain
        x_full = pd.concat([x_train, x_val], axis=0, ignore_index=True)
        y_full = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)

        final_pipeline = train_final_model(
            best_model, best_info["best_params"], x_full, y_full
        )

        # Evaluate
        test_metrics = evaluate_model(final_pipeline, x_test, y_test, best_model)

        # Register
        register_model_in_mlflow(
            BEST_MODEL_PATH,
            best_model,
            test_metrics.get("f1_macro", 0.0),
            mlflow_run_active=True,
        )

        duration = time.time() - start_time
        mlflow.log_metric("training_duration_sec", duration)

        meta_info = {
            "best_model": best_model.value,
            "best_params": best_info["best_params"],
            "best_val_f1_macro": best_info["best_value"],
            "test_metrics": test_metrics,
            "sizes": {"train": len(x_train), "val": len(x_val), "test": len(x_test)},
            "duration_sec": duration,
        }

        save_artifacts(
            final_pipeline,
            best_model,
            best_info["best_params"],
            baseline_stats,
            meta_info,
            y_test,
            x_test,
        )


def main() -> None:
    """Точка входа: парсит аргументы командной строки и запускает run()."""
    parser = argparse.ArgumentParser(description="Обучение моделей с Optuna и MLflow")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительное переобучение даже если модель существует",
    )
    args = parser.parse_args()

    run(force=args.force)


if __name__ == "__main__":
    main()

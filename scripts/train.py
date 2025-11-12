"""Пайплайн обучения с оптимизацией гиперпараметров Optuna и отслеживанием MLflow."""

import json
import logging
import os
import signal
import sys
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from scripts.config import (
    BEST_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_ARTEFACTS_DIR,
    MODEL_DIR,
    N_FOLDS,
    OPTUNA_STORAGE,
    SEED,
    SELECTED_MODEL_KINDS,
    STUDY_BASE_NAME,
    TRAIN_DEVICE,
)
from scripts.constants import NUMERIC_COLS
from scripts.logging_config import get_logger
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind
from scripts.train_modules import (
    ModelBuilderFactory,
    compute_metrics,
    load_splits,
    log_confusion_matrix,
    optimize_model,
)

log = get_logger("train")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.ERROR)
logging.getLogger("git").setLevel(logging.ERROR)

EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "kindle_experiment")

_model_sig = "_".join([m.value[:3] for m in sorted(SELECTED_MODEL_KINDS)])
OPTUNA_STUDY_NAME = f"{STUDY_BASE_NAME}_{_model_sig}"


def log_artifact_safe(path: Path, artifact_name: str) -> None:
    """Безопасное логирование артефакта в MLflow."""
    try:
        mlflow.log_artifact(str(path))
    except Exception as e:
        log.warning("Не удалось залогировать %s: %s", artifact_name, e)


def build_pipeline(
    trial: optuna.Trial, model_name, fixed_solver: str | None = None
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
        feature_names.extend([vocab_inv.get(i, f"tok_{i}") for i in range(text_dim)])
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
    return feature_names


def _get_model_coefficients(model) -> np.ndarray | None:
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


def run():
    def signal_handler(signum, frame):
        log.warning("Получен сигнал %d, завершаю обучение...", signum)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    from scripts.config import FORCE_TRAIN

    force_train = FORCE_TRAIN
    log.info(
        "FORCE_TRAIN=%s, наличие best_model=%s", force_train, BEST_MODEL_PATH.exists()
    )

    from scripts.model_registry import load_old_model_metric

    old_model_metric = load_old_model_metric()

    if BEST_MODEL_PATH.exists() and not force_train:
        log.info("Модель уже существует и FORCE_TRAIN=False — пропуск")
        return

    x_train, x_val, x_test, y_train, y_val, y_test = load_splits()
    log.info(
        "Размеры: train=%d, val=%d, test=%d", len(x_train), len(x_val), len(x_test)
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    start_time = time.time()

    with mlflow.start_run(run_name="classical_pipeline"):
        # общие параметры
        mlflow.log_params(
            {
                "seed": SEED,
                "numeric_cols": ",".join(
                    [c for c in NUMERIC_COLS if c in x_train.columns]
                ),
                "text_clean_stage": "spark_process",
                "cv_n_folds": N_FOLDS,
            }
        )

        # версии библиотек
        mlflow.log_params(
            {
                "version_sklearn": sklearn.__version__,
                "version_optuna": optuna.__version__,
                "version_mlflow": mlflow.__version__,
                "version_pandas": pd.__version__,
            }
        )

        # baseline статистики числовых признаков
        from scripts.utils import get_baseline_stats

        baseline_stats = get_baseline_stats(x_train)

        # Оптимизируем каждую модель
        per_model_results: dict[ModelKind, dict[str, object]] = {}

        for model_kind in SELECTED_MODEL_KINDS:
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

        if not per_model_results:
            log.error("Нет ни одного успешного результата оптимизации — выход")
            raise RuntimeError("Оптимизация не дала успешных результатов")

        # Выбираем лучшую модель по best_value
        best_model = max(per_model_results.items(), key=lambda x: x[1]["best_value"])[0]
        best_info = per_model_results[best_model]
        new_model_metric = best_info["best_value"]

        from scripts.model_registry import should_replace_model

        if not should_replace_model(
            new_model_metric, old_model_metric, best_model.value
        ):
            return

        log.info(
            "Лучшая модель: %s (val_f1_macro=%.4f)", best_model.value, new_model_metric
        )
        mlflow.log_param("best_model", best_model.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_info["best_params"].items()})
        mlflow.log_metric("best_val_f1_macro", best_info["best_value"])

        # Retrain на train+val
        x_full = pd.concat([x_train, x_val], axis=0, ignore_index=True)
        y_full = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
        best_params = best_info["best_params"]

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

        joblib.dump(final_pipeline, BEST_MODEL_PATH)
        log_artifact_safe(BEST_MODEL_PATH, "best_model")

        bs_path = MODEL_ARTEFACTS_DIR / "baseline_numeric_stats.json"
        bs_path.write_text(
            json.dumps(baseline_stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_artifact_safe(bs_path, "baseline_stats")

        # Тестовая оценка (до регистрации в MLflow Registry)
        if best_model is ModelKind.distilbert:
            test_preds = final_pipeline.predict(x_test["reviewText"])
        else:
            test_preds = final_pipeline.predict(x_test)
        test_metrics = compute_metrics(y_test, test_preds)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        from scripts.model_registry import register_model_in_mlflow

        register_model_in_mlflow(
            BEST_MODEL_PATH,
            best_model,
            test_metrics.get("f1_macro", 0.0),
            mlflow_run_active=True,
        )

        cm_path = MODEL_ARTEFACTS_DIR / "confusion_matrix_test.png"
        log_confusion_matrix(y_test, test_preds, cm_path)
        log_artifact_safe(cm_path, "confusion_matrix")
        cr_txt = classification_report(y_test, test_preds)

        cr_path = MODEL_ARTEFACTS_DIR / "classification_report_test.txt"
        cr_path.write_text(cr_txt, encoding="utf-8")
        log_artifact_safe(cr_path, "classification_report")

        # Feature importances (для классических моделей)
        if best_model is not ModelKind.distilbert:
            try:
                use_svd_flag = False
                try:
                    # Предпочитаем детектировать по финальному пайплайну
                    pre: ColumnTransformer | None = final_pipeline.named_steps.get(
                        "pre"
                    )
                    if pre is not None:
                        text_pipe: Pipeline = pre.named_transformers_["text"]
                        use_svd_flag = "svd" in getattr(text_pipe, "named_steps", {})
                except (KeyError, AttributeError):
                    # Фолбэк по best_params
                    use_svd_flag = bool(best_params.get("use_svd", False))

                fi_list = _extract_feature_importances(final_pipeline, use_svd_flag)
                if fi_list:
                    fi_path = MODEL_ARTEFACTS_DIR / "feature_importances.json"
                    with open(fi_path, "w", encoding="utf-8") as f:
                        json.dump(fi_list, f, ensure_ascii=False, indent=2)
                    log_artifact_safe(fi_path, "feature_importances")
            except (OSError, ValueError, KeyError, AttributeError) as e:
                log.warning("Не удалось сохранить feature importances: %s", e)

        # Снимок лучших трейлов Optuna (top-K)
        try:
            top_k = int(os.environ.get("OPTUNA_TOPK_EXPORT", "20"))
            study_name = best_info.get("study_name")
            if isinstance(study_name, str) and study_name:
                study = optuna.load_study(storage=OPTUNA_STORAGE, study_name=study_name)
                valid_trials = [t for t in study.trials if t.value is not None]
                valid_trials.sort(key=lambda t: t.value, reverse=True)
                top_trials = valid_trials[:top_k]
                if top_trials:
                    import pandas as _pd

                    # Собираем плоский датафрейм: number, value, затем параметры
                    all_param_keys = sorted(
                        {key for t in top_trials for key in t.params}
                    )
                    rows = []
                    for t in top_trials:
                        row = {"number": t.number, "value": t.value}
                        for k in all_param_keys:
                            row[k] = t.params.get(k)
                        rows.append(row)
                    df = _pd.DataFrame(rows)

                    csv_path = MODEL_ARTEFACTS_DIR / "optuna_top_trials.csv"
                    df.to_csv(csv_path, index=False)
                    log_artifact_safe(csv_path, "optuna_top_trials")
        except (ValueError, KeyError, AttributeError, OSError) as e:
            log.warning("Не удалось сохранить топ-K трейлов Optuna: %s", e)

        # Схема входа/выхода модели и реально использованные фичи
        try:
            schema: dict[str, object] = {"input": {}, "output": {}}
            if best_model is ModelKind.distilbert:
                schema["input"] = {"text_column": "reviewText"}
                classes = sorted(set(y_full.tolist()))
                schema["output"] = {"target_dtype": "int", "classes": classes}
            else:
                pre: ColumnTransformer = final_pipeline.named_steps.get("pre")
                text_info: dict[str, object] = {"text_column": "reviewText"}
                numeric_cols_used: list[str] = []
                text_dim = None
                if pre is not None:
                    try:
                        # numeric фичи брались из второго трансформера
                        numeric_cols_used = list(pre.transformers_[1][2])
                    except (KeyError, IndexError, AttributeError):
                        numeric_cols_used = []
                    try:
                        text_pipe: Pipeline = pre.named_transformers_["text"]
                        if "svd" in text_pipe.named_steps:
                            text_dim = int(text_pipe.named_steps["svd"].n_components)
                        else:
                            tfidf: TfidfVectorizer = text_pipe.named_steps["tfidf"]
                            vocab_size = (
                                len(tfidf.vocabulary_)
                                if hasattr(tfidf, "vocabulary_") and tfidf.vocabulary_
                                else 0
                            )
                            text_dim = int(vocab_size)
                    except (KeyError, AttributeError):
                        pass
                text_info["text_dim"] = text_dim if text_dim is not None else "unknown"
                schema["input"] = {
                    "text": text_info,
                    "numeric_features": numeric_cols_used,
                }
                classes = sorted(set(y_full.tolist()))
                schema["output"] = {"target_dtype": "int", "classes": classes}

            schema_path = MODEL_ARTEFACTS_DIR / "model_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
            log_artifact_safe(schema_path, "model_schema")
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
            log.warning("Не удалось сохранить схему модели: %s", e)

        # 4) ROC/PR кривые по тесту (если у модели есть predict_proba)
        try:
            if best_model is ModelKind.distilbert:
                x_for_proba = x_test["reviewText"]
            else:
                x_for_proba = x_test
            if hasattr(final_pipeline, "predict_proba"):
                from sklearn.metrics import (
                    auc,
                    average_precision_score,
                    precision_recall_curve,
                    roc_curve,
                )
                from sklearn.preprocessing import label_binarize

                y_score = final_pipeline.predict_proba(x_for_proba)
                classes = sorted(set(y_test.tolist()))
                y_true_bin = label_binarize(y_test, classes=classes)
                # Защита: некоторые модели возвращают proba без последнего класса
                if y_score.shape[1] != y_true_bin.shape[1]:
                    # Приведём к общему виду, заполняя недостающие классы нулями
                    import numpy as _np

                    proba_aligned = _np.zeros(
                        (y_score.shape[0], y_true_bin.shape[1]), dtype=float
                    )
                    # Предполагаем порядок классов как в model.classes_, если доступен
                    try:
                        cls_model = list(getattr(final_pipeline, "classes_", []))
                    except AttributeError:
                        cls_model = []
                    for j, c in enumerate(classes):
                        if cls_model and c in cls_model:
                            src_idx = cls_model.index(c)
                            if src_idx < y_score.shape[1]:
                                proba_aligned[:, j] = y_score[:, src_idx]
                    y_score = proba_aligned

                # ROC micro-average
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                ax_roc.plot(fpr, tpr, label=f"micro-avg ROC (AUC={roc_auc:.3f})")
                ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
                ax_roc.set_xlabel("FPR")
                ax_roc.set_ylabel("TPR")
                ax_roc.set_title("ROC Curve (micro)")
                ax_roc.legend(loc="lower right", fontsize=8)

                roc_path = MODEL_ARTEFACTS_DIR / "roc_curve_test.png"
                fig_roc.tight_layout()
                fig_roc.savefig(roc_path)
                plt.close(fig_roc)
                log_artifact_safe(roc_path, "roc_curve")

                # PR micro-average
                precision, recall, _ = precision_recall_curve(
                    y_true_bin.ravel(), y_score.ravel()
                )
                ap_micro = average_precision_score(y_true_bin, y_score, average="micro")
                fig_pr, ax_pr = plt.subplots(figsize=(4, 4))
                ax_pr.plot(recall, precision, label=f"micro-avg PR (AP={ap_micro:.3f})")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision-Recall Curve (micro)")
                ax_pr.legend(loc="lower left", fontsize=8)

                pr_path = MODEL_ARTEFACTS_DIR / "pr_curve_test.png"
                fig_pr.tight_layout()
                fig_pr.savefig(pr_path)
                plt.close(fig_pr)
                log_artifact_safe(pr_path, "pr_curve")
        except (ValueError, OSError, RuntimeError) as e:
            log.warning("Не удалось построить ROC/PR кривые: %s", e)

        # Ошибки классификации (первые 200)
        mis_idx = np.where(test_preds != y_test)[0]
        if len(mis_idx):
            mis_samples = x_test.iloc[mis_idx].copy()
            mis_samples["true"] = y_test[mis_idx]
            mis_samples["pred"] = test_preds[mis_idx]

            mis_path = MODEL_ARTEFACTS_DIR / "misclassified_samples_test.csv"
            mis_samples.head(200).to_csv(mis_path, index=False)
            log_artifact_safe(mis_path, "misclassified_samples")

        duration = time.time() - start_time
        mlflow.log_metric("training_duration_sec", duration)

        meta = {
            "best_model": (
                best_model.value if hasattr(best_model, "value") else str(best_model)
            ),
            "best_params": best_params,
            "best_val_f1_macro": best_info["best_value"],
            "test_metrics": test_metrics,
            "sizes": {"train": len(x_train), "val": len(x_val), "test": len(x_test)},
            "duration_sec": duration,
        }
        _meta_path = MODEL_ARTEFACTS_DIR / "best_model_meta.json"
        _meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    run()

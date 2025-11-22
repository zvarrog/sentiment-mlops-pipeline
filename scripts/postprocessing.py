"""Общий модуль для формирования пакета артефактов лучшей модели.

Функция generate_best_bundle() создает артефакты:
- baseline_numeric_stats.json
- confusion_matrix_test.png
- classification_report_test.txt
- feature_importances.json (для классических моделей)
- model_schema.json
- roc_curve_test.png, pr_curve_test.png (если доступна predict_proba)
- misclassified_samples_test.csv
- best_model_meta.json (единый формат)

Предполагается, что модель уже сохранена как best_model.joblib в MODEL_FILE_DIR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from scripts.config import MODEL_ARTEFACTS_DIR
from scripts.logging_config import get_logger

log = get_logger("postprocessing")


def _save_confusion_and_report(y_true, y_pred, out_dir: Path) -> tuple[Path, Path]:
    from scripts.train_modules.evaluation import log_confusion_matrix

    out_dir = Path(out_dir)
    cm_path = out_dir / "confusion_matrix_test.png"
    log_confusion_matrix(y_true, y_pred, cm_path)

    cr_txt = classification_report(y_true, y_pred)
    cr_path = out_dir / "classification_report_test.txt"
    cr_path.write_text(cr_txt, encoding="utf-8")
    return cm_path, cr_path


def _save_feature_importances(pipeline, out_dir: Path) -> Path | None:
    from scripts.train import _extract_feature_importances

    try:
        pre = (
            getattr(pipeline, "named_steps", {}).get("pre")
            if hasattr(pipeline, "named_steps")
            else None
        )
        use_svd_flag = False
        if (
            pre is not None
            and hasattr(pre, "named_transformers_")
            and "text" in pre.named_transformers_
        ):
            text_pipe = pre.named_transformers_["text"]
            use_svd_flag = "svd" in getattr(text_pipe, "named_steps", {})

        fi_list = _extract_feature_importances(pipeline, use_svd_flag)
        if fi_list:
            fi_path = Path(out_dir) / "feature_importances.json"
            with open(fi_path, "w", encoding="utf-8") as f:
                json.dump(fi_list, f, ensure_ascii=False, indent=2)
            return fi_path
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        log.warning("Не удалось сохранить feature importances: %s", e)
    return None


def _save_schema(pipeline, classes, x_train: pd.DataFrame, out_dir: Path) -> Path:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline as SkPipeline

    schema: dict[str, Any] = {"input": {}, "output": {}}
    if hasattr(pipeline, "named_steps") and "pre" in pipeline.named_steps:
        pre: ColumnTransformer = pipeline.named_steps.get("pre")  # type: ignore
        text_info: dict[str, Any] = {"text_column": "reviewText"}
        numeric_cols_used: list[str] = []
        text_dim = None
        if pre is not None:
            try:
                numeric_cols_used = list(pre.transformers_[1][2])
            except (KeyError, IndexError, AttributeError, TypeError):
                numeric_cols_used = []
            try:
                text_pipe: SkPipeline = pre.named_transformers_["text"]
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
        schema["input"] = {"text": text_info, "numeric_features": numeric_cols_used}
        schema["output"] = {"target_dtype": "int", "classes": sorted(set(classes))}
    else:
        schema["input"] = {"text_column": "reviewText"}
        schema["output"] = {"target_dtype": "int", "classes": sorted(set(classes))}

    schema_path = Path(out_dir) / "model_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    return schema_path


def _save_roc_pr(
    pipeline, x_test: pd.DataFrame, y_test, out_dir: Path
) -> tuple[Path | None, Path | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            auc,
            average_precision_score,
            precision_recall_curve,
            roc_curve,
        )
        from sklearn.preprocessing import label_binarize

        if not hasattr(pipeline, "predict_proba"):
            return None, None

        x_for_proba = (
            x_test
            if "pre" in getattr(pipeline, "named_steps", {})
            else x_test["reviewText"]
        )
        y_score = pipeline.predict_proba(x_for_proba)
        classes = sorted(set(y_test.tolist()))
        y_true_bin = label_binarize(y_test, classes=classes)

        if y_score.shape[1] != y_true_bin.shape[1]:
            proba_aligned = np.zeros(
                (y_score.shape[0], y_true_bin.shape[1]), dtype=float
            )
            try:
                cls_model = list(getattr(pipeline, "classes_", []))
            except AttributeError:
                cls_model = []
            for j, c in enumerate(classes):
                if cls_model and c in cls_model:
                    src_idx = cls_model.index(c)
                    if src_idx < y_score.shape[1]:
                        proba_aligned[:, j] = y_score[:, src_idx]
            y_score = proba_aligned

        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
        ax_roc.plot(fpr, tpr, label=f"micro-avg ROC (AUC={roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("ROC Curve (micro)")
        ax_roc.legend(loc="lower right", fontsize=8)
        roc_path = Path(out_dir) / "roc_curve_test.png"
        fig_roc.tight_layout()
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)

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
        pr_path = Path(out_dir) / "pr_curve_test.png"
        fig_pr.tight_layout()
        fig_pr.savefig(pr_path)
        plt.close(fig_pr)

        return roc_path, pr_path
    except (ValueError, OSError, RuntimeError) as e:
        log.warning("Не удалось построить ROC/PR: %s", e)
        return None, None


def _save_misclassified(
    x_test: pd.DataFrame, y_test, y_pred, out_dir: Path
) -> Path | None:
    try:
        mis_idx = np.where(y_pred != y_test)[0]
        if len(mis_idx) == 0:
            return None
        mis_samples = x_test.iloc[mis_idx].copy()
        mis_samples["true"] = y_test[mis_idx]
        mis_samples["pred"] = y_pred[mis_idx]
        out_path = Path(out_dir) / "misclassified_samples_test.csv"
        mis_samples.head(200).to_csv(out_path, index=False)
        return out_path
    except (OSError, ValueError, TypeError) as e:
        log.warning("Не удалось сохранить ошибки классификации: %s", e)
        return None


def generate_best_bundle(
    best_model: str,
    best_params: dict[str, Any],
    best_val_f1_macro: float,
    pipeline_path: Path,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train,
    y_val,
    y_test,
    artefacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Генерирует пакет артефактов и пишет best_model_meta.json.

    Returns:
        dict с meta (для удобства тестирования/логирования).
    """
    out_dir = Path(artefacts_dir or MODEL_ARTEFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.utils import atomic_write_json, get_baseline_stats

    baseline_stats = get_baseline_stats(x_train)
    baseline_path = out_dir / "baseline_numeric_stats.json"
    atomic_write_json(baseline_path, baseline_stats)

    pipeline = joblib.load(pipeline_path)
    try:
        x_for_test = (
            x_test
            if "pre" in getattr(pipeline, "named_steps", {})
            else x_test["reviewText"]
        )
        test_preds = pipeline.predict(x_for_test)
    except (ValueError, TypeError, AttributeError, RuntimeError):
        test_preds = None

    test_metrics: dict[str, Any] = {}
    if test_preds is not None:
        from scripts.train_modules.evaluation import compute_metrics

        _save_confusion_and_report(y_test, test_preds, out_dir)
        test_metrics = compute_metrics(y_test, test_preds)

    # 4) схема и FI
    classes = sorted(
        set(
            pd.Series(y_train).tolist()
            + pd.Series(y_val).tolist()
            + pd.Series(y_test).tolist()
        )
    )
    _save_schema(pipeline, classes, x_train, out_dir)
    _save_feature_importances(pipeline, out_dir)

    # 5) ROC/PR и ошибки
    if test_preds is not None:
        _save_roc_pr(pipeline, x_test, y_test, out_dir)
        _save_misclassified(x_test, y_test, test_preds, out_dir)

    # 6) формируем meta
    sizes = {"train": len(x_train), "val": len(x_val), "test": len(x_test)}
    meta = {
        "best_model": best_model,
        "best_params": best_params,
        "best_val_f1_macro": float(best_val_f1_macro),
        "test_metrics": test_metrics,
        "sizes": sizes,
    }
    meta_path = out_dir / "best_model_meta.json"
    atomic_write_json(meta_path, meta)

    return meta

"""Полноценный training pipeline классических моделей с Optuna + MLflow.

Что добавлено относительно базовой версии:
 - Корректное разделение: train.parquet / val.parquet / test.parquet (валидация только на val)
 - TF-IDF вместо Hashing (контролируемое max_features) + опциональный TruncatedSVD (понижение размерности)
 - Стандартизация числовых признаков (StandardScaler)
 - Выбор метрики оптимизации: macro F1 (лучше для несбалансированных классов)
 - Логирование в MLflow: параметры, метрики (train/val/test), артефакты (confusion matrix, classification report)
 - Optuna c SQLite storage (возобновляемость) + MedianPruner
 - Retrain лучшего пайплайна на train+val и финальная оценка на test
 - Метаданные: размеры датасетов, время выполнения, версии библиотек
 - Детальная структура кода без временных демо (для production impression)

Файл не содержит тестового «вспомогательного» кода; всё экспериментальное должно располагаться в tests/.
"""

import json
import logging
import os
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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# локальные реализации метрик и визуализаций ниже
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind
from scripts.settings import (
    DISTILBERT_TIMEOUT_SEC,
    EARLY_STOP_PATIENCE,
    FORCE_SVD_THRESHOLD_MB,
    FORCE_TRAIN,
    MIN_TRIALS_BEFORE_EARLY_STOP,
    N_FOLDS,
    OPTUNA_N_TRIALS,
    OPTUNA_STORAGE,
    OPTUNA_TIMEOUT_SEC,
    SEED,
    SELECTED_MODEL_KINDS,
    STUDY_BASE_NAME,
    TFIDF_MAX_FEATURES_MAX,
    TFIDF_MAX_FEATURES_MIN,
    TFIDF_MAX_FEATURES_STEP,
    TRAIN_DEVICE,
    log,
)
from scripts.train_modules.data_loading import load_splits
from scripts.train_modules.feature_space import NUMERIC_COLS, DenseTransformer
from scripts.train_modules.models import SimpleMLP

# Подавляем избыточные предупреждения
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

# Подавляем MLflow/Git логи
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.ERROR)
logging.getLogger("git").setLevel(logging.ERROR)

EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "kindle_experiment")

_model_sig = "_".join([m.value[:3] for m in sorted(SELECTED_MODEL_KINDS)])
OPTUNA_STUDY_NAME = f"{STUDY_BASE_NAME}_{_model_sig}"


def _configure_mlflow_tracking() -> None:
    """Устанавливает безопасный file-store для MLflow внутри контейнера/локально."""

    original_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    log.info("MLflow URI до обработки: '%s'", original_uri or "НЕ ЗАДАНО")

    # Если внешний URI задан и нет явного принуждения, уважаем его
    force_local = os.environ.get("MLFLOW_FORCE_LOCAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if original_uri and not force_local:
        try:
            mlflow.set_tracking_uri(original_uri)
            actual = mlflow.get_tracking_uri()
            log.info("MLflow URI оставлен без изменений: '%s'", actual)
            return
        except Exception as e:
            log.warning(
                "Не удалось установить внешний MLflow URI '%s': %s. Перехожу на локальный file-store.",
                original_uri,
                e,
            )

    # Пытаемся использовать путь в контейнере, иначе — локальный путь в проекте
    target_path: Path
    airflow_mlruns = Path("/opt/airflow/mlruns")
    if airflow_mlruns.exists():
        target_path = airflow_mlruns
        log.info("Обнаружен /opt/airflow/mlruns — настраиваю локальный file-store")
    else:
        target_path = Path("mlruns").resolve()
        log.info("Использую локальный mlruns: %s", target_path)

    # Создаём директорию и проверяем права на запись; при ошибке — фолбэк в локальный каталог проекта
    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning(
            "Не удалось создать директорию '%s': %s. Фолбэк в локальный путь.",
            target_path,
            e,
        )
        target_path = Path("mlruns").resolve()
        target_path.mkdir(parents=True, exist_ok=True)

    can_write = os.access(str(target_path), os.W_OK)
    if not can_write:
        try:
            test_file = target_path / ".mlflow_write_test"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            try:
                test_file.unlink()
            except Exception:
                pass
            can_write = True
        except Exception as e:
            log.info(
                "Нет прав на запись в '%s': %s. Перехожу на локальный путь.",
                target_path,
                e,
            )
            target_path = Path("mlruns").resolve()
            target_path.mkdir(parents=True, exist_ok=True)

    # Устанавливаем URI в явном file:// формате
    file_uri = target_path.as_uri()
    mlflow.set_tracking_uri(file_uri)
    os.environ["MLFLOW_TRACKING_URI"] = file_uri

    actual_uri = mlflow.get_tracking_uri()
    log.info("MLflow URI после установки: '%s'", actual_uri)


def build_pipeline(
    trial: optuna.Trial, model_name, fixed_solver: str | None = None
) -> Pipeline:
    """
    Строит sklearn Pipeline для trial.

    Состав:
        - ColumnTransformer: text -> TF-IDF(+опц. SVD), numeric -> Impute+Scale
        - (опц.) DenseTransformer для hist_gb
        - Финальная модель

    ---
    Гиперпараметры DistilBERT и их влияние:

    - db_epochs: увеличение числа эпох повышает качество, но сильно замедляет обучение и увеличивает расход GPU-памяти.
        Рекомендация: для быстрой проверки — 1 эпоха, для максимального качества — 2 и более.
    - db_lr: уменьшение learning rate (lr) обычно повышает качество, но замедляет сходимость; слишком маленький lr может привести к очень долгому обучению.
        Рекомендация: 1e-5 — медленно, но качественно; 5e-5 — быстрее, но возможна потеря качества.
    - db_max_len: увеличение max_len позволяет учитывать больше контекста, повышает качество, но увеличивает расход памяти и время обработки.
        Рекомендация: 96 — быстро, 192 — качественно, но требует больше ресурсов.
    - db_use_bigrams: включение биграмм может повысить качество, но замедляет обучение.
        Рекомендация: включать только если есть запас по времени/ресурсам.
    ---
    """
    # Приводим имя модели к enum (обратная совместимость со строками)
    model_kind: ModelKind
    if isinstance(model_name, ModelKind):
        model_kind = model_name
    else:
        model_kind = ModelKind(str(model_name))
    # DistilBERT отдельный
    if model_kind is ModelKind.distilbert:
        # db_epochs: увеличение числа эпох повышает качество, но сильно замедляет обучение и увеличивает расход GPU-памяти.
        epochs = trial.suggest_int("db_epochs", 1, 2)
        # db_lr: слишком маленький lr — долгое обучение, слишком большой — риск плохой сходимости.
        lr = trial.suggest_float("db_lr", 1e-5, 5e-5, log=True)
        # db_max_len: увеличение max_len повышает качество, но увеличивает расход памяти и время обработки.
        max_len = trial.suggest_int("db_max_len", 96, 192, step=32)
        # db_use_bigrams: включение биграмм может повысить качество, но замедляет обучение.
        use_bi = trial.suggest_categorical("db_use_bigrams", [False, True])
        clf = DistilBertClassifier(
            epochs=epochs,
            lr=lr,
            max_len=max_len,
            device=TRAIN_DEVICE,
            use_bigrams=use_bi,
        )
        return Pipeline([("distilbert", clf)])

    # Общая лёгкая предобработка TF-IDF: удаление стоп-слов, опциональный стемминг
    # Вынесено в отдельный модуль, чтобы модель ссылалась на лёгкий код без mlflow
    from scripts.train_modules.text_analyzers import make_tfidf_analyzer

    use_stemming = trial.suggest_categorical("use_stemming", [False, True])

    # Общие параметры TF-IDF для всех моделей (кроме DistilBERT)
    text_max_features = trial.suggest_int(
        "tfidf_max_features",
        TFIDF_MAX_FEATURES_MIN,
        TFIDF_MAX_FEATURES_MAX,
        step=TFIDF_MAX_FEATURES_STEP,
    )

    # Базовый TF-IDF для всех моделей
    tfidf = TfidfVectorizer(
        max_features=text_max_features,
        ngram_range=(1, 2),
        dtype=np.float32,
        stop_words="english",
        analyzer=make_tfidf_analyzer(use_stemming),
    )

    if model_kind is ModelKind.logreg:
        # LogReg не использует SVD для простоты
        use_svd = False
        text_steps = [("tfidf", tfidf)]
    else:
        # Оценка памяти ~ nnz * 4 байта (float32); для двуграмм плотность выше, возьмём коэффициент 1.5
        # nnz ~ n_samples * avg_terms_per_doc * 1.5
        n_samples_est = int(trial.user_attrs.get("n_train_samples", 20000))
        avg_terms = 120  # консервативная средняя оценка после очистки
        bigram_coef = 1.5
        estimated_nnz = n_samples_est * avg_terms * bigram_coef
        estimated_size_mb = (estimated_nnz * 4) / (1024 * 1024)
        force_svd = estimated_size_mb > FORCE_SVD_THRESHOLD_MB

        if force_svd:
            log.warning(
                "Принудительно включаю SVD: оценка памяти TF-IDF ~ %.1f MB (порог=%d MB, n≈%d, terms≈%d)",
                estimated_size_mb,
                FORCE_SVD_THRESHOLD_MB,
                n_samples_est,
                avg_terms,
            )
            use_svd = True
            svd_components = trial.suggest_int("svd_components", 20, 100, step=20)
        else:
            use_svd = trial.suggest_categorical("use_svd", [False, True])
            if use_svd:
                svd_components = trial.suggest_int("svd_components", 20, 100, step=20)
            else:
                svd_components = (
                    None  # Значение по умолчанию, когда SVD не используется
                )

        text_steps = [("tfidf", tfidf)]
        if use_svd and svd_components is not None:
            text_steps.append(
                ("svd", TruncatedSVD(n_components=svd_components, random_state=SEED))
            )
    text_pipeline = Pipeline(text_steps)

    numeric_available = [
        c
        for c in NUMERIC_COLS
        if c in trial.user_attrs.get("numeric_cols", NUMERIC_COLS)
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    # Веса для балансировки важности text vs numeric признаков
    text_weight = trial.suggest_float("text_weight", 0.1, 1.0)
    numeric_weight = trial.suggest_float("numeric_weight", 1.0, 10.0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, "reviewText"),
            ("num", numeric_pipeline, numeric_available),
        ],
        transformer_weights={"text": text_weight, "num": numeric_weight},
        sparse_threshold=0.3,
    )
    steps = [("pre", preprocessor)]

    if model_kind is ModelKind.logreg:
        # ВНИМАНИЕ: solver 'saga' очень медленно сходится на больших данных.
        # Если хотите протестировать все варианты — раскомментируйте 'saga' выше, но будьте готовы к долгому ожиданию.
        C = trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
        # Если fixed_solver задан, фиксируем выбор в trial через single-choice
        if fixed_solver is not None:
            solver = trial.suggest_categorical("logreg_solver", [fixed_solver])
        else:
            solver = trial.suggest_categorical(
                "logreg_solver", ["lbfgs", "liblinear", "saga"]
            )
        # Упрощённый подход: сначала выбираем регуляризацию, затем фильтруем по solver
        penalty_options = ["l1", "l2"]
        if solver == "lbfgs":
            # LBFGS не поддерживает L1 (недифференцируемость)
            penalty_options = ["l2"]

        penalty = trial.suggest_categorical("logreg_penalty", penalty_options)
        # lbfgs не поддерживает sparse-входы, а наш ColumnTransformer может возвращать sparse,
        # поэтому перед моделью принудительно конвертируем в dense
        if solver == "lbfgs":
            steps.append(("to_dense", DenseTransformer()))
        clf = LogisticRegression(
            max_iter=2500,
            C=C,
            class_weight="balanced",
            solver=solver,
            penalty=penalty,
        )
    elif model_kind is ModelKind.rf:
        # Если хотите более точную модель — увеличьте n_estimators и max_depth,
        # но это значительно увеличит время обучения и расход памяти.
        n_estimators = trial.suggest_int("rf_n_estimators", 100, 300, step=50)
        max_depth = trial.suggest_int("rf_max_depth", 6, 18, step=2)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        bootstrap = trial.suggest_categorical("rf_bootstrap", [True, False])
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=SEED,
        )
    elif model_kind is ModelKind.mlp:
        # Увеличение hidden_dim и epochs повышает качество, но замедляет обучение.
        # Если хотите максимальное качество — расширьте диапазоны в settings.py.
        hidden = trial.suggest_int("mlp_hidden", 64, 256, step=64)
        epochs = trial.suggest_int("mlp_epochs", 3, 8)
        lr = trial.suggest_float("mlp_lr", 1e-4, 5e-3, log=True)
        # обязателен dense
        steps.append(("to_dense", DenseTransformer()))
        clf = SimpleMLP(hidden_dim=hidden, epochs=epochs, lr=lr, device=TRAIN_DEVICE)
    else:  # hist_gb
        # Увеличение max_iter и уменьшение min_samples_leaf повышает качество, но замедляет обучение.
        # Если хотите максимальное качество — расширьте диапазоны в settings.py.
        lr = trial.suggest_float("hist_gb_lr", 0.01, 0.3, log=True)
        max_iter = trial.suggest_int("hist_gb_max_iter", 60, 160, step=20)
        l2 = trial.suggest_float("hist_gb_l2_regularization", 0.0, 1.0)
        min_leaf = trial.suggest_int("hist_gb_min_samples_leaf", 10, 60, step=10)
        clf = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=max_iter,
            l2_regularization=l2,
            min_samples_leaf=min_leaf,
            random_state=SEED,
        )
        steps.append(("to_dense", DenseTransformer()))

    steps.append(("model", clf))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Считает accuracy и F1 (macro/weighted) по вектору предсказаний."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def log_confusion_matrix(y_true, y_pred, path: Path):
    """Рисует и сохраняет матрицу ошибок в PNG по указанному пути."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def objective(
    trial: optuna.Trial,
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    fixed_solver: str | None = None,
):
    """Objective для Optuna: возвращает macro-F1 (CV или hold-out на val) для заданной модели."""
    model_kind: ModelKind = (
        model_name if isinstance(model_name, ModelKind) else ModelKind(str(model_name))
    )
    trial.set_user_attr(
        "numeric_cols", [c for c in NUMERIC_COLS if c in X_train.columns]
    )
    # Для эвристики по памяти/включению SVD передаём размер train в user_attrs
    try:
        trial.set_user_attr("n_train_samples", int(len(X_train)))
    except Exception:
        pass
    # Логируем модель один раз для trial
    mlflow.log_param("model", model_kind.value)

    # Вспомогательная функция: единообразно считает метрику через KFold или hold-out
    def _evaluate_with_cv_or_holdout(is_distilbert: bool) -> float:
        """Возвращает macro-F1 по CV (если N_FOLDS>1) или по hold-out (val).

        - Для DistilBERT используем сырой текст (Series/ndarray), иначе — DataFrame.
        - Логирует соответствующие метрики в MLflow (cv_f1_macro или accuracy/f1_*).
        """
        if N_FOLDS > 1:
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            f1_scores: list[float] = []
            if is_distilbert:
                texts = X_train["reviewText"].values
                for tr_idx, va_idx in skf.split(texts, y_train):
                    X_tr, X_va = texts[tr_idx], texts[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    pipe = build_pipeline(trial, model_kind, fixed_solver)
                    pipe.fit(X_tr, y_tr)
                    preds_fold = pipe.predict(X_va)
                    f1_scores.append(f1_score(y_va, preds_fold, average="macro"))
            else:
                for tr_idx, va_idx in skf.split(X_train, y_train):
                    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    pipe = build_pipeline(trial, model_kind, fixed_solver)
                    pipe.fit(X_tr, y_tr)
                    preds_fold = pipe.predict(X_va)
                    f1_scores.append(f1_score(y_va, preds_fold, average="macro"))

            mean_f1 = float(np.mean(f1_scores))
            mlflow.log_metric("cv_f1_macro", mean_f1)
            return mean_f1
        pipe = build_pipeline(trial, model_kind, fixed_solver)
        if is_distilbert:
            pipe.fit(X_train["reviewText"], y_train)
            preds = pipe.predict(X_val["reviewText"])
        else:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        mlflow.log_metrics(metrics)
        return metrics["f1_macro"]

    # Если distilbert – обучаем напрямую без ColumnTransformer
    if model_kind is ModelKind.distilbert:
        try:
            return _evaluate_with_cv_or_holdout(is_distilbert=True)
        except ImportError as e:
            # В DistilBERT возможны отсутствующие зависимости (torch/transformers и т.п.)
            mlflow.log_param("skipped", f"missing_deps: {e}")
            return 0.0

    # Классические модели: общая функция
    return _evaluate_with_cv_or_holdout(is_distilbert=False)


def _early_stop_callback(patience: int, min_trials: int):
    """Коллбек ранней остановки Optuna с учётом паузы (patience) и минимума успешных трейлов."""
    # не позволяет patience < 1
    if patience is None or patience < 1:
        log.warning("patience<1 в early stop, принудительно устанавливаю в 1")
        patience = 1
    best = {"value": -1, "since": 0, "count": 0}

    def cb(study: optuna.Study, trial: optuna.Trial):
        if trial.value is None:
            return
        best["count"] += 1
        if trial.value > best["value"] + 1e-9:
            best["value"] = trial.value
            best["since"] = 0
        else:
            best["since"] += 1
        # Не останавливаемся, пока не наберём минимум успешных трейлов
        if best["count"] < max(1, min_trials):
            return
        if best["since"] >= patience:
            log.info("Early stop: нет улучшений %d трейлов", patience)
            try:
                study.set_user_attr("early_stopped", True)
            except Exception:
                pass
            study.stop()

    return cb


def _extract_feature_importances(
    pipeline: Pipeline, use_svd: bool
) -> list[dict[str, float]]:
    """Возвращает список топ‑важностей фич из финального пайплайна (coef_/feature_importances_)."""
    res: list[dict[str, float]] = []
    try:
        if "pre" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            return res
        model = pipeline.named_steps["model"]
        pre: ColumnTransformer = pipeline.named_steps["pre"]
        text_pipe: Pipeline = pre.named_transformers_["text"]
        tfidf: TfidfVectorizer = text_pipe.named_steps["tfidf"]
        vocab_inv = (
            {idx: tok for tok, idx in tfidf.vocabulary_.items()}
            if hasattr(tfidf, "vocabulary_")
            else {}
        )
        text_dim = len(vocab_inv) if vocab_inv else 0
        numeric_cols = pre.transformers_[1][2]
        feature_names: list[str] = []
        if not use_svd and vocab_inv:
            feature_names.extend(
                [vocab_inv.get(i, f"tok_{i}") for i in range(text_dim)]
            )
        elif use_svd and "svd" in text_pipe.named_steps:
            svd_comp = text_pipe.named_steps["svd"].n_components
            feature_names.extend([f"svd_{i}" for i in range(svd_comp)])
        feature_names.extend(list(numeric_cols))

        if hasattr(model, "coef_"):
            coefs = np.mean(np.abs(model.coef_), axis=0)
        elif hasattr(model, "feature_importances_"):
            coefs = model.feature_importances_
        else:
            return res
        top_idx = np.argsort(coefs)[::-1][:50]
        for i in top_idx:
            if i < len(feature_names):
                res.append({"feature": feature_names[i], "importance": float(coefs[i])})
    except (KeyError, AttributeError, ValueError) as e:
        log.warning("Не удалось извлечь важности: %s", e)
    return res


def run():
    """Основной сценарий обучения: Optuna‑подбор, выбор лучшего, дообучение и логирование артефактов."""
    # Настраиваем логирование для обучения
    from .logging_config import setup_training_logging

    setup_training_logging()

    # Настройка MLflow трекинга до любых логов
    _configure_mlflow_tracking()

    from scripts.settings import MODEL_ARTEFACTS_DIR, MODEL_FILE_DIR

    MODEL_FILE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_FILE_DIR / "best_model.joblib"
    best_meta_path = MODEL_ARTEFACTS_DIR / "best_model_meta.json"

    log.info("FORCE_TRAIN=%s, best_model exists=%s", FORCE_TRAIN, best_path.exists())

    # Загружаем метрики старой модели для сравнения
    old_model_metric = None
    if best_meta_path.exists():
        try:
            with open(best_meta_path, encoding="utf-8") as f:
                old_meta = json.load(f)
                old_model_metric = old_meta.get("best_val_f1_macro")
                if old_model_metric:
                    log.info(
                        "Найдена предыдущая модель с val_f1_macro=%.4f",
                        old_model_metric,
                    )
        except Exception as e:
            log.warning("Не удалось загрузить метаданные старой модели: %s", e)

    if best_path.exists() and not FORCE_TRAIN:
        log.info("Модель уже существует и FORCE_TRAIN=False — пропуск")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    log.info(
        "Размеры: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    start_time = time.time()

    with mlflow.start_run(run_name="classical_pipeline"):
        # общие параметры
        mlflow.log_params(
            {
                "seed": SEED,
                "numeric_cols": ",".join(
                    [c for c in NUMERIC_COLS if c in X_train.columns]
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
        baseline_stats: dict[str, dict[str, float]] = {}
        for c in [c for c in NUMERIC_COLS if c in X_train.columns]:
            s = X_train[c]
            baseline_stats[c] = {"mean": float(s.mean()), "std": float(s.std() or 0.0)}
        _baseline_stats_cached = baseline_stats

        # Перебираем модели и оптимизируем каждую в своей study
        per_model_results: dict[ModelKind, dict[str, object]] = {}
        # Расширяем цели поиска: для logreg создаём отдельные study по solver, чтобы избежать динамики дистрибуций в одной study
        search_targets: list[tuple[ModelKind, str | None]] = []
        for model_name in SELECTED_MODEL_KINDS:
            if model_name is ModelKind.logreg:
                search_targets.extend(
                    [
                        (model_name, "lbfgs"),
                        # (model_name, "liblinear"),
                        # (model_name, "saga"),  # Очень медленно сходится на больших данных
                    ]
                )
            else:
                search_targets.append((model_name, None))

        for model_name, fixed_solver in search_targets:
            # 5 стартовых трейлов без прунинга
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
            study_name = f"{STUDY_BASE_NAME}_{model_name.value}{('_' + fixed_solver) if fixed_solver else ''}_{_model_sig}"
            with mlflow.start_run(nested=True, run_name=f"model={model_name.value}"):
                study = optuna.create_study(
                    direction="maximize",
                    pruner=pruner,
                    storage=OPTUNA_STORAGE,
                    study_name=study_name,
                    load_if_exists=True,
                )
                mlflow.log_param("study_name", study_name)
                mlflow.log_param(
                    "existing_trials",
                    len([t for t in study.trials if t.value is not None]),
                )

                def opt_obj(trial, model_name=model_name, fixed_solver=fixed_solver):
                    with mlflow.start_run(nested=True):
                        result = objective(
                            trial,
                            model_name,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            fixed_solver,
                        )
                        # Логируем каждый trial как INFO
                        log.info(
                            "Trial %d (%s%s): value=%.4f, params=%s",
                            trial.number,
                            model_name.value,
                            "/" + str(fixed_solver) if fixed_solver else "",
                            result,
                            trial.params,
                        )
                        return result

                stop_reason = None
                try:
                    # Персональный таймаут для тяжёлых моделей
                    timeout_sec = (
                        DISTILBERT_TIMEOUT_SEC
                        if model_name is ModelKind.distilbert
                        else OPTUNA_TIMEOUT_SEC
                    )
                    study.optimize(
                        opt_obj,
                        n_trials=OPTUNA_N_TRIALS,
                        timeout=timeout_sec,
                        callbacks=[
                            _early_stop_callback(
                                EARLY_STOP_PATIENCE, MIN_TRIALS_BEFORE_EARLY_STOP
                            )
                        ],
                        show_progress_bar=False,
                    )
                    if study.user_attrs.get("early_stopped", False):
                        stop_reason = "early_stop"
                except KeyboardInterrupt:
                    stop_reason = "keyboard_interrupt"
                except (RuntimeError, ValueError, optuna.exceptions.OptunaError) as e:
                    stop_reason = f"error: {e}"

                # Явно логируем причину остановки
                if stop_reason == "early_stop":
                    log.info(
                        "Optuna: остановка по ранней остановке (patience=%d)",
                        EARLY_STOP_PATIENCE,
                    )
                elif stop_reason == "keyboard_interrupt":
                    log.info("Optuna: остановка по KeyboardInterrupt")
                elif stop_reason:
                    log.info("Optuna: остановка по ошибке: %s", stop_reason)
                else:
                    log.info("Optuna: успешное завершение всех trial'ов")

                # Проверка best_trial
                best_trial = None
                try:
                    if len([t for t in study.trials if t.value is not None]) > 0:
                        best_trial = study.best_trial
                except (ValueError, optuna.exceptions.OptunaError) as e:
                    log.warning(
                        "%s: ошибка получения best_trial: %s", model_name.value, e
                    )

                if not best_trial or best_trial.value is None:
                    log.warning("%s: нет успешных trial'ов — пропуск", model_name.value)
                else:
                    cur = per_model_results.get(model_name)
                    new_entry = {
                        "best_value": best_trial.value,
                        "best_params": best_trial.params,
                        "study_name": study_name,
                    }
                    if cur is None or new_entry["best_value"] > cur["best_value"]:
                        per_model_results[model_name] = new_entry

        if not per_model_results:
            log.error("Нет ни одного успешного результата оптимизации — выход")
            raise RuntimeError("Оптимизация не дала успешных результатов")

        # Выбираем лучшую модель по best_value
        best_model = max(per_model_results.items(), key=lambda x: x[1]["best_value"])[0]
        best_info = per_model_results[best_model]
        new_model_metric = best_info["best_value"]

        # Сравниваем с предыдущей моделью
        if old_model_metric is not None:
            if new_model_metric <= old_model_metric:
                log.info(
                    "Новая модель %s (val_f1_macro=%.4f) НЕ лучше предыдущей (%.4f) — сохраняем старую",
                    best_model.value,
                    new_model_metric,
                    old_model_metric,
                )
                log.info("Обучение завершено без замены модели")
                return
            log.info(
                "Новая модель %s (val_f1_macro=%.4f) лучше предыдущей (%.4f) — заменяем",
                best_model.value,
                new_model_metric,
                old_model_metric,
            )

        log.info(
            "Лучшая модель: %s (val_f1_macro=%.4f)", best_model.value, new_model_metric
        )
        mlflow.log_param("best_model", best_model.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_info["best_params"].items()})
        mlflow.log_metric("best_val_f1_macro", best_info["best_value"])

        # Retrain на train+val
        X_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
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
            final_pipeline.fit(X_full["reviewText"], y_full)
        else:
            fixed_trial = optuna.trial.FixedTrial(best_params)
            # Прокидываем список доступных числовых колонок через user_attrs
            fixed_trial.set_user_attr(
                "numeric_cols", [c for c in NUMERIC_COLS if c in X_full.columns]
            )
            final_pipeline = build_pipeline(fixed_trial, best_model)
            final_pipeline.fit(X_full, y_full)
        # Атомарная запись модели
        tmp_model_path = best_path.with_suffix(".joblib.tmp")
        joblib.dump(final_pipeline, tmp_model_path)
        tmp_model_path.replace(best_path)
        mlflow.log_artifact(str(best_path))

        # Сохраняем baseline статистики только когда модель обновляется/создаётся
        from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

        bs_path = _MR / "baseline_numeric_stats.json"
        tmp_bs = bs_path.with_suffix(".json.tmp")
        with open(tmp_bs, "w", encoding="utf-8") as f:
            json.dump(_baseline_stats_cached, f, ensure_ascii=False, indent=2)
        tmp_bs.replace(bs_path)
        mlflow.log_artifact(str(bs_path))

        # Тестовая оценка
        if best_model is ModelKind.distilbert:
            test_preds = final_pipeline.predict(X_test["reviewText"])
        else:
            test_preds = final_pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, test_preds)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Артефакты: confusion matrix + classification report
        from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

        cm_path = _MR / "confusion_matrix_test.png"
        log_confusion_matrix(y_test, test_preds, cm_path)
        mlflow.log_artifact(str(cm_path))
        cr_txt = classification_report(y_test, test_preds)
        from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

        cr_path = _MR / "classification_report_test.txt"
        cr_path.write_text(cr_txt, encoding="utf-8")
        mlflow.log_artifact(str(cr_path))

        # Важности признаков (для классических моделей)
        if best_model is not ModelKind.distilbert:
            try:
                use_svd_flag = False
                try:
                    # Предпочитаем детектировать по финальному пайплайну
                    pre: ColumnTransformer = final_pipeline.named_steps.get("pre")
                    if pre is not None:
                        text_pipe: Pipeline = pre.named_transformers_["text"]
                        use_svd_flag = "svd" in getattr(text_pipe, "named_steps", {})
                except Exception:
                    # Фолбэк по best_params
                    use_svd_flag = bool(best_params.get("use_svd", False))

                fi_list = _extract_feature_importances(final_pipeline, use_svd_flag)
                if fi_list:
                    from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

                    fi_path = _MR / "feature_importances.json"
                    with open(fi_path, "w", encoding="utf-8") as f:
                        json.dump(fi_list, f, ensure_ascii=False, indent=2)
                    mlflow.log_artifact(str(fi_path))
            except Exception as e:
                log.warning("Не удалось сохранить важности признаков: %s", e)

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
                    import pandas as _pd  # локальный импорт, чтобы не засорять глобалы

                    # Собираем плоский датафрейм: number, value, затем параметры
                    all_param_keys = sorted(
                        {key for t in top_trials for key in t.params.keys()}
                    )
                    rows = []
                    for t in top_trials:
                        row = {"number": t.number, "value": t.value}
                        for k in all_param_keys:
                            row[k] = t.params.get(k)
                        rows.append(row)
                    df = _pd.DataFrame(rows)
                    from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

                    csv_path = _MR / "optuna_top_trials.csv"
                    df.to_csv(csv_path, index=False)
                    mlflow.log_artifact(str(csv_path))
        except Exception as e:
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
                    except Exception:
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
                    except Exception:
                        pass
                text_info["text_dim"] = text_dim if text_dim is not None else "unknown"
                schema["input"] = {
                    "text": text_info,
                    "numeric_features": numeric_cols_used,
                }
                classes = sorted(set(y_full.tolist()))
                schema["output"] = {"target_dtype": "int", "classes": classes}

            from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

            schema_path = _MR / "model_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(str(schema_path))
        except Exception as e:
            log.warning("Не удалось сохранить схему модели: %s", e)

        # 4) ROC/PR кривые по тесту (если у модели есть predict_proba)
        try:
            if best_model is ModelKind.distilbert:
                x_for_proba = X_test["reviewText"]
            else:
                x_for_proba = X_test
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
                # Защита: некоторые модели возвращают proba без последнего класса (редко) — проверим форму
                if y_score.shape[1] != y_true_bin.shape[1]:
                    # Приведём к общему виду, заполняя недостающие классы нулями
                    import numpy as _np

                    proba_aligned = _np.zeros(
                        (y_score.shape[0], y_true_bin.shape[1]), dtype=float
                    )
                    # Предполагаем порядок классов как в model.classes_, если доступен
                    try:
                        cls_model = list(getattr(final_pipeline, "classes_", []))
                    except Exception:
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
                from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

                roc_path = _MR / "roc_curve_test.png"
                fig_roc.tight_layout()
                fig_roc.savefig(roc_path)
                plt.close(fig_roc)
                mlflow.log_artifact(str(roc_path))

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
                from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

                pr_path = _MR / "pr_curve_test.png"
                fig_pr.tight_layout()
                fig_pr.savefig(pr_path)
                plt.close(fig_pr)
                mlflow.log_artifact(str(pr_path))
        except Exception as e:
            log.warning("Не удалось построить ROC/PR кривые: %s", e)

        # Ошибки классификации (первые 200)
        mis_idx = np.where(test_preds != y_test)[0]
        if len(mis_idx):
            mis_samples = X_test.iloc[mis_idx].copy()
            mis_samples["true"] = y_test[mis_idx]
            mis_samples["pred"] = test_preds[mis_idx]
            from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

            mis_path = _MR / "misclassified_samples_test.csv"
            mis_samples.head(200).to_csv(mis_path, index=False)
            mlflow.log_artifact(str(mis_path))

        duration = time.time() - start_time
        mlflow.log_metric("training_duration_sec", duration)

        meta = {
            # Сохраняем строковое значение Enum для JSON-сериализации
            "best_model": (
                best_model.value if hasattr(best_model, "value") else str(best_model)
            ),
            "best_params": best_params,
            "best_val_f1_macro": best_info["best_value"],
            "test_metrics": test_metrics,
            "sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
            "duration_sec": duration,
        }
        # Атомарная запись метаданных
        from scripts.settings import MODEL_ARTEFACTS_DIR as _MR

        _meta_path = _MR / "best_model_meta.json"
        _meta_tmp = _meta_path.with_suffix(".json.tmp")
        with open(_meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _meta_tmp.replace(_meta_path)
    log.info("Завершено. Лучшая модель сохранена: %s", best_path)


if __name__ == "__main__":
    run()

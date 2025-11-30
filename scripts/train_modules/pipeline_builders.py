"""Строители пайплайнов для разных типов моделей."""

from abc import ABC, abstractmethod

import numpy as np
from optuna.trial import BaseTrial
from scipy import sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from scripts.config import (
    DISTILBERT_MAX_EPOCHS,
    DISTILBERT_MIN_EPOCHS,
    NUMERIC_COLS,
    SEED,
    TFIDF_MAX_FEATURES_MAX,
    TFIDF_MAX_FEATURES_MIN,
    TFIDF_MAX_FEATURES_STEP,
    TRAIN_DEVICE,
)
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind
from scripts.train_modules.models import SimpleMLP
from scripts.train_modules.text_analyzers import make_tfidf_analyzer


def _to_dense(x):
    """Преобразует sparse матрицу в dense."""
    return x.toarray() if sp.issparse(x) else x


class PipelineBuilder(ABC):
    """Базовый класс для построения sklearn Pipeline.

    Args:
        trial: Optuna trial для сэмплирования гиперпараметров.
        numeric_cols: Список доступных числовых колонок в данных.
    """

    def __init__(self, trial: BaseTrial, numeric_cols: list[str] | None = None):
        self.trial = trial
        self._numeric_cols = numeric_cols if numeric_cols is not None else list(NUMERIC_COLS)

    @abstractmethod
    def build(self) -> Pipeline:
        """Строит sklearn Pipeline для данного типа модели."""

    def _build_preprocessor(
        self, use_stemming: bool, text_max_features: int, skip_svd: bool = False
    ) -> ColumnTransformer:
        analyzer = make_tfidf_analyzer(use_stemming)
        tfidf_params = {
            "max_features": text_max_features,
            "dtype": np.float32,
            "analyzer": analyzer,
        }
        if analyzer == "word":
            tfidf_params["ngram_range"] = (1, 2)
            tfidf_params["stop_words"] = None

        tfidf = TfidfVectorizer(**tfidf_params)

        text_steps = [("tfidf", tfidf)]

        if not skip_svd:
            use_svd = self.trial.suggest_categorical("use_svd", [False, True])
            if use_svd:
                svd_components = self.trial.suggest_int("svd_components", 20, 100, step=20)
                text_steps.append(
                    (
                        "svd",
                        TruncatedSVD(n_components=svd_components, random_state=SEED),
                    )
                )

        text_pipeline = Pipeline(text_steps)

        numeric_available = self._numeric_cols
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]
        )

        text_weight = self.trial.suggest_float("text_weight", 0.1, 1.0)
        numeric_weight = self.trial.suggest_float("numeric_weight", 1.0, 10.0)

        return ColumnTransformer(
            transformers=[
                ("text", text_pipeline, "reviewText"),
                ("num", numeric_pipeline, numeric_available),
            ],
            transformer_weights={"text": text_weight, "num": numeric_weight},
            sparse_threshold=0.3,
        )


class DistilBertBuilder(PipelineBuilder):
    """Builder для DistilBERT модели (text-only, без числовых признаков)."""

    def build(self) -> Pipeline:
        epochs = self.trial.suggest_int("db_epochs", DISTILBERT_MIN_EPOCHS, DISTILBERT_MAX_EPOCHS)
        lr = self.trial.suggest_float("db_lr", 1e-5, 5e-5, log=True)
        max_len = self.trial.suggest_int("db_max_len", 96, 192, step=32)

        clf = DistilBertClassifier(
            epochs=epochs,
            lr=lr,
            max_len=max_len,
            device=TRAIN_DEVICE,
        )
        return Pipeline([("model", clf)])


class LogRegBuilder(PipelineBuilder):
    """Буилдер для Logistic Regression с TF-IDF и числовыми признаками."""

    def __init__(
        self,
        trial: BaseTrial,
        numeric_cols: list[str] | None = None,
        fixed_solver: str | None = None,
    ):
        super().__init__(trial, numeric_cols)
        self.fixed_solver = fixed_solver

    def build(self) -> Pipeline:
        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features",
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_MAX,
            step=TFIDF_MAX_FEATURES_STEP,
        )

        preprocessor = self._build_preprocessor(use_stemming, text_max_features, skip_svd=True)
        steps = [("pre", preprocessor)]

        c_value = self.trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
        if self.fixed_solver is not None:
            solver = self.trial.suggest_categorical("logreg_solver", [self.fixed_solver])
        else:
            solver = self.trial.suggest_categorical(
                "logreg_solver",
                ["lbfgs", "liblinear"],  # "saga" медленно сходится на больших данных
            )

        pen_others = self.trial.suggest_categorical("logreg_penalty_liblinear_saga", ["l1", "l2"])
        penalty = "l2" if solver == "lbfgs" else pen_others

        if solver == "lbfgs":
            steps.append(("to_dense", FunctionTransformer(_to_dense)))

        clf = LogisticRegression(
            max_iter=2500,
            C=c_value,
            solver=solver,
            penalty=penalty,
        )
        steps.append(("model", clf))
        return Pipeline(steps)


class RandomForestBuilder(PipelineBuilder):
    def build(self) -> Pipeline:
        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features",
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_MAX,
            step=TFIDF_MAX_FEATURES_STEP,
        )

        preprocessor = self._build_preprocessor(use_stemming, text_max_features)
        steps = [("pre", preprocessor)]

        n_estimators = self.trial.suggest_int("rf_n_estimators", 100, 300, step=50)
        max_depth = self.trial.suggest_int("rf_max_depth", 6, 18, step=2)
        min_samples_split = self.trial.suggest_int("rf_min_samples_split", 2, 10)
        bootstrap = self.trial.suggest_categorical("rf_bootstrap", [True, False])

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            n_jobs=1,
            random_state=SEED,
        )
        steps.append(("model", clf))
        return Pipeline(steps)


class MLPBuilder(PipelineBuilder):
    def build(self) -> Pipeline:
        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features",
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_MAX,
            step=TFIDF_MAX_FEATURES_STEP,
        )

        preprocessor = self._build_preprocessor(use_stemming, text_max_features)
        steps = [("pre", preprocessor)]

        hidden = self.trial.suggest_int("mlp_hidden", 64, 256, step=64)
        epochs = self.trial.suggest_int("mlp_epochs", 3, 8)
        lr = self.trial.suggest_float("mlp_lr", 1e-4, 5e-3, log=True)

        steps.append(("to_dense", FunctionTransformer(_to_dense)))
        clf = SimpleMLP(hidden_dim=hidden, epochs=epochs, lr=lr, device=TRAIN_DEVICE)
        steps.append(("model", clf))
        return Pipeline(steps)


class HistGBBuilder(PipelineBuilder):
    def build(self) -> Pipeline:
        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features",
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_MAX,
            step=TFIDF_MAX_FEATURES_STEP,
        )

        preprocessor = self._build_preprocessor(use_stemming, text_max_features)
        steps = [("pre", preprocessor)]

        lr = self.trial.suggest_float("hist_gb_lr", 0.01, 0.3, log=True)
        max_iter = self.trial.suggest_int("hist_gb_max_iter", 60, 160, step=20)
        l2 = self.trial.suggest_float("hist_gb_l2_regularization", 0.0, 1.0)
        min_leaf = self.trial.suggest_int("hist_gb_min_samples_leaf", 10, 60, step=10)

        clf = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=max_iter,
            l2_regularization=l2,
            min_samples_leaf=min_leaf,
            random_state=SEED,
        )
        steps.append(("to_dense", FunctionTransformer(_to_dense)))
        steps.append(("model", clf))
        return Pipeline(steps)


class ModelBuilderFactory:
    """Фабрика для создания PipelineBuilder по типу модели."""

    @staticmethod
    def get_builder(
        model_kind: ModelKind,
        trial: BaseTrial,
        numeric_cols: list[str] | None = None,
        fixed_solver: str | None = None,
    ) -> PipelineBuilder:
        """Создаёт Builder для указанного типа модели.

        Args:
            model_kind: Тип модели из ModelKind enum.
            trial: Optuna trial для сэмплирования гиперпараметров.
            numeric_cols: Список доступных числовых колонок (если None — берётся из config).
            fixed_solver: Фиксированный solver для LogReg (опционально).

        Returns:
            PipelineBuilder для указанного типа модели.
        """
        match model_kind:
            case ModelKind.distilbert:
                return DistilBertBuilder(trial, numeric_cols)
            case ModelKind.logreg:
                return LogRegBuilder(trial, numeric_cols, fixed_solver)
            case ModelKind.rf:
                return RandomForestBuilder(trial, numeric_cols)
            case ModelKind.mlp:
                return MLPBuilder(trial, numeric_cols)
            case ModelKind.hist_gb:
                return HistGBBuilder(trial, numeric_cols)
            case _:
                raise ValueError(f"Неизвестный тип модели: {model_kind}")

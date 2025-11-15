"""Строители пайплайнов для разных типов моделей."""

from abc import ABC, abstractmethod

import numpy as np
import optuna
from scipy import sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from scripts.config import NUMERIC_COLS, SEED, TRAIN_DEVICE
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind
from scripts.train_modules.models import SimpleMLP
from scripts.train_modules.text_analyzers import make_tfidf_analyzer


def _to_dense(x):
    """Преобразует sparse матрицу в dense."""
    return x.toarray() if sp.issparse(x) else x


class PipelineBuilder(ABC):
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    @abstractmethod
    def build(self) -> Pipeline:
        pass

    def _build_preprocessor(
        self, use_stemming: bool, text_max_features: int
    ) -> ColumnTransformer:
        tfidf = TfidfVectorizer(
            max_features=text_max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            stop_words="english",
            analyzer=make_tfidf_analyzer(use_stemming),
        )

        use_svd = self.trial.suggest_categorical("use_svd", [False, True])
        text_steps = [("tfidf", tfidf)]

        if use_svd:
            svd_components = self.trial.suggest_int("svd_components", 20, 100, step=20)
            text_steps.append(
                ("svd", TruncatedSVD(n_components=svd_components, random_state=SEED))
            )

        text_pipeline = Pipeline(text_steps)

        numeric_available = [
            c
            for c in NUMERIC_COLS
            if c in self.trial.user_attrs.get("numeric_cols", NUMERIC_COLS)
        ]
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
            # sparse_threshold: порог разреженности для вывода sparse матрицы (0.3 = 30% ненулевых значений)
            sparse_threshold=0.3,
        )


class DistilBertBuilder(PipelineBuilder):
    def build(self) -> Pipeline:
        from scripts.config import DISTILBERT_MAX_EPOCHS, DISTILBERT_MIN_EPOCHS

        epochs = self.trial.suggest_int(
            "db_epochs", DISTILBERT_MIN_EPOCHS, DISTILBERT_MAX_EPOCHS
        )
        lr = self.trial.suggest_float("db_lr", 1e-5, 5e-5, log=True)
        max_len = self.trial.suggest_int("db_max_len", 96, 192, step=32)
        use_bi = self.trial.suggest_categorical("db_use_bigrams", [False, True])

        clf = DistilBertClassifier(
            epochs=epochs,
            lr=lr,
            max_len=max_len,
            device=TRAIN_DEVICE,
            use_bigrams=use_bi,
        )
        return Pipeline([("model", clf)])


class LogRegBuilder(PipelineBuilder):
    def __init__(self, trial: optuna.Trial, fixed_solver: str | None = None):
        super().__init__(trial)
        self.fixed_solver = fixed_solver

    def build(self) -> Pipeline:
        from scripts.config import (
            TFIDF_MAX_FEATURES_MAX,
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_STEP,
        )

        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features",
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_MAX,
            step=TFIDF_MAX_FEATURES_STEP,
        )

        # LogisticRegression не использует SVD, поэтому создаем упрощенный препроцессор
        tfidf = TfidfVectorizer(
            max_features=text_max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            stop_words="english",
            analyzer=make_tfidf_analyzer(use_stemming),
        )

        text_pipeline = Pipeline([("tfidf", tfidf)])

        numeric_available = [
            c
            for c in NUMERIC_COLS
            if c in self.trial.user_attrs.get("numeric_cols", NUMERIC_COLS)
        ]
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]
        )

        text_weight = self.trial.suggest_float("text_weight", 0.1, 1.0)
        numeric_weight = self.trial.suggest_float("numeric_weight", 1.0, 10.0)

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_pipeline, "reviewText"),
                ("num", numeric_pipeline, numeric_available),
            ],
            transformer_weights={"text": text_weight, "num": numeric_weight},
            # sparse_threshold: порог разреженности для вывода sparse матрицы (0.3 = 30% ненулевых значений)
            sparse_threshold=0.3,
        )

        steps = [("pre", preprocessor)]

        c_value = self.trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
        if self.fixed_solver is not None:
            solver = self.trial.suggest_categorical(
                "logreg_solver", [self.fixed_solver]
            )
        else:
            solver = self.trial.suggest_categorical(
                "logreg_solver", ["lbfgs", "liblinear", "saga"]
            )

        pen_others = self.trial.suggest_categorical(
            "logreg_penalty_liblinear_saga", ["l1", "l2"]
        )
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
        from scripts.config import (
            TFIDF_MAX_FEATURES_MAX,
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_STEP,
        )

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
            n_jobs=-1,
            random_state=SEED,
        )
        steps.append(("model", clf))
        return Pipeline(steps)


class MLPBuilder(PipelineBuilder):
    def build(self) -> Pipeline:
        from scripts.config import (
            TFIDF_MAX_FEATURES_MAX,
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_STEP,
        )

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
        from scripts.config import (
            TFIDF_MAX_FEATURES_MAX,
            TFIDF_MAX_FEATURES_MIN,
            TFIDF_MAX_FEATURES_STEP,
        )

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
    @staticmethod
    def get_builder(
        model_kind: ModelKind, trial: optuna.Trial, fixed_solver: str | None = None
    ) -> PipelineBuilder:
        match model_kind:
            case ModelKind.distilbert:
                return DistilBertBuilder(trial)
            case ModelKind.logreg:
                return LogRegBuilder(trial, fixed_solver)
            case ModelKind.rf:
                return RandomForestBuilder(trial)
            case ModelKind.mlp:
                return MLPBuilder(trial)
            case ModelKind.hist_gb:
                return HistGBBuilder(trial)
            case _:
                raise ValueError(f"Неизвестный тип модели: {model_kind}")

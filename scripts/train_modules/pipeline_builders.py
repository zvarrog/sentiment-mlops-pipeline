"""Строители пайплайнов для разных типов моделей.

Паттерн Builder/Factory для инкапсуляции логики создания пайплайнов.
Каждый билдер отвечает за конфигурацию конкретного типа модели.
"""

from abc import ABC, abstractmethod

import numpy as np
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.config import (
    FORCE_SVD_THRESHOLD_MB,
    SEED,
    SVD_ESTIMATION_AVG_TERMS,
    SVD_ESTIMATION_BIGRAM_COEF,
    TRAIN_DEVICE,
    log,
)
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind
from scripts.train_modules.feature_space import NUMERIC_COLS, DenseTransformer
from scripts.train_modules.models import SimpleMLP


def make_tfidf_analyzer(use_stemming: bool):
    if not use_stemming:
        return "word"

    from nltk.stem.porter import PorterStemmer

    stemmer = PorterStemmer()

    def stemmed_analyzer(doc: str) -> list[str]:
        return [stemmer.stem(w) for w in doc.split()]

    return stemmed_analyzer


class PipelineBuilder(ABC):
    def __init__(self, trial: optuna.Trial):
        """Инициализирует строитель с trial Optuna.

        Args:
            trial: Объект trial для предложения гиперпараметров.
        """
        self.trial = trial

    @abstractmethod
    def build(self) -> Pipeline:
        """Создаёт и возвращает готовый Pipeline для модели.

        Returns:
            Настроенный sklearn Pipeline.
        """

    def _build_preprocessor(
        self, use_stemming: bool, text_max_features: int, force_svd: bool = False
    ) -> ColumnTransformer:
        """Создаёт препроцессор с TF-IDF и числовыми признаками.

        Args:
            use_stemming: Использовать ли стемминг в TF-IDF.
            text_max_features: Максимальное количество признаков TF-IDF.
            force_svd: Принудительно использовать SVD даже без оптимизации.

        Returns:
            ColumnTransformer с текстовым и числовым пайплайнами.
        """
        tfidf = TfidfVectorizer(
            max_features=text_max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            stop_words="english",
            analyzer=make_tfidf_analyzer(use_stemming),
        )

        # Логика SVD
        use_svd = force_svd
        svd_components = None

        if not force_svd:
            n_samples_est = int(self.trial.user_attrs.get("n_train_samples", 20000))
            estimated_nnz = (
                n_samples_est * SVD_ESTIMATION_AVG_TERMS * SVD_ESTIMATION_BIGRAM_COEF
            )
            estimated_size_mb = (estimated_nnz * 4) / (1024 * 1024)
            auto_force_svd = estimated_size_mb > FORCE_SVD_THRESHOLD_MB

            if auto_force_svd:
                log.warning(
                    "Принудительно включаю SVD: оценка памяти TF-IDF ~ %.1f MB (порог=%d MB, n≈%d, terms≈%d)",
                    estimated_size_mb,
                    FORCE_SVD_THRESHOLD_MB,
                    n_samples_est,
                    SVD_ESTIMATION_AVG_TERMS,
                )
                use_svd = True
                svd_components = self.trial.suggest_int(
                    "svd_components", 20, 100, step=20
                )
            else:
                use_svd = self.trial.suggest_categorical("use_svd", [False, True])
                if use_svd:
                    svd_components = self.trial.suggest_int(
                        "svd_components", 20, 100, step=20
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
            sparse_threshold=0.3,
        )


class DistilBertBuilder(PipelineBuilder):
    """Строитель для DistilBERT классификатора."""

    def build(self) -> Pipeline:
        """Создаёт Pipeline с DistilBERT моделью.

        Returns:
            Pipeline содержащий только DistilBERT классификатор.
        """
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
    """Строитель для логистической регрессии."""

    def __init__(self, trial: optuna.Trial, fixed_solver: str | None = None):
        """Инициализирует строитель LogReg.

        Args:
            trial: Объект trial для предложения гиперпараметров.
            fixed_solver: Фиксированный солвер (для повторных запусков).
        """
        super().__init__(trial)
        self.fixed_solver = fixed_solver

    def build(self) -> Pipeline:
        """Создаёт Pipeline с логистической регрессией.

        Returns:
            Pipeline с препроцессором и LogisticRegression (без SVD).
        """
        from scripts.config import get_tfidf_max_features_range

        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        n_samples = int(self.trial.user_attrs.get("n_train_samples", 20000))
        min_f, max_f, step_f = get_tfidf_max_features_range(n_samples)
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features", min_f, max_f, step=step_f
        )

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
            sparse_threshold=0.3,
        )

        steps = [("pre", preprocessor)]

        C = self.trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
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
            steps.append(("to_dense", DenseTransformer()))

        clf = LogisticRegression(
            max_iter=2500,
            C=C,
            class_weight="balanced",
            solver=solver,
            penalty=penalty,
        )
        steps.append(("model", clf))
        return Pipeline(steps)


class RandomForestBuilder(PipelineBuilder):
    """Строитель для Random Forest."""

    def build(self) -> Pipeline:
        """Создаёт Pipeline с Random Forest классификатором.

        Returns:
            Pipeline с препроцессором и RandomForestClassifier.
        """
        from scripts.config import get_tfidf_max_features_range

        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        n_samples = int(self.trial.user_attrs.get("n_train_samples", 20000))
        min_f, max_f, step_f = get_tfidf_max_features_range(n_samples)
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features", min_f, max_f, step=step_f
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
            class_weight="balanced_subsample",
            random_state=SEED,
        )
        steps.append(("model", clf))
        return Pipeline(steps)


class MLPBuilder(PipelineBuilder):
    """Строитель для многослойного перцептрона."""

    def build(self) -> Pipeline:
        """Создаёт Pipeline с SimpleMLP моделью.

        Returns:
            Pipeline с препроцессором, DenseTransformer и SimpleMLP.
        """
        from scripts.config import get_tfidf_max_features_range

        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        n_samples = int(self.trial.user_attrs.get("n_train_samples", 20000))
        min_f, max_f, step_f = get_tfidf_max_features_range(n_samples)
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features", min_f, max_f, step=step_f
        )

        preprocessor = self._build_preprocessor(use_stemming, text_max_features)
        steps = [("pre", preprocessor)]

        hidden = self.trial.suggest_int("mlp_hidden", 64, 256, step=64)
        epochs = self.trial.suggest_int("mlp_epochs", 3, 8)
        lr = self.trial.suggest_float("mlp_lr", 1e-4, 5e-3, log=True)

        steps.append(("to_dense", DenseTransformer()))
        clf = SimpleMLP(hidden_dim=hidden, epochs=epochs, lr=lr, device=TRAIN_DEVICE)
        steps.append(("model", clf))
        return Pipeline(steps)


class HistGBBuilder(PipelineBuilder):
    """Строитель для Histogram-based Gradient Boosting."""

    def build(self) -> Pipeline:
        """Создаёт Pipeline с HistGradientBoostingClassifier.

        Returns:
            Pipeline с препроцессором, DenseTransformer и HistGradientBoostingClassifier.
        """
        from scripts.config import get_tfidf_max_features_range

        use_stemming = self.trial.suggest_categorical("use_stemming", [False, True])
        n_samples = int(self.trial.user_attrs.get("n_train_samples", 20000))
        min_f, max_f, step_f = get_tfidf_max_features_range(n_samples)
        text_max_features = self.trial.suggest_int(
            "tfidf_max_features", min_f, max_f, step=step_f
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
        steps.append(("to_dense", DenseTransformer()))
        steps.append(("model", clf))
        return Pipeline(steps)


class ModelBuilderFactory:
    """Фабрика для создания строителей пайплайнов."""

    @staticmethod
    def get_builder(
        model_kind: ModelKind, trial: optuna.Trial, fixed_solver: str | None = None
    ) -> PipelineBuilder:
        """Возвращает строитель для указанного типа модели.

        Args:
            model_kind: Тип модели из ModelKind enum.
            trial: Объект trial для предложения гиперпараметров.
            fixed_solver: Фиксированный солвер (только для LogReg).

        Returns:
            Экземпляр соответствующего строителя.

        Raises:
            ValueError: Если model_kind неизвестен.
        """
        if model_kind is ModelKind.distilbert:
            return DistilBertBuilder(trial)
        if model_kind is ModelKind.logreg:
            return LogRegBuilder(trial, fixed_solver)
        if model_kind is ModelKind.rf:
            return RandomForestBuilder(trial)
        if model_kind is ModelKind.mlp:
            return MLPBuilder(trial)
        if model_kind is ModelKind.hist_gb:
            return HistGBBuilder(trial)

        raise ValueError(f"Неизвестный тип модели: {model_kind}")

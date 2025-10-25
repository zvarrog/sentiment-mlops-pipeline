"""Unit tests для ключевых модулей проекта sentiment-mlops-pipeline.

Покрываемые модули:
- scripts.train_modules.data_loading: загрузка splits
- scripts.train_modules.feature_space: DenseTransformer, NUMERIC_COLS
- scripts.feature_contract: FeatureContract
- scripts.data_validation: валидация схемы и качества
- scripts.drift_monitor: вычисление PSI
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from scripts.train_modules.evaluation import compute_metrics


class TestDataLoading:
    """Тесты для scripts.train_modules.data_loading."""

    def test_load_splits_returns_six_arrays(self, sample_parquet_files):
        """load_splits возвращает 6 объектов: X_train, X_val, X_test, y_train, y_val, y_test."""
        from scripts.train_modules.data_loading import load_splits

        x_train, x_val, x_test, y_train, y_val, y_test = load_splits()

        assert x_train is not None
        assert len(x_train) > 0
        assert len(y_train) == len(x_train)
        assert len(x_val) > 0
        assert len(x_test) > 0

    def test_load_splits_has_required_columns(self, sample_parquet_files):
        """X содержит reviewText и числовые признаки."""
        from scripts.train_modules.data_loading import load_splits

        x_train, _, _, _, _, _ = load_splits()

        assert "reviewText" in x_train.columns
        assert len(x_train.columns) > 1


class TestFeatureSpace:
    """Тесты для scripts.train_modules.feature_space."""

    def test_dense_transformer_converts_sparse_to_dense(self):
        """DenseTransformer корректно конвертирует sparse матрицу в dense."""
        from scipy.sparse import csr_matrix

        from scripts.train_modules.feature_space import DenseTransformer

        sparse_data = csr_matrix(np.array([[1, 0, 2], [0, 3, 0]]))
        transformer = DenseTransformer()

        dense_result = transformer.fit_transform(sparse_data)

        assert isinstance(dense_result, np.ndarray)
        assert dense_result.shape == (2, 3)
        np.testing.assert_array_equal(dense_result, np.array([[1, 0, 2], [0, 3, 0]]))

    def test_dense_transformer_handles_dense_input(self):
        """DenseTransformer корректно обрабатывает уже плотные данные."""
        from scripts.train_modules.feature_space import DenseTransformer

        dense_data = np.array([[1, 2], [3, 4]])
        transformer = DenseTransformer()

        result = transformer.fit_transform(dense_data)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, dense_data)

    def test_numeric_cols_is_list(self):
        """NUMERIC_COLS является списком строк."""
        from scripts.train_modules.feature_space import NUMERIC_COLS

        assert isinstance(NUMERIC_COLS, list)
        assert len(NUMERIC_COLS) > 0
        assert all(isinstance(col, str) for col in NUMERIC_COLS)

    def test_numeric_cols_are_numeric_type(self, sample_parquet_files):
        """Все NUMERIC_COLS имеют числовой dtype после загрузки сплитов."""
        from scripts.train_modules.data_loading import load_splits
        from scripts.train_modules.feature_space import NUMERIC_COLS

        x_train, _, _, _, _, _ = load_splits()

        for col in NUMERIC_COLS:
            if col in x_train.columns:
                assert pd.api.types.is_numeric_dtype(x_train[col]), (
                    f"{col} имеет нечисловой тип: {x_train[col].dtype}"
                )


class TestFeatureContract:
    """Тесты для scripts.feature_contract."""

    def test_feature_contract_from_artifacts_with_baseline(self, tmp_path):
        """FeatureContract.from_model_artifacts загружает baseline_numeric_stats."""
        baseline_stats = {
            "text_len": {"mean": 100.0, "std": 20.0},
            "word_count": {"mean": 50.0, "std": 10.0},
        }

        baseline_path = tmp_path / "baseline_numeric_stats.json"
        baseline_path.write_text(json.dumps(baseline_stats), encoding="utf-8")

        from scripts.feature_contract import FeatureContract

        contract = FeatureContract.from_model_artifacts(tmp_path)

        assert contract.expected_numeric_columns == ["text_len", "word_count"]
        assert contract.baseline_stats == baseline_stats

    def test_feature_contract_validate_input_detects_missing_columns(self):
        """validate_input_data детектирует отсутствующие обязательные колонки."""
        from scripts.feature_contract import FeatureContract

        contract = FeatureContract(
            required_text_columns=["reviewText"],
            expected_numeric_columns=["text_len", "word_count"],
            baseline_stats=None,
        )

        data = {"text_len": 100}
        warnings = contract.validate_input_data(data)

        assert "missing_required_columns" in warnings
        assert "reviewText" in warnings["missing_required_columns"]

    def test_feature_contract_validate_input_detects_missing_numeric(self):
        """validate_input_data детектирует отсутствующие числовые колонки."""
        from scripts.feature_contract import FeatureContract

        contract = FeatureContract(
            required_text_columns=["reviewText"],
            expected_numeric_columns=["text_len", "word_count"],
            baseline_stats=None,
        )

        data = {"reviewText": "test", "text_len": 100}
        warnings = contract.validate_input_data(data)

        assert "missing_numeric_columns" in warnings
        assert "word_count" in warnings["missing_numeric_columns"]


class TestDataValidation:
    """Тесты для scripts.data_validation."""

    def test_validate_column_schema_detects_missing_required(self):
        """validate_column_schema детектирует отсутствующие обязательные колонки."""
        from scripts.data_validation import DataSchema, validate_column_schema

        df = pd.DataFrame({"reviewText": ["test"]})
        schema = DataSchema(
            required_columns={"reviewText", "overall"},
            optional_columns=set(),
            column_types={},
            numeric_ranges={},
            text_constraints={},
        )

        errors, warnings = validate_column_schema(df, schema)

        assert len(errors) > 0
        assert any("overall" in err for err in errors)

    def test_validate_column_schema_detects_type_mismatch(self):
        """validate_column_schema детектирует несоответствие типов."""
        from scripts.data_validation import DataSchema, validate_column_schema

        df = pd.DataFrame({"overall": ["text_instead_of_int"]})
        schema = DataSchema(
            required_columns={"overall"},
            optional_columns=set(),
            column_types={"overall": "int64"},
            numeric_ranges={},
            text_constraints={},
        )

        errors, warnings = validate_column_schema(df, schema)

        assert len(warnings) > 0

    def test_validate_data_quality_detects_empty_dataset(self):
        """validate_data_quality детектирует пустой датасет."""
        from scripts.data_validation import DataSchema, validate_data_quality

        df = pd.DataFrame()
        schema = DataSchema(
            required_columns=set(),
            optional_columns=set(),
            column_types={},
            numeric_ranges={},
            text_constraints={},
        )

        errors, warnings = validate_data_quality(df, schema)

        assert len(errors) > 0
        assert any("пуст" in err.lower() for err in errors)

    def test_validate_data_quality_detects_nulls_in_required(self):
        """validate_data_quality детектирует пропуски в обязательных колонках."""
        from scripts.data_validation import DataSchema, validate_data_quality

        df = pd.DataFrame({"reviewText": ["test", None, "test2"], "overall": [1, 2, 3]})
        schema = DataSchema(
            required_columns={"reviewText", "overall"},
            optional_columns=set(),
            column_types={},
            numeric_ranges={},
            text_constraints={},
        )

        errors, warnings = validate_data_quality(df, schema)

        assert len(errors) > 0
        assert any("reviewText" in err for err in errors)


class TestEvaluation:
    """Тесты для метрик качества моделей."""

    def test_compute_metrics_perfect_prediction(self):
        """Идеальное предсказание даёт метрики 1.0."""
        y_true_arr = np.array([1, 2, 3, 4, 5] * 20)
        y_pred_arr = y_true_arr.copy()
        m = compute_metrics(y_true_arr, y_pred_arr)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["f1_weighted"] == 1.0

    def test_compute_metrics_worst_prediction(self):
        """Всегда неверные предсказания → accuracy 0, F1 близко к 0."""
        y_true_arr = np.array([1] * 100)
        y_pred_arr = np.array([5] * 100)
        m = compute_metrics(y_true_arr, y_pred_arr)
        assert m["accuracy"] == 0.0
        assert m["f1_macro"] <= 0.01

    def test_compute_metrics_matches_sklearn(self):
        """Наш f1_macro совпадает со sklearn f1_score(average='macro')."""
        rng = np.random.default_rng(42)
        y_true_arr = rng.integers(1, 6, size=1000)
        y_pred_arr = rng.integers(1, 6, size=1000)
        our = compute_metrics(y_true_arr, y_pred_arr)["f1_macro"]
        skl = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
        assert abs(our - skl) < 1e-6


class TestDriftMonitor:
    """Тесты для scripts.drift_monitor."""

    def test_psi_identical_distributions_returns_zero(self):
        """PSI для идентичных распределений близок к нулю."""
        from scripts.drift_monitor import psi

        expected = np.random.normal(0, 1, 1000)
        actual = expected.copy()

        result = psi(expected, actual, bins=10)

        assert result < 0.01

    def test_psi_different_distributions_returns_positive(self):
        """PSI для различных распределений положителен."""
        from scripts.drift_monitor import psi

        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)

        result = psi(expected, actual, bins=10)

        assert result > 0.1

    def test_psi_handles_nan_values(self):
        """PSI корректно обрабатывает NaN значения."""
        from scripts.drift_monitor import psi

        expected = np.array([1, 2, 3, np.nan, 4, 5])
        actual = np.array([1, 2, 3, 4, 5, np.nan])

        result = psi(expected, actual, bins=3)

        assert not np.isnan(result)
        assert result >= 0

    def test_psi_with_custom_cuts(self):
        """PSI работает с пользовательскими границами бинов."""
        from scripts.drift_monitor import psi

        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual = np.array([1, 2, 3, 4, 5, 11, 12, 13, 14, 15])
        cuts = np.array([0, 5, 10, 20])

        result = psi(expected, actual, cuts=cuts)

        assert result > 0


class TestTrainModules:
    """Smoke tests для обучающих модулей."""

    def test_simple_mlp_import(self):
        """SimpleMLP импортируется корректно."""
        from scripts.train_modules.models import SimpleMLP

        assert SimpleMLP is not None


# Fixtures
@pytest.fixture(scope="session")
def sample_parquet_files(tmp_path_factory):
    """Создаёт временные parquet файлы для тестов."""
    import os

    data_dir = tmp_path_factory.mktemp("data") / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Минимальный датасет для тестов
    df_train = pd.DataFrame(
        {
            "reviewText": ["good product"] * 100,
            "overall": [5] * 100,
            "text_len": [12.0] * 100,
            "word_count": [2.0] * 100,
        }
    )
    df_val = df_train.iloc[:20].copy()
    df_test = df_train.iloc[:20].copy()

    df_train.to_parquet(data_dir / "train.parquet")
    df_val.to_parquet(data_dir / "val.parquet")
    df_test.to_parquet(data_dir / "test.parquet")

    # Устанавливаем переменную окружения для тестов
    os.environ["PROCESSED_DATA_DIR"] = str(data_dir)

    yield data_dir

    # Cleanup
    os.environ.pop("PROCESSED_DATA_DIR", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

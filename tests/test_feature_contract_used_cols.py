import json

import pytest

from scripts.feature_contract import FeatureContract


def test_contract_uses_schema_numeric_cols_if_present(tmp_path, monkeypatch):
    # Подготовим временный каталог с артефактами
    arte = tmp_path / "artefacts" / "model_artefacts"
    arte.mkdir(parents=True, exist_ok=True)
    # Схема с конкретными фичами
    schema = {
        "input": {
            "text": {"text_column": "reviewText", "text_dim": 100},
            "numeric_features": ["text_len", "word_count", "caps_ratio"],
        },
        "output": {"target_dtype": "int", "classes": [1, 2, 3]},
    }
    (arte / "model_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    # Базовые статистики (минимальные)
    baseline = {"text_len": {"mean": 10.0, "std": 5.0}}
    (arte / "baseline_numeric_stats.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )

    # Контракт должен подхватить именно эти 3 колонки
    contract = FeatureContract.from_model_artifacts(arte)
    assert contract.expected_numeric_columns == [
        "text_len",
        "word_count",
        "caps_ratio",
    ]
    assert contract.baseline_stats is not None


def test_contract_requires_artifacts_when_no_schema(tmp_path):
    """Тест что контракт выбрасывает ошибку при отсутствии артефактов."""
    arte = tmp_path / "artefacts" / "model_artefacts"
    arte.mkdir(parents=True, exist_ok=True)
    # Пустая папка артефактов

    # Должна быть ошибка при отсутствии baseline_numeric_stats.json или model_schema.json
    with pytest.raises(
        RuntimeError, match="Не удалось определить список числовых признаков"
    ):
        FeatureContract.from_model_artifacts(arte)


def test_contract_works_with_baseline_only(tmp_path):
    """Тест что контракт работает с одним baseline_numeric_stats.json."""
    arte = tmp_path / "artefacts" / "model_artefacts"
    arte.mkdir(parents=True, exist_ok=True)
    # Только baseline, без схемы
    (arte / "baseline_numeric_stats.json").write_text(
        json.dumps(
            {
                "text_len": {"mean": 100, "std": 50},
                "word_count": {"mean": 20, "std": 10},
            }
        ),
        encoding="utf-8",
    )
    contract = FeatureContract.from_model_artifacts(arte)
    assert contract.expected_numeric_columns == ["text_len", "word_count"]
    assert contract.expected_numeric_columns == ["text_len", "word_count"]

"""
Тестирует загрузку настроек из переменных окружения.
"""

import importlib
import sys
from pathlib import Path


def _import_settings_module():
    """Импортирует scripts.settings как пакет, форсируя наш корень в sys.path."""
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    settings = importlib.import_module("scripts.settings")
    # Перезагрузка, чтобы учесть новые env переменные
    return importlib.reload(settings)


def test_settings_env_overrides(monkeypatch):
    # Подменяем переменные окружения
    monkeypatch.setenv("RAW_DATA_DIR", "data/raw_env")
    monkeypatch.setenv("PROCESSED_DATA_DIR", "data/processed_env")
    monkeypatch.setenv("MODEL_DIR", "model_env")
    monkeypatch.setenv("SEED", "123")
    monkeypatch.setenv("FORCE_TRAIN", "1")
    monkeypatch.setenv("OPTUNA_N_TRIALS", "7")
    monkeypatch.setenv("OPTUNA_STORAGE", "sqlite:///env_optuna.db")
    monkeypatch.setenv("STUDY_BASE_NAME", "env_study")

    # Загружаем модуль настроек после подмены окружения
    settings = _import_settings_module()

    assert Path("data/raw_env") == settings.RAW_DATA_DIR
    assert Path("data/processed_env") == settings.PROCESSED_DATA_DIR
    assert Path("model_env") == settings.MODEL_DIR
    assert settings.SEED == 123
    assert settings.FORCE_TRAIN is True
    assert settings.OPTUNA_N_TRIALS == 7
    assert settings.OPTUNA_STORAGE == "sqlite:///env_optuna.db"
    assert settings.STUDY_BASE_NAME == "env_study"

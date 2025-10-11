from pathlib import Path

from scripts.train import run


def test_run_creates_artifacts(tmp_path, monkeypatch):
    # Перенаправим каталоги артефактов на временную папку
    monkeypatch.setenv("MODEL_FILE_DIR", str(tmp_path))
    monkeypatch.setenv("ARTEFACTS_DIR", str(tmp_path / "artefacts"))
    import importlib

    import scripts.settings as settings_mod
    import scripts.train as train_mod

    # Перезагрузим settings/train, чтобы подтянуть новые переменные окружения
    importlib.reload(settings_mod)
    importlib.reload(train_mod)

    monkeypatch.setattr(train_mod, "FORCE_TRAIN", True)
    run()
    p = Path(tmp_path)
    arte_root = p / "artefacts"  # MODEL_FILE_DIR теперь artefacts
    assert (arte_root / "best_model.joblib").exists()
    # мета теперь в artefacts/model_artefacts внутри tmp
    assert (arte_root / "model_artefacts" / "best_model_meta.json").exists()

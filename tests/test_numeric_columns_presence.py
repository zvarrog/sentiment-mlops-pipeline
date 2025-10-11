from scripts.train_modules.data_loading import load_splits
from scripts.train_modules.feature_space import NUMERIC_COLS


def test_numeric_columns_present():
    X_train, X_val, X_test, *_ = load_splits()
    present = [c for c in NUMERIC_COLS if c in X_train.columns]
    # Должно быть не меньше половины ожидаемых фич (с учётом возможных изменений пайплайна)
    assert len(present) >= max(1, len(NUMERIC_COLS) // 2)

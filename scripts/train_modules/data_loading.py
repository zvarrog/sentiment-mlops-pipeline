"""Модуль загрузки и подготовки данных для обучения моделей."""

import logging

import pandas as pd

from scripts.config import DATA_PATHS, SEED

from .feature_space import NUMERIC_COLS

log = logging.getLogger("data_loading")


def load_splits() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    """Загружает train/val/test сплиты; если отсутствуют — формирует стратифицированно."""
    from sklearn.model_selection import train_test_split

    log.info("Загрузка данных...")

    if not DATA_PATHS.train.exists():
        raise FileNotFoundError(f"Нет train.parquet: {DATA_PATHS.train}")
    df_train = pd.read_parquet(DATA_PATHS.train)
    if not {"reviewText", "overall"}.issubset(df_train.columns):
        raise KeyError("Ожидаются колонки 'reviewText' и 'overall'")

    frames = {"train": df_train}
    if DATA_PATHS.val.exists():
        frames["val"] = pd.read_parquet(DATA_PATHS.val)
    if DATA_PATHS.test.exists():
        frames["test"] = pd.read_parquet(DATA_PATHS.test)

    if "val" not in frames or "test" not in frames:
        full = df_train.copy()
        if "val" in frames:
            full = pd.concat([full, frames["val"]], ignore_index=True)
        if "test" in frames:
            full = pd.concat([full, frames["test"]], ignore_index=True)
        full = full.dropna(subset=["reviewText", "overall"])
        y_full = full["overall"].astype(int)
        temp_x, x_test, temp_y, y_test = train_test_split(
            full, y_full, test_size=0.15, stratify=y_full, random_state=SEED
        )
        x_train, x_val, y_train, y_val = train_test_split(
            temp_x, temp_y, test_size=0.17647, stratify=temp_y, random_state=SEED
        )
    else:
        x_train = frames["train"]
        x_val = frames["val"]
        x_test = frames["test"]
        y_train = x_train["overall"].astype(int)
        y_val = x_val["overall"].astype(int)
        y_test = x_test["overall"].astype(int)

    def clean(df: pd.DataFrame):
        keep = ["reviewText", "overall"] + [c for c in NUMERIC_COLS if c in df.columns]
        return df[keep].dropna(subset=["reviewText", "overall"])

    x_train = clean(x_train)
    x_val = clean(x_val)
    x_test = clean(x_test)
    y_train = x_train["overall"].astype(int)
    y_val = x_val["overall"].astype(int)
    y_test = x_test["overall"].astype(int)

    return x_train, x_val, x_test, y_train, y_val, y_test

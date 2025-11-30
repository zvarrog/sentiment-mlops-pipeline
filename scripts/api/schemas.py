"""Pydantic-схемы для API запросов и ответов."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from scripts.config import MAX_BATCH_SIZE, MAX_TEXT_LENGTH, NUMERIC_COLS
from scripts.types import FeatureInfo, HealthStatus, ModelInfo


class PredictRequest(BaseModel):
    """Запрос на предсказание тональности."""

    texts: list[Annotated[str, StringConstraints(max_length=MAX_TEXT_LENGTH)]] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE
    )
    numeric_features: dict[str, list[float]] | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": ["This Kindle is amazing!", "Battery life is poor."],
                "numeric_features": None,
            }
        }
    )


class BatchItem(BaseModel):
    """Элемент батч-запроса с текстом и опциональными числовыми признаками.

    Числовые признаки должны соответствовать config.NUMERIC_COLS.
    Дополнительные поля принимаются через extra="allow".
    """

    reviewText: str
    model_config = ConfigDict(extra="allow")

    def get_numeric_features(self) -> dict[str, float]:
        """Извлекает числовые признаки, соответствующие NUMERIC_COLS."""
        data = self.model_dump(exclude={"reviewText"})
        return {k: v for k, v in data.items() if k in NUMERIC_COLS and v is not None}


class BatchPredictRequest(BaseModel):
    """Батч-запрос на предсказание."""

    data: list[BatchItem] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class PredictResponse(BaseModel):
    """Ответ с предсказаниями."""

    labels: list[int]
    probs: list[list[float]] | None = None
    warnings: dict[str, list[str]] | None = None


class MetadataResponse(BaseModel):
    """Метаданные модели и сервиса."""

    model_info: ModelInfo
    feature_contract: FeatureInfo
    health: HealthStatus


class HealthResponse(BaseModel):
    """Статус готовности сервиса."""

    status: Literal["alive", "ready", "not_ready"]
    model_loaded: bool = False
    best_model: str | None = None
    message: str | None = None

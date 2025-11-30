"""
Модуль моделей для обучения: классические, SimpleMLP, DistilBERT.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from scripts.config import SEED

log = logging.getLogger("models")

# Torch
try:
    import torch
except ImportError:
    torch = None


def _select_device(preferred=None):
    """CPU/CUDA выбор устройства."""
    if preferred in {"cpu", "cuda"}:
        return preferred
    if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda"
    return "cpu"


class SimpleMLP(BaseEstimator, ClassifierMixin):
    """
    Простейшая MLP (sklearn-совместимая) поверх плотных признаков.
    """

    _device_logged_globally = False

    def __init__(self, hidden_dim=128, epochs=5, lr=1e-3, batch_size=256, seed=SEED, device=None):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._fitted = False
        self.classes_ = None  # sklearn convention: публичный атрибут
        self._model = None
        self._device = device
        self._device_actual = None
        if torch is None:
            log.warning("SimpleMLP: torch не установлен — модель будет недоступна при fit()")

    def fit(self, X, y):
        """Обучает MLP на плотных признаках."""
        if torch is None:
            raise ImportError("Для SimpleMLP требуется пакет torch. Установите torch.")
        from torch import nn

        torch.Generator().manual_seed(self.seed)
        device_str = _select_device(self._device)
        if not SimpleMLP._device_logged_globally:
            log.info("SimpleMLP: обучение на устройстве %s", device_str)
            SimpleMLP._device_logged_globally = True
        device = torch.device(device_str)

        # Классы и индексация целевой переменной
        unique_labels = np.unique(y)
        self.classes_ = unique_labels
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_idx = np.vectorize(label2idx.get)(y).astype(int)

        # Данные на устройство
        X_ = torch.tensor(X.astype(np.float32), device=device)
        y_ = torch.tensor(y_idx, device=device)
        in_dim = X_.shape[1]
        n_classes = len(unique_labels)

        # Модель
        model = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_classes),
        )
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        # Обучение
        model.train()
        for _ in range(self.epochs):
            for i in range(0, len(X_), self.batch_size):
                xb = X_[i : i + self.batch_size]
                yb = y_[i : i + self.batch_size]
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

        # Финализация
        self._model = model
        self._device_actual = device
        self._fitted = True
        return self

    def predict(self, X):
        """Предсказывает классы."""
        self._ensure_fitted()
        if torch is None:
            raise RuntimeError("torch недоступен во время predict")
        with torch.no_grad():
            device = getattr(self, "_device_actual", torch.device("cpu"))
            t = torch.tensor(X.astype(np.float32), device=device)
            logits = self._model(t)
            pred_idx = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[pred_idx]

    def predict_proba(self, X):
        """Возвращает вероятности классов (softmax по логитам)."""
        self._ensure_fitted()
        if torch is None:
            raise RuntimeError("torch недоступен во время predict_proba")
        with torch.no_grad():
            import torch.nn.functional as F

            device = getattr(self, "_device_actual", torch.device("cpu"))
            t = torch.tensor(X.astype(np.float32), device=device)
            logits = self._model(t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _ensure_fitted(self):
        """Проверка, что модель обучена."""
        if not self._fitted:
            raise RuntimeError("SimpleMLP не обучена")


# DistilBERT вынесен в отдельный модуль scripts/models/distilbert.py

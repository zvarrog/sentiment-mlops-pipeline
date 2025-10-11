import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModel, AutoTokenizer


class DistilBertClassifier(BaseEstimator, ClassifierMixin):
    """Лёгкая обёртка DistilBERT: замороженный encoder + линейная голова."""

    def __init__(
        self,
        epochs=1,
        lr=1e-4,
        max_len=128,
        batch_size=8,
        seed=42,
        use_bigrams=False,
        device=None,
    ):
        """Инициализация DistilBERT классификатора.

        Args:
            epochs: Количество эпох обучения (по умолчанию 1)
            lr: Learning rate (по умолчанию 1e-4)
            max_len: Максимальная длина токенизированного текста (по умолчанию 128)
            batch_size: Размер батча (по умолчанию 8)
            seed: Random seed для воспроизводимости (по умолчанию 42)
            use_bigrams: Добавлять ли биграммы к текстам (по умолчанию False)
            device: Устройство для обучения ('cpu', 'cuda' или None для автовыбора)
        """
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.batch_size = batch_size
        self.seed = seed
        self.use_bigrams = use_bigrams
        self.device = device
        self._fitted = False
        self._tokenizer = None
        self._base_model = None
        self._head = None
        self._device_actual = None
        self._classes_ = None

    def _tokenize(self, texts, return_tensors: str = "pt"):
        """Вызов токенизатора без clean_up_tokenization_spaces."""
        return self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors=return_tensors,
        )

    @staticmethod
    def _augment_texts(texts):
        """Добавляет биграммы к текстам для улучшения качества классификации.

        Args:
            texts: Массив текстов для аугментации

        Returns:
            np.ndarray: Тексты с добавленными биграммами (до 20 штук на текст)
        """

        def _augment_bigrams(s: str) -> str:
            ws = s.split()
            bigrams = [f"{ws[i]}_{ws[i + 1]}" for i in range(len(ws) - 1)]
            return s + (" " + " ".join(bigrams[:20]) if bigrams else "")

        return np.array([_augment_bigrams(t) for t in texts])

    def _get_device(self):
        """Определяет устройство для операций модели.

        Логика:
        1) Если модель загружена — берём device из параметров.
        2) Если не получается — используем _device_actual.
        3) Фоллбек на CPU.
        """
        if self._head is not None:
            try:
                return next(self._head.parameters()).device
            except (StopIteration, RuntimeError):
                pass
        if self._base_model is not None:
            try:
                return next(self._base_model.parameters()).device
            except (StopIteration, RuntimeError):
                pass
        return self._device_actual or torch.device("cpu")

    def fit(self, X, y):
        """Обучает классификатор на текстах с фиксированным DistilBERT-encoder.

        Args:
            X: Тексты (list, np.ndarray или DataFrame)
            y: Метки классов

        Returns:
            self: Обученная модель
        """
        torch.manual_seed(self.seed)
        texts = X if isinstance(X, (list, np.ndarray)) else X.values
        if self.use_bigrams:
            texts = self._augment_texts(texts)
        self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self._base_model = AutoModel.from_pretrained("distilbert-base-uncased")
        device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Отладка: логируем значение устройства
        import logging

        log = logging.getLogger(__name__)
        log.info(
            f"DistilBERT device: self.device='{self.device}', computed device_str='{device_str}'"
        )

        # Проверяем корректность device_str перед созданием torch.device
        # Если запрошен CUDA-устройcтво, но CUDA недоступна — переходим на CPU
        if device_str not in ["cpu", "cuda"] and not device_str.startswith("cuda:"):
            log.warning(
                f"Некорректное устройство '{device_str}', переключаюсь на 'cpu'"
            )
            device_str = "cpu"

        device = torch.device(device_str)
        for p in self._base_model.parameters():
            p.requires_grad = False
        unique_labels = np.unique(y)
        self._classes_ = unique_labels
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}
        hidden = self._base_model.config.hidden_size
        n_classes = len(unique_labels)
        self._head = torch.nn.Linear(hidden, n_classes).to(device)
        self._base_model.to(device)
        optimizer = torch.optim.Adam(self._head.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self._base_model.eval()
        self._head.train()

        def batch_iter():
            for i in range(0, len(texts), self.batch_size):
                yield texts[i : i + self.batch_size], y[i : i + self.batch_size]

        for _ in range(self.epochs):
            for bt, by_raw in batch_iter():
                by = np.vectorize(label2idx.get)(by_raw).astype(int)
                enc = self._tokenize(bt)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = self._base_model(**enc)
                    cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                loss = loss_fn(
                    logits, torch.tensor(by, dtype=torch.long, device=device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self._device_actual = device
        self._head.eval()
        self._fitted = True
        return self

    def predict(self, X):
        """Предсказывает классы для текстов.

        Args:
            X: Тексты (list, np.ndarray или DataFrame)

        Returns:
            np.ndarray: Предсказанные классы

        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self._fitted:
            raise RuntimeError("DistilBertClassifier не обучен")
        texts = X if isinstance(X, (list, np.ndarray)) else X.values
        if self.use_bigrams:
            texts = self._augment_texts(texts)
        preds = []
        device = self._get_device()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenize(batch)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._base_model(**enc)
                cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        preds = np.array(preds)
        return self._classes_[preds]

    def predict_proba(self, X):
        """Вероятности классов (softmax по логитам) в порядке self._classes_."""
        if not self._fitted:
            raise RuntimeError("DistilBertClassifier не обучен")
        import torch.nn.functional as F  # локальный импорт

        texts = X if isinstance(X, (list, np.ndarray)) else X.values
        if self.use_bigrams:
            texts = self._augment_texts(texts)
        device = self._get_device()
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenize(batch)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._base_model(**enc)
                cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        if not all_probs:
            return np.zeros((0, len(self._classes_)), dtype=float)
        return np.vstack(all_probs)

    def __getstate__(self):
        """Перед сериализацией переносим веса на CPU, чтобы обеспечить безопасную загрузку в окружениях без CUDA."""
        state = self.__dict__.copy()
        try:
            if self._base_model is not None:
                self._base_model.to("cpu")
            if self._head is not None:
                self._head.to("cpu")
            # Фиксируем девайс как CPU в сохранённом состоянии
            state["_device_actual"] = torch.device("cpu")
        except Exception:
            pass
        return state

    def __setstate__(self, state):
        """Восстанавливаем состояние; гарантируем CPU-девайс по умолчанию после загрузки."""
        self.__dict__.update(state)
        try:
            self._device_actual = torch.device("cpu")
            # Модели уже на CPU (см. __getstate__), но на всякий случай
            if self._base_model is not None:
                self._base_model.to("cpu")
            if self._head is not None:
                self._head.to("cpu")
        except Exception:
            pass

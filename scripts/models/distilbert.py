import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModel, AutoTokenizer


class DistilBertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        epochs=1,
        lr=1e-4,
        max_len=128,
        batch_size=8,
        seed=42,
        use_bigrams: bool = False,
        device=None,
    ):
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
        return self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors=return_tensors,
        )

    def _get_device(self):
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

    def fit(self, x, y, x_val=None, y_val=None):
        torch.manual_seed(self.seed)
        texts = x if isinstance(x, (list, np.ndarray)) else x.values

        self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self._base_model = AutoModel.from_pretrained("distilbert-base-uncased")

        # Автовыбор устройства
        device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        import logging

        log = logging.getLogger(__name__)
        log.info(f"DistilBERT training on device: {device_str}")

        # Валидация устройства
        if device_str not in ["cpu", "cuda"] and not device_str.startswith("cuda:"):
            log.warning(f"Invalid device '{device_str}', falling back to 'cpu'")
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

        # Подготовка validation set для ранней остановки
        val_texts = None
        val_labels = None
        if x_val is not None and y_val is not None:
            val_texts = x_val if isinstance(x_val, (list, np.ndarray)) else x_val.values
            val_labels = np.vectorize(label2idx.get)(y_val).astype(int)

        def batch_iter(texts_batch, labels_batch):
            for i in range(0, len(texts_batch), self.batch_size):
                yield (
                    texts_batch[i : i + self.batch_size],
                    labels_batch[i : i + self.batch_size],
                )

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 3

        for epoch in range(self.epochs):
            self._head.train()
            train_loss = 0.0
            n_batches = 0

            for bt, by_raw in batch_iter(texts, y):
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
                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0.0

            # Валидация
            if val_texts is not None:
                self._head.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for bt, by in batch_iter(val_texts, val_labels):
                        enc = self._tokenize(bt)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        out = self._base_model(**enc)
                        cls = out.last_hidden_state[:, 0]
                        logits = self._head(cls)
                        loss = loss_fn(
                            logits, torch.tensor(by, dtype=torch.long, device=device)
                        )
                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
                log.info(
                    f"Epoch {epoch + 1}/{self.epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
                )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        log.info(
                            f"Early stopping: no improvement for {patience} epochs"
                        )
                        break
            else:
                log.info(
                    f"Epoch {epoch + 1}/{self.epochs}: train_loss={avg_train_loss:.4f}"
                )

        self._device_actual = device
        self._head.eval()
        self._fitted = True
        return self

    def predict(self, x):
        if not self._fitted:
            raise RuntimeError("DistilBertClassifier не обучен")
        texts = x if isinstance(x, (list, np.ndarray)) else x.values
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

    def predict_proba(self, x):
        if not self._fitted:
            raise RuntimeError("DistilBertClassifier не обучен")
        import torch.nn.functional as func

        texts = x if isinstance(x, (list, np.ndarray)) else x.values
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
                probs = func.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        if not all_probs:
            return np.zeros((0, len(self._classes_)), dtype=float)
        return np.vstack(all_probs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_device_actual"] = torch.device("cpu")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._device_actual = torch.device("cpu")

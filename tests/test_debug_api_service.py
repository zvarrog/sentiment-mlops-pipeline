from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from fastapi.testclient import TestClient


class DummyModel:
    def predict(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np
        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


def test_debug_predict_response_body(tmp_path):
    from scripts.api_service import create_app
    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]

    with (
        patch("scripts.api_service.BEST_MODEL_PATH.exists", return_value=True),
        patch("scripts.api_service.joblib.load", return_value=DummyModel()),
        patch("scripts.api_service.FeatureContract") as mock_contract_cls,
    ):
        mock_contract_cls.from_model_artifacts.return_value = contract
        app = create_app(defer_artifacts=False)
        client = TestClient(app)
        app.state.META = {"best_model": "logreg"}
        app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}
        app.state.FEATURE_CONTRACT = contract
        payload = {"texts": ["great product"], "numeric_features": {"text_len": [13.0], "word_count": [2.0]}}
        resp = client.post("/predict", json=payload)
        print("DEBUG PREDICT STATUS:", resp.status_code)
        print("DEBUG PREDICT BODY:\n", resp.text)
        assert resp.status_code == 200


def test_call_internal_validate_and_predict_directly():
    from scripts.api_service import _validate_and_predict
    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]
    app = SimpleNamespace(state=SimpleNamespace())
    app.state.MODEL = DummyModel()
    app.state.META = {"best_model": "logreg"}
    app.state.FEATURE_CONTRACT = contract
    app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}
    try:
        labels, probs, warnings = _validate_and_predict(app, "/predict", ["great product"], {"text_len": [13.0], "word_count": [2.0]})
        print("INTERNAL_PREDICT_OK", labels, probs, warnings)
    except Exception as e:
        import traceback as _tb
        print("INTERNAL_PREDICT_ERROR", type(e).__name__, e)
        print(_tb.format_exc())
        raise
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from fastapi.testclient import TestClient


class DummyModel:
    def predict(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np
        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


def test_debug_predict_response_body(tmp_path):
    from scripts.api_service import create_app
    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]

    with (
        patch("scripts.api_service.BEST_MODEL_PATH.exists", return_value=True),
        patch("scripts.api_service.joblib.load", return_value=DummyModel()),
        patch("scripts.api_service.FeatureContract") as mock_contract_cls,
    ):
        mock_contract_cls.from_model_artifacts.return_value = contract
        app = create_app(defer_artifacts=False)
        client = TestClient(app)
        app.state.META = {"best_model": "logreg"}
        app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}
        app.state.FEATURE_CONTRACT = contract
        payload = {"texts": ["great product"], "numeric_features": {"text_len": [13.0], "word_count": [2.0]}}
        resp = client.post("/predict", json=payload)
        print("DEBUG PREDICT STATUS:", resp.status_code)
        print("DEBUG PREDICT BODY:\n", resp.text)
        assert resp.status_code == 200


def test_call_internal_validate_and_predict_directly():
    from scripts.api_service import _validate_and_predict
    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]
    app = SimpleNamespace(state=SimpleNamespace())
    app.state.MODEL = DummyModel()
    app.state.META = {"best_model": "logreg"}
    app.state.FEATURE_CONTRACT = contract
    app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}
    try:
        labels, probs, warnings = _validate_and_predict(app, "/predict", ["great product"], {"text_len": [13.0], "word_count": [2.0]})
        print("INTERNAL_PREDICT_OK", labels, probs, warnings)
    except Exception as e:
        import traceback as _tb
        print("INTERNAL_PREDICT_ERROR", type(e).__name__, e)
        print(_tb.format_exc())
        raise
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from fastapi.testclient import TestClient


class DummyModel:
    def predict(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np

        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np

        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


def test_debug_predict_response_body(tmp_path):
    """Debug test: prints /predict response text to show stack trace in 500s."""
    from scripts.api_service import create_app

    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]

    with (
        patch("scripts.api_service.BEST_MODEL_PATH.exists", return_value=True),
        patch("scripts.api_service.joblib.load", return_value=DummyModel()),
        patch("scripts.api_service.FeatureContract") as mock_contract_cls,
    ):
        mock_contract_cls.from_model_artifacts.return_value = contract
        app = create_app(defer_artifacts=False)
        client = TestClient(app)
        app.state.META = {"best_model": "logreg"}
        app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}
        app.state.FEATURE_CONTRACT = contract
        payload = {"texts": ["great product"], "numeric_features": {"text_len": [13.0], "word_count": [2.0]}}
        resp = client.post("/predict", json=payload)
        print("DEBUG PREDICT STATUS:", resp.status_code)
        print("DEBUG PREDICT BODY:\n", resp.text)
        assert resp.status_code == 200


def test_call_internal_validate_and_predict_directly():
    from scripts.api_service import _validate_and_predict

    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]

    app = SimpleNamespace(state=SimpleNamespace())
    app.state.MODEL = DummyModel()
    app.state.META = {"best_model": "logreg"}
    app.state.FEATURE_CONTRACT = contract
    app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}

    try:
        labels, probs, warnings = _validate_and_predict(app, "/predict", ["great product"], {"text_len": [13.0], "word_count": [2.0]})
        print("INTERNAL_PREDICT_OK", labels, probs, warnings)
    except Exception as e:
        import traceback as _tb
        print("INTERNAL_PREDICT_ERROR", type(e).__name__, e)
        print(_tb.format_exc())
        raise
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


class DummyModel:
    def predict(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np

        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np

        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


def test_debug_predict_response_body(tmp_path):
    """Debug test: prints /predict response text to show stack trace in 500s."""
    from scripts.api_service import create_app

    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]

    with (
        patch("scripts.api_service.BEST_MODEL_PATH.exists", return_value=True),
        patch("scripts.api_service.joblib.load", return_value=DummyModel()),
        patch("scripts.api_service.FeatureContract") as mock_contract_cls,
    ):
        mock_contract_cls.from_model_artifacts.return_value = contract
        app = create_app(defer_artifacts=False)
        client = TestClient(app)
        app.state.META = {"best_model": "logreg"}
        app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}
        app.state.FEATURE_CONTRACT = contract
        payload = {"texts": ["great product"], "numeric_features": {"text_len": [13.0], "word_count": [2.0]}}
        resp = client.post("/predict", json=payload)
        print("DEBUG PREDICT STATUS:", resp.status_code)
        print("DEBUG PREDICT BODY:\n", resp.text)
        assert resp.status_code == 200


def test_call_internal_validate_and_predict_directly():
    from types import SimpleNamespace
    from scripts.api_service import _validate_and_predict

    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]

    app = SimpleNamespace(state=SimpleNamespace())
    app.state.MODEL = DummyModel()
    app.state.META = {"best_model": "logreg"}
    app.state.FEATURE_CONTRACT = contract
    app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}

    try:
        labels, probs, warnings = _validate_and_predict(app, '/predict', ['great product'], {'text_len': [13.0], 'word_count': [2.0]})
        print('INTERNAL_PREDICT_OK', labels, probs, warnings)
    except Exception as e:
        import traceback as _tb
        print('INTERNAL_PREDICT_ERROR', type(e).__name__, e)
        print(_tb.format_exc())
        raise
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


class DummyModel:
    def predict(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np

        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        import numpy as np

        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


def test_debug_predict_response_body(tmp_path):
    """Debug test: prints /predict response text to show stack trace in 500s."""
    from scripts.api_service import create_app

    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]

    with (
        patch("scripts.api_service.BEST_MODEL_PATH.exists", return_value=True),
        patch("scripts.api_service.joblib.load", return_value=DummyModel()),
        patch("scripts.api_service.FeatureContract") as mock_contract_cls,
    ):
        mock_contract_cls.from_model_artifacts.return_value = contract
        app = create_app(defer_artifacts=False)
        client = TestClient(app)
        app.state.META = {"best_model": "logreg"}
        app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}
        app.state.FEATURE_CONTRACT = contract
        payload = {"texts": ["great product"], "numeric_features": {"text_len": [13.0], "word_count": [2.0]}}
        resp = client.post("/predict", json=payload)
        print("DEBUG PREDICT STATUS:", resp.status_code)
        print("DEBUG PREDICT BODY:\n", resp.text)
        assert resp.status_code == 200


    def test_call_internal_validate_and_predict_directly():
        from types import SimpleNamespace
        from scripts.api_service import _validate_and_predict

        contract = MagicMock()
        contract.validate_input_data.return_value = {}
        contract.required_text_columns = ["reviewText"]
        contract.expected_numeric_columns = ["text_len", "word_count"]

        app = SimpleNamespace(state=SimpleNamespace())
        app.state.MODEL = DummyModel()
        app.state.META = {"best_model": "logreg"}
        app.state.FEATURE_CONTRACT = contract
        app.state.NUMERIC_DEFAULTS = {"text_len": {"mean": 10.0}, "word_count": {"mean": 2.0}}

    try:
            labels, probs, warnings = _validate_and_predict(app, '/predict', ['great product'], {'text_len': [13.0], 'word_count': [2.0]})
            print('INTERNAL_PREDICT_OK', labels, probs, warnings)
        except Exception as e:
            import traceback as _tb
            print('INTERNAL_PREDICT_ERROR', type(e).__name__, e)
            print(_tb.format_exc())
        raise

"""Unit tests for the prediction API.

Tests the offline-safe parts: preprocessing, payload decoding, softmax, /health,
and /predict. No real model or GCS access, so the suite runs fast in CI.
"""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from mlops_eurosat.api import CLASS_NAMES, _decode_instance, _softmax, preprocess


def _make_b64_image() -> str:
    """Return a base64 JPEG, simulating a client request payload."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _mock_session(predicted_class: int = 1) -> MagicMock:
    """Return a mock ort.InferenceSession whose run() returns high-confidence logits."""
    logits = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
    logits[0, predicted_class] = 10.0
    session = MagicMock()
    session.run.return_value = [logits]
    return session


def test_preprocess_output_shape():
    """Any input size must be resized and batched to (1, 3, 64, 64)."""
    arr = preprocess(Image.new("RGB", (128, 128)))
    assert arr.shape == (1, 3, 64, 64)


def test_preprocess_output_dtype():
    """Output must be float32 for onnxruntime."""
    arr = preprocess(Image.new("RGB", (64, 64)))
    assert arr.dtype == np.float32


def test_decode_instance_from_dict():
    """Accepts the structured payload {"image_b64": "..."} -> RGB image."""
    img = _decode_instance({"image_b64": _make_b64_image()})
    assert img.mode == "RGB"


def test_decode_instance_from_string():
    """Also accepts a bare base64 string -> RGB image."""
    img = _decode_instance(_make_b64_image())
    assert img.mode == "RGB"


def test_softmax_sums_to_one():
    probs = _softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert abs(probs.sum() - 1.0) < 1e-6


def test_softmax_output_shape():
    x = np.random.randn(10).astype(np.float32)
    assert _softmax(x).shape == (10,)


def test_health_endpoint():
    """/health returns 200 + {"status": "ok"} without hitting GCS."""
    from mlops_eurosat.api import app

    with patch("mlops_eurosat.api._load_session_from_gcs", return_value=_mock_session()):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_class_name():
    """/predict returns the correct class name for a mocked session."""
    from mlops_eurosat.api import app

    with patch("mlops_eurosat.api._load_session_from_gcs", return_value=_mock_session(predicted_class=1)):
        with patch("mlops_eurosat.api._log_image"):
            with TestClient(app) as client:
                response = client.post("/predict", json={"instances": [{"image_b64": _make_b64_image()}]})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["class_name"] == "Forest"
    assert predictions[0]["prediction"] == 1


def test_predict_endpoint_probabilities():
    """/predict response includes a probability for every class."""
    from mlops_eurosat.api import app

    with patch("mlops_eurosat.api._load_session_from_gcs", return_value=_mock_session()):
        with patch("mlops_eurosat.api._log_image"):
            with TestClient(app) as client:
                response = client.post("/predict", json={"instances": [{"image_b64": _make_b64_image()}]})

    probs = response.json()["predictions"][0]["probabilities"]
    assert set(probs.keys()) == set(CLASS_NAMES)
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_predict_endpoint_multiple_instances():
    """/predict handles a batch of multiple instances."""
    from mlops_eurosat.api import app

    with patch("mlops_eurosat.api._load_session_from_gcs", return_value=_mock_session()):
        with patch("mlops_eurosat.api._log_image") as mock_log:
            with TestClient(app) as client:
                response = client.post(
                    "/predict",
                    json={"instances": [{"image_b64": _make_b64_image()}, {"image_b64": _make_b64_image()}]},
                )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert mock_log.call_count == 2


def test_metrics_endpoint_no_redirect():
    """/metrics answers 200 directly; the GMP sidecar cannot follow a 307 redirect."""
    from mlops_eurosat.api import app

    with patch("mlops_eurosat.api._load_session_from_gcs", return_value=_mock_session()):
        with TestClient(app) as client:
            response = client.get("/metrics", follow_redirects=False)

    assert response.status_code == 200
    assert "prediction_requests_total" in response.text

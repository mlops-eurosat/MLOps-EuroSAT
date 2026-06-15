"""Unit tests for the prediction API.

Tests the offline-safe parts: preprocessing, payload decoding, and /health.
No real model or GCS access, so the suite runs fast in CI.
"""

import base64
import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from mlops_eurosat.api import _decode_instance, preprocess


def _make_b64_image() -> str:
    """Return a base64 JPEG, simulating a client request payload (in-memory)."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_preprocess_output_shape():
    """Any input size must be normalized to the model's (3, 64, 64) input."""
    tensor = preprocess(Image.new("RGB", (128, 128)))
    assert tensor.shape == (3, 64, 64)


def test_decode_instance_from_dict():
    """Accepts the structured payload {"image_b64": "..."} -> RGB image."""
    img = _decode_instance({"image_b64": _make_b64_image()})
    assert img.mode == "RGB"


def test_decode_instance_from_string():
    """Also accepts a bare base64 string -> RGB image (dual-format contract)."""
    img = _decode_instance(_make_b64_image())
    assert img.mode == "RGB"


def test_health_endpoint():
    """/health returns 200 + {"status": "ok"}.

    Patch the GCS loader to return a valid fake checkpoint so app startup
    (lifespan) can load weights without hitting the network, then drive the
    real route through TestClient.
    """
    from mlops_eurosat.api import app
    from mlops_eurosat.model import Model

    mock_checkpoint = {"state_dict": Model().state_dict()}
    with patch("mlops_eurosat.api._load_checkpoint_from_gcs", return_value=mock_checkpoint):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

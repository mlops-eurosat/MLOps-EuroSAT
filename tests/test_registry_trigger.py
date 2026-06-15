from unittest.mock import patch

from fastapi.testclient import TestClient

from mlops_eurosat.registry_trigger import app


def test_health():
    """/health returns {"status": "ok"} — basic liveness check."""
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_handle_event_ignored_for_unknown_method():
    """Events for irrelevant GCP operations are ignored, not acted on.

    The trigger reacts to a specific audit-log methodName; anything else
    (here "SomeOtherMethod") should return status "ignored" and do nothing.
    """
    with TestClient(app) as client:
        r = client.post("/", json={"protoPayload": {"methodName": "SomeOtherMethod"}})
    assert r.json()["status"] == "ignored"


def test_handle_event_triggers_gate_on_upload():
    """An "UploadModel" event fires the promote/deploy gate exactly once.

    Patch gate_promote_deploy so the real Vertex AI promotion never runs — we
    only test the routing: that the right event triggers the right call and the
    response reflects its result ("promoted").
    """
    with patch(
        "mlops_eurosat.registry_trigger.model_registry.gate_promote_deploy",
        return_value="promoted",
    ) as mock:
        with TestClient(app) as client:
            r = client.post("/", json={"protoPayload": {"methodName": "UploadModel"}})
    mock.assert_called_once()
    assert r.json()["status"] == "promoted"

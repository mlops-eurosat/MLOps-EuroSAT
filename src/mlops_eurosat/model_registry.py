"""Model registry helpers for EuroSAT.

The Vertex AI Model Registry is the source of truth for which model version is
live. W&B is used only for experiment tracking.

Workflow:
  * a freshly trained model is uploaded as a new *version* with alias ``staging``
  * a champion/challenger check promotes it to ``production`` only if its
    ``val_acc`` beats the current production model.
  * the production version is then deployed to the ``eurosat-api`` Cloud Run
    service by updating its ``AIP_STORAGE_URI`` environment variable.

The model weights themselves live in GCS (``artifact_uri``); the registry stores
metadata + aliases and points at that artifact.
"""

from __future__ import annotations

from google.cloud import aiplatform

PROJECT_ID = "mlops-eurosat-496913"
REGION = "europe-west3"

MODEL_DISPLAY_NAME = "eurosat-classifier"

STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"

METRIC_LABEL = "val_acc_bp"

CLOUD_RUN_SERVICE = "eurosat-api"

# Serving container metadata stored in the registry (used at registration time).
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/mlops-eurosat/api:latest"
SERVING_PORT = 8080
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"


def _init() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION)


def _encode_metric(val_acc: float) -> str:
    return str(round(val_acc * 10000))


def _decode_metric(label_value: str | None) -> float | None:
    if not label_value:
        return None
    try:
        return int(label_value) / 10000.0
    except (TypeError, ValueError):
        return None


def _get_model() -> aiplatform.Model | None:
    """Return the (latest) registry entry for our model, or None if not created yet."""
    _init()
    models = aiplatform.Model.list(filter=f'display_name="{MODEL_DISPLAY_NAME}"')
    return models[0] if models else None


def _versioned_model(alias: str) -> aiplatform.Model:
    """Return the model version carrying `alias`."""
    base = _get_model()
    if base is None:
        raise RuntimeError(f"No '{MODEL_DISPLAY_NAME}' model found in the registry.")
    return aiplatform.Model(model_name=base.resource_name, version=alias)


def register_candidate(artifact_uri: str, val_acc: float) -> aiplatform.Model:
    """Upload a freshly trained checkpoint as a new version aliased ``staging``.

    Args:
        artifact_uri: GCS dir holding the checkpoint, e.g.
            ``gs://eurosat_models/checkpoints/<run-id>/``.
        val_acc: validation accuracy, stored as a label for the promotion check.

    Returns:
        The uploaded model version.
    """
    _init()
    existing = _get_model()
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=artifact_uri,
        serving_container_image_uri=SERVING_IMAGE,
        serving_container_predict_route=PREDICT_ROUTE,
        serving_container_health_route=HEALTH_ROUTE,
        serving_container_ports=[SERVING_PORT],
        parent_model=existing.resource_name if existing else None,
        version_aliases=[STAGING_ALIAS],
        labels={METRIC_LABEL: _encode_metric(val_acc)},
    )
    print(f"[registry] uploaded version {model.version_id} aliased '{STAGING_ALIAS}' (val_acc={val_acc:.4f})")
    return model


def get_current_production() -> tuple[str | None, float | None]:
    """Return (version_id, val_acc) of the version aliased ``production``, or (None, None)."""
    try:
        prod = _versioned_model(PRODUCTION_ALIAS)
    except Exception:
        return None, None
    return prod.version_id, _decode_metric(prod.labels.get(METRIC_LABEL))


def promote_if_better(staging_version: str | None = None) -> bool:
    """Promote the staging model to ``production`` iff it beats the current one.

    Compares the ``val_acc`` labels of candidate and current production model.

    Args:
        staging_version: Alias or version id of the candidate; defaults to ``staging``.

    Returns:
        True if the candidate was promoted, False if production was kept.
    """
    model = _get_model()
    if model is None:
        raise RuntimeError(f"No '{MODEL_DISPLAY_NAME}' model found in the registry.")

    alias = staging_version or STAGING_ALIAS
    candidate = _versioned_model(alias)
    candidate_metric = _decode_metric(candidate.labels.get(METRIC_LABEL))
    if candidate_metric is None:
        raise ValueError(f"Candidate version has no numeric '{METRIC_LABEL}' label; cannot promote.")

    _, current_metric = get_current_production()
    if current_metric is not None and candidate_metric <= current_metric:
        print(f"[registry] candidate {candidate_metric:.4f} <= production {current_metric:.4f} -- keeping production.")
        return False

    model.versioning_registry.add_version_aliases(
        new_aliases=[PRODUCTION_ALIAS],
        version=candidate.version_id,
    )
    prev = f"{current_metric:.4f}" if current_metric is not None else "none"
    print(
        f"[registry] promoted version {candidate.version_id} ({candidate_metric:.4f}) to '{PRODUCTION_ALIAS}' "
        f"(previous: {prev})."
    )
    return True


def deploy_to_cloud_run(version_alias: str = PRODUCTION_ALIAS) -> str:
    """Point the eurosat-api Cloud Run service at the promoted model version.

    Updates AIP_STORAGE_URI to the model's GCS artifact directory, which
    triggers a new Cloud Run revision that loads the new weights on startup.
    Returns the service URL.
    """
    from google.cloud import run_v2

    model = _versioned_model(version_alias)
    if not model.uri:
        raise RuntimeError(f"Model version '{version_alias}' has no artifact_uri.")

    client = run_v2.ServicesClient()
    service_name = f"projects/{PROJECT_ID}/locations/{REGION}/services/{CLOUD_RUN_SERVICE}"

    service = client.get_service(name=service_name)
    container = service.template.containers[0]

    # Replace AIP_STORAGE_URI, keep all other env vars intact
    container.env = [e for e in container.env if e.name != "AIP_STORAGE_URI"]
    container.env.append(run_v2.EnvVar(name="AIP_STORAGE_URI", value=model.uri))

    result = client.update_service(service=service).result()
    print(f"[registry] {CLOUD_RUN_SERVICE} now serves model {model.version_id} ({model.uri}) -> {result.uri}")
    return result.uri


def performance_gate(max_latency_s: float = 5.0, num_predictions: int = 100, alias: str = STAGING_ALIAS) -> bool:
    """Latency check: time forward passes of the candidate ONNX model on CPU.

    Args:
        max_latency_s: Total time budget for all predictions together.
        num_predictions: Number of single-image forward passes to run.
        alias: Model version to check; defaults to ``staging``.

    Returns:
        True if all predictions finished within the budget.
    """
    import tempfile
    import time

    import numpy as np
    import onnxruntime as ort  # type: ignore[import-untyped]
    from google.cloud import storage  # type: ignore[attr-defined]

    model_ref = _versioned_model(alias)
    uri = model_ref.uri
    if not uri:
        raise RuntimeError(f"Model version '{alias}' has no artifact_uri.")
    bucket_name, _, prefix = uri[len("gs://") :].partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix.rstrip('/')}/model.onnx")

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        blob.download_to_filename(f.name)
        session = ort.InferenceSession(f.name)

    x = np.random.randn(1, 3, 64, 64).astype(np.float32)
    start = time.perf_counter()
    for _ in range(num_predictions):
        session.run(["logits"], {"image": x})
    elapsed = time.perf_counter() - start

    print(f"[gate] {num_predictions} predictions in {elapsed:.3f}s (limit {max_latency_s}s)")
    return elapsed < max_latency_s


def gate_promote_deploy(max_latency_s: float = 5.0) -> str:
    """Run the full registry-change reaction: gate -> promote -> deploy to Cloud Run.

    Returns:
        One of ``gate_failed``, ``not_better`` or ``promoted_and_deployed``.
    """
    if not performance_gate(max_latency_s=max_latency_s):
        print("[registry-trigger] staging failed the performance gate; not promoting.")
        return "gate_failed"

    if not promote_if_better():
        return "not_better"

    deploy_to_cloud_run()
    return "promoted_and_deployed"

"""Vertex AI Model Registry helpers for EuroSAT (GCP-native registry).

In the architecture the *source of truth* for "which model is live" lives in
the Vertex AI Model Registry, while W&B is used only for experiment tracking.

Workflow :
  * a freshly trained model is uploaded as a new *version* with alias ``staging``
  * a champion/challenger check promotes it to ``production`` only if its
    ``val_acc`` beats the current production model.

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

# Custom serving container used when registering the model.
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/mlops-eurosat/api:latest"
SERVING_PORT = 8080
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"

ENDPOINT_DISPLAY_NAME = "eurosat-endpoint"


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
    print(f"[vertex] uploaded version {model.version_id} aliased '{STAGING_ALIAS}' (val_acc={val_acc:.4f})")
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

    Args:
        staging_version: version id to promote; defaults to whatever currently
            carries the ``staging`` alias.
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
        print(f"[vertex] candidate {candidate_metric:.4f} <= production {current_metric:.4f} -- keeping production.")
        return False

    model.versioning_registry.add_version_aliases(
        new_aliases=[PRODUCTION_ALIAS],
        version=candidate.version_id,
    )
    prev = f"{current_metric:.4f}" if current_metric is not None else "none"
    print(
        f"[vertex] promoted version {candidate.version_id} ({candidate_metric:.4f}) to '{PRODUCTION_ALIAS}' "
        f"(previous: {prev})."
    )
    return True


def deploy_to_endpoint(
    version_alias: str = PRODUCTION_ALIAS,
    machine_type: str = "n1-standard-32",
    min_replicas: int = 1,
    max_replicas: int = 1,
) -> aiplatform.Endpoint:
    """Deploy the given model version to the shared Vertex Endpoint.

    Re-uses the endpoint if it exists. Any previously deployed models are removed
    first so replicas don't accumulate.
    """
    model = _versioned_model(version_alias)

    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"')
    endpoint = endpoints[0] if endpoints else aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)

    if endpoint.list_models():
        endpoint.undeploy_all(sync=True)

    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
        sync=False,
    )
    print(
        f"[vertex] started async deploy of version {model.version_id} to endpoint "
        f"'{ENDPOINT_DISPLAY_NAME}' ({endpoint.resource_name})"
    )
    return endpoint


def performance_gate(max_latency_s: float = 5.0, num_predictions: int = 100, alias: str = STAGING_ALIAS) -> bool:
    """Time `num_predictions` forward passes of the aliased model; True if under the limit.

    Loads the checkpoint from the model version's GCS ``artifact_uri`` and runs it
    on CPU.
    """

    import tempfile
    import time

    import torch
    from google.cloud import storage  # type: ignore[attr-defined]

    from mlops_eurosat.model import Model

    model_ref = _versioned_model(alias)
    uri = model_ref.uri
    if not uri:
        raise RuntimeError(f"Model version '{alias}' has no artifact_uri.")
    bucket_name, _, prefix = uri[len("gs://") :].partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix.rstrip('/')}/model.ckpt")

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        blob.download_to_filename(f.name)
        ckpt = torch.load(f.name, map_location="cpu")

    model = Model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_predictions):
            model(x)
    elapsed = time.perf_counter() - start

    print(f"[gate] {num_predictions} predictions in {elapsed:.3f}s (limit {max_latency_s}s)")
    return elapsed < max_latency_s


def gate_promote_deploy(max_latency_s: float = 5.0) -> str:
    """Run the full registry-change reaction: gate -> promote -> deploy.

    Returns a short status string describing what happened.
    """
    if not performance_gate(max_latency_s=max_latency_s):
        print("[registry-trigger] staging failed the performance gate; not promoting.")
        return "gate_failed"

    if not promote_if_better():
        return "not_better"

    deploy_to_endpoint()
    return "promoted_and_deployed"

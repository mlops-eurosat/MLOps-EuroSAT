"""Model registry helpers for EuroSAT (team entity, W&B Registry).

Champion/challenger promotion: a model only becomes the new "best" if it beats
the current best on val_acc.
"""

from __future__ import annotations

import wandb

MODEL_NAME = "eurosat-classifier"

BEST_ALIAS = "best"

METRIC_KEY = "val_acc"

ORG_ENTITY_NAME = "wandb-eurosat"

REGISTRY_TARGET = f"{ORG_ENTITY_NAME}/wandb-registry-model/{MODEL_NAME}"


def get_current_best() -> tuple[str | None, float | None]:
    """Return (version, val_acc) of the model currently aliased 'best', or (None, None)."""
    api = wandb.Api()
    try:
        art = api.artifact(f"{REGISTRY_TARGET}:{BEST_ALIAS}", type="model")
    except Exception:
        return None, None
    return art.version, art.metadata.get(METRIC_KEY)


def promote_if_better(entity: str, project: str, candidate_run_id: str) -> bool:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{candidate_run_id}")
    cand_art = next(a for a in run.logged_artifacts() if a.type == "model")

    candidate_metric = cand_art.metadata.get(METRIC_KEY)
    if not isinstance(candidate_metric, (int, float)):
        raise ValueError(f"Artifact has no numeric '{METRIC_KEY}' in metadata; cannot promote.")

    _, current_metric = get_current_best()
    if current_metric is not None and candidate_metric <= current_metric:
        print(f"[registry] candidate {candidate_metric:.4f} <= current best {current_metric:.4f} -- keeping best.")
        return False

    link_run = wandb.init(entity=entity, project=project, job_type="promote", reinit=True)
    link_run.link_artifact(artifact=cand_art, target_path=REGISTRY_TARGET, aliases=[BEST_ALIAS])
    link_run.finish()

    prev = f"{current_metric:.4f}" if current_metric is not None else "none"
    print(f"[registry] promoted {candidate_metric:.4f} to 'best' (previous: {prev}).")
    return True


def download_best(root: str = "models/best") -> str:
    """Download the current best model from the registry to `root`. Returns the path."""
    api = wandb.Api()
    art = api.artifact(f"{REGISTRY_TARGET}:{BEST_ALIAS}", type="model")
    path = art.download(root=root)
    print(f"[registry] downloaded best ({art.name}, {METRIC_KEY}={art.metadata.get(METRIC_KEY)}) -> {path}")
    return path

"""Profile the EuroSAT training pipeline.

Reports the data-loading vs. compute split, top Python functions, and top
torch operators. Uses the same Hydra config (and overrides) as train.py.

Run with:
    python src/mlops_eurosat/profiling.py [+profiling.steps=100]
"""

import cProfile
import io
import pstats
import warnings
from pathlib import Path
from time import perf_counter

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat.model import Model


def _section(title: str) -> None:
    """Print a section banner."""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def _select_device() -> torch.device:
    """Pick the fastest available device (CUDA > Apple MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    """Block until queued device work is done so timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _load_dataset(path: str) -> TensorDataset:
    """Load the preprocessed tensors once so every pass can reuse them."""
    data = torch.load(path, weights_only=False)
    return TensorDataset(data["images"], data["targets"])


def _loader(dataset: TensorDataset, batch_size: int, num_workers: int) -> DataLoader:
    """Build a DataLoader over an already-loaded dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _train_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    warmup: int,
    measure: bool = False,
) -> tuple[float, float]:
    """Run ``warmup + steps`` training iterations.

    When ``measure`` is set, returns the accumulated (data_time, compute_time) in
    seconds over the post-warmup steps; otherwise returns ``(0.0, 0.0)``.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    data_time = 0.0
    compute_time = 0.0
    total = warmup + steps

    data_iter = iter(loader)
    tick = perf_counter()
    for i in range(total):
        try:
            imgs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, targets = next(data_iter)
        imgs, targets = imgs.to(device), targets.to(device)
        if measure:
            _synchronize(device)
            after_data = perf_counter()

        optimizer.zero_grad()
        loss = loss_fn(model(imgs), targets)
        loss.backward()
        optimizer.step()

        if measure:
            _synchronize(device)
            after_compute = perf_counter()
            if i >= warmup:
                data_time += after_data - tick
                compute_time += after_compute - after_data
            tick = perf_counter()

    return data_time, compute_time


def _benchmark(
    model: nn.Module, dataset: TensorDataset, cfg: DictConfig, device: torch.device, steps: int, warmup: int
) -> None:
    """Report the wall-clock data-loading vs. compute split (configured workers)."""
    loader = _loader(dataset, cfg.training.batch_size, cfg.training.num_workers)
    data_time, compute_time = _train_loop(model, loader, device, steps, warmup, measure=True)
    total = data_time + compute_time or 1.0
    data_pct = 100 * data_time / total
    compute_pct = 100 * compute_time / total
    imgs_per_s = steps * cfg.training.batch_size / total

    _section("1. Bottleneck  -  data loading vs. compute")
    print(f"  data loading      {data_time:6.3f} s    {data_pct:5.1f} %")
    print(f"  compute fwd+bwd   {compute_time:6.3f} s    {compute_pct:5.1f} %")
    print(f"  throughput        {imgs_per_s:7.1f} images/s")


def _profile_cprofile(
    model: nn.Module, dataset: TensorDataset, cfg: DictConfig, device: torch.device, steps: int, warmup: int
) -> None:
    """Function-level cProfile pass; prints the top Python functions."""
    # num_workers=0 keeps data loading in the main process so it is visible to cProfile.
    loader = _loader(dataset, cfg.training.batch_size, 0)

    profiler = cProfile.Profile()
    profiler.enable()
    _train_loop(model, loader, device, steps, warmup)
    profiler.disable()

    # strip directory paths, sort by tottime, capture to a string to print as one block
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(12)

    _section("2. cProfile  -  top functions (by tottime)")
    print(buf.getvalue().strip())


def _profile_torch(
    model: nn.Module, dataset: TensorDataset, cfg: DictConfig, device: torch.device, steps: int, warmup: int
) -> None:
    """Operator-level torch.profiler pass; prints the top operators."""
    from torch.profiler import ProfilerActivity, profile, schedule

    loader = _loader(dataset, cfg.training.batch_size, cfg.training.num_workers)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sched = schedule(wait=0, warmup=warmup, active=steps, repeat=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Profiler clears events.*")
        with profile(activities=activities, schedule=sched, profile_memory=True) as prof:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()
            data_iter = iter(loader)
            for _ in range(warmup + steps):
                try:
                    imgs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    imgs, targets = next(data_iter)
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(imgs), targets)
                loss.backward()
                optimizer.step()
                prof.step()

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    _section(f"3. torch.profiler  -  top operators (by {sort_key})")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the profiling passes with the configured settings.

    Args:
        cfg: Hydra configuration; reads ``profiling.steps``, ``profiling.warmup``,
            ``profiling.torch``, and the ``training`` settings.
    """
    steps = int(OmegaConf.select(cfg, "profiling.steps", default=50))
    warmup = int(OmegaConf.select(cfg, "profiling.warmup", default=5))
    run_torch = bool(OmegaConf.select(cfg, "profiling.torch", default=True))

    device = _select_device()

    print("\n" + "=" * 60)
    print("  PROFILING  -  EuroSAT training pipeline")
    print(
        f"  device={device}  steps={steps}  warmup={warmup}  "
        f"batch_size={cfg.training.batch_size}  num_workers={cfg.training.num_workers}"
    )
    print("=" * 60)

    model = Model(num_classes=cfg.model.num_classes, lr=cfg.training.lr).to(device)
    dataset = _load_dataset(str(Path(cfg.data_dir) / "train.pt"))

    _benchmark(model, dataset, cfg, device, steps, warmup)
    _profile_cprofile(model, dataset, cfg, device, steps, warmup)
    if run_torch:
        _profile_torch(model, dataset, cfg, device, steps, warmup)
    print()


if __name__ == "__main__":
    main()

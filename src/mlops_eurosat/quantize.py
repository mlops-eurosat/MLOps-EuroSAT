"""Quantize the EuroSAT ONNX model and benchmark fp32 vs int8 inference (M31).

Applies ONNX Runtime *dynamic* quantization (int8 weights) to the exported model
and reports model size, inference latency, and prediction agreement, so the
speed/size trade-off of quantization is measurable on the same artifact we serve.

    # export a fresh model, quantize it, benchmark, and check agreement on the test set
    python src/mlops_eurosat/quantize.py

    # quantize a specific exported model
    python src/mlops_eurosat/quantize.py --onnx models/model.onnx --data data/processed/test.pt
"""

import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
import typer

from mlops_eurosat.model import Model

INPUT_NAME = "image"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (3, 64, 64)


def _section(title: str) -> None:
    """Print a section banner"""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def export_fp32(onnx_path: Path, checkpoint: Path | None = None) -> Path:
    """Export the model to fp32 ONNX, loading trained weights if a checkpoint is given."""
    model = Model()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.randn(1, *INPUT_SHAPE),),
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}},
        dynamo=False,
    )
    src = "checkpoint" if checkpoint else "random init"
    print(f"Exported fp32 ONNX ({src}) -> {onnx_path}")
    return onnx_path


def quantize_dynamic_int8(fp32_path: Path, int8_path: Path) -> Path:
    """Dynamically quantize an fp32 ONNX model to int8 weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Pre-process (shape inference + graph opt) so the quantizer sees a clean graph.
    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prepped.onnx"
        quant_pre_process(str(fp32_path), str(prepped))
        quantize_dynamic(str(prepped), str(int8_path), weight_type=QuantType.QInt8)
    print(f"Quantized int8 ONNX -> {int8_path}")
    return int8_path


def _session(onnx_path: Path) -> ort.InferenceSession:
    """Open a CPU inference session (matches the serving setup in api.py)."""
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def benchmark(onnx_path: Path, batch_size: int, runs: int, warmup: int) -> tuple[float, float]:
    """Return (mean, std) latency in milliseconds over ``runs`` timed forward passes."""
    session = _session(onnx_path)
    x = np.random.randn(batch_size, *INPUT_SHAPE).astype(np.float32)

    for _ in range(warmup):
        session.run([OUTPUT_NAME], {INPUT_NAME: x})

    timings = []
    for _ in range(runs):
        tick = perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: x})
        timings.append((perf_counter() - tick) * 1000.0)
    return float(np.mean(timings)), float(np.std(timings))


def _predict(onnx_path: Path, images: np.ndarray, batch_size: int) -> np.ndarray:
    """Return predicted class indices for every image, batched."""
    session = _session(onnx_path)
    preds = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        logits = session.run([OUTPUT_NAME], {INPUT_NAME: chunk})[0]
        preds.append(logits.argmax(axis=1))
    return np.concatenate(preds)


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main(
    onnx: Path = typer.Option(None, help="fp32 ONNX model to quantize; exports a fresh one if omitted."),
    checkpoint: Path = typer.Option(None, help="Lightning .ckpt to load weights from when exporting."),
    data: Path = typer.Option(
        Path("data/processed/test.pt"), help="Preprocessed .pt for the fp32-vs-int8 agreement check."
    ),
    batch_size: int = typer.Option(1, help="Batch size for the latency benchmark (serving uses 1)."),
    runs: int = typer.Option(200, help="Number of timed forward passes."),
    warmup: int = typer.Option(20, help="Untimed warmup passes before timing."),
) -> None:
    """Quantize the ONNX model and report the size / latency / accuracy trade-off."""
    fp32_path = onnx if onnx is not None else export_fp32(Path("models/model.onnx"), checkpoint)
    int8_path = fp32_path.with_suffix(".quant.onnx")
    quantize_dynamic_int8(fp32_path, int8_path)

    fp32_size, int8_size = _size_mb(fp32_path), _size_mb(int8_path)
    fp32_ms, fp32_std = benchmark(fp32_path, batch_size, runs, warmup)
    int8_ms, int8_std = benchmark(int8_path, batch_size, runs, warmup)

    _section("Model size  -  on disk")
    print(f"  fp32   {fp32_size:7.3f} MB")
    print(f"  int8   {int8_size:7.3f} MB    ({fp32_size / int8_size:.2f}x smaller)")

    _section(f"Latency  -  batch_size={batch_size}, {runs} runs")
    print(f"  fp32   {fp32_ms:7.3f} +/- {fp32_std:.3f} ms")
    print(f"  int8   {int8_ms:7.3f} +/- {int8_std:.3f} ms    ({fp32_ms / int8_ms:.2f}x faster)")

    if data is not None and Path(data).exists():
        payload = torch.load(data, weights_only=False)
        images = payload["images"].numpy().astype(np.float32)
        targets = payload["targets"].numpy()
        fp32_pred = _predict(fp32_path, images, max(batch_size, 64))
        int8_pred = _predict(int8_path, images, max(batch_size, 64))

        _section(f"Prediction quality  -  {len(targets)} test images")
        print(f"  fp32 accuracy        {(fp32_pred == targets).mean():6.2%}")
        print(f"  int8 accuracy        {(int8_pred == targets).mean():6.2%}")
        print(f"  fp32 vs int8 agree   {(fp32_pred == int8_pred).mean():6.2%}")
    else:
        print(f"\n(skipping accuracy check: {data} not found)")
    print()


if __name__ == "__main__":
    typer.run(main)

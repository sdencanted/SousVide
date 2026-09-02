#!/usr/bin/env python3
"""Profile the HistNet and CommNet architectures used by the training notebook.

The script builds the networks from a pilot JSON configuration, creates synthetic
inputs with the configured shapes, benchmarks them, and prints a torch.profiler
operator table. Checkpoint loading is optional because weights do not change the
network's compute profile.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sousvide.control.network_factory import (  # noqa: E402
    generate_network,
    get_network_load_path,
)
from sousvide.synthesize.image_modality import VISUAL_MODALITIES  # noqa: E402


NETWORK_NAMES = ("histNet", "commNet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the HistNet and CommNet defined by a pilot config."
    )
    parser.add_argument(
        "--pilot",
        default="Maverick",
        help="Pilot name used to select configs/pilots/<pilot>.json (default: Maverick).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Pilot JSON path. Overrides --pilot when supplied.",
    )
    parser.add_argument(
        "--cohort",
        default="robustness",
        help="Cohort containing optional checkpoints (default: robustness).",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        choices=NETWORK_NAMES,
        default=list(NETWORK_NAMES),
        help="Networks to profile (default: histNet commNet).",
    )
    parser.add_argument(
        "--image-modality",
        choices=VISUAL_MODALITIES,
        default="kronecker_delta",
        help="CommNet checkpoint modality (default: kronecker_delta).",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load checkpoints from cohorts/<cohort>/roster/<pilot> when available.",
    )
    parser.add_argument(
        "--mode",
        choices=("inference", "training"),
        default="inference",
        help="Profile forward inference or a forward/backward/Adam step.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Autocast precision for CUDA execution (default: float32).",
    )
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--warmup", type=nonnegative_int, default=5)
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument(
        "--profile-iterations",
        type=positive_int,
        default=10,
        help="Steps captured by torch.profiler (default: 10).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--num-threads",
        type=positive_int,
        help="Override the number of PyTorch CPU threads.",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=15,
        help="Number of operator rows shown per network (default: 15).",
    )
    parser.add_argument(
        "--profile-input-transfer",
        action="store_true",
        help=(
            "Include a synthetic CPU-to-device copy in every step. This diagnoses "
            "transfer cost, but not DataLoader worker or storage latency."
        ),
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="Use pinned inputs and non-blocking CPU-to-CUDA copies.",
    )
    parser.add_argument(
        "--compile",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
        help="Optionally profile torch.compile with the selected mode.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Optional directory for Chrome trace JSON files.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable summary.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.image_modality == "event_cloud" and (
            args.precision != "float32" or args.compile != "none"):
        parser.error(
            "event_cloud SECNet profiling currently requires "
            "--precision float32 --compile none")
    return args


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def resolve_from_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def make_inputs(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Create inputs from the deployment dimensions calculated by BaseNet."""
    input_dims = model.get_io_dims()["xdp"]
    inputs = {
        name: torch.randn((batch_size, *dims), device=device)
        for name, dims in input_dims.items()
    }
    if pin_memory:
        inputs = {name: tensor.pin_memory() for name, tensor in inputs.items()}
    return inputs


def build_network(
    network_config: dict[str, Any],
    name: str,
    model_dir: Path,
    image_modality: str,
    load_weights: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, Path | None]:
    """Load a mapped checkpoint or construct the architecture via the factory."""
    if load_weights:
        checkpoint_path = Path(
            get_network_load_path(str(model_dir), name, image_modality)
        )
        if checkpoint_path.is_file():
            model = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            return model, checkpoint_path

    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        model = generate_network(
            network_config,
            name,
            str(model_dir),
            image_modality=image_modality,
        )
    return model, None


def prediction_loss(
    model: torch.nn.Module, outputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Construct a synthetic MSE-like loss over the configured prediction outputs."""
    prediction_names = model.io_idxs["ypd"].keys()
    losses = [outputs[name].float().square().mean() for name in prediction_names]
    if not losses:
        raise RuntimeError("The network config has no prediction outputs")
    return sum(losses)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_step(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    mode: str,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    transfer_inputs: bool = False,
    non_blocking: bool = False,
    annotate: bool = False,
    precision: str = "float32",
) -> dict[str, torch.Tensor]:
    phase = torch.profiler.record_function if annotate else contextlib.nullcontext
    if transfer_inputs:
        with phase("phase:input_transfer"):
            step_inputs = {
                name: tensor.to(device, non_blocking=non_blocking)
                for name, tensor in inputs.items()
            }
    else:
        step_inputs = inputs

    if mode == "inference":
        with torch.inference_mode():
            with phase("phase:forward"):
                with autocast_context(device, precision):
                    return model(step_inputs)

    if optimizer is None:
        raise RuntimeError("Training mode requires an optimizer")
    with phase("phase:zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with phase("phase:forward"):
        with autocast_context(device, precision):
            outputs = model(step_inputs)
    with phase("phase:loss"):
        loss = prediction_loss(model, outputs)
    with phase("phase:backward"):
        loss.backward()
    with phase("phase:optimizer"):
        optimizer.step()
    return outputs


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def percentile(samples: list[float], percentage: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentage * len(ordered)) - 1)
    return ordered[index]


def format_count(value: float) -> str:
    for suffix in ("", "K", "M", "G", "T"):
        if abs(value) < 1000.0:
            return f"{value:.3f}{suffix}"
        value /= 1000.0
    return f"{value:.3f}P"


def format_bytes(value: int | float) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0:
            return f"{amount:.2f} {suffix}"
        amount /= 1024.0
    return f"{amount:.2f} PiB"


COMPUTE_TOKENS = (
    "conv",
    "gemm",
    "matmul",
    "addmm",
    "bmm",
    "linear",
    "cudnn",
    "cutlass",
)
POINTWISE_TOKENS = (
    "relu",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "dropout",
    "elementwise",
    "pointwise",
    "aten::add",
    "aten::mul",
    "aten::sub",
    "aten::div",
    "aten::pow",
)
MEMORY_TOKENS = (
    "pool",
    "aten::cat",
    "aten::stack",
    "aten::permute",
    "aten::transpose",
    "aten::contiguous",
)
TRANSFER_TOKENS = (
    "memcpy",
    "memset",
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
)
SYNC_TOKENS = (
    "synchronize",
    "cudaeventsynchronize",
    "cudadevicesynchronize",
    "cudastreamsynchronize",
    "aten::item",
    "_local_scalar_dense",
)


def operator_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in COMPUTE_TOKENS):
        return "compute"
    if any(token in lowered for token in POINTWISE_TOKENS):
        return "pointwise"
    if any(token in lowered for token in MEMORY_TOKENS):
        return "memory-like"
    if any(token in lowered for token in TRANSFER_TOKENS):
        return "copy/cast"
    if any(token in lowered for token in SYNC_TOKENS):
        return "synchronization"
    return "other"


def event_metric(event: Any, attribute: str) -> float:
    value = getattr(event, attribute, 0)
    return float(value or 0)


def build_diagnosis(
    name: str,
    events: Any,
    args: argparse.Namespace,
    device: torch.device,
    peak_memory_ratio: float | None,
    benchmark_mean_ms: float,
) -> dict[str, Any]:
    """Classify the profile using transparent, deliberately conservative heuristics."""
    profile_steps = args.profile_iterations
    root_name = f"{name}_{args.mode}_step"
    usable_events = [
        event
        for event in events
        if event.key != root_name
        and not event.key.startswith("phase:")
        and not event.key.startswith("ProfilerStep")
    ]
    total_device_us = sum(
        event_metric(event, "self_device_time_total") for event in usable_events
    )
    total_cpu_us = sum(
        event_metric(event, "self_cpu_time_total") for event in usable_events
    )
    benchmark_step_wall_us = benchmark_mean_ms * 1000.0

    category_device_us: dict[str, float] = {}
    category_calls: dict[str, int] = {}
    device_operator_calls = 0
    small_device_operator_calls = 0
    sync_calls = 0
    for event in usable_events:
        category = operator_category(event.key)
        device_us = event_metric(event, "self_device_time_total")
        category_device_us[category] = category_device_us.get(category, 0.0) + device_us
        category_calls[category] = category_calls.get(category, 0) + event.count
        if device_us > 0:
            device_operator_calls += event.count
            if device_us / max(event.count, 1) <= 20.0:
                small_device_operator_calls += event.count
        if category == "synchronization":
            sync_calls += event.count

    def device_fraction(category: str) -> float:
        return category_device_us.get(category, 0.0) / max(total_device_us, 1e-9)

    compute_fraction = device_fraction("compute")
    pointwise_fraction = device_fraction("pointwise")
    memory_like_fraction = device_fraction("memory-like")
    bandwidth_fraction = pointwise_fraction + memory_like_fraction
    copy_fraction = device_fraction("copy/cast")
    small_kernel_fraction = small_device_operator_calls / max(device_operator_calls, 1)
    device_active_proxy = min(
        1.0,
        (total_device_us / profile_steps) / max(benchmark_step_wall_us, 1e-9),
    )

    sort_attribute = (
        "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    top_operators = []
    for event in sorted(
        usable_events,
        key=lambda item: event_metric(item, sort_attribute),
        reverse=True,
    )[: args.top]:
        top_operators.append(
            {
                "name": event.key,
                "category": operator_category(event.key),
                "calls_per_step": event.count / profile_steps,
                "self_cpu_us_per_step": (
                    event_metric(event, "self_cpu_time_total") / profile_steps
                ),
                "self_device_us_per_step": (
                    event_metric(event, "self_device_time_total") / profile_steps
                ),
                "device_time_percent": (
                    100.0
                    * event_metric(event, "self_device_time_total")
                    / max(total_device_us, 1e-9)
                ),
                "flops_per_step": float(event.flops or 0) / profile_steps,
            }
        )

    phase_breakdown = {}
    for event in events:
        if event.key.startswith("phase:"):
            phase_breakdown[event.key.removeprefix("phase:")] = {
                "cpu_total_us_per_step": (
                    event_metric(event, "cpu_time_total") / profile_steps
                ),
                "device_total_us_per_step": (
                    event_metric(event, "device_time_total") / profile_steps
                ),
            }

    input_transfer_device_us = phase_breakdown.get("input_transfer", {}).get(
        "device_total_us_per_step", 0.0
    )
    input_transfer_fraction = min(
        1.0,
        input_transfer_device_us / max(total_device_us / profile_steps, 1e-9),
    )

    metrics = {
        "benchmark_step_wall_us": benchmark_step_wall_us,
        "self_cpu_us_per_step": total_cpu_us / profile_steps,
        "self_device_us_per_step": total_device_us / profile_steps,
        "device_active_time_proxy": device_active_proxy,
        "compute_device_time_fraction": compute_fraction,
        "pointwise_device_time_fraction": pointwise_fraction,
        "memory_like_device_time_fraction": memory_like_fraction,
        "bandwidth_device_time_fraction": bandwidth_fraction,
        "copy_cast_device_time_fraction": copy_fraction,
        "input_transfer_device_time_fraction": input_transfer_fraction,
        "small_device_operator_call_fraction": small_kernel_fraction,
        "device_operator_calls_per_step": device_operator_calls / profile_steps,
        "explicit_synchronization_calls_per_step": sync_calls / profile_steps,
    }

    evidence: list[str] = []
    recommendations: list[str] = []
    if device.type != "cuda":
        regime = "CPU-only run; GPU regime undetermined"
        evidence.append(
            "CUDA kernels and host/device overlap were not captured, so compute-, "
            "bandwidth-, and launch-overhead regimes cannot be separated reliably."
        )
        if top_operators:
            evidence.append(
                f"The largest CPU operator was {top_operators[0]['name']} at "
                f"{top_operators[0]['self_cpu_us_per_step']:.1f} us/step."
            )
        recommendations.append("Re-run with --device cuda on the target training GPU.")
    else:
        overhead_signal = (
            (small_kernel_fraction >= 0.50 and device_active_proxy < 0.75)
            or device_active_proxy < 0.40
        )
        bandwidth_signal = (
            bandwidth_fraction >= 0.25
            or (
                (
                    category_calls.get("pointwise", 0)
                    + category_calls.get("memory-like", 0)
                )
                / max(device_operator_calls, 1)
                >= 0.50
            )
        )
        compute_signal = compute_fraction >= 0.60 and device_active_proxy >= 0.50

        if args.profile_input_transfer and input_transfer_fraction >= 0.20:
            regime = "input-transfer-bound candidate"
        elif overhead_signal:
            regime = "launch/framework-overhead-bound candidate"
        elif bandwidth_signal and not compute_signal:
            regime = "memory-bandwidth-bound candidate"
        elif compute_signal:
            regime = "compute-bound candidate"
        else:
            regime = "mixed or inconclusive GPU bottleneck"

        evidence.extend(
            [
                f"Compute-like operators account for {compute_fraction:.1%} of "
                "self device time.",
                f"Pointwise and memory-like operators account for "
                f"{bandwidth_fraction:.1%} of self device time.",
                f"{small_kernel_fraction:.1%} of device-associated operator calls "
                "average 20 us or less.",
                f"Device work occupies roughly {device_active_proxy:.1%} of the "
                "benchmarked step wall time (a profiler-derived proxy, not nvidia-smi utilization).",
                "torch.profiler does not measure total HBM bytes, so the bandwidth "
                "classification uses operator mix and kernel size rather than a true "
                "FLOPs/byte roofline measurement.",
            ]
        )
        if args.profile_input_transfer and input_transfer_fraction >= 0.05:
            evidence.append(
                f"The annotated input-copy phase consumes {input_transfer_fraction:.1%} "
                "of self device time."
            )

        if regime == "compute-bound candidate":
            recommendations.append(
                "Concentrate on the dominant convolution/matmul operators; test AMP/TF32 "
                "and backend autotuning before rewriting pointwise code."
            )
        elif regime == "memory-bandwidth-bound candidate":
            recommendations.append(
                "Test --compile reduce-overhead to fuse pointwise chains and reduce "
                "round-trips through device memory."
            )
        elif regime == "launch/framework-overhead-bound candidate":
            recommendations.append(
                "Test --compile reduce-overhead, increase batch size, and inspect the "
                "Chrome trace for gaps between short kernels."
            )
        elif regime == "input-transfer-bound candidate":
            recommendations.append(
                "Use pinned DataLoader memory and non-blocking device copies, then overlap "
                "transfer with compute where the training loop permits."
            )
        else:
            recommendations.append(
                "Open the Chrome trace and inspect the dominant operators and gaps before "
                "choosing an optimization."
            )

        if sync_calls:
            recommendations.append(
                "Remove or batch .item(), .cpu(), and explicit synchronization calls from "
                "the hot loop where correctness allows."
            )
        if peak_memory_ratio is not None and peak_memory_ratio >= 0.85:
            recommendations.append(
                "Peak allocated memory exceeds 85% of device capacity; reduce batch size "
                "or use activation checkpointing before testing faster kernels."
            )

    if not args.profile_input_transfer:
        recommendations.append(
            "This run isolates model compute. Use --profile-input-transfer to measure "
            "synthetic H2D cost; profile the real DataLoader separately for I/O stalls."
        )
    if args.trace_dir is None:
        recommendations.append(
            "Add --trace-dir profiles/traces and inspect the trace in Perfetto or "
            "chrome://tracing to verify kernel gaps and overlap."
        )

    return {
        "regime": regime,
        "heuristic": True,
        "metrics": metrics,
        "phase_breakdown": phase_breakdown,
        "evidence": evidence,
        "recommendations": recommendations,
        "top_operators": top_operators,
    }


def profile_network(
    name: str,
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], str]:
    training = args.mode == "training"
    model.train(training)
    optimizer = torch.optim.Adam(model.parameters()) if training else None

    outputs: dict[str, torch.Tensor] = {}
    for _ in range(args.warmup):
        outputs = run_step(
            model,
            inputs,
            args.mode,
            optimizer,
            device,
            transfer_inputs=args.profile_input_transfer,
            non_blocking=args.non_blocking,
            precision=args.precision,
        )
    synchronize(device)

    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        synchronize(device)
        start = time.perf_counter()
        outputs = run_step(
            model,
            inputs,
            args.mode,
            optimizer,
            device,
            transfer_inputs=args.profile_input_transfer,
            non_blocking=args.non_blocking,
            precision=args.precision,
        )
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler:
        for _ in range(args.profile_iterations):
            with torch.profiler.record_function(f"{name}_{args.mode}_step"):
                outputs = run_step(
                    model,
                    inputs,
                    args.mode,
                    optimizer,
                    device,
                    transfer_inputs=args.profile_input_transfer,
                    non_blocking=args.non_blocking,
                    annotate=True,
                    precision=args.precision,
                )

    events = profiler.key_averages()
    estimated_flops = sum((event.flops or 0) for event in events)
    estimated_flops /= args.profile_iterations
    cpu_allocation_bytes = sum(
        max(event.self_cpu_memory_usage, 0) for event in events
    ) / args.profile_iterations

    trace_path = None
    if args.trace_dir is not None:
        trace_dir = resolve_from_repo(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{name}_{args.mode}.json"
        profiler.export_chrome_trace(str(trace_path))

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    mean_ms = statistics.fmean(latencies_ms)
    peak_memory_ratio = None
    if device.type == "cuda":
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        total_device_bytes = torch.cuda.get_device_properties(device).total_memory
        peak_memory_ratio = peak_allocated_bytes / total_device_bytes

    diagnosis = build_diagnosis(
        name, events, args, device, peak_memory_ratio, mean_ms
    )
    summary: dict[str, Any] = {
        "network": name,
        "network_type": getattr(model, "network_type", "compiled"),
        "mode": args.mode,
        "compile_mode": args.compile,
        "precision": args.precision,
        "device": str(device),
        "batch_size": args.batch_size,
        "profile_scope": (
            "model plus synthetic input transfer"
            if args.profile_input_transfer
            else "model compute only"
        ),
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_bytes": parameter_bytes,
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "latency_ms": {
            "mean": mean_ms,
            "median": statistics.median(latencies_ms),
            "p95": percentile(latencies_ms, 0.95),
            "minimum": min(latencies_ms),
            "maximum": max(latencies_ms),
        },
        "samples_per_second": args.batch_size / (mean_ms / 1000.0),
        "estimated_flops_per_step": estimated_flops,
        "positive_cpu_allocation_bytes_per_profiled_step": cpu_allocation_bytes,
        "diagnosis": diagnosis,
    }
    if device.type == "cuda":
        summary["peak_cuda_allocated_bytes"] = peak_allocated_bytes
        summary["total_cuda_memory_bytes"] = total_device_bytes
        summary["peak_cuda_memory_fraction"] = peak_memory_ratio
    if trace_path is not None:
        summary["trace_path"] = str(trace_path)

    sort_by = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = events.table(sort_by=sort_by, row_limit=args.top)
    return summary, table


def print_summary(summary: dict[str, Any], table: str) -> None:
    latency = summary["latency_ms"]
    print(f"\n{'=' * 88}")
    print(
        f"{summary['network']} ({summary['network_type']}) | "
        f"{summary['mode']} | {summary['device']} | batch {summary['batch_size']} | "
        f"compile={summary['compile_mode']} | precision={summary['precision']}"
    )
    print(f"{'=' * 88}")
    print(f"Inputs:  {summary['input_shapes']}")
    print(f"Outputs: {summary['output_shapes']}")
    print(
        "Parameters: "
        f"{summary['parameters']:,} ({format_bytes(summary['parameter_bytes'])}); "
        f"trainable {summary['trainable_parameters']:,}"
    )
    print(
        "Latency: "
        f"mean {latency['mean']:.3f} ms | median {latency['median']:.3f} ms | "
        f"p95 {latency['p95']:.3f} ms | "
        f"{summary['samples_per_second']:.2f} samples/s"
    )
    print(
        "Estimated profiler FLOPs/step: "
        f"{format_count(summary['estimated_flops_per_step'])} "
        "(supported operators only)"
    )
    print(
        "Positive CPU allocation volume/profiled step: "
        f"{format_bytes(summary['positive_cpu_allocation_bytes_per_profiled_step'])}"
    )
    if "peak_cuda_allocated_bytes" in summary:
        print(
            "Peak CUDA memory allocated: "
            f"{format_bytes(summary['peak_cuda_allocated_bytes'])} "
            f"({summary['peak_cuda_memory_fraction']:.1%} of device memory)"
        )
    if "trace_path" in summary:
        print(f"Chrome trace: {summary['trace_path']}")

    diagnosis = summary["diagnosis"]
    metrics = diagnosis["metrics"]
    print("\nBottleneck diagnosis (heuristic):")
    print(f"  Regime: {diagnosis['regime']}")
    print(f"  Scope:  {summary['profile_scope']}")
    if summary["device"].startswith("cuda"):
        print(
            "  GPU mix: "
            f"compute {metrics['compute_device_time_fraction']:.1%} | "
            f"pointwise/memory {metrics['bandwidth_device_time_fraction']:.1%} | "
            f"copy/cast {metrics['copy_cast_device_time_fraction']:.1%}"
        )
        if summary["profile_scope"] == "model plus synthetic input transfer":
            print(
                "  Input copy: "
                f"{metrics['input_transfer_device_time_fraction']:.1%} of "
                "self device time"
            )
        print(
            "  Launch clues: "
            f"short-call fraction {metrics['small_device_operator_call_fraction']:.1%} | "
            f"device-active proxy {metrics['device_active_time_proxy']:.1%} | "
            f"{metrics['device_operator_calls_per_step']:.1f} device calls/step"
        )

    phases = diagnosis["phase_breakdown"]
    if phases:
        print("  Phases (CPU total / device total per profiled step):")
        for phase_name, phase in phases.items():
            print(
                f"    {phase_name:14s} "
                f"{phase['cpu_total_us_per_step']:10.1f} us / "
                f"{phase['device_total_us_per_step']:10.1f} us"
            )
    print("  Evidence:")
    for item in diagnosis["evidence"]:
        print(f"    - {item}")
    print("  Suggested next checks:")
    for item in diagnosis["recommendations"]:
        print(f"    - {item}")

    print("  Categorized hot operators:")
    for operator in diagnosis["top_operators"][:5]:
        hot_time = (
            operator["self_device_us_per_step"]
            if summary["device"].startswith("cuda")
            else operator["self_cpu_us_per_step"]
        )
        print(
            f"    - [{operator['category']:15s}] {operator['name']}: "
            f"{hot_time:.1f} us/step, {operator['calls_per_step']:.1f} calls/step"
        )
    print("\nTop operators:")
    print(table)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    device = select_device(args.device)
    if args.profile_input_transfer and device.type != "cuda":
        raise ValueError("--profile-input-transfer requires a CUDA device")
    if args.non_blocking and not args.profile_input_transfer:
        raise ValueError("--non-blocking requires --profile-input-transfer")
    if args.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("--precision bfloat16 requires a CUDA device")
    config_path = (
        resolve_from_repo(args.config)
        if args.config is not None
        else REPO_ROOT / "configs" / "pilots" / f"{args.pilot}.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Pilot config not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    print(f"Config: {config_path}")
    print(f"PyTorch: {torch.__version__} | device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    summaries: list[dict[str, Any]] = []
    if args.load_weights:
        model_dir_context = contextlib.nullcontext(
            REPO_ROOT / "cohorts" / args.cohort / "roster" / args.pilot
        )
    else:
        model_dir_context = tempfile.TemporaryDirectory(
            prefix="sousvide-profile-untrained-"
        )

    with model_dir_context as model_dir_value:
        model_dir = Path(model_dir_value)
        for name in args.networks:
            network_config = config.get("networks", {}).get(name)
            if network_config is None:
                print(f"Skipping {name}: it is not present in {config_path}")
                continue

            model, checkpoint_path = build_network(
                network_config,
                name,
                model_dir,
                args.image_modality,
                args.load_weights,
                device,
            )
            if args.load_weights:
                if checkpoint_path is None:
                    print(f"No checkpoint found for {name}; profiling initialized weights.")
                else:
                    print(f"Loaded {name} checkpoint: {checkpoint_path}")
            model = model.to(device)
            input_device = (
                torch.device("cpu") if args.profile_input_transfer else device
            )
            inputs = make_inputs(
                model,
                args.batch_size,
                input_device,
                pin_memory=args.non_blocking,
            )
            if args.compile != "none":
                if args.compile == "default":
                    model = torch.compile(model)
                else:
                    model = torch.compile(model, mode=args.compile)
            summary, table = profile_network(name, model, inputs, args, device)
            print_summary(summary, table)
            summaries.append(summary)

            del inputs, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not summaries:
        raise RuntimeError("None of the requested networks were found in the config")

    if args.json_output is not None:
        output_path = resolve_from_repo(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_path),
            "torch_version": torch.__version__,
            "device": str(device),
            "results": summaries,
        }
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.write("\n")
        print(f"\nJSON summary: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

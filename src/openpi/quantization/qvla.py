"""QVLA-style fake quantization for openpi PyTorch models.

This module adapts the QVLA algorithmic pieces to openpi models. It does not
load OpenVLA/HuggingFace checkpoints; callers should load openpi policies first
and then call these helpers on the PyTorch model instance.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import heapq
import json
import logging
import math
import pathlib
from typing import Any
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TargetPreset = Literal["pi05_backbones", "all_linear_conv"]
MismatchPolicy = Literal["median", "skip", "error"]
ActivationGranularity = Literal["dynamic-token", "dynamic-tensor", "calibrated-tensor"]


_PI05_TARGET_PREFIXES = (
    "paligemma_with_expert.paligemma.model.language_model.",
    "paligemma_with_expert.paligemma.model.vision_tower.",
    "paligemma_with_expert.gemma_expert.model.",
)

_ALWAYS_EXCLUDE_NAME_PARTS = (
    ".lm_head",
    "multi_modal_projector",
)

_PI05_EXCLUDE_NAME_PARTS = (
    "input_layernorm.dense",
    "post_attention_layernorm.dense",
    ".norm.dense",
)


@dataclasses.dataclass(frozen=True)
class InjectionReport:
    """Summary of a fake-quant injection pass."""

    loaded_gate_layers: int
    target_layers: int
    injected_layers: int
    missing_gate_layers: tuple[str, ...]
    mismatched_layers: tuple[str, ...]
    unused_gate_layers: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"loaded_gate_layers={self.loaded_gate_layers}, "
            f"target_layers={self.target_layers}, "
            f"injected_layers={self.injected_layers}, "
            f"missing_gate_layers={len(self.missing_gate_layers)}, "
            f"mismatched_layers={len(self.mismatched_layers)}, "
            f"unused_gate_layers={len(self.unused_gate_layers)}"
        )


@dataclasses.dataclass(frozen=True)
class ActivationInjectionReport:
    """Summary of an activation fake-quant hook injection pass."""

    activation_bits: int
    activation_granularity: ActivationGranularity
    target_layers: int
    injected_layers: int
    scale_layers: int
    missing_scale_layers: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"activation_bits={self.activation_bits}, "
            f"activation_granularity={self.activation_granularity}, "
            f"target_layers={self.target_layers}, "
            f"injected_layers={self.injected_layers}, "
            f"scale_layers={self.scale_layers}, "
            f"missing_scale_layers={len(self.missing_scale_layers)}"
        )


def is_target_module(name: str, module: nn.Module, target: TargetPreset = "pi05_backbones") -> bool:
    """Return whether a module should receive QVLA fake quantization."""

    if not isinstance(module, (nn.Linear, nn.Conv2d)):
        return False
    if any(part in name for part in _ALWAYS_EXCLUDE_NAME_PARTS):
        return False

    if target == "all_linear_conv":
        return True
    if target == "pi05_backbones":
        if any(part in name for part in _PI05_EXCLUDE_NAME_PARTS):
            return False
        return any(name.startswith(prefix) for prefix in _PI05_TARGET_PREFIXES)

    raise ValueError(f"Unknown QVLA target preset: {target}")


def iter_target_modules(
    model: nn.Module,
    target: TargetPreset = "pi05_backbones",
) -> list[tuple[str, nn.Module]]:
    """List target modules in model order."""

    return [(name, module) for name, module in model.named_modules() if is_target_module(name, module, target)]


def _load_raw_gate_file(gates_path: str | pathlib.Path) -> Mapping[str, Any]:
    gates_path = pathlib.Path(gates_path).expanduser()
    if gates_path.suffix == ".pt":
        raw = torch.load(gates_path, map_location="cpu")
    else:
        with gates_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected gate file to contain a mapping, got {type(raw)!r}")
    if isinstance(raw.get("assign"), Mapping):
        raw = raw["assign"]
    return raw


def load_gate_assignments(
    gates_path: str | pathlib.Path,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Load QVLA per-layer channel gates.

    Supported formats:
    - raw mapping: {"module.name": [0, 4, 8, ...]}
    - QVLA allocation output: {"assign": {"module.name": [...]}, "stats": ...}
    - ``.pt`` files with the same structures.
    """

    torch_device = None if device is None else torch.device(device)
    raw = _load_raw_gate_file(gates_path)
    gates: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            continue
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(dtype=torch.int64)
        else:
            tensor = torch.tensor(value, dtype=torch.int64)
        if torch_device is not None:
            tensor = tensor.to(torch_device)
        gates[str(key)] = tensor
    return gates


def _gate_name_candidates(module_name: str) -> tuple[str, ...]:
    candidates = [module_name]

    prefix = "paligemma_with_expert.paligemma.model.language_model."
    if module_name.startswith(prefix):
        suffix = module_name[len(prefix) :]
        candidates.extend((f"language_model.{suffix}", suffix))

    prefix = "paligemma_with_expert.paligemma.model.vision_tower."
    if module_name.startswith(prefix):
        suffix = module_name[len(prefix) :]
        candidates.extend((f"vision_backbone.{suffix}", f"vision_tower.{suffix}"))

    prefix = "paligemma_with_expert.gemma_expert.model."
    if module_name.startswith(prefix):
        suffix = module_name[len(prefix) :]
        candidates.extend((f"gemma_expert.model.{suffix}", f"action_expert.{suffix}"))

    return tuple(dict.fromkeys(candidates))


def find_gate_for_module(module_name: str, gates: Mapping[str, torch.Tensor]) -> tuple[str, torch.Tensor] | None:
    """Find the matching gate tensor for a module name, including common aliases."""

    for candidate in _gate_name_candidates(module_name):
        if candidate in gates:
            return candidate, gates[candidate]
    return None


@torch.no_grad()
def fake_quantize_tensor_sym(
    x: torch.Tensor,
    num_bits: int,
    *,
    scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Symmetric fake quantization with an optional precomputed quantization scale."""

    if num_bits >= 16:
        return x
    if num_bits <= 0:
        return torch.zeros_like(x)
    if not torch.is_floating_point(x):
        return x

    qmax = (1 << (num_bits - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"num_bits must be 0 or at least 2, got {num_bits}")

    work = x.float()
    if scale is None:
        quant_scale = work.abs().max().clamp_min(1e-8) / float(qmax)
    else:
        quant_scale = torch.as_tensor(scale, device=x.device, dtype=torch.float32).clamp_min(1e-8)
    quantized = torch.round(work / quant_scale).clamp_(min=-(qmax + 1), max=qmax)
    return (quantized * quant_scale).to(dtype=x.dtype)


@torch.no_grad()
def fake_quantize_activation_sym(
    x: torch.Tensor,
    num_bits: int,
    *,
    amax: torch.Tensor | float | None = None,
    granularity: ActivationGranularity = "dynamic-token",
) -> torch.Tensor:
    """Symmetric activation fake quantization.

    ``dynamic-token`` uses a separate scale for each token/row for Linear inputs
    and each sample for Conv2d inputs. ``calibrated-tensor`` uses a precomputed
    layer-wide max-absolute activation value.
    """

    if num_bits >= 16:
        return x
    if num_bits <= 1:
        raise ValueError(f"activation bits must be at least 2 or 16, got {num_bits}")
    if not torch.is_floating_point(x):
        return x

    qmax = (1 << (num_bits - 1)) - 1
    if granularity == "calibrated-tensor":
        if amax is None:
            return fake_quantize_tensor_sym(x, num_bits)
        quant_scale = torch.as_tensor(amax, device=x.device, dtype=torch.float32).clamp_min(1e-8) / float(qmax)
    elif granularity == "dynamic-tensor":
        quant_scale = x.float().abs().max().clamp_min(1e-8) / float(qmax)
    elif granularity == "dynamic-token":
        work = x.float()
        if work.ndim >= 4:
            reduce_dims = tuple(range(1, work.ndim))
            amax_dynamic = work.abs().amax(dim=reduce_dims, keepdim=True)
        elif work.ndim >= 2:
            amax_dynamic = work.abs().amax(dim=-1, keepdim=True)
        else:
            amax_dynamic = work.abs().max()
        quant_scale = amax_dynamic.clamp_min(1e-8) / float(qmax)
    else:
        raise ValueError(f"Unknown activation granularity: {granularity}")
    return fake_quantize_tensor_sym(x, num_bits, scale=quant_scale)


@torch.no_grad()
def estimate_activation_amax_from_histogram(
    histogram: torch.Tensor,
    max_abs: float,
    num_bits: int,
    *,
    grid_size: int = 256,
) -> float:
    """Estimate a static activation clipping value by minimizing histogram MSE.

    The histogram is over ``abs(activation)`` values on ``[0, max_abs]``. The
    returned value is the calibrated max-absolute value to use with symmetric
    fake quantization.
    """

    if num_bits >= 16:
        return float(max_abs)
    if num_bits <= 1:
        raise ValueError(f"activation bits must be at least 2 or 16, got {num_bits}")
    if max_abs <= 0:
        return 0.0

    counts = histogram.detach().to(dtype=torch.float64, device="cpu").flatten()
    if counts.numel() == 0 or float(counts.sum().item()) <= 0:
        return float(max_abs)

    bins = int(counts.numel())
    max_value = float(max_abs)
    bin_width = max_value / float(bins)
    centers = (torch.arange(bins, dtype=torch.float64) + 0.5) * bin_width

    grid_size = max(1, min(int(grid_size), bins))
    candidate_indices = torch.linspace(1, bins, steps=grid_size, dtype=torch.float64).round().to(torch.int64)
    candidate_indices = torch.unique(candidate_indices.clamp_(1, bins))
    thresholds = candidate_indices.to(torch.float64) * bin_width

    qmax = (1 << (num_bits - 1)) - 1
    total_count = counts.sum().clamp_min(1.0)
    best_amax = max_value
    best_mse = float("inf")

    # Keep the loop explicit to avoid a large thresholds x bins temporary when
    # using fine histograms for hundreds of layers.
    for threshold in thresholds:
        scale = threshold / float(qmax)
        clipped = centers.clamp(max=float(threshold.item()))
        quantized = torch.round(clipped / scale).clamp_(min=0, max=qmax)
        dequantized = quantized * scale
        mse = float((((centers - dequantized) ** 2) * counts).sum().item() / float(total_count.item()))
        if mse < best_mse:
            best_mse = mse
            best_amax = float(threshold.item())

    return max(best_amax, 1e-8)


@torch.no_grad()
def apply_weight_only_fake_quant(
    module: nn.Module,
    gates: torch.Tensor,
    *,
    mismatch_policy: MismatchPolicy = "median",
) -> bool:
    """Apply channel-wise fake quantization to one Linear or Conv2d module."""

    if not isinstance(module, (nn.Linear, nn.Conv2d)):
        return False

    weight = module.weight.data
    out_channels = int(weight.shape[0])
    gate_tensor = gates.to(device=weight.device, dtype=torch.int64).flatten()

    if gate_tensor.numel() != out_channels:
        if mismatch_policy == "error":
            raise ValueError(f"Gate length {gate_tensor.numel()} does not match out_channels {out_channels}")
        if mismatch_policy == "skip":
            return False
        if gate_tensor.numel() == 0:
            return False
        median_bit = int(gate_tensor.float().median().round().item())
        gate_tensor = torch.full((out_channels,), median_bit, device=weight.device, dtype=torch.int64)

    for channel_idx in range(out_channels):
        bit_width = int(gate_tensor[channel_idx].item())
        if bit_width >= 16:
            continue
        if bit_width <= 0:
            weight[channel_idx].zero_()
            continue
        weight[channel_idx].copy_(fake_quantize_tensor_sym(weight[channel_idx], bit_width))

    return True


@torch.no_grad()
def inject_weight_fake_quant(
    model: nn.Module,
    gates_path: str | pathlib.Path,
    *,
    target: TargetPreset = "pi05_backbones",
    mismatch_policy: MismatchPolicy = "median",
) -> InjectionReport:
    """Inject QVLA-style weight-only fake quantization into an already-loaded model."""

    gates = load_gate_assignments(gates_path)
    used_gate_names: set[str] = set()
    target_modules = iter_target_modules(model, target=target)
    missing_gate_layers: list[str] = []
    mismatched_layers: list[str] = []
    injected_layers = 0

    for module_name, module in target_modules:
        matched_gate = find_gate_for_module(module_name, gates)
        if matched_gate is None:
            missing_gate_layers.append(module_name)
            continue

        gate_name, gate_tensor = matched_gate
        used_gate_names.add(gate_name)
        expected_channels = int(module.weight.shape[0])
        if gate_tensor.numel() != expected_channels:
            mismatched_layers.append(module_name)

        applied = apply_weight_only_fake_quant(module, gate_tensor, mismatch_policy=mismatch_policy)
        if applied:
            injected_layers += 1

    unused_gate_layers = tuple(sorted(set(gates) - used_gate_names))
    report = InjectionReport(
        loaded_gate_layers=len(gates),
        target_layers=len(target_modules),
        injected_layers=injected_layers,
        missing_gate_layers=tuple(missing_gate_layers),
        mismatched_layers=tuple(mismatched_layers),
        unused_gate_layers=unused_gate_layers,
    )
    logger.info("QVLA fake weight injection: %s", report.summary())
    return report


def _load_activation_amax_payload(scales_path: str | pathlib.Path) -> Mapping[str, Any]:
    scales_path = pathlib.Path(scales_path).expanduser()
    if scales_path.suffix == ".pt":
        raw = torch.load(scales_path, map_location="cpu")
    else:
        with scales_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected activation scale file to contain a mapping, got {type(raw)!r}")
    return raw


def _load_raw_activation_amax_file(scales_path: str | pathlib.Path) -> Mapping[str, Any]:
    raw = _load_activation_amax_payload(scales_path)
    if isinstance(raw.get("activations"), Mapping):
        raw = raw["activations"]
    if isinstance(raw.get("activation_amax"), Mapping):
        raw = raw["activation_amax"]
    return raw


def load_activation_amax_metadata(scales_path: str | pathlib.Path) -> dict[str, Any]:
    """Load non-activation metadata from a calibrated activation scale file."""

    raw = _load_activation_amax_payload(scales_path)
    return {
        str(key): value
        for key, value in raw.items()
        if key not in {"activations", "activation_amax"}
    }


def load_activation_amax(
    scales_path: str | pathlib.Path,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Load calibrated max-absolute activation values.

    Supported formats:
    - raw mapping: {"module.name": 12.3}
    - wrapped mapping: {"activations": {"module.name": {"amax": 12.3}}}
    - wrapped mapping: {"activation_amax": {"module.name": 12.3}}
    """

    torch_device = None if device is None else torch.device(device)
    raw = _load_raw_activation_amax_file(scales_path)
    amax: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            if "amax" not in value:
                continue
            value = value["amax"]
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.numel() != 1:
            tensor = tensor.flatten()[0]
        if torch_device is not None:
            tensor = tensor.to(torch_device)
        amax[str(key)] = tensor.reshape(())
    return amax


def find_activation_amax_for_module(
    module_name: str,
    activation_amax: Mapping[str, torch.Tensor],
) -> tuple[str, torch.Tensor] | None:
    """Find a calibrated activation scale for a module name, including common aliases."""

    for candidate in _gate_name_candidates(module_name):
        if candidate in activation_amax:
            return candidate, activation_amax[candidate]
    return None


def inject_activation_fake_quant(
    model: nn.Module,
    *,
    num_bits: int,
    target: TargetPreset = "pi05_backbones",
    activation_scales_path: str | pathlib.Path | None = None,
    activation_granularity: ActivationGranularity = "dynamic-token",
) -> ActivationInjectionReport:
    """Register forward pre-hooks that fake-quantize target layer input activations."""

    if num_bits >= 16:
        target_layers = len(iter_target_modules(model, target=target))
        return ActivationInjectionReport(
            activation_bits=int(num_bits),
            activation_granularity=activation_granularity,
            target_layers=target_layers,
            injected_layers=0,
            scale_layers=0,
            missing_scale_layers=(),
        )
    if num_bits <= 1:
        raise ValueError(f"activation bits must be at least 2 or 16, got {num_bits}")

    if activation_granularity == "calibrated-tensor" and activation_scales_path is None:
        raise ValueError("--qvla-activation-scales-path is required for calibrated-tensor activation quantization")
    activation_amax = None
    if activation_granularity == "calibrated-tensor" and activation_scales_path is not None:
        metadata = load_activation_amax_metadata(activation_scales_path)
        scale_bits = metadata.get("activation_bits")
        if scale_bits is not None and int(scale_bits) != int(num_bits):
            raise ValueError(
                f"Activation scale file was calibrated for {scale_bits} bits, "
                f"but {num_bits} bits were requested."
            )
        activation_amax = load_activation_amax(activation_scales_path)
    target_modules = iter_target_modules(model, target=target)
    missing_scale_layers: list[str] = []
    handles = []

    for module_name, module in target_modules:
        matched_scale = None if activation_amax is None else find_activation_amax_for_module(module_name, activation_amax)
        if activation_amax is not None and matched_scale is None:
            missing_scale_layers.append(module_name)
        amax = None if matched_scale is None else matched_scale[1]

        def _hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            *,
            activation_amax_ref: torch.Tensor | None = amax,
            activation_bits: int = int(num_bits),
            granularity_ref: ActivationGranularity = activation_granularity,
        ) -> tuple[object, ...]:
            if not inputs:
                return inputs
            first, *rest = inputs
            if torch.is_tensor(first):
                first = fake_quantize_activation_sym(
                    first,
                    activation_bits,
                    amax=activation_amax_ref,
                    granularity=granularity_ref,
                )
            return (first, *rest)

        handles.append(module.register_forward_pre_hook(_hook))

    existing_handles = list(getattr(model, "_qvla_activation_quant_handles", []))
    setattr(model, "_qvla_activation_quant_handles", [*existing_handles, *handles])

    report = ActivationInjectionReport(
        activation_bits=int(num_bits),
        activation_granularity=activation_granularity,
        target_layers=len(target_modules),
        injected_layers=len(handles),
        scale_layers=0 if activation_amax is None else len(activation_amax),
        missing_scale_layers=tuple(missing_scale_layers),
    )
    logger.info("QVLA fake activation injection: %s", report.summary())
    return report


class HessianProxy:
    """Accumulate the Hessian-proxy input covariance for one Linear or Conv2d layer."""

    def __init__(self, layer: nn.Module, device: torch.device | str):
        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            raise TypeError(f"HessianProxy only supports Linear/Conv2d, got {type(layer)!r}")
        self.layer = layer
        self.device = torch.device(device)
        weight = layer.weight.detach()
        if isinstance(layer, nn.Conv2d):
            weight = weight.flatten(1)
        self.rows = int(weight.shape[0])
        self.columns = int(weight.shape[1])
        self.hessian = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        """Add a layer-input batch to the proxy covariance."""

        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batch_size = int(inp.shape[0])

        if isinstance(self.layer, nn.Linear):
            if inp.ndim == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        elif isinstance(self.layer, nn.Conv2d):
            padding = self.layer.padding
            if isinstance(padding, str):
                inp = _pad_conv2d_string_padding(inp, self.layer)
                padding = 0
            inp = F.unfold(
                inp,
                kernel_size=self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=padding,
                stride=self.layer.stride,
            )
            inp = inp.permute([1, 0, 2]).flatten(1)

        inp = inp.to(device=self.device, dtype=torch.float32)
        self.hessian *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = torch.sqrt(torch.tensor(2.0 / self.nsamples, device=self.device)) * inp
        self.hessian += inp.matmul(inp.t())

    @torch.no_grad()
    def diag_hinv(self, percdamp: float = 0.01) -> torch.Tensor:
        """Return the Cholesky diagonal of the damped inverse Hessian proxy."""

        hessian = self.hessian.clone()
        diag = torch.arange(self.columns, device=self.device)
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        damp = percdamp * torch.mean(torch.diag(hessian))
        hessian[diag, diag] += damp
        chol = torch.linalg.cholesky(hessian)
        hessian_inv = torch.cholesky_inverse(chol)
        chol_inv = torch.linalg.cholesky(hessian_inv, upper=True)
        return torch.diag(chol_inv)


def _pad_conv2d_string_padding(inp: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
    """Apply Conv2d string padding before F.unfold, which only accepts ints."""

    if layer.padding == "valid":
        return inp
    if layer.padding != "same":
        raise ValueError(f"Unsupported Conv2d padding string for QVLA proxy: {layer.padding!r}")

    input_h, input_w = int(inp.shape[-2]), int(inp.shape[-1])
    kernel_h, kernel_w = _as_pair(layer.kernel_size)
    stride_h, stride_w = _as_pair(layer.stride)
    dilation_h, dilation_w = _as_pair(layer.dilation)

    out_h = math.ceil(input_h / stride_h)
    out_w = math.ceil(input_w / stride_w)
    pad_h = max((out_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h, 0)
    pad_w = max((out_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(inp, (pad_left, pad_right, pad_top, pad_bottom))


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


@torch.no_grad()
def compute_proxy_for_bits(
    layer: nn.Module,
    diag_hinv: torch.Tensor,
    bits: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Compute QVLA per-output-channel proxy losses for candidate bit widths."""

    weight = layer.weight.detach()
    if isinstance(layer, nn.Conv2d):
        weight = weight.flatten(1)
    weight = weight.float()
    d2 = (diag_hinv.float() ** 2).clamp_min(1e-12).to(weight.device)
    proxies: dict[int, torch.Tensor] = {}
    for bit_width in bits:
        bit_width = int(bit_width)
        if bit_width >= 16:
            proxies[bit_width] = torch.zeros(weight.shape[0], device=weight.device)
            continue
        quantized = torch.stack([fake_quantize_tensor_sym(weight[i], bit_width) for i in range(weight.shape[0])])
        loss = ((weight - quantized.float()) ** 2) / d2.unsqueeze(0)
        proxies[bit_width] = loss.sum(dim=1)
    return proxies


def parse_bits(bits: str) -> list[int]:
    parsed = sorted({int(value) for value in bits.split(",") if value.strip()})
    if not parsed:
        raise ValueError("At least one bit width must be provided.")
    return parsed


def load_proxy_file(proxy_path: str | pathlib.Path, bits: Iterable[int]) -> dict[str, dict[int, torch.Tensor]]:
    """Load a proxy file produced by the QVLA Hessian proxy script."""

    raw = torch.load(pathlib.Path(proxy_path).expanduser(), map_location="cpu")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected proxy file to contain a mapping, got {type(raw)!r}")

    bit_list = [int(bit) for bit in bits]
    proxies: dict[str, dict[int, torch.Tensor]] = {}
    for layer_name, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        layer_proxies: dict[int, torch.Tensor] = {}
        for bit_width in bit_list:
            proxy_key = f"proxy_{bit_width}"
            if proxy_key in value:
                layer_proxies[bit_width] = value[proxy_key].float()
        if layer_proxies:
            proxies[str(layer_name)] = layer_proxies
    return proxies


def _next_lower_bit(bit_width: int, bit_list_desc: list[int]) -> int:
    for candidate in bit_list_desc:
        if candidate < bit_width:
            return candidate
    return -1


def greedy_allocate(
    proxies: Mapping[str, Mapping[int, torch.Tensor]],
    bit_list: Iterable[int],
    target_avg_bits: float,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Greedy global channel-wise bit allocation from per-channel proxy losses."""

    bit_list_desc = sorted({int(bit) for bit in bit_list}, reverse=True)
    if not bit_list_desc:
        raise ValueError("bit_list must not be empty")
    highest_bit = bit_list_desc[0]

    layer_bits: dict[str, list[int]] = {}
    total_bits = 0
    total_channels = 0
    heap: list[tuple[float, int, str, int, int]] = []
    step_id = 0

    def push_candidate(layer_name: str, channel_idx: int, current_bit: int) -> None:
        nonlocal step_id
        next_bit = _next_lower_bit(current_bit, bit_list_desc)
        if next_bit < 0 or next_bit not in proxies[layer_name]:
            return
        saving = current_bit - next_bit
        if saving <= 0:
            return
        current_proxy = (
            float(proxies[layer_name][current_bit][channel_idx]) if current_bit in proxies[layer_name] else 0.0
        )
        next_proxy = float(proxies[layer_name][next_bit][channel_idx])
        cost = max(0.0, next_proxy - current_proxy)
        heapq.heappush(heap, (cost / saving, step_id, layer_name, channel_idx, next_bit))
        step_id += 1

    for layer_name, layer_proxy in proxies.items():
        any_proxy = next(iter(layer_proxy.values()))
        channels = int(any_proxy.numel())
        layer_bits[layer_name] = [highest_bit] * channels
        total_bits += highest_bit * channels
        total_channels += channels
        for channel_idx in range(channels):
            push_candidate(layer_name, channel_idx, highest_bit)

    if total_channels == 0:
        raise ValueError("No channels found in proxies")

    avg_bits = total_bits / total_channels
    while heap and avg_bits > target_avg_bits:
        _, _, layer_name, channel_idx, next_bit = heapq.heappop(heap)
        current_bit = layer_bits[layer_name][channel_idx]
        if next_bit >= current_bit:
            continue
        saving = current_bit - next_bit
        total_bits -= saving
        layer_bits[layer_name][channel_idx] = next_bit
        avg_bits = total_bits / total_channels
        push_candidate(layer_name, channel_idx, next_bit)

    histogram: dict[int, int] = {}
    for bits in layer_bits.values():
        for bit_width in bits:
            histogram[bit_width] = histogram.get(bit_width, 0) + 1

    stats = {
        "target_avg_bits": float(target_avg_bits),
        "final_avg_bits": float(avg_bits),
        "total_channels": int(total_channels),
        "initial_total_bits": int(highest_bit * total_channels),
        "final_total_bits": int(total_bits),
        "bit_hist": {int(key): int(value) for key, value in sorted(histogram.items())},
    }
    return layer_bits, stats


def save_gate_assignment(
    out_path: str | pathlib.Path,
    *,
    proxy_path: str | pathlib.Path,
    bits: Iterable[int],
    layer_bits: Mapping[str, list[int]],
    stats: Mapping[str, Any],
) -> None:
    """Save gate assignment JSON in a format understood by ``load_gate_assignments``."""

    out_path = pathlib.Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "proxy_pt": str(proxy_path),
        "bits": [int(bit) for bit in bits],
        "assign": layer_bits,
        "stats": dict(stats),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_activation_amax(
    out_path: str | pathlib.Path,
    *,
    target: TargetPreset,
    samples: int,
    activation_amax: Mapping[str, float],
    nsamples: Mapping[str, int] | None = None,
    activation_bits: int | None = None,
    calibration_method: str = "max_abs",
    hist_bins: int | None = None,
    mse_grid_size: int | None = None,
) -> None:
    """Save calibrated max-absolute activation values for target modules."""

    out_path = pathlib.Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "stat": "calibrated_max_abs",
        "calibration_method": calibration_method,
        "samples": int(samples),
        "activations": {
            str(name): {
                "amax": float(value),
                "nsamples": int(nsamples.get(name, 0)) if nsamples is not None else 0,
            }
            for name, value in sorted(activation_amax.items())
        },
    }
    if activation_bits is not None:
        payload["activation_bits"] = int(activation_bits)
    if hist_bins is not None:
        payload["hist_bins"] = int(hist_bins)
    if mse_grid_size is not None:
        payload["mse_grid_size"] = int(mse_grid_size)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

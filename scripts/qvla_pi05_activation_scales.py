"""Calibrate QVLA activation fake-quant scales for openpi pi05 LIBERO policies."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import urllib.parse

import numpy as np
import torch

# The activation calibration workflow is PyTorch-only. Keep JAX on CPU to avoid
# initializing a second CUDA stack before the openpi config/policy imports.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import qvla_pi05_hessian_proxy as _proxy_helpers

from openpi.policies import policy_config
from openpi.quantization import qvla
from openpi.training import config as _config

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


def _resolve_local_or_uri(path: str) -> str:
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme:
        return path
    return str(pathlib.Path(path).expanduser())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_libero", help="openpi training config name.")
    parser.add_argument(
        "--checkpoint-dir",
        default="~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
        help="Converted openpi PyTorch checkpoint directory containing model.safetensors.",
    )
    parser.add_argument("--calib-jsonl", help="LIBERO calibration JSONL.")
    parser.add_argument("--image-root", help="Base directory for relative image paths in calibration JSONL.")
    parser.add_argument("--out-path", required=True, help="Output activation scale JSON path.")
    parser.add_argument(
        "--target",
        default="pi05_vlm_backbones",
        choices=["pi05_backbones", "pi05_vlm_backbones", "pi05_action_expert", "all_linear_conv"],
    )
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--fake-calib-samples", type=int, default=0, help="Use random LIBERO examples for smoke tests.")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow denoising steps during calibration inference.")
    parser.add_argument("--activation-bits", type=int, default=8, help="Activation bit width this scale file is for.")
    parser.add_argument(
        "--calibration-method",
        default="mse",
        choices=["mse", "max"],
        help="Static activation clipping calibration. 'mse' minimizes histogram reconstruction error.",
    )
    parser.add_argument("--hist-bins", type=int, default=2048, help="Histogram bins for MSE activation calibration.")
    parser.add_argument("--mse-grid-size", type=int, default=256, help="Candidate clipping thresholds for MSE search.")
    parser.add_argument(
        "--pytorch-compile-mode",
        default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode for calibration inference. Defaults to disabled because calibration hooks are compile-hostile.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default-prompt", default="do something")
    parser.add_argument("--state-dim", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args.activation_bits <= 1 and args.activation_bits != 16:
        raise ValueError("--activation-bits must be at least 2, or 16 for no activation fake quantization")
    if args.hist_bins <= 0:
        raise ValueError("--hist-bins must be positive")
    if args.mse_grid_size <= 0:
        raise ValueError("--mse-grid-size must be positive")

    train_config = _config.get_config(args.config_name)
    compile_mode = None if args.pytorch_compile_mode == "none" else args.pytorch_compile_mode
    if hasattr(train_config.model, "pytorch_compile_mode"):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=compile_mode),
        )
    checkpoint_dir = _resolve_local_or_uri(args.checkpoint_dir)

    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )
    if not getattr(policy, "_is_pytorch_model", False):
        raise ValueError("qvla_pi05_activation_scales.py requires a converted PyTorch checkpoint with model.safetensors.")

    model = policy._model  # noqa: SLF001
    target_modules = qvla.iter_target_modules(model, target=args.target)
    if args.max_layers is not None:
        target_modules = target_modules[: args.max_layers]
    if not target_modules:
        raise RuntimeError(f"No target modules found for target preset {args.target!r}")

    if args.fake_calib_samples > 0:
        calib_examples = _proxy_helpers._make_fake_calib_samples(args.fake_calib_samples)  # noqa: SLF001
    elif args.calib_jsonl:
        calib_examples = _proxy_helpers._load_calib_jsonl(  # noqa: SLF001
            args.calib_jsonl,
            image_root=args.image_root,
            max_samples=args.max_samples,
            default_prompt=args.default_prompt,
            state_dim=args.state_dim,
        )
    else:
        raise ValueError("Provide --calib-jsonl, or use --fake-calib-samples for a smoke test.")

    logging.info(
        (
            "Calibrating QVLA activation scales: config=%s checkpoint=%s target=%s "
            "layers=%s samples=%s activation_bits=%s method=%s"
        ),
        args.config_name,
        checkpoint_dir,
        args.target,
        len(target_modules),
        len(calib_examples),
        args.activation_bits,
        args.calibration_method,
    )

    activation_amax = {name: 0.0 for name, _ in target_modules}
    nsamples = {name: 0 for name, _ in target_modules}
    nvalues = {name: 0 for name, _ in target_modules}

    rng = np.random.default_rng(args.seed)
    action_shape = (train_config.model.action_horizon, train_config.model.action_dim)
    # Reuse the exact same denoising noise in the max and histogram passes so
    # the histogram range matches the observed activation maxima.
    calibration_noises = [rng.standard_normal(action_shape).astype(np.float32) for _ in calib_examples]

    def run_calibration_pass(desc: str) -> None:
        with torch.no_grad():
            for example, noise in tqdm(
                zip(calib_examples, calibration_noises, strict=True),
                desc=desc,
                dynamic_ncols=True,
                total=len(calib_examples),
            ):
                policy.infer(example, noise=noise)

    handles = []
    for layer_name, module in target_modules:

        def _hook(
            _module: torch.nn.Module,
            inputs: tuple[object, ...],
            *,
            name: str = layer_name,
        ) -> None:
            if not inputs:
                return
            first = inputs[0]
            if not torch.is_tensor(first) or not torch.is_floating_point(first):
                return
            value = float(first.detach().float().abs().max().cpu().item())
            if value > activation_amax[name]:
                activation_amax[name] = value
            nsamples[name] += 1
            nvalues[name] += int(first.numel())

        handles.append(module.register_forward_pre_hook(_hook))

    try:
        run_calibration_pass("[qvla-act-calib] max")
    finally:
        for handle in handles:
            handle.remove()

    if args.calibration_method == "mse":
        activation_histograms = {
            name: torch.zeros(args.hist_bins, dtype=torch.float64) for name, _ in target_modules
        }

        handles = []
        for layer_name, module in target_modules:

            def _hook(
                _module: torch.nn.Module,
                inputs: tuple[object, ...],
                *,
                name: str = layer_name,
            ) -> None:
                if not inputs:
                    return
                first = inputs[0]
                if not torch.is_tensor(first) or not torch.is_floating_point(first):
                    return
                max_value = float(activation_amax[name])
                if max_value <= 0.0:
                    return
                values = first.detach().float().abs()
                hist = torch.histc(values, bins=args.hist_bins, min=0.0, max=max_value)
                activation_histograms[name] += hist.to(dtype=torch.float64, device="cpu")

            handles.append(module.register_forward_pre_hook(_hook))

        try:
            run_calibration_pass("[qvla-act-calib] hist")
        finally:
            for handle in handles:
                handle.remove()

        activation_amax = {
            name: qvla.estimate_activation_amax_from_histogram(
                activation_histograms[name],
                max_value,
                args.activation_bits,
                grid_size=args.mse_grid_size,
            )
            for name, max_value in activation_amax.items()
        }

    nsamples = {
        name: value
        for name, value in nsamples.items()
    }

    out_path = pathlib.Path(args.out_path).expanduser()
    qvla.save_activation_amax(
        out_path,
        target=args.target,
        samples=len(calib_examples),
        activation_amax=activation_amax,
        nsamples=nsamples,
        activation_bits=args.activation_bits,
        calibration_method=args.calibration_method,
        hist_bins=args.hist_bins if args.calibration_method == "mse" else None,
        mse_grid_size=args.mse_grid_size if args.calibration_method == "mse" else None,
    )
    logging.info(
        "Saved QVLA activation scales for %s layers to %s (values_seen=%s)",
        len(activation_amax),
        out_path,
        sum(nvalues.values()),
    )


if __name__ == "__main__":
    main()

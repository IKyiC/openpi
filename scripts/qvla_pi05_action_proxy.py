"""Build experimental action-space Taylor proxy sensitivities for pi05 LIBERO.

This script is not the official QVLA public-code path. The official QVLA
repository builds gates with ``sensitivity_hessian_proxy.py``. Keep this script
only for exploratory ablations.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import json
import logging
import os
import pathlib
import urllib.parse

import jax
import numpy as np
import torch
from torch import nn
from torch.nn import grad as nn_grad

# The proxy workflow is PyTorch-only. Keep JAX on CPU to avoid initializing a
# second CUDA stack before the openpi config/policy imports.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.quantization import qvla
from openpi.training import config as _config
from openpi import transforms as _transforms

import qvla_pi05_hessian_proxy as _calib_helpers

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


def _save_proxy(out_path: pathlib.Path, proxy: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(proxy), out_path)
    logging.info("Saved action proxy layers=%s to %s", len(proxy), out_path)


def _prepare_observation(policy, example: dict, device: str) -> _model.Observation:
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    return _model.Observation.from_dict(inputs)


def _sample_actions_with_grad(model, device: str, observation: _model.Observation, *, noise: torch.Tensor, num_steps: int):
    """Differentiable copy of PI0Pytorch.sample_actions.

    The upstream method is decorated with ``torch.no_grad`` for serving. QVLA's
    action-space proxy needs gradients from the final sampled action back to the
    target layer outputs, so the sampling loop is reproduced here without that
    decorator.
    """

    bsize = observation.state.shape[0]
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks) if hasattr(model, "make_att_2d_masks") else None
    if prefix_att_2d_masks is None:
        from openpi.models_pytorch import pi0_pytorch

        prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = model.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )
        x_t = x_t + dt * v_t
        time += dt
    return x_t


def _find_action_output_scale(policy, action_dim: int, device: str) -> torch.Tensor:
    """Return action unnormalization scale for action-space sensitivity."""

    output_transform = getattr(policy, "_output_transform", None)
    transforms = getattr(output_transform, "transforms", ())
    for transform in transforms:
        if isinstance(transform, _transforms.Unnormalize) and transform.norm_stats is not None:
            norm_stats = transform.norm_stats
            if isinstance(norm_stats, Mapping) and "actions" in norm_stats:
                std = np.asarray(norm_stats["actions"].std, dtype=np.float32)
                return torch.as_tensor(std[:action_dim], dtype=torch.float32, device=device)
    return torch.ones((action_dim,), dtype=torch.float32, device=device)


@torch.no_grad()
def _quantized_weight_delta(weight: torch.Tensor, bit_width: int) -> torch.Tensor:
    if bit_width >= 16:
        return torch.zeros_like(weight, dtype=torch.float32)
    quantized = torch.stack([qvla.fake_quantize_tensor_sym(weight[i].float(), bit_width) for i in range(weight.shape[0])])
    return quantized.float() - weight.float()


def _linear_channel_contrib(
    module: nn.Linear,
    inp: torch.Tensor,
    grad_out: torch.Tensor,
    bits: list[int],
) -> dict[int, torch.Tensor]:
    x = inp.float()
    g = grad_out.float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if g.ndim == 2:
        g = g.unsqueeze(0)
    x = x.reshape(-1, x.shape[-1])
    g = g.reshape(-1, g.shape[-1])
    cross = x.t().matmul(g).t()
    weight = module.weight.detach().float()
    return {
        bit: (_quantized_weight_delta(weight, bit).to(cross.device) * cross).sum(dim=1)
        for bit in bits
    }


def _conv_channel_contrib(
    module: nn.Conv2d,
    inp: torch.Tensor,
    grad_out: torch.Tensor,
    bits: list[int],
) -> dict[int, torch.Tensor]:
    x = inp.float()
    g = grad_out.float()
    padding = module.padding
    if isinstance(padding, str):
        x = qvla._pad_conv2d_string_padding(x, module)  # noqa: SLF001
        padding = 0
    grad_weight = nn_grad.conv2d_weight(
        x,
        module.weight.shape,
        g,
        stride=module.stride,
        padding=padding,
        dilation=module.dilation,
        groups=module.groups,
    ).flatten(1)
    weight = module.weight.detach().flatten(1).float()
    return {
        bit: (_quantized_weight_delta(weight, bit).to(grad_weight.device) * grad_weight).sum(dim=1)
        for bit in bits
    }


def _record_contribs(
    module: nn.Module,
    records: list[tuple[torch.Tensor, torch.Tensor]],
    grads: tuple[torch.Tensor | None, ...],
    bits: list[int],
) -> dict[int, torch.Tensor]:
    device = next(module.parameters()).device
    channels = int(module.weight.shape[0])
    total = {bit: torch.zeros(channels, dtype=torch.float32, device=device) for bit in bits}
    for (inp, _out), grad_out in zip(records, grads, strict=True):
        if grad_out is None:
            continue
        if isinstance(module, nn.Linear):
            contrib = _linear_channel_contrib(module, inp, grad_out, bits)
        elif isinstance(module, nn.Conv2d):
            contrib = _conv_channel_contrib(module, inp, grad_out, bits)
        else:
            raise TypeError(f"Unsupported module type for action proxy: {type(module)!r}")
        for bit in bits:
            total[bit] += contrib[bit]
    return total


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
    parser.add_argument("--out-path", required=True, help="Output action-space proxy .pt path.")
    parser.add_argument("--bits", default="0,2,4,8,16", help="Candidate bit widths.")
    parser.add_argument(
        "--target",
        default="pi05_vlm_backbones",
        choices=["pi05_backbones", "pi05_vlm_backbones", "pi05_action_expert", "all_linear_conv"],
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--fake-calib-samples", type=int, default=0, help="Use random LIBERO examples for smoke tests.")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--num-layer-shards", type=int, default=1, help="Split target layers across this many jobs.")
    parser.add_argument("--layer-shard-index", type=int, default=0, help="Layer shard index for this job.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow denoising steps during proxy inference.")
    parser.add_argument("--num-probes", type=int, default=1, help="Hutchinson probes per calibration sample.")
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimensions used for LIBERO action-space proxy.")
    parser.add_argument(
        "--pytorch-compile-mode",
        default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode. Defaults to disabled because action proxy hooks need autograd graphs.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default-prompt", default="do something")
    parser.add_argument("--state-dim", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")
    if args.action_dim <= 0:
        raise ValueError("--action-dim must be positive")
    if args.num_layer_shards <= 0:
        raise ValueError("--num-layer-shards must be positive")
    if args.layer_shard_index < 0 or args.layer_shard_index >= args.num_layer_shards:
        raise ValueError("--layer-shard-index must be in [0, num_layer_shards)")

    bits = qvla.parse_bits(args.bits)
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
        raise ValueError("qvla_pi05_action_proxy.py requires a converted PyTorch checkpoint with model.safetensors.")

    model = policy._model  # noqa: SLF001
    target_modules = qvla.iter_target_modules(model, target=args.target)
    if args.max_layers is not None:
        target_modules = target_modules[: args.max_layers]
    if not target_modules:
        raise RuntimeError(f"No target modules found for target preset {args.target!r}")
    total_target_layers = len(target_modules)
    if args.num_layer_shards > 1:
        target_modules = [
            item
            for layer_index, item in enumerate(target_modules)
            if layer_index % args.num_layer_shards == args.layer_shard_index
        ]
    if not target_modules:
        raise RuntimeError(
            f"No target modules assigned to shard {args.layer_shard_index}/{args.num_layer_shards}."
        )

    if args.fake_calib_samples > 0:
        calib_examples = _calib_helpers._make_fake_calib_samples(args.fake_calib_samples)  # noqa: SLF001
    elif args.calib_jsonl:
        calib_examples = _calib_helpers._load_calib_jsonl(  # noqa: SLF001
            args.calib_jsonl,
            image_root=args.image_root,
            max_samples=args.max_samples,
            default_prompt=args.default_prompt,
            state_dim=args.state_dim,
        )
    else:
        raise ValueError("Provide --calib-jsonl, or use --fake-calib-samples for a smoke test.")

    action_scale = _find_action_output_scale(policy, args.action_dim, args.device)
    logging.info(
        (
            "Building QVLA action proxy: config=%s checkpoint=%s target=%s layers=%s/%s "
            "shard=%s/%s samples=%s probes=%s action_dim=%s bits=%s"
        ),
        args.config_name,
        checkpoint_dir,
        args.target,
        len(target_modules),
        total_target_layers,
        args.layer_shard_index,
        args.num_layer_shards,
        len(calib_examples),
        args.num_probes,
        args.action_dim,
        bits,
    )

    out_path = pathlib.Path(args.out_path).expanduser()
    proxy_out: dict[str, dict[str, torch.Tensor]] = {}
    if args.resume and out_path.exists():
        loaded = torch.load(out_path, map_location="cpu")
        if isinstance(loaded, Mapping):
            proxy_out = dict(loaded)
            logging.info("Resuming from %s with %s completed layers", out_path, len(proxy_out))

    rng = np.random.default_rng(args.seed)
    calibration_noises = [
        rng.standard_normal((train_config.model.action_horizon, train_config.model.action_dim)).astype(np.float32)
        for _ in calib_examples
    ]
    probe_generator = torch.Generator(device=args.device)
    probe_generator.manual_seed(int(args.seed))

    processed_since_save = 0
    for layer_name, module in tqdm(target_modules, desc="[qvla-action-proxy] layers", dynamic_ncols=True):
        if layer_name in proxy_out:
            continue

        layer_device = next(module.parameters()).device
        layer_scores = {
            bit: torch.zeros(int(module.weight.shape[0]), dtype=torch.float64, device="cpu")
            for bit in bits
        }
        for example, noise_np in tqdm(
            zip(calib_examples, calibration_noises, strict=True),
            desc=f"[{layer_name}] samples",
            leave=False,
            dynamic_ncols=True,
            total=len(calib_examples),
        ):
            observation = _prepare_observation(policy, example, args.device)
            noise = torch.from_numpy(noise_np).to(args.device)[None, ...]
            for _ in range(args.num_probes):
                records: list[tuple[torch.Tensor, torch.Tensor]] = []

                def _hook(
                    _module: nn.Module,
                    inputs: tuple[object, ...],
                    output: object,
                    records_ref: list[tuple[torch.Tensor, torch.Tensor]] = records,
                ) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                        return
                    records_ref.append((inputs[0].detach(), output))

                handle = module.register_forward_hook(_hook)
                try:
                    with torch.enable_grad():
                        actions = _sample_actions_with_grad(
                            model,
                            args.device,
                            observation,
                            noise=noise,
                            num_steps=args.num_steps,
                        )
                        actions = actions[..., : args.action_dim].float() * action_scale
                        probe = torch.randint(
                            low=0,
                            high=2,
                            size=actions.shape,
                            generator=probe_generator,
                            device=actions.device,
                        ).to(dtype=actions.dtype)
                        probe = probe.mul_(2.0).sub_(1.0)
                        scalar = (actions * probe).sum()
                        outputs = [out for _, out in records]
                        if not outputs:
                            logging.warning("No outputs captured for layer %s on one sample/probe; skipping", layer_name)
                            continue
                        grads = torch.autograd.grad(scalar, outputs, allow_unused=True)
                        contribs = _record_contribs(module, records, grads, bits)
                finally:
                    handle.remove()

                for bit in bits:
                    layer_scores[bit] += (contribs[bit].detach().float().cpu().double() ** 2)
                del actions, scalar, records, grads, contribs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del observation, noise

        denom = float(len(calib_examples) * args.num_probes)
        proxy_out[layer_name] = {
            f"proxy_{bit}": (layer_scores[bit] / denom).to(dtype=torch.float32)
            for bit in bits
        }
        processed_since_save += 1
        if args.save_every > 0 and processed_since_save >= args.save_every:
            _save_proxy(out_path, proxy_out)
            processed_since_save = 0

    _save_proxy(out_path, proxy_out)


if __name__ == "__main__":
    main()

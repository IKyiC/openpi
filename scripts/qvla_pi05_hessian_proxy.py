"""Build QVLA Hessian-proxy sensitivities for openpi pi05 LIBERO PyTorch policies."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import logging
import pathlib
import urllib.parse

import numpy as np
import torch

from openpi.policies import policy_config
from openpi.quantization import qvla
from openpi.training import config as _config

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


_MISSING = object()


def _resolve_local_or_uri(path: str) -> str:
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme:
        return path
    return str(pathlib.Path(path).expanduser())


def _lookup(data: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in data:
            return data[key]

        cursor: object = data
        found = True
        for part in key.split("/"):
            if isinstance(cursor, Mapping) and part in cursor:
                cursor = cursor[part]
            else:
                found = False
                break
        if found:
            return cursor
    return _MISSING


def _load_image(value: object, image_root: pathlib.Path) -> np.ndarray:
    if isinstance(value, str):
        from PIL import Image

        path = pathlib.Path(value).expanduser()
        if not path.is_absolute():
            path = image_root / path
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _load_calib_jsonl(
    calib_jsonl: str,
    *,
    image_root: str | None,
    max_samples: int,
    default_prompt: str,
    state_dim: int,
) -> list[dict]:
    calib_path = pathlib.Path(calib_jsonl).expanduser()
    root = pathlib.Path(image_root).expanduser() if image_root is not None else calib_path.parent
    examples: list[dict] = []

    with calib_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, Mapping):
                    continue

                base_value = _lookup(item, "observation/image", "image", "base_image")
                if base_value is _MISSING:
                    raise ValueError("missing observation/image")
                base_image = _load_image(base_value, root)

                wrist_value = _lookup(item, "observation/wrist_image", "wrist_image")
                wrist_image = np.zeros_like(base_image) if wrist_value is _MISSING else _load_image(wrist_value, root)

                state_value = _lookup(item, "observation/state", "state")
                state = (
                    np.zeros((state_dim,), dtype=np.float32)
                    if state_value is _MISSING
                    else np.asarray(state_value, dtype=np.float32)
                )

                prompt_value = _lookup(item, "prompt", "text", "language_instruction")
                prompt = default_prompt if prompt_value is _MISSING else str(prompt_value)

                examples.append(
                    {
                        "observation/image": base_image,
                        "observation/wrist_image": wrist_image,
                        "observation/state": state,
                        "prompt": prompt,
                    }
                )
            except Exception as exc:
                logging.warning("Skipping invalid calibration line %s: %s", line_number, exc)

            if len(examples) >= max_samples:
                break

    if not examples:
        raise RuntimeError(f"No calibration examples were loaded from {calib_jsonl}")
    return examples


def _make_fake_calib_samples(count: int) -> list[dict]:
    from openpi.policies import libero_policy

    return [libero_policy.make_libero_example() for _ in range(count)]


def _save_proxy(out_path: pathlib.Path, proxy: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(proxy), out_path)
    logging.info("Saved proxy layers=%s to %s", len(proxy), out_path)


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
    parser.add_argument("--out-path", required=True, help="Output proxy .pt path.")
    parser.add_argument("--bits", default="0,2,4,8,16", help="Candidate bit widths.")
    parser.add_argument("--target", default="pi05_backbones", choices=["pi05_backbones", "all_linear_conv"])
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--fake-calib-samples", type=int, default=0, help="Use random LIBERO examples for smoke tests.")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow denoising steps during calibration inference.")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default-prompt", default="do something")
    parser.add_argument("--state-dim", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")

    bits = qvla.parse_bits(args.bits)
    train_config = _config.get_config(args.config_name)
    checkpoint_dir = _resolve_local_or_uri(args.checkpoint_dir)

    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )
    if not getattr(policy, "_is_pytorch_model", False):
        raise ValueError("qvla_pi05_hessian_proxy.py requires a converted PyTorch checkpoint with model.safetensors.")

    model = policy._model  # noqa: SLF001
    target_modules = qvla.iter_target_modules(model, target=args.target)
    if args.max_layers is not None:
        target_modules = target_modules[: args.max_layers]
    if not target_modules:
        raise RuntimeError(f"No target modules found for target preset {args.target!r}")

    if args.fake_calib_samples > 0:
        calib_examples = _make_fake_calib_samples(args.fake_calib_samples)
    elif args.calib_jsonl:
        calib_examples = _load_calib_jsonl(
            args.calib_jsonl,
            image_root=args.image_root,
            max_samples=args.max_samples,
            default_prompt=args.default_prompt,
            state_dim=args.state_dim,
        )
    else:
        raise ValueError("Provide --calib-jsonl, or use --fake-calib-samples for a smoke test.")

    logging.info(
        "Building QVLA proxy: config=%s checkpoint=%s target=%s layers=%s samples=%s bits=%s",
        args.config_name,
        checkpoint_dir,
        args.target,
        len(target_modules),
        len(calib_examples),
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
    processed_since_save = 0
    action_shape = (train_config.model.action_horizon, train_config.model.action_dim)

    for layer_name, module in tqdm(target_modules, desc="[qvla-proxy] layers", dynamic_ncols=True):
        if layer_name in proxy_out:
            continue

        layer_device = next(module.parameters()).device
        proxy = qvla.HessianProxy(module, device=layer_device)

        def _hook(
            _module: torch.nn.Module,
            inputs: tuple[object, ...],
            _output: object,
            proxy_ref: qvla.HessianProxy = proxy,
        ) -> None:
            inp = inputs[0] if isinstance(inputs, tuple) and inputs else inputs
            if torch.is_tensor(inp):
                proxy_ref.add_batch(inp.detach())

        handle = module.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                for example in calib_examples:
                    noise = rng.standard_normal(action_shape).astype(np.float32)
                    policy.infer(example, noise=noise)
        finally:
            handle.remove()

        if proxy.nsamples == 0:
            logging.warning("No samples captured for layer %s; skipping", layer_name)
            continue

        diag_hinv = proxy.diag_hinv(percdamp=args.percdamp)
        layer_proxy = qvla.compute_proxy_for_bits(module, diag_hinv, bits)
        proxy_out[layer_name] = {
            f"proxy_{bit_width}": value.detach().cpu() for bit_width, value in layer_proxy.items()
        }

        processed_since_save += 1
        if args.save_every > 0 and processed_since_save >= args.save_every:
            _save_proxy(out_path, proxy_out)
            processed_since_save = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save_proxy(out_path, proxy_out)


if __name__ == "__main__":
    main()

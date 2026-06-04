"""Merge QVLA proxy shard files produced by qvla_pi05_hessian_proxy.py."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", required=True, help="Merged proxy .pt path.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow later shards to overwrite duplicate layers.",
    )
    parser.add_argument("proxy_shards", nargs="+", help="Proxy shard .pt files.")
    args = parser.parse_args()

    merged: dict[str, object] = {}
    for shard_path_str in args.proxy_shards:
        shard_path = pathlib.Path(shard_path_str).expanduser()
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, Mapping):
            raise ValueError(f"Expected {shard_path} to contain a mapping, got {type(shard)!r}")

        for layer_name, layer_proxy in shard.items():
            layer_name = str(layer_name)
            if layer_name in merged and not args.allow_overwrite:
                raise ValueError(f"Duplicate layer {layer_name!r} found in {shard_path}")
            merged[layer_name] = layer_proxy

    out_path = pathlib.Path(args.out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    print(f"Merged {len(args.proxy_shards)} shards with {len(merged)} layers -> {out_path}")


if __name__ == "__main__":
    main()

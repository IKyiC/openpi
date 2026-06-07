"""Assign QVLA channel-wise bit gates from Hessian proxy values."""

from __future__ import annotations

import argparse
import json

from openpi.quantization import qvla


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy bit allocation from QVLA proxy values.")
    parser.add_argument("--proxy-pt", required=True, help="Proxy .pt produced by qvla_pi05_hessian_proxy.py.")
    parser.add_argument(
        "--bits",
        default="0,2,4,8,16",
        help="Available bit widths, comma-separated. 16 means no fake quantization.",
    )
    parser.add_argument(
        "--target-filter",
        default="pi05_vlm_backbones",
        choices=["all_linear_conv", "pi05_backbones", "pi05_vlm_backbones", "pi05_action_expert"],
        help="Filter proxy layers before assigning gates. pi05_vlm_backbones matches the official QVLA target pattern.",
    )
    parser.add_argument("--target-avg-bits", type=float, default=8.0, help="Target global average bit width.")
    parser.add_argument("--out-json", required=True, help="Output gate assignment JSON.")
    args = parser.parse_args()

    bits = qvla.parse_bits(args.bits)
    proxies = qvla.load_proxy_file(args.proxy_pt, bits)
    proxies = qvla.filter_proxy_layers(proxies, args.target_filter)
    if not proxies:
        raise ValueError(f"No proxy values for bits {bits} and target {args.target_filter!r} were found in {args.proxy_pt}")

    layer_bits, stats = qvla.greedy_allocate(proxies, bits, args.target_avg_bits)
    qvla.save_gate_assignment(
        args.out_json,
        proxy_path=args.proxy_pt,
        bits=bits,
        layer_bits=layer_bits,
        stats=stats,
    )

    print(f"Saved QVLA gates to {args.out_json}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

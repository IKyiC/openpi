"""Backward-compatible wrapper for the fixed LIBERO baseline split."""

import logging

import tyro

try:
    from examples.libero import fixed_libero_baseline_eval as _fixed_baseline
except ImportError:
    import fixed_libero_baseline_eval as _fixed_baseline

Args = _fixed_baseline.Args


def run_baseline(args: Args) -> None:
    _fixed_baseline.run_baseline(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(run_baseline)

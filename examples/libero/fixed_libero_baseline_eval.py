"""Run the fixed LIBERO split used for all quantization baseline experiments."""

import dataclasses
import json
import logging
import pathlib
from typing import Optional

import tyro

try:
    from examples.libero import main as _libero_main
except ImportError:
    import main as _libero_main

BASELINE_TASK_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
BASELINE_NUM_TASKS = 10
BASELINE_NUM_TRIALS_PER_TASK = 20


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10
    seed: int = 7

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/fixed_baseline_videos"
    summary_out_path: Optional[str] = None


def run_baseline(args: Args) -> None:
    """Evaluate four LIBERO suites with the fixed baseline sample selection."""

    results = []
    logging.info(
        "Running fixed LIBERO baseline: suites=%s, first_tasks=%s, trials_per_task=%s",
        ",".join(BASELINE_TASK_SUITES),
        BASELINE_NUM_TASKS,
        BASELINE_NUM_TRIALS_PER_TASK,
    )
    for task_suite_name in BASELINE_TASK_SUITES:
        logging.info("Starting suite: %s", task_suite_name)
        suite_args = _libero_main.Args(
            host=args.host,
            port=args.port,
            resize_size=args.resize_size,
            replan_steps=args.replan_steps,
            task_suite_name=task_suite_name,
            num_tasks=BASELINE_NUM_TASKS,
            num_steps_wait=args.num_steps_wait,
            num_trials_per_task=BASELINE_NUM_TRIALS_PER_TASK,
            video_out_path=str(pathlib.Path(args.video_out_path) / task_suite_name),
            seed=args.seed,
        )
        results.append(_libero_main.eval_libero(suite_args))

    average_success_rate = sum(result["success_rate"] for result in results) / len(results)
    summary = {
        "task_suites": list(BASELINE_TASK_SUITES),
        "num_tasks_per_suite": BASELINE_NUM_TASKS,
        "num_trials_per_task": BASELINE_NUM_TRIALS_PER_TASK,
        "total_episodes": sum(result["total_episodes"] for result in results),
        "total_successes": sum(result["total_successes"] for result in results),
        "average_success_rate": average_success_rate,
        "suite_results": results,
    }

    summary_out_path = (
        pathlib.Path(args.summary_out_path)
        if args.summary_out_path is not None
        else pathlib.Path(args.video_out_path) / "summary.json"
    )
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Fixed LIBERO baseline summary:")
    for result in results:
        logging.info(
            "%s: %.2f%% (%s/%s)",
            result["task_suite_name"],
            100.0 * result["success_rate"],
            result["total_successes"],
            result["total_episodes"],
        )
    logging.info("Average: %.2f%%", 100.0 * average_success_rate)
    logging.info("Saved summary to %s", summary_out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(run_baseline)

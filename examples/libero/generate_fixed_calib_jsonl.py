"""Generate calibration JSONL from the fixed LIBERO baseline split."""

import dataclasses
import json
import logging
import math
import pathlib
import sys
from typing import Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _path in (
    _REPO_ROOT / "third_party" / "libero",
    _REPO_ROOT / "packages" / "openpi-client" / "src",
):
    if _path.exists():
        sys.path.insert(0, str(_path))

try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError as exc:
    if exc.name == "libero":
        raise ModuleNotFoundError(
            "Could not import LIBERO. From the openpi repo root, run "
            "`git submodule update --init --recursive third_party/libero`, then "
            "install the LIBERO eval environment from examples/libero/README.md."
        ) from exc
    raise
import numpy as np
from openpi_client import image_tools
from PIL import Image
import tqdm
import tyro

BASELINE_TASK_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
BASELINE_NUM_TASKS = 10
BASELINE_NUM_TRIALS_PER_TASK = 20
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    out_jsonl: str = "out/baselines/libero_fixed_calib/calib.jsonl"
    image_dir: str = "out/baselines/libero_fixed_calib/images"
    resize_size: int = 224
    num_steps_wait: int = 10
    seed: int = 7
    max_examples: Optional[int] = None


def generate_calib_jsonl(args: Args) -> None:
    np.random.seed(args.seed)
    out_jsonl = pathlib.Path(args.out_jsonl)
    image_dir = pathlib.Path(args.image_dir)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    examples_written = 0

    with out_jsonl.open("w", encoding="utf-8") as jsonl:
        for task_suite_name in BASELINE_TASK_SUITES:
            task_suite = benchmark_dict[task_suite_name]()
            num_tasks_to_eval = min(BASELINE_NUM_TASKS, task_suite.n_tasks)
            logging.info("Generating calibration examples for %s", task_suite_name)

            for task_id in tqdm.tqdm(range(num_tasks_to_eval), desc=task_suite_name):
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                if BASELINE_NUM_TRIALS_PER_TASK > len(initial_states):
                    raise ValueError(
                        f"Requested {BASELINE_NUM_TRIALS_PER_TASK} trials for task {task_id}, "
                        f"but only {len(initial_states)} initial states are available."
                    )

                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                try:
                    for episode_idx in range(BASELINE_NUM_TRIALS_PER_TASK):
                        env.reset()
                        obs = env.set_init_state(initial_states[episode_idx])
                        for _ in range(args.num_steps_wait):
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                        base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        base_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(base_img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32)

                        stem = f"{task_suite_name}_task_{task_id:02d}_episode_{episode_idx:02d}"
                        base_path = image_dir / f"{stem}_base.png"
                        wrist_path = image_dir / f"{stem}_wrist.png"
                        Image.fromarray(base_img).save(base_path)
                        Image.fromarray(wrist_img).save(wrist_path)

                        row = {
                            "observation/image": str(base_path.resolve()),
                            "observation/wrist_image": str(wrist_path.resolve()),
                            "observation/state": state.tolist(),
                            "prompt": str(task_description),
                            "task_suite_name": task_suite_name,
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                        }
                        jsonl.write(json.dumps(row) + "\n")
                        examples_written += 1

                        if args.max_examples is not None and examples_written >= args.max_examples:
                            logging.info("Wrote %s calibration examples to %s", examples_written, out_jsonl)
                            return
                finally:
                    env.close()

    logging.info("Wrote %s calibration examples to %s", examples_written, out_jsonl)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_calib_jsonl(tyro.cli(Args))

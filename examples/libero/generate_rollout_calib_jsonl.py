"""Generate calibration JSONL from policy-driven LIBERO rollouts.

Unlike ``generate_fixed_calib_jsonl.py``, which records only the initial
post-wait observation for each fixed episode, this script follows a running
policy server and records the observations that are actually sent to the policy
at replan time. This gives QVLA proxy estimation a calibration distribution
closer to evaluation rollouts.
"""

from __future__ import annotations

import collections
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
            "Could not import LIBERO. From the openpi repo root, install the "
            "LIBERO eval environment from examples/libero/README.md."
        ) from exc
    raise

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
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
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7

    task_suites: str = ",".join(BASELINE_TASK_SUITES)
    num_tasks: int = BASELINE_NUM_TASKS
    num_trials_per_task: int = BASELINE_NUM_TRIALS_PER_TASK
    max_examples: int = 800
    max_examples_per_episode: Optional[int] = None

    out_jsonl: str = "out/baselines/libero_rollout_calib/calib.jsonl"
    image_dir: str = "out/baselines/libero_rollout_calib/images"
    video_out_path: Optional[str] = None


def generate_rollout_calib_jsonl(args: Args) -> None:
    np.random.seed(args.seed)

    out_jsonl = pathlib.Path(args.out_jsonl)
    image_dir = pathlib.Path(args.image_dir)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    if args.video_out_path is not None:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    task_suite_names = tuple(name.strip() for name in args.task_suites.split(",") if name.strip())
    if not task_suite_names:
        raise ValueError("--task-suites must contain at least one suite name")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    benchmark_dict = benchmark.get_benchmark_dict()
    examples_written = 0

    logging.info(
        "Generating rollout calibration: suites=%s num_tasks=%s trials=%s max_examples=%s server=%s:%s",
        ",".join(task_suite_names),
        args.num_tasks,
        args.num_trials_per_task,
        args.max_examples,
        args.host,
        args.port,
    )

    with out_jsonl.open("w", encoding="utf-8") as jsonl:
        for task_suite_name in task_suite_names:
            task_suite = benchmark_dict[task_suite_name]()
            num_tasks_to_eval = min(args.num_tasks, task_suite.n_tasks)
            max_steps = _max_steps_for_suite(task_suite_name)

            for task_id in tqdm.tqdm(range(num_tasks_to_eval), desc=task_suite_name):
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                if args.num_trials_per_task > len(initial_states):
                    raise ValueError(
                        f"Requested {args.num_trials_per_task} trials for task {task_id}, "
                        f"but only {len(initial_states)} initial states are available."
                    )

                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                try:
                    for episode_idx in range(args.num_trials_per_task):
                        env.reset()
                        obs = env.set_init_state(initial_states[episode_idx])
                        action_plan = collections.deque()
                        replay_images = []
                        episode_examples = 0
                        done = False
                        t = 0

                        while t < max_steps + args.num_steps_wait:
                            if t < args.num_steps_wait:
                                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                                t += 1
                                continue

                            if not action_plan:
                                element, base_img, wrist_img = _policy_input_from_obs(
                                    obs,
                                    task_description,
                                    args.resize_size,
                                )
                                replay_images.append(base_img)

                                _write_example(
                                    jsonl=jsonl,
                                    image_dir=image_dir,
                                    example=element,
                                    task_suite_name=task_suite_name,
                                    task_id=task_id,
                                    episode_idx=episode_idx,
                                    timestep=t,
                                    example_idx=examples_written,
                                    base_img=base_img,
                                    wrist_img=wrist_img,
                                )
                                examples_written += 1
                                episode_examples += 1

                                action_chunk = client.infer(element)["actions"]
                                if len(action_chunk) < args.replan_steps:
                                    raise ValueError(
                                        f"Policy returned {len(action_chunk)} actions, "
                                        f"but replan_steps={args.replan_steps}."
                                    )
                                action_plan.extend(action_chunk[: args.replan_steps])

                                if examples_written >= args.max_examples:
                                    _write_video_if_requested(
                                        args.video_out_path,
                                        task_suite_name,
                                        task_id,
                                        episode_idx,
                                        replay_images,
                                        done,
                                    )
                                    logging.info("Wrote %s rollout calibration examples to %s", examples_written, out_jsonl)
                                    return
                                if (
                                    args.max_examples_per_episode is not None
                                    and episode_examples >= args.max_examples_per_episode
                                ):
                                    break

                            action = action_plan.popleft()
                            obs, _, done, _ = env.step(action.tolist())
                            t += 1
                            if done:
                                break

                        _write_video_if_requested(
                            args.video_out_path,
                            task_suite_name,
                            task_id,
                            episode_idx,
                            replay_images,
                            done,
                        )
                finally:
                    env.close()

    logging.info("Wrote %s rollout calibration examples to %s", examples_written, out_jsonl)


def _policy_input_from_obs(obs, task_description: str, resize_size: int):
    base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return (
        {
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": str(task_description),
        },
        base_img,
        wrist_img,
    )


def _write_example(
    *,
    jsonl,
    image_dir: pathlib.Path,
    example: dict,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    timestep: int,
    example_idx: int,
    base_img: np.ndarray,
    wrist_img: np.ndarray,
) -> None:
    stem = f"{example_idx:06d}_{task_suite_name}_task_{task_id:02d}_episode_{episode_idx:02d}_t_{timestep:04d}"
    base_path = image_dir / f"{stem}_base.png"
    wrist_path = image_dir / f"{stem}_wrist.png"
    Image.fromarray(base_img).save(base_path)
    Image.fromarray(wrist_img).save(wrist_path)

    row = {
        "observation/image": str(base_path.resolve()),
        "observation/wrist_image": str(wrist_path.resolve()),
        "observation/state": np.asarray(example["observation/state"], dtype=np.float32).tolist(),
        "prompt": str(example["prompt"]),
        "task_suite_name": task_suite_name,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "timestep": int(timestep),
    }
    jsonl.write(json.dumps(row) + "\n")


def _write_video_if_requested(
    video_out_path: str | None,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    replay_images: list[np.ndarray],
    done: bool,
) -> None:
    if video_out_path is None or not replay_images:
        return
    suffix = "success" if done else "failure"
    video_path = (
        pathlib.Path(video_out_path)
        / task_suite_name
        / f"rollout_task_{task_id:02d}_episode_{episode_idx:02d}_{suffix}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


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
    generate_rollout_calib_jsonl(tyro.cli(Args))

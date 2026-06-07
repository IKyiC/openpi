import dataclasses
import enum
import logging
import socket
from typing import Literal

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.quantization import qvla as _qvla
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

PytorchCompileMode = Literal["none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a
    # default prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Optional QVLA gate assignment file for PyTorch checkpoints.
    qvla_gates_path: str | None = None
    # Target module preset used when applying QVLA gates.
    qvla_target: _qvla.TargetPreset = "pi05_vlm_backbones"
    # How to handle gate length mismatches.
    qvla_mismatch_policy: _qvla.MismatchPolicy = "median"
    # Optional QVLA activation fake-quant bit width, e.g. 4 or 8.
    qvla_activation_bits: int | None = None
    # Optional calibrated activation max-abs scale file.
    qvla_activation_scales_path: str | None = None
    # Activation quantization granularity.
    qvla_activation_granularity: _qvla.ActivationGranularity = "dynamic-token"
    # torch.compile mode for PyTorch policies. Disabled by default for predictable serving startup.
    pytorch_compile_mode: PytorchCompileMode = "none"

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def get_config(config_name: str, pytorch_compile_mode: PytorchCompileMode) -> _config.TrainConfig:
    train_config = _config.get_config(config_name)
    compile_mode = None if pytorch_compile_mode == "none" else pytorch_compile_mode
    if hasattr(train_config.model, "pytorch_compile_mode"):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=compile_mode),
        )
    return train_config


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    qvla_gates_path: str | None = None,
    qvla_target: _qvla.TargetPreset = "pi05_vlm_backbones",
    qvla_mismatch_policy: _qvla.MismatchPolicy = "median",
    qvla_activation_bits: int | None = None,
    qvla_activation_scales_path: str | None = None,
    qvla_activation_granularity: _qvla.ActivationGranularity = "dynamic-token",
    pytorch_compile_mode: PytorchCompileMode = "none",
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            get_config(checkpoint.config, pytorch_compile_mode),
            checkpoint.dir,
            default_prompt=default_prompt,
            qvla_gates_path=qvla_gates_path,
            qvla_target=qvla_target,
            qvla_mismatch_policy=qvla_mismatch_policy,
            qvla_activation_bits=qvla_activation_bits,
            qvla_activation_scales_path=qvla_activation_scales_path,
            qvla_activation_granularity=qvla_activation_granularity,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                get_config(args.policy.config, args.pytorch_compile_mode),
                args.policy.dir,
                default_prompt=args.default_prompt,
                qvla_gates_path=args.qvla_gates_path,
                qvla_target=args.qvla_target,
                qvla_mismatch_policy=args.qvla_mismatch_policy,
                qvla_activation_bits=args.qvla_activation_bits,
                qvla_activation_scales_path=args.qvla_activation_scales_path,
                qvla_activation_granularity=args.qvla_activation_granularity,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                qvla_gates_path=args.qvla_gates_path,
                qvla_target=args.qvla_target,
                qvla_mismatch_policy=args.qvla_mismatch_policy,
                qvla_activation_bits=args.qvla_activation_bits,
                qvla_activation_scales_path=args.qvla_activation_scales_path,
                qvla_activation_granularity=args.qvla_activation_granularity,
                pytorch_compile_mode=args.pytorch_compile_mode,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

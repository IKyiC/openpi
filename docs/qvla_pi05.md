# QVLA for pi05 LIBERO PyTorch

This workflow adapts QVLA-style training-free fake quantization to the openpi
`pi05_libero` PyTorch policy path. Weight quantization uses QVLA channel-wise
gate assignment, while activation quantization uses fixed-bit input activation
fake quantization on the same target modules. The formal calibrated activation
path uses static per-layer scales selected by histogram MSE calibration
(`calibrated-tensor`).

It does not load OpenVLA checkpoints and does not modify the original checkpoint
directory. The model must be an openpi-converted PyTorch checkpoint containing
`model.safetensors`.

For the full 2x2 baseline organization, see `docs/baseline_matrix.md`.

## Target Model

- config: `pi05_libero`
- checkpoint: `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
- loader: `openpi.policies.policy_config.create_trained_policy`

## Calibration JSONL

`--calib-jsonl` points to a small JSONL file used by QVLA proxy estimation. It
is not the full LIBERO dataset. If the full dataset is not available, generate
this file from the fixed LIBERO simulator split:

```bash
git submodule update --init --recursive third_party/libero
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH

python examples/libero/generate_fixed_calib_jsonl.py \
  --out-jsonl out/baselines/libero_fixed_calib/calib.jsonl \
  --image-dir out/baselines/libero_fixed_calib/images
```

This writes 800 calibration entries by default: four suites, first 10 tasks per
suite, first 20 initial states per task. The JSONL and PNGs are written under
`out/baselines/libero_fixed_calib/`. The generated JSONL can be reused by the
QVLA proxy command:

```bash
export CALIB=out/baselines/libero_fixed_calib/calib.jsonl
```

Each JSONL line describes one LIBERO-style policy input. Image paths may be
absolute or relative to the JSONL file directory, or to `--image-root`.

```json
{
  "observation/image": "images/base_000001.png",
  "observation/wrist_image": "images/wrist_000001.png",
  "observation/state": [0, 0, 0, 0, 0, 0, 0, 0],
  "prompt": "pick up the object"
}
```

Accepted aliases:

- base image: `observation/image`, `image`, `base_image`
- wrist image: `observation/wrist_image`, `wrist_image`
- state: `observation/state`, `state`
- prompt: `prompt`, `text`, `language_instruction`

## 1. Build Hessian Proxy

Run this on the Linux server with the converted PyTorch checkpoint available.
This command is the fast input-covariance proxy path. It is useful for smoke
tests and ablations, but the stricter QVLA action-space proxy is described in
the next section.

```bash
JAX_PLATFORMS=cpu uv run scripts/qvla_pi05_hessian_proxy.py \
  --config-name pi05_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --calib-jsonl /path/to/libero_calib.jsonl \
  --out-path out/baselines/qvla/pi05_libero/proxy.pt \
  --bits 0,2,4,8,16 \
  --target pi05_backbones \
  --device cuda:0 \
  --max-samples 800
```

The proxy command disables `torch.compile` by default (`--pytorch-compile-mode
none`) because QVLA forward hooks and per-layer sweeps make PyTorch
`max-autotune` compile overhead dominate runtime.

For a wiring-only smoke test, use `--fake-calib-samples 2` instead of
`--calib-jsonl`, or use a small `--max-samples` value such as `32`. Do not use
fake or small calibration for real gate assignment.

For multi-GPU proxy building, run one shard per GPU. For example, with 4 GPUs:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i JAX_PLATFORMS=cpu uv run scripts/qvla_pi05_hessian_proxy.py \
    --config-name pi05_libero \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    --calib-jsonl /path/to/libero_calib.jsonl \
    --out-path out/baselines/qvla/pi05_libero/proxy_shard_${i}.pt \
    --bits 0,2,4,8,16 \
    --target pi05_backbones \
    --device cuda:0 \
    --max-samples 800 \
    --num-layer-shards 4 \
    --layer-shard-index $i &
done
wait

uv run scripts/qvla_merge_proxy_shards.py \
  --out-path out/baselines/qvla/pi05_libero/proxy.pt \
  out/baselines/qvla/pi05_libero/proxy_shard_*.pt
```

## 1b. Build Action-Space Proxy

The paper-faithful QVLA path estimates sensitivity in action space. For each
target layer, this script measures how quantizing each output channel changes
the final sampled action, using a first-order Taylor/Jacobian proxy with random
action projections. This is much slower than the Hessian/input-covariance proxy,
so use layer sharding across GPUs.

Single-GPU smoke test:

```bash
JAX_PLATFORMS=cpu uv run python scripts/qvla_pi05_action_proxy.py \
  --config-name pi05_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --calib-jsonl out/baselines/libero_fixed_calib/calib.jsonl \
  --out-path out/baselines/qvla/pi05_libero/action_proxy_smoke.pt \
  --bits 0,2,4,8,16 \
  --target pi05_vlm_backbones \
  --device cuda:0 \
  --max-samples 2 \
  --max-layers 1 \
  --num-probes 1
```

Multi-GPU run:

```bash
export CKPT=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
export CALIB=out/baselines/libero_fixed_calib/calib.jsonl
export OUT=out/baselines/qvla/pi05_libero/action_proxy_vlm
mkdir -p $OUT

GPU_IDS=(1 3 4 5)
NGPUS=${#GPU_IDS[@]}
for idx in "${!GPU_IDS[@]}"; do
  gpu=${GPU_IDS[$idx]}
  CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  uv run python scripts/qvla_pi05_action_proxy.py \
    --config-name pi05_libero \
    --checkpoint-dir $CKPT \
    --calib-jsonl $CALIB \
    --out-path $OUT/action_proxy_shard_${idx}.pt \
    --bits 0,2,4,8,16 \
    --target pi05_vlm_backbones \
    --device cuda:0 \
    --max-samples 128 \
    --num-probes 1 \
    --save-every 1 \
    --num-layer-shards $NGPUS \
    --layer-shard-index $idx \
    > $OUT/action_proxy_shard_${idx}.log 2>&1 &
  sleep 20
done
wait

uv run python scripts/qvla_merge_proxy_shards.py \
  --out-path out/baselines/qvla/pi05_libero/action_proxy_vlm.pt \
  $OUT/action_proxy_shard_*.pt
```

After the 128-sample run is validated, increase `--max-samples` to 800 for the
full paper-faithful proxy. The action-space proxy is substantially slower than
the Hessian/input-covariance proxy because it runs differentiable action
sampling and backward passes.

## 2. Assign Weight Gates

```bash
uv run scripts/qvla_assign_gates.py \
  --proxy-pt out/baselines/qvla/pi05_libero/proxy.pt \
  --bits 0,2,4,8,16 \
  --target-avg-bits 8.0 \
  --out-json out/baselines/qvla/pi05_libero/gates_w8.json
```

The output JSON stores channel-wise gates under `assign` and can be passed
directly to policy loading.

For the action-space proxy, use `action_proxy_vlm.pt` instead of `proxy.pt`:

```bash
uv run python scripts/qvla_assign_gates.py \
  --proxy-pt out/baselines/qvla/pi05_libero/action_proxy_vlm.pt \
  --bits 0,2,4,8,16 \
  --target-filter pi05_vlm_backbones \
  --target-avg-bits 8.0 \
  --out-json out/baselines/qvla/pi05_libero/gates_w8_vlm_action.json
```

For a W4A4 run, reuse the same proxy and assign a second gate file:

```bash
uv run scripts/qvla_assign_gates.py \
  --proxy-pt out/baselines/qvla/pi05_libero/proxy.pt \
  --bits 0,2,4,8,16 \
  --target-avg-bits 4.0 \
  --out-json out/baselines/qvla/pi05_libero/gates_w4.json
```

If W4 allocation collapses the pi05 action expert, assign W4 gates only over
the PaliGemma VLM backbone layers. This matches QVLA's OpenVLA target more
closely because the action module is left unquantized:

```bash
uv run scripts/qvla_assign_gates.py \
  --proxy-pt out/baselines/qvla/pi05_libero/proxy.pt \
  --bits 0,2,4,8,16 \
  --target-filter pi05_vlm_backbones \
  --target-avg-bits 4.0 \
  --out-json out/baselines/qvla/pi05_libero/gates_w4_vlm.json
```

## 3. Calibrate Static Activation Scales

The formal W8A8/W4A8 path uses static calibrated activation scales. The
calibration script runs full policy inference on the fixed calibration set,
collects input activation histograms for the quantized layers, and chooses a
per-layer clipping value by minimizing reconstruction MSE.

A8 scales:

```bash
JAX_PLATFORMS=cpu uv run python scripts/qvla_pi05_activation_scales.py \
  --config-name pi05_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --calib-jsonl out/baselines/libero_fixed_calib/calib.jsonl \
  --out-path out/baselines/qvla/pi05_libero/activation_amax_w8_mse.json \
  --target pi05_backbones \
  --device cuda:0 \
  --max-samples 800 \
  --activation-bits 8 \
  --calibration-method mse
```

For A4 ablations, run the same command with `--activation-bits 4` and an
output such as `activation_amax_w4_mse.json`.

## 4. Serve Quantized Policy

W8A8:

```bash
JAX_PLATFORMS=cpu uv run scripts/serve_policy.py \
  --qvla-gates-path out/baselines/qvla/pi05_libero/gates_w8.json \
  --qvla-activation-bits 8 \
  --qvla-activation-granularity calibrated-tensor \
  --qvla-activation-scales-path out/baselines/qvla/pi05_libero/activation_amax_w8_mse.json \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

W4A8:

```bash
JAX_PLATFORMS=cpu uv run scripts/serve_policy.py \
  --qvla-gates-path out/baselines/qvla/pi05_libero/gates_w4_vlm.json \
  --qvla-target pi05_vlm_backbones \
  --qvla-activation-bits 8 \
  --qvla-activation-granularity calibrated-tensor \
  --qvla-activation-scales-path out/baselines/qvla/pi05_libero/activation_amax_w8_mse.json \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

For W4A4, use `--qvla-activation-bits 4` with
`activation_amax_w4_mse.json`.

For dynamic-scale ablations, use:

```bash
--qvla-activation-granularity dynamic-token
```

## 5. Run Fixed LIBERO Baseline Evaluation

The pi05 + QVLA baseline uses the same fixed sample selection as the other
baselines:

- suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- first 10 tasks in each suite
- first 20 initial states per task
- 200 episodes per suite, 800 episodes total

Run the client against the policy server:

```bash
python examples/libero/fixed_libero_baseline_eval.py \
  --host 0.0.0.0 \
  --port 8000
```

For a single-suite debug run with the same first-task/first-state semantics:

```bash
python examples/libero/main.py \
  --task-suite-name libero_spatial \
  --num-tasks 10 \
  --num-trials-per-task 20
```

## Target Preset

The default `pi05_backbones` target applies fake quantization to:

- `paligemma_with_expert.paligemma.model.language_model.*`
- `paligemma_with_expert.paligemma.model.vision_tower.*`
- `paligemma_with_expert.gemma_expert.model.*`

It excludes `multi_modal_projector`, `lm_head`, pi05 AdaRMS condition dense
layers, and the small top-level action projection/time MLP layers. This keeps
the adaptation close to QVLA's backbone quantization intent while including
pi05's action expert transformer attention and MLP blocks.

The `pi05_vlm_backbones` target applies fake quantization only to:

- `paligemma_with_expert.paligemma.model.language_model.*`
- `paligemma_with_expert.paligemma.model.vision_tower.*`

Use this target for conservative W4A8 runs when the global allocator assigns
nearly all `gemma_expert` channels to 0-bit.

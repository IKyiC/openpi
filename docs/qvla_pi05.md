# QVLA for pi05 LIBERO PyTorch

This workflow adapts QVLA-style training-free, weight-only fake quantization to
the openpi `pi05_libero` PyTorch policy path.

It does not load OpenVLA checkpoints and does not modify the original checkpoint
directory. The model must be an openpi-converted PyTorch checkpoint containing
`model.safetensors`.

For the full 2x2 baseline organization, see `docs/baseline_matrix.md`.

## Target Model

- config: `pi05_libero`
- checkpoint: `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
- loader: `openpi.policies.policy_config.create_trained_policy`

## Calibration JSONL

Each line should describe one LIBERO-style policy input. Image paths may be
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

```bash
uv run scripts/qvla_pi05_hessian_proxy.py \
  --config-name pi05_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --calib-jsonl /path/to/libero_calib.jsonl \
  --out-path out/baselines/qvla/pi05_libero/proxy.pt \
  --bits 0,2,4,8,16 \
  --target pi05_backbones \
  --device cuda:0 \
  --max-samples 32
```

For a wiring-only smoke test, use `--fake-calib-samples 2` instead of
`--calib-jsonl`. Do not use fake calibration for real gate assignment.

For multi-GPU proxy building, run one shard per GPU. For example, with 4 GPUs:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i uv run scripts/qvla_pi05_hessian_proxy.py \
    --config-name pi05_libero \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    --calib-jsonl /path/to/libero_calib.jsonl \
    --out-path out/baselines/qvla/pi05_libero/proxy_shard_${i}.pt \
    --bits 0,2,4,8,16 \
    --target pi05_backbones \
    --device cuda:0 \
    --max-samples 32 \
    --num-layer-shards 4 \
    --layer-shard-index $i &
done
wait

uv run scripts/qvla_merge_proxy_shards.py \
  --out-path out/baselines/qvla/pi05_libero/proxy.pt \
  out/baselines/qvla/pi05_libero/proxy_shard_*.pt
```

## 2. Assign Gates

```bash
uv run scripts/qvla_assign_gates.py \
  --proxy-pt out/baselines/qvla/pi05_libero/proxy.pt \
  --bits 0,2,4,8,16 \
  --target-avg-bits 8.0 \
  --out-json out/baselines/qvla/pi05_libero/gates_w8.json
```

The output JSON stores channel-wise gates under `assign` and can be passed
directly to policy loading.

## 3. Serve Quantized Policy

```bash
uv run scripts/serve_policy.py \
  --qvla-gates-path out/baselines/qvla/pi05_libero/gates_w8.json \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

## 4. Run Fixed LIBERO Baseline Evaluation

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

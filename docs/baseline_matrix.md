# Quantization Baseline Matrix

This project is organized around four LIBERO baselines:

| Method | Model | Suggested branch | Code owner repo |
| --- | --- | --- | --- |
| QVLA | pi05 LIBERO PyTorch | `qvla-pi05` | `openpi` |
| QVLA | GROOT 1.5 | `qvla-groot15` | GROOT repo |
| OMEGAQVLA | pi05 LIBERO PyTorch | `omegaqvla-pi05` | `openpi` |
| OMEGAQVLA | GROOT 1.5 | `omegaqvla-groot15` | GROOT repo |

Keep method/model combinations isolated. Do not mix QVLA and OMEGAQVLA
implementation files on the same experiment branch.

## Shared LIBERO Evaluation Split

All four baselines must use the same fixed LIBERO split:

- task suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- tasks: first 10 tasks in each suite
- samples: first 20 initial states per task
- total: 200 episodes per suite, 800 episodes overall

For openpi clients, use:

```bash
python examples/libero/fixed_libero_baseline_eval.py \
  --host 0.0.0.0 \
  --port 8000
```

## Artifact Layout

Use a method/model directory so output names stay comparable across branches:

```text
out/baselines/
  qvla/
    pi05_libero/
      proxy.pt
      gates_w8.json
      eval/
    groot15_libero/
      proxy.pt
      gates_w8.json
      eval/
  omegaqvla/
    pi05_libero/
      proxy.pt
      gates_w8.json
      eval/
    groot15_libero/
      proxy.pt
      gates_w8.json
      eval/
```

This directory should live outside git or remain ignored. Checkpoints should not
be modified in place.

## openpi pi05 Branches

Use `pi05-quant-base` as the common base for pi05 quantization branches.

For QVLA + pi05:

```bash
uv run scripts/qvla_pi05_hessian_proxy.py \
  --config-name pi05_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --calib-jsonl /path/to/libero_calib.jsonl \
  --out-path out/baselines/qvla/pi05_libero/proxy.pt

uv run scripts/qvla_assign_gates.py \
  --proxy-pt out/baselines/qvla/pi05_libero/proxy.pt \
  --target-avg-bits 8.0 \
  --out-json out/baselines/qvla/pi05_libero/gates_w8.json

uv run scripts/serve_policy.py \
  --qvla-gates-path out/baselines/qvla/pi05_libero/gates_w8.json \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

For multi-GPU proxy generation, shard target layers across jobs with
`--num-layer-shards` and `--layer-shard-index`, then merge with
`scripts/qvla_merge_proxy_shards.py`.

OMEGAQVLA + pi05 should get its own branch and scripts/module names, for
example:

```text
src/openpi/quantization/omegaqvla.py
scripts/omegaqvla_pi05_*.py
docs/omegaqvla_pi05.md
```

## Server Worktree Layout

Prefer separate worktrees on the server so four experiments can coexist without
frequent branch switching:

```text
~/code/openpi.git-worktree-root/
~/experiments/baselines/openpi-qvla-pi05/
~/experiments/baselines/openpi-omegaqvla-pi05/
~/experiments/baselines/groot-qvla-groot15/
~/experiments/baselines/groot-omegaqvla-groot15/
```

Create worktrees from an existing clone:

```bash
cd ~/code/openpi
git fetch origin --prune
git worktree add ~/experiments/baselines/openpi-qvla-pi05 qvla-pi05
git worktree add ~/experiments/baselines/openpi-omegaqvla-pi05 omegaqvla-pi05
```

If a branch exists only on the remote:

```bash
git worktree add \
  ~/experiments/baselines/openpi-omegaqvla-pi05 \
  origin/omegaqvla-pi05
```

If you need to create a new method branch from the base branch:

```bash
git switch pi05-quant-base
git pull --ff-only
git switch -c omegaqvla-pi05
git push -u origin omegaqvla-pi05
```

Run `git status --short --branch` before every experiment to confirm that the
worktree is on the intended branch.

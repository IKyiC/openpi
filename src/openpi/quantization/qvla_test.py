import json

import torch
from torch import nn

from openpi.quantization import qvla


class _TinyPi05LikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.model = nn.Module()
        self.paligemma_with_expert.paligemma.model.language_model = nn.Module()
        self.paligemma_with_expert.paligemma.model.language_model.proj = nn.Linear(3, 2, bias=False)
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Module()
        self.paligemma_with_expert.gemma_expert.model.proj = nn.Linear(3, 2, bias=False)


def test_load_gate_assignments_accepts_qvla_wrapper(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({"assign": {"layer": [0, 4]}, "stats": {"final_avg_bits": 2.0}}))

    gates = qvla.load_gate_assignments(path)

    assert list(gates) == ["layer"]
    assert gates["layer"].tolist() == [0, 4]


def test_pi05_vlm_target_excludes_action_expert():
    model = _TinyPi05LikeModel()

    target_names = [name for name, _ in qvla.iter_target_modules(model, target="pi05_vlm_backbones")]

    assert "paligemma_with_expert.paligemma.model.language_model.proj" in target_names
    assert "paligemma_with_expert.gemma_expert.model.proj" not in target_names


def test_filter_proxy_layers_can_exclude_action_expert():
    proxies = {
        "paligemma_with_expert.paligemma.model.language_model.proj": {8: torch.zeros(2)},
        "paligemma_with_expert.gemma_expert.model.proj": {8: torch.zeros(2)},
    }

    filtered = qvla.filter_proxy_layers(proxies, "pi05_vlm_backbones")

    assert list(filtered) == ["paligemma_with_expert.paligemma.model.language_model.proj"]


def test_inject_weight_fake_quant_applies_channel_gates(tmp_path):
    model = _TinyPi05LikeModel()
    layer = model.paligemma_with_expert.gemma_expert.model.proj
    layer.weight.data = torch.tensor([[1.0, -2.0, 0.5], [0.3, -0.8, 0.1]])
    path = tmp_path / "gates.json"
    path.write_text(
        json.dumps(
            {
                "assign": {
                    "paligemma_with_expert.gemma_expert.model.proj": [0, 2],
                }
            }
        )
    )

    report = qvla.inject_weight_fake_quant(model, path)

    assert report.injected_layers == 1
    assert torch.equal(layer.weight.data[0], torch.zeros(3))
    assert torch.allclose(layer.weight.data[1], torch.tensor([0.0, -0.8, 0.0]))


def test_inject_activation_fake_quant_quantizes_layer_inputs(tmp_path):
    model = _TinyPi05LikeModel()
    layer = model.paligemma_with_expert.gemma_expert.model.proj
    layer.weight.data = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    path = tmp_path / "activation_amax.json"
    path.write_text(
        json.dumps(
            {
                "activations": {
                    "paligemma_with_expert.gemma_expert.model.proj": {
                        "amax": 1.0,
                    }
                }
            }
        )
    )

    report = qvla.inject_activation_fake_quant(
        model,
        num_bits=2,
        activation_scales_path=path,
        activation_granularity="calibrated-tensor",
    )
    out = layer(torch.tensor([[0.25, 0.75, -0.75]]))

    assert report.injected_layers == 1
    assert torch.allclose(out[0, 0], torch.tensor(0.0))


def test_calibrated_activation_scale_rejects_bit_mismatch(tmp_path):
    model = _TinyPi05LikeModel()
    path = tmp_path / "activation_amax.json"
    path.write_text(
        json.dumps(
            {
                "activation_bits": 4,
                "activations": {
                    "paligemma_with_expert.gemma_expert.model.proj": {
                        "amax": 1.0,
                    }
                },
            }
        )
    )

    try:
        qvla.inject_activation_fake_quant(
            model,
            num_bits=8,
            activation_scales_path=path,
            activation_granularity="calibrated-tensor",
        )
    except ValueError as exc:
        assert "calibrated for 4 bits" in str(exc)
    else:
        raise AssertionError("Expected calibrated activation bit mismatch to fail")


def test_dynamic_token_activation_quant_uses_row_local_scale():
    x = torch.tensor([[0.25, 0.75], [10.0, 20.0]])

    quantized = qvla.fake_quantize_activation_sym(x, 2, granularity="dynamic-token")

    assert torch.allclose(quantized[0], torch.tensor([0.0, 0.75]))
    assert torch.allclose(quantized[1], torch.tensor([0.0, 20.0]))


def test_histogram_activation_calibration_can_clip_outlier():
    histogram = torch.zeros(100)
    histogram[0] = 100_000
    histogram[-1] = 1

    amax = qvla.estimate_activation_amax_from_histogram(
        histogram,
        max_abs=100.0,
        num_bits=4,
        grid_size=100,
    )

    assert 0.0 < amax < 100.0


def test_greedy_allocate_reduces_cheapest_channels_first():
    proxies = {
        "a": {
            0: torch.tensor([100.0, 1.0]),
            4: torch.tensor([10.0, 0.1]),
            8: torch.zeros(2),
        }
    }

    layer_bits, stats = qvla.greedy_allocate(proxies, [0, 4, 8], target_avg_bits=4.0)

    assert layer_bits["a"] == [8, 0]
    assert stats["final_avg_bits"] == 4.0


def test_greedy_allocate_matches_official_absolute_proxy_costs():
    proxies = {
        "a": {
            0: torch.tensor([32.0, 32.0]),
            4: torch.tensor([32.0, 0.0]),
            8: torch.zeros(2),
        }
    }

    layer_bits, stats = qvla.greedy_allocate(proxies, [0, 4, 8], target_avg_bits=2.0)

    assert layer_bits["a"] == [4, 0]
    assert stats["final_avg_bits"] == 2.0


def test_hessian_proxy_accepts_conv2d_valid_padding_string():
    layer = nn.Conv2d(3, 4, kernel_size=2, padding="valid", bias=False)
    proxy = qvla.HessianProxy(layer, device="cpu")

    proxy.add_batch(torch.randn(1, 3, 8, 8))

    assert proxy.nsamples == 1
    assert proxy.hessian.shape == (12, 12)

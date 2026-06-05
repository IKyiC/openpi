import json

import torch
from torch import nn

from openpi.quantization import qvla


class _TinyPi05LikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Module()
        self.paligemma_with_expert.gemma_expert.model.proj = nn.Linear(3, 2, bias=False)


def test_load_gate_assignments_accepts_qvla_wrapper(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({"assign": {"layer": [0, 4]}, "stats": {"final_avg_bits": 2.0}}))

    gates = qvla.load_gate_assignments(path)

    assert list(gates) == ["layer"]
    assert gates["layer"].tolist() == [0, 4]


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

    report = qvla.inject_activation_fake_quant(model, num_bits=2, activation_scales_path=path)
    out = layer(torch.tensor([[0.25, 0.75, -0.75]]))

    assert report.injected_layers == 1
    assert torch.allclose(out[0, 0], torch.tensor(0.0))


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


def test_hessian_proxy_accepts_conv2d_valid_padding_string():
    layer = nn.Conv2d(3, 4, kernel_size=2, padding="valid", bias=False)
    proxy = qvla.HessianProxy(layer, device="cpu")

    proxy.add_batch(torch.randn(1, 3, 8, 8))

    assert proxy.nsamples == 1
    assert proxy.hessian.shape == (12, 12)

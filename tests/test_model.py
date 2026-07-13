import pytest
import torch

from config import ProjectConfig
from src.model import FlowReasoningLM


def small_config(**overrides: object) -> ProjectConfig:
    values = {
        "device": "cpu",
        "dim": 16,
        "num_heads": 2,
        "seq_length": 8,
        "max_seq_len": 8,
        "flow_steps": 2,
    }
    values.update(overrides)
    return ProjectConfig(**values).normalize()


def test_single_path_forward_is_finite() -> None:
    model = FlowReasoningLM(small_config(), vocab_size=12)
    tokens = torch.randint(0, 12, (2, 8))

    output = model(tokens, return_output=True)

    assert output.logits.shape == (2, 8, 12)
    assert torch.isfinite(output.logits).all()
    assert output.diagnostics["mode"] == "single"


def test_multi_path_forward_reports_aggregation_diagnostics() -> None:
    config = small_config(executor_mode="paths", num_paths=3)
    model = FlowReasoningLM(config, vocab_size=12)
    tokens = torch.randint(0, 12, (2, 8))

    output = model(tokens, return_output=True)

    assert output.diagnostics["path_weights"].shape == (2, 3)
    assert torch.allclose(output.diagnostics["path_weights"].sum(dim=1), torch.ones(2))
    assert output.diagnostics["effective_sample_size"].shape == (2,)


def test_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        small_config(dim=18, num_heads=4)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"num_heads": 0}, "num_heads"),
        ({"dropout": 1.0}, "dropout"),
        ({"learning_rate": 0.0}, "learning_rate"),
    ],
)
def test_rejects_invalid_hyperparameters(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        small_config(**override)

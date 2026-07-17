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

def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(7)

    config = small_config(
        dropout=0.0,
        executor_mode="single",
    )
    model = FlowReasoningLM(config, vocab_size=12)
    model.eval()

    first = torch.randint(0, 12, (1, 8))
    second = first.clone()

    # Change only positions 6 and 7.
    second[:, 6:] = (second[:, 6:] + 1) % 12

    first_logits = model(first)
    second_logits = model(second)

    torch.testing.assert_close(
        first_logits[:, :6],
        second_logits[:, :6],
        rtol=1e-5,
        atol=1e-6,
    )

def test_active_path_model_parameters_receive_gradients() -> None:
    torch.manual_seed(9)

    config = small_config(
        executor_mode="paths",
        num_paths=3,
        dropout=0.0,
    )
    model = FlowReasoningLM(config, vocab_size=12)
    model.train()

    tokens = torch.randint(0, 12, (2, 8))
    targets = torch.randint(0, 12, (2, 8))

    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    loss.backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]

    assert missing == []

def test_effective_branch_count_has_valid_bounds() -> None:
    config = small_config(
        executor_mode="paths",
        num_paths=3,
    )
    model = FlowReasoningLM(config, vocab_size=12)
    model.eval()

    tokens = torch.randint(0, 12, (2, 8))
    output = model(tokens, return_output=True)

    count = output.diagnostics["effective_branch_count"]

    assert torch.all(count >= 1.0)
    assert torch.all(count <= 3.0 + 1e-6)

def test_zero_flow_steps_is_supported() -> None:
    config = small_config(flow_steps=0)
    model = FlowReasoningLM(config, vocab_size=12)

    tokens = torch.randint(0, 12, (2, 8))
    output = model(tokens, return_output=True)

    assert torch.isfinite(output.logits).all()
    assert output.diagnostics["mode"] == "single"

def test_greedy_generation_is_deterministic() -> None:
    config = small_config()
    model = FlowReasoningLM(config, vocab_size=12)

    prompt = torch.randint(0, 12, (1, 4))

    first = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.0,
    )
    second = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.0,
    )

    torch.testing.assert_close(first, second)

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
    assert output.diagnostics["effective_branch_count"].shape == (2,)


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

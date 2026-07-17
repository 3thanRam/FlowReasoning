import torch
import math


from config import ProjectConfig
from src.model import FlowReasoningLM
from src.train import evaluate_model,train_one_model


def small_config(**overrides: object) -> ProjectConfig:
    values = {
        "device": "cpu",
        "dim": 16,
        "num_heads": 2,
        "seq_length": 8,
        "max_seq_len": 8,
        "flow_steps": 1,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ProjectConfig(**values).normalize()


def test_evaluation_uses_fixed_validation_windows() -> None:
    torch.manual_seed(5)

    config = small_config()
    model = FlowReasoningLM(
        config,
        vocab_size=12,
    )

    validation_tokens = torch.randint(
        0,
        12,
        (200,),
        dtype=torch.long,
    )

    first = evaluate_model(
        model,
        validation_tokens,
        batch_size=2,
        seq_length=8,
        batches=4,
        seed=123,
        device=torch.device("cpu"),
    )

    second = evaluate_model(
        model,
        validation_tokens,
        batch_size=2,
        seq_length=8,
        batches=4,
        seed=123,
        device=torch.device("cpu"),
    )

    assert first == second




def test_training_saves_best_and_final_checkpoints(
    tmp_path,
) -> None:
    best_path = tmp_path / "flow_reasoning.pt"

    config = ProjectConfig(
        device="cpu",
        dim=16,
        num_heads=2,
        seq_length=8,
        max_seq_len=8,
        flow_steps=1,
        training_steps=3,
        evaluation_interval=1,
        evaluation_batches=1,
        validation_fraction=0.2,
        batch_size=2,
        model_save_dir=str(best_path),
    )

    result = train_one_model(
        config,
        verbose=False,
    )

    final_path = (
        tmp_path / "flow_reasoning.last.pt"
    )

    assert best_path.exists()
    assert final_path.exists()

    assert 1 <= result["best_step"] <= 3
    assert math.isfinite(
        result["best_validation_loss"]
    )
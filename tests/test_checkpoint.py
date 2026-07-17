import torch

from config import ProjectConfig
from src.train import load_checkpoint, train_one_model


def test_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    config = ProjectConfig(
        device="cpu",
        dim=16,
        num_heads=2,
        seq_length=8,
        max_seq_len=8,
        flow_steps=1,
        training_steps=1,
        batch_size=2,
        model_save_dir=str(checkpoint),
    )

    result = train_one_model(config, verbose=False)
    loaded_config, tokenizer, model = load_checkpoint(str(checkpoint))
    tokens = tokenizer.encode("Flow").unsqueeze(0)

    assert result["model_path"] == str(checkpoint)
    assert loaded_config.device == "cpu"
    assert torch.isfinite(model(tokens)).all()

def test_checkpoint_records_validation_and_data_metadata(
    tmp_path,
) -> None:
    checkpoint_path = (
        tmp_path / "flow_reasoning.pt"
    )

    config = ProjectConfig(
        device="cpu",
        dim=16,
        num_heads=2,
        seq_length=8,
        max_seq_len=8,
        flow_steps=1,
        training_steps=1,
        evaluation_interval=1,
        evaluation_batches=1,
        validation_fraction=0.2,
        batch_size=2,
        model_save_dir=str(checkpoint_path),
    )

    train_one_model(
        config,
        verbose=False,
    )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert "validation" in payload
    assert "data" in payload
    assert "corpus_sha256" in payload["data"]
    assert payload["data"]["training_tokens"] > 0
    assert payload["data"]["validation_tokens"] > 0
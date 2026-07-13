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

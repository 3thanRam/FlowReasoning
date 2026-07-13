import torch

from src.data_loading import CharTokenizer, make_text_batch


def test_tokenizer_round_trip() -> None:
    text = "latent flow\n"
    tokenizer = CharTokenizer.from_text(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode("?")) == "�"


def test_batch_targets_are_shifted_by_one() -> None:
    tokenizer = CharTokenizer.from_text("abcdef")
    encoded = tokenizer.encode("abcdef")

    torch.manual_seed(0)
    x, y = make_text_batch(encoded, batch_size=3, seq_length=3, device="cpu")

    assert x.shape == y.shape == (3, 3)
    assert torch.equal(x[:, 1:], y[:, :-1])

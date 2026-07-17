from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib 
import torch

DEFAULT_CORPUS = """
FlowReasoning is a compact language-model experiment. It maps characters to
vectors, evolves those vectors through a shared latent operator, and predicts the
next character. Each update is deliberately small so that the latent trajectory
remains stable.

The operator mixes information in three ways. Causal attention connects each
position to its history. A spectral convolution captures longer patterns in the
sequence. A gated memory carries a summary between updates. After the final
update, a normalized projection produces logits over the vocabulary.

This bundled text is only a smoke-test corpus. Useful language modelling requires
a larger dataset, a validation split, and comparison with a strong baseline.
""".strip()


@dataclass
class CharTokenizer:
    tokens: list[str]

    def __post_init__(self) -> None:
        if "<unk>" not in self.tokens:
            self.tokens = ["<unk>"] + list(self.tokens)
        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls(chars)

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(list(data["tokens"]))

    def to_dict(self) -> dict:
        return {"type": "char", "tokens": self.tokens}

    def encode(self, text: str, device: torch.device | str | None = None) -> torch.Tensor:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        pieces = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            pieces.append("�" if token == "<unk>" else token)
        return "".join(pieces)


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_CORPUS
    text = Path(path).expanduser().read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Text file is empty: {path}")
    return text

@dataclass
class PreparedTextData:
    tokenizer: CharTokenizer
    train_tokens: torch.Tensor
    validation_tokens: torch.Tensor
    split_index: int
    corpus_sha256: str


def prepare_text_dataset(
    path: str | None,
    device: torch.device | str,
    validation_fraction: float,
) -> PreparedTextData:
    text = load_text(path)

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")

    split_index = int(len(text) * (1.0 - validation_fraction))

    train_text = text[:split_index]
    validation_text = text[split_index:]

    if len(train_text) < 2:
        raise ValueError("training split is too short")

    if len(validation_text) < 2:
        raise ValueError("validation split is too short")

    # Building the vocabulary from training data avoids validation leakage.
    tokenizer = CharTokenizer.from_text(train_text)

    train_tokens = tokenizer.encode(train_text, device=device)
    validation_tokens = tokenizer.encode(validation_text, device=device)

    corpus_sha256 = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return PreparedTextData(
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        split_index=split_index,
        corpus_sha256=corpus_sha256,
    )

def make_text_batch(
    encoded: torch.Tensor,
    batch_size: int,
    seq_length: int,
    device: torch.device | str,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x/y next-token batches of shape [batch, seq_length]."""
    if encoded.ndim != 1:
        raise ValueError("encoded text must be a flat token tensor")

    # Very small corpora are repeated so smoke tests still work.
    min_len = seq_length + 2
    if encoded.numel() < min_len:
        repeats = (min_len // max(1, encoded.numel())) + 1
        encoded = encoded.repeat(repeats)

    max_start = encoded.numel() - seq_length - 1
    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )
    offsets = torch.arange(seq_length, device=device)
    positions = starts[:, None] + offsets[None, :]
    x = encoded[positions]
    y = encoded[positions + 1]
    return x.contiguous(), y.contiguous()

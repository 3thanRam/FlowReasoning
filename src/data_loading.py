from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_CORPUS = """
Observation: A reasoning model receives tokens and embeds them into a latent space.
Assumptions and objectives: The latent state should evolve through several small stable steps before logits are produced.
Expectation vs reality: If the next-token loss falls, the latent evolution is useful. If it does not, revise the number of steps, capacity, data, or optimisation.
Revision: Keep the architecture simple, measure diagnostics, and only add complexity when the observations justify it.

Observation: Structured reasoning records what the model or researcher can see.
Assumptions and objectives: It states what must be true and what must be achieved.
Expectation vs reality: It compares predicted behaviour against measured behaviour.
Revision: It updates the next experiment instead of pretending the first assumption was perfect.
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


def prepare_text_dataset(path: str | None, device: torch.device | str) -> tuple[CharTokenizer, torch.Tensor]:
    text = load_text(path)
    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text, device=device)
    return tokenizer, encoded


def make_text_batch(
    encoded: torch.Tensor,
    batch_size: int,
    seq_length: int,
    device: torch.device | str,
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
    starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
    offsets = torch.arange(seq_length, device=device)
    positions = starts[:, None] + offsets[None, :]
    x = encoded[positions]
    y = encoded[positions + 1]
    return x.contiguous(), y.contiguous()

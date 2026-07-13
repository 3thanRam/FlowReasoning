from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

ExecutorMode = Literal["single", "paths"]


@dataclass
class ProjectConfig:
    """Configuration for the FlowReasoning language-model prototype.

    The model is intentionally small by default so that a smoke test can run on
    CPU, while still preserving the intended pipeline:

        text -> tokens -> latent states -> flow evolution -> output logits
    """

    # Reproducibility / device
    seed: int = 42
    device: str = "auto"
    num_threads: int = 1

    # Data
    data_path: str | None = None
    seq_length: int = 128
    batch_size: int = 16

    # Latent model
    dim: int = 128
    num_heads: int = 4
    max_seq_len: int = 256
    flow_steps: int = 4
    max_flow_steps: int = 16
    max_step_size: float = 0.25
    dz_clip: float = 3.0
    executor_mode: ExecutorMode = "single"
    num_paths: int = 1
    dropout: float = 0.0

    # Optimisation
    training_steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 20

    # Generation / checkpointing
    model_save_dir: str = "checkpoints/flow_reasoning.pt"
    prompt: str = "The latent state"
    generate_tokens: int = 200
    temperature: float = 0.9
    top_k: int = 40

    def normalize(self) -> "ProjectConfig":
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_threads = int(self.num_threads)
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)

        self.seq_length = int(self.seq_length)
        self.max_seq_len = max(int(self.max_seq_len), self.seq_length)
        self.batch_size = int(self.batch_size)
        self.dim = int(self.dim)
        self.num_heads = int(self.num_heads)
        self.flow_steps = max(0, int(self.flow_steps))
        self.max_flow_steps = max(self.flow_steps, int(self.max_flow_steps))
        self.num_paths = max(1, int(self.num_paths))
        self.training_steps = max(1, int(self.training_steps))
        self.log_interval = max(1, int(self.log_interval))

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.dim % 2 != 0:
            raise ValueError("dim must be even; the diagonal unitary memory splits real/imag parts")
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if (self.dim // self.num_heads) % 2 != 0:
            raise ValueError("dim / num_heads must be even for rotary embeddings")
        if self.seq_length < 2:
            raise ValueError("seq_length must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.executor_mode not in {"single", "paths"}:
            raise ValueError("executor_mode must be 'single' or 'paths'")
        if self.executor_mode == "paths" and self.num_paths < 2:
            self.num_paths = 2

        save_path = Path(self.model_save_dir)
        if save_path.suffix == "":
            self.model_save_dir = str(save_path / "flow_reasoning.pt")

        return self

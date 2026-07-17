from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ProjectConfig
from .core import FlowEvolver, RMSNorm


@dataclass
class ModelOutput:
    logits: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class FlowReasoningLM(nn.Module):
    """A small language model with flow-like latent evolution.

    Pipeline:
        token ids -> token/position embeddings -> latent z0
        -> FlowEvolver(z0) for structured latent computation
        -> normalised latent state -> logits over the vocabulary
    """

    def __init__(self, config: ProjectConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size)
        self.token_embed = nn.Embedding(self.vocab_size, config.dim)
        self.position_embed = nn.Embedding(config.max_seq_len, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.evolver = FlowEvolver(
            dim=config.dim,
            max_seq_len=config.max_seq_len,
            num_heads=config.num_heads,
            max_flow_steps=config.max_flow_steps,
            max_step_size=config.max_step_size,
            dz_clip=config.dz_clip,
        )
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, self.vocab_size, bias=False)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_output: bool = False,
        flow_steps: int | None = None,
    ) -> torch.Tensor | ModelOutput:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)
        z0 = self.token_embed(tokens) + self.position_embed(positions)
        z0 = self.dropout(z0)

        flow = self.evolver(
            z0,
            flow_steps=(
                self.config.flow_steps
                if flow_steps is None
                else flow_steps
            ),
            mode=self.config.executor_mode,
            num_paths=self.config.num_paths,
        )
        hidden = self.final_norm(flow.z)
        logits = self.lm_head(hidden)
        if return_output:
            return ModelOutput(
                logits=logits,
                diagnostics=flow.diagnostics,
            )
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids.clone()
        for _ in range(max_new_tokens):
            context = out[:, -self.config.max_seq_len :]
            logits = self(context)
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0 and top_k < next_logits.shape[-1]:
                    values, _ = torch.topk(next_logits, top_k, dim=-1)
                    cutoff = values[:, [-1]]
                    next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)
        return out

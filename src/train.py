from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

from config import ProjectConfig
from .data_loading import CharTokenizer, make_text_batch, prepare_text_dataset
from .model import FlowReasoningLM
from .utils import count_parameters, diagnostics_to_dict, ensure_parent_dir, set_seed


def text_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.view(-1, vocab), targets.reshape(-1))


def train_one_model(config: ProjectConfig, verbose: bool = True) -> dict[str, Any]:
    config = copy.deepcopy(config).normalize()
    set_seed(config.seed)
    device = torch.device(config.device)

    tokenizer, encoded = prepare_text_dataset(config.data_path, device)
    model = FlowReasoningLM(config, tokenizer.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    started = time.time()
    last_loss = float("nan")
    last_diag: dict[str, Any] = {}

    if verbose:
        print("Observation: training text has", encoded.numel(), "tokens and", tokenizer.vocab_size, "unique symbols")
        print(
            "Assumptions and objectives: minimise next-token loss while checking that latent flow diagnostics stay finite"
        )

    for step in range(1, config.training_steps + 1):
        model.train()
        x, y = make_text_batch(encoded, config.batch_size, config.seq_length, device)
        output = model(x, return_output=True)
        loss = text_loss(output.logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()

        last_loss = float(loss.detach().item())
        last_diag = diagnostics_to_dict(output.diagnostics)
        perplexity = float(math.exp(min(last_loss, 20.0)))

        should_log = step == 1 or step == config.training_steps or step % config.log_interval == 0
        if verbose and should_log:
            diag_text = " ".join(
                f"{key}={value:.4f}" for key, value in last_diag.items() if isinstance(value, float)
            )
            print(
                f"Expectation vs reality: step={step:5d}/{config.training_steps} "
                f"loss={last_loss:.4f} perplexity={perplexity:.2f} {diag_text}"
            )

    ensure_parent_dir(config.model_save_dir)
    checkpoint = {
        "config": asdict(config),
        "tokenizer": tokenizer.to_dict(),
        "state_dict": model.state_dict(),
        "param_count": count_parameters(model),
        "final_loss": last_loss,
        "diagnostics": last_diag,
    }
    torch.save(checkpoint, config.model_save_dir)

    result: dict[str, Any] = {
        "loss": last_loss,
        "perplexity": float(math.exp(min(last_loss, 20.0))),
        "param_count": count_parameters(model),
        "seconds": round(time.time() - started, 3),
        "model_path": config.model_save_dir,
        "vocab_size": tokenizer.vocab_size,
    }
    result.update(last_diag)

    if verbose:
        print(
            "Revision: saved checkpoint; next useful revisions are more data, longer training, or comparing single vs paths mode"
        )

    return result


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[ProjectConfig, CharTokenizer, FlowReasoningLM]:
    checkpoint = torch.load(path, map_location=map_location)
    config = ProjectConfig(**checkpoint["config"]).normalize()
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    model = FlowReasoningLM(config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(config.device))
    model.eval()
    return config, tokenizer, model

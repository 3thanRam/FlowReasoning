from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from typing import Any
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ProjectConfig
from .data_loading import CharTokenizer, make_text_batch, prepare_text_dataset, PreparedTextData
from .model import FlowReasoningLM
from .utils import count_parameters, diagnostics_to_dict, ensure_parent_dir, set_seed

@torch.inference_mode()
def evaluate_model(
    model: FlowReasoningLM,
    encoded: torch.Tensor,
    *,
    batch_size: int,
    seq_length: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on a fixed set of randomly sampled validation windows."""

    was_training = model.training
    model.eval()

    generator = torch.Generator(device=device).manual_seed(seed)

    losses: list[float] = []

    for _ in range(batches):
        x, y = make_text_batch(
            encoded,
            batch_size,
            seq_length,
            device,
            generator=generator,
        )

        logits = model(x)
        loss = text_loss(logits, y)
        losses.append(float(loss.item()))

    if was_training:
        model.train()

    mean_loss = sum(losses) / len(losses)

    return {
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(min(mean_loss, 20.0)),
        "validation_bits_per_character": mean_loss / math.log(2.0),
    }

def save_checkpoint(
    path: str,
    *,
    model: FlowReasoningLM,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    tokenizer: CharTokenizer,
    step: int,
    training_loss: float,
    validation: dict[str, float],
    best_validation_loss: float,
    best_step: int,
    data: PreparedTextData,
    diagnostics: dict[str, Any],
) -> None:
    ensure_parent_dir(path)

    checkpoint = {
        "schema_version": 2,
        "config": asdict(config),
        "tokenizer": tokenizer.to_dict(),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "param_count": count_parameters(model),
        "step": step,
        "training_loss": training_loss,
        "validation": validation,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "diagnostics": diagnostics,
        "data": {
            "corpus_sha256": data.corpus_sha256,
            "split_index": data.split_index,
            "training_tokens": int(data.train_tokens.numel()),
            "validation_tokens": int(data.validation_tokens.numel()),
        },
    }

    torch.save(checkpoint, path)

def text_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.view(-1, vocab), targets.reshape(-1))


def train_one_model(
    config: ProjectConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    config = copy.deepcopy(config).normalize()
    set_seed(config.seed)

    device = torch.device(config.device)

    data = prepare_text_dataset(
        config.data_path,
        device,
        validation_fraction=config.validation_fraction,
    )

    model = FlowReasoningLM(
        config,
        data.tokenizer.vocab_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    training_generator = torch.Generator(
        device=device
    ).manual_seed(config.seed + 1_000)

    best_path = str(config.model_save_dir)
    best_path_object = Path(best_path)

    last_path = str(
        best_path_object.with_name(
            f"{best_path_object.stem}.last{best_path_object.suffix}"
        )
    )

    started = time.perf_counter()

    last_training_loss = float("nan")
    last_diagnostics: dict[str, Any] = {}
    last_validation: dict[str, float] = {}

    best_validation_loss = float("inf")
    best_step = 0

    history: list[dict[str, Any]] = []

    if verbose:
        print(
            f"Training tokens: {data.train_tokens.numel():,} | "
            f"validation tokens: {data.validation_tokens.numel():,} | "
            f"vocabulary: {data.tokenizer.vocab_size} | "
            f"parameters: {count_parameters(model):,} | "
            f"device: {device}",
            flush=True,
        )

    for step in range(1, config.training_steps + 1):
        model.train()

        x, y = make_text_batch(
            data.train_tokens,
            config.batch_size,
            config.seq_length,
            device,
            generator=training_generator,
        )

        output = model(
            x,
            return_output=True,
        )

        loss = text_loss(
            output.logits,
            y,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite training loss at step {step}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip,
            )
        else:
            gradient_norm = torch.tensor(
                float("nan"),
                device=device,
            )

        optimizer.step()

        last_training_loss = float(loss.detach().item())
        last_diagnostics = diagnostics_to_dict(
            output.diagnostics
        )

        should_evaluate = (
            step == 1
            or step == config.training_steps
            or step % config.evaluation_interval == 0
        )

        should_log = (
            step == 1
            or step == config.training_steps
            or step % config.log_interval == 0
        )

        if should_evaluate:
            last_validation = evaluate_model(
                model,
                data.validation_tokens,
                batch_size=config.batch_size,
                seq_length=config.seq_length,
                batches=config.evaluation_batches,
                seed=config.validation_seed,
                device=device,
            )

            current_validation_loss = last_validation[
                "validation_loss"
            ]

            improved = (
                current_validation_loss
                < best_validation_loss
            )

            if improved:
                best_validation_loss = (
                    current_validation_loss
                )
                best_step = step

                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    tokenizer=data.tokenizer,
                    step=step,
                    training_loss=last_training_loss,
                    validation=last_validation,
                    best_validation_loss=best_validation_loss,
                    best_step=best_step,
                    data=data,
                    diagnostics=last_diagnostics,
                )

            event = {
                "step": step,
                "training_loss": last_training_loss,
                "training_perplexity": math.exp(
                    min(last_training_loss, 20.0)
                ),
                "gradient_norm": float(
                    gradient_norm.detach().item()
                ),
                **last_validation,
                **last_diagnostics,
            }

            history.append(event)

            if verbose:
                elapsed = time.perf_counter() - started

                print(
                    f"step {step:5d}/{config.training_steps} | "
                    f"train loss {last_training_loss:.4f} | "
                    f"validation loss "
                    f"{last_validation['validation_loss']:.4f} | "
                    f"validation BPC "
                    f"{last_validation['validation_bits_per_character']:.4f} | "
                    f"best step {best_step} | "
                    f"elapsed {elapsed:.1f}s",
                    flush=True,
                )

        elif verbose and should_log:
            elapsed = time.perf_counter() - started
            training_perplexity = math.exp(
                min(last_training_loss, 20.0)
            )

            diagnostic_text = " ".join(
                f"{key}={value:.4f}"
                for key, value in last_diagnostics.items()
                if isinstance(value, float)
            )

            print(
                f"step {step:5d}/{config.training_steps} | "
                f"train loss {last_training_loss:.4f} | "
                f"train perplexity {training_perplexity:.2f} | "
                f"elapsed {elapsed:.1f}s"
                + (
                    f" | {diagnostic_text}"
                    if diagnostic_text
                    else ""
                ),
                flush=True,
            )

    # Save the final model separately, regardless of whether it is the best.
    save_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=data.tokenizer,
        step=config.training_steps,
        training_loss=last_training_loss,
        validation=last_validation,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        data=data,
        diagnostics=last_diagnostics,
    )

    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "training_loss": last_training_loss,
        "training_perplexity": math.exp(
            min(last_training_loss, 20.0)
        ),
        "best_validation_loss": best_validation_loss,
        "best_validation_perplexity": math.exp(
            min(best_validation_loss, 20.0)
        ),
        "best_validation_bits_per_character": (
            best_validation_loss / math.log(2.0)
        ),
        "best_step": best_step,
        "param_count": count_parameters(model),
        "seconds": round(elapsed, 3),
        "best_model_path": best_path,
        "last_model_path": last_path,
        "vocab_size": data.tokenizer.vocab_size,
        "training_tokens": int(
            data.train_tokens.numel()
        ),
        "validation_tokens": int(
            data.validation_tokens.numel()
        ),
        "corpus_sha256": data.corpus_sha256,
        "history": history,
    }

    if verbose:
        print(
            f"Best checkpoint: {best_path} "
            f"(step {best_step}, "
            f"validation loss {best_validation_loss:.4f})",
            flush=True,
        )
        print(
            f"Final checkpoint: {last_path}",
            flush=True,
        )

    return result

def load_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[
    ProjectConfig,
    CharTokenizer,
    FlowReasoningLM,
]:
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    config_values = dict(checkpoint["config"])
    config_values["device"] = str(map_location)

    config = ProjectConfig(
        **config_values
    ).normalize()

    tokenizer = CharTokenizer.from_dict(
        checkpoint["tokenizer"]
    )

    model = FlowReasoningLM(
        config,
        tokenizer.vocab_size,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.to(
        torch.device(config.device)
    )
    model.eval()

    return config, tokenizer, model

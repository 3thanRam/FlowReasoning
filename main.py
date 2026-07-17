from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any

import torch

from config import ProjectConfig
from src.train import load_checkpoint, train_one_model


def _coerce_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowReasoning: experimental language model with iterative latent-state evolution")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a character-level FlowReasoning model")
    for field in fields(ProjectConfig):
        default = getattr(ProjectConfig(), field.name)
        arg_name = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            train.add_argument(arg_name, action="store_true")
        else:
            train.add_argument(arg_name, default=None)

    generate = sub.add_parser("generate", help="Generate text from a saved checkpoint")
    generate.add_argument("--checkpoint", default=ProjectConfig().model_save_dir)
    generate.add_argument("--prompt", default=None)
    generate.add_argument("--max-new-tokens", type=int, default=None)
    generate.add_argument("--temperature", type=float, default=None)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--device", default=None)

    return parser


def config_from_args(args: argparse.Namespace) -> ProjectConfig:
    config = ProjectConfig()
    for field in fields(ProjectConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(config, field.name, _coerce_value(value, getattr(config, field.name)))
    return config.normalize()


def run_train(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    result = train_one_model(config, verbose=True)

    print(
        f"Finished | "
        f"train loss {result['training_loss']:.4f} | "
        f"best validation loss "
        f"{result['best_validation_loss']:.4f} | "
        f"validation BPC "
        f"{result['best_validation_bits_per_character']:.4f} | "
        f"time {result['seconds']:.1f}s"
    )

    print(
        "Generate with: "
        f'python main.py generate '
        f'--checkpoint {result["best_model_path"]} '
        f'--prompt "The latent state"'
    )


def run_generate(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint
    map_location = args.device or "cpu"
    config, tokenizer, model = load_checkpoint(checkpoint_path, map_location=map_location)
    if args.device is not None:
        config.device = args.device
        model.to(torch.device(args.device))

    prompt = args.prompt if args.prompt is not None else config.prompt
    max_new = args.max_new_tokens if args.max_new_tokens is not None else config.generate_tokens
    temperature = args.temperature if args.temperature is not None else config.temperature
    top_k = args.top_k if args.top_k is not None else config.top_k

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, device=device).unsqueeze(0)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
    )[0]
    print(tokenizer.decode(generated))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "generate":
        run_generate(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

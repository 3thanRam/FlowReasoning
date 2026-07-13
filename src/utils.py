from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random-number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def diagnostics_to_dict(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Convert lightweight diagnostics to checkpoint-safe Python values."""
    return {
        key: tensor_to_python(value)
        for key, value in diagnostics.items()
        if key != "z_paths"
    }

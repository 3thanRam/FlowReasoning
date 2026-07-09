from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_to_float(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def diagnostics_to_dict(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {key: tensor_to_float(value) for key, value in diagnostics.items() if key != "z_paths"}


def format_structured_status(
    *,
    observation: str,
    assumptions_objectives: str,
    expectation: str,
    reality: str,
    revision: str,
) -> str:
    return (
        f"Observation: {observation}\n"
        f"Assumptions and objectives: {assumptions_objectives}\n"
        f"Expectation vs reality: {expectation} / {reality}\n"
        f"Revision: {revision}"
    )

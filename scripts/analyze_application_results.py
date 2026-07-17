from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


ARCHITECTURE_ORDER = {"single-1": 0, "single-4": 1, "paths-4": 2}
MATCHED_CONFIG_FIELDS = (
    "seq_length",
    "batch_size",
    "validation_fraction",
    "validation_seed",
    "evaluation_batches",
    "dim",
    "num_heads",
    "max_seq_len",
    "max_flow_steps",
    "max_step_size",
    "dz_clip",
    "dropout",
    "training_steps",
    "learning_rate",
    "weight_decay",
    "grad_clip",
)


@dataclass(frozen=True)
class RunResult:
    run: str
    architecture: str
    seed: int
    mode: str
    flow_steps: int
    branches: int
    parameters: int
    best_step: int
    training_loss_at_best: float
    validation_loss: float
    validation_bpc: float
    validation_perplexity: float
    mean_update_norm: float | None
    branch_variance: float | None
    effective_branch_count: float | None
    wall_seconds: float | None
    training_tokens: int
    validation_tokens: int
    corpus_sha256: str
    split_index: int
    validation_seed: int
    checkpoint: str


@dataclass(frozen=True)
class AggregateResult:
    architecture: str
    mode: str
    flow_steps: int
    branches: int
    seeds: int
    validation_bpc_mean: float
    validation_bpc_sd: float
    validation_loss_mean: float
    validation_loss_sd: float
    wall_minutes_mean: float | None
    wall_minutes_sd: float | None
    parameters_mean: float
    best_step_mean: float
    branch_variance_mean: float | None
    effective_branch_count_mean: float | None


@dataclass(frozen=True)
class PairwiseDelta:
    seed: int
    comparison: str
    baseline: str
    candidate: str
    validation_bpc_delta: float
    validation_loss_delta: float
    wall_time_ratio: float | None


def architecture_name(config: dict[str, Any]) -> str:
    mode = str(config["executor_mode"])
    steps = int(config["flow_steps"])
    branches = int(config["num_paths"])
    if mode == "single":
        return f"single-{steps}"
    if mode == "paths":
        return f"paths-{branches}"
    return f"{mode}-{steps}-{branches}"


def flatten_numeric(value: Any) -> list[float]:
    values: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            values.extend(float(x) for x in item.detach().cpu().reshape(-1).tolist())
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(float(item))

    visit(value)
    return values


def numeric_mean(value: Any) -> float | None:
    values = flatten_numeric(value)
    return statistics.fmean(values) if values else None


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def optional_mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def optional_sd(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sample_sd(present)


def format_optional(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def read_wall_seconds(run_dir: Path) -> float | None:
    path = run_dir / "wall_seconds.txt"
    if not path.exists():
        return None
    try:
        value = float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def load_run(checkpoint: Path) -> tuple[RunResult, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "config",
        "param_count",
        "step",
        "training_loss",
        "validation",
        "diagnostics",
        "data",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{checkpoint} is missing keys: {', '.join(missing)}")

    config = dict(payload["config"])
    validation = dict(payload["validation"])
    diagnostics = dict(payload.get("diagnostics", {}))
    data = dict(payload["data"])

    result = RunResult(
        run=checkpoint.parent.name,
        architecture=architecture_name(config),
        seed=int(config["seed"]),
        mode=str(config["executor_mode"]),
        flow_steps=int(config["flow_steps"]),
        branches=int(config["num_paths"]),
        parameters=int(payload["param_count"]),
        best_step=int(payload["step"]),
        training_loss_at_best=float(payload["training_loss"]),
        validation_loss=float(validation["validation_loss"]),
        validation_bpc=float(validation["validation_bits_per_character"]),
        validation_perplexity=float(validation["validation_perplexity"]),
        mean_update_norm=numeric_mean(diagnostics.get("mean_dz_norm")),
        branch_variance=numeric_mean(diagnostics.get("latent_path_variance")),
        effective_branch_count=numeric_mean(
            diagnostics.get("effective_branch_count")
        ),
        wall_seconds=read_wall_seconds(checkpoint.parent),
        training_tokens=int(data["training_tokens"]),
        validation_tokens=int(data["validation_tokens"]),
        corpus_sha256=str(data["corpus_sha256"]),
        split_index=int(data["split_index"]),
        validation_seed=int(config["validation_seed"]),
        checkpoint=str(checkpoint.resolve()),
    )
    return result, config


def discover_runs(runs_dir: Path) -> tuple[list[RunResult], dict[str, dict[str, Any]]]:
    checkpoints = sorted(runs_dir.glob("*/flow_reasoning.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No */flow_reasoning.pt checkpoints found in {runs_dir.resolve()}"
        )

    results: list[RunResult] = []
    configs: dict[str, dict[str, Any]] = {}
    for checkpoint in checkpoints:
        result, config = load_run(checkpoint)
        results.append(result)
        configs[result.run] = config

    results.sort(
        key=lambda row: (
            ARCHITECTURE_ORDER.get(row.architecture, 99),
            row.architecture,
            row.seed,
        )
    )
    return results, configs


def validate_consistency(
    results: list[RunResult], configs: dict[str, dict[str, Any]]
) -> list[str]:
    warnings: list[str] = []

    def check_one(label: str, values: dict[str, Any]) -> None:
        unique = {json.dumps(value, sort_keys=True) for value in values.values()}
        if len(unique) > 1:
            details = ", ".join(
                f"{run}={value!r}" for run, value in sorted(values.items())
            )
            warnings.append(f"{label} differs across runs: {details}")

    check_one("corpus SHA-256", {r.run: r.corpus_sha256 for r in results})
    check_one("split index", {r.run: r.split_index for r in results})
    check_one("training token count", {r.run: r.training_tokens for r in results})
    check_one("validation token count", {r.run: r.validation_tokens for r in results})
    check_one("validation seed", {r.run: r.validation_seed for r in results})

    for field in MATCHED_CONFIG_FIELDS:
        check_one(
            f"config.{field}",
            {run: config.get(field) for run, config in configs.items()},
        )

    by_seed: dict[int, set[str]] = defaultdict(set)
    for row in results:
        by_seed[row.seed].add(row.architecture)

    required = {"single-1", "single-4", "paths-4"}
    for seed, architectures in sorted(by_seed.items()):
        missing = sorted(required - architectures)
        if missing:
            warnings.append(
                f"seed {seed} is missing architecture runs: {', '.join(missing)}"
            )

    for row in results:
        value = row.effective_branch_count
        if value is not None and not (1.0 - 1e-6 <= value <= row.branches + 1e-6):
            warnings.append(
                f"{row.run} effective branch count {value:.6f} is outside "
                f"[1, {row.branches}]"
            )

    return warnings


def aggregate_runs(results: list[RunResult]) -> list[AggregateResult]:
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for row in results:
        groups[row.architecture].append(row)

    aggregates: list[AggregateResult] = []
    for architecture, group in groups.items():
        bpc = [row.validation_bpc for row in group]
        losses = [row.validation_loss for row in group]
        wall_minutes = [
            None if row.wall_seconds is None else row.wall_seconds / 60.0
            for row in group
        ]
        aggregates.append(
            AggregateResult(
                architecture=architecture,
                mode=group[0].mode,
                flow_steps=group[0].flow_steps,
                branches=group[0].branches,
                seeds=len(group),
                validation_bpc_mean=statistics.fmean(bpc),
                validation_bpc_sd=sample_sd(bpc),
                validation_loss_mean=statistics.fmean(losses),
                validation_loss_sd=sample_sd(losses),
                wall_minutes_mean=optional_mean(wall_minutes),
                wall_minutes_sd=optional_sd(wall_minutes),
                parameters_mean=statistics.fmean(row.parameters for row in group),
                best_step_mean=statistics.fmean(row.best_step for row in group),
                branch_variance_mean=optional_mean(
                    row.branch_variance for row in group
                ),
                effective_branch_count_mean=optional_mean(
                    row.effective_branch_count for row in group
                ),
            )
        )

    aggregates.sort(
        key=lambda row: (
            ARCHITECTURE_ORDER.get(row.architecture, 99),
            row.architecture,
        )
    )
    return aggregates


def pairwise_deltas(results: list[RunResult]) -> list[PairwiseDelta]:
    by_seed: dict[int, dict[str, RunResult]] = defaultdict(dict)
    for row in results:
        by_seed[row.seed][row.architecture] = row

    comparisons = (
        ("repeated updates", "single-1", "single-4"),
        ("learned branches", "single-4", "paths-4"),
    )
    deltas: list[PairwiseDelta] = []

    for seed, rows in sorted(by_seed.items()):
        for comparison, baseline_name, candidate_name in comparisons:
            if baseline_name not in rows or candidate_name not in rows:
                continue
            baseline = rows[baseline_name]
            candidate = rows[candidate_name]
            wall_ratio = None
            if (
                baseline.wall_seconds is not None
                and candidate.wall_seconds is not None
                and baseline.wall_seconds > 0
            ):
                wall_ratio = candidate.wall_seconds / baseline.wall_seconds
            deltas.append(
                PairwiseDelta(
                    seed=seed,
                    comparison=comparison,
                    baseline=baseline_name,
                    candidate=candidate_name,
                    validation_bpc_delta=candidate.validation_bpc
                    - baseline.validation_bpc,
                    validation_loss_delta=candidate.validation_loss
                    - baseline.validation_loss,
                    wall_time_ratio=wall_ratio,
                )
            )
    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_table_markdown(results: list[RunResult]) -> str:
    lines = [
        "| Run | Seed | Parameters | Best step | Validation BPC | Validation loss | Wall time | Branch variance | Effective branches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        wall = "-" if row.wall_seconds is None else f"{row.wall_seconds / 60.0:.1f} min"
        lines.append(
            f"| {row.run} | {row.seed} | {row.parameters:,} | {row.best_step} | "
            f"{row.validation_bpc:.4f} | {row.validation_loss:.4f} | {wall} | "
            f"{format_optional(row.branch_variance, 5)} | "
            f"{format_optional(row.effective_branch_count, 2)} |"
        )
    return "\n".join(lines)


def aggregate_table_markdown(aggregates: list[AggregateResult]) -> str:
    lines = [
        "| Architecture | Flow steps | Branches | Seeds | Validation BPC | Validation loss | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        wall = "-"
        if row.wall_minutes_mean is not None:
            wall = (
                f"{row.wall_minutes_mean:.1f} +/- "
                f"{(row.wall_minutes_sd or 0.0):.1f} min"
            )
        lines.append(
            f"| {row.architecture} | {row.flow_steps} | {row.branches} | "
            f"{row.seeds} | {row.validation_bpc_mean:.4f} +/- "
            f"{row.validation_bpc_sd:.4f} | {row.validation_loss_mean:.4f} +/- "
            f"{row.validation_loss_sd:.4f} | {wall} |"
        )
    return "\n".join(lines)


def delta_table_markdown(deltas: list[PairwiseDelta]) -> str:
    lines = [
        "| Seed | Comparison | Candidate - baseline BPC | Candidate - baseline loss | Wall-time ratio |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row.seed} | {row.candidate} vs {row.baseline} | "
            f"{row.validation_bpc_delta:+.6f} | "
            f"{row.validation_loss_delta:+.6f} | "
            f"{format_optional(row.wall_time_ratio, 2)} |"
        )
    return "\n".join(lines)


def delta_summary_markdown(deltas: list[PairwiseDelta]) -> str:
    grouped: dict[str, list[PairwiseDelta]] = defaultdict(list)
    for row in deltas:
        grouped[row.comparison].append(row)

    lines: list[str] = []
    for comparison in ("repeated updates", "learned branches"):
        group = grouped.get(comparison, [])
        if not group:
            continue
        bpc = [row.validation_bpc_delta for row in group]
        loss = [row.validation_loss_delta for row in group]
        wall = [row.wall_time_ratio for row in group if row.wall_time_ratio is not None]
        mean_bpc = statistics.fmean(bpc)
        direction = (
            "favours the candidate"
            if mean_bpc < 0
            else "favours the baseline"
            if mean_bpc > 0
            else "is neutral"
        )
        wall_text = "not available" if not wall else f"{statistics.fmean(wall):.2f}x"
        lines.append(
            f"- **{comparison.capitalize()}** (`{group[0].candidate}` minus "
            f"`{group[0].baseline}`): validation BPC "
            f"{mean_bpc:+.6f} +/- {sample_sd(bpc):.6f}; validation loss "
            f"{statistics.fmean(loss):+.6f} +/- {sample_sd(loss):.6f}; "
            f"mean wall-time ratio {wall_text}. Lower BPC is better, so this "
            f"{direction}."
        )
    return "\n".join(lines)


def generate_samples(
    results: list[RunResult],
    repository_root: Path,
    prompt: str,
    tokens: int,
    device: str,
) -> dict[str, str]:
    main_path = repository_root / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Cannot find {main_path}")

    representative: dict[str, RunResult] = {}
    for row in results:
        current = representative.get(row.architecture)
        if current is None or row.seed < current.seed:
            representative[row.architecture] = row

    samples: dict[str, str] = {}
    for architecture in sorted(
        representative,
        key=lambda name: (ARCHITECTURE_ORDER.get(name, 99), name),
    ):
        row = representative[architecture]
        command = [
            sys.executable,
            str(main_path),
            "generate",
            "--checkpoint",
            row.checkpoint,
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(tokens),
            "--temperature",
            "0",
            "--device",
            device,
        ]
        completed = subprocess.run(
            command,
            cwd=repository_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        samples[architecture] = completed.stdout.strip()
        if completed.returncode != 0:
            samples[architecture] = (
                f"[generation failed with exit code {completed.returncode}]\n"
                + samples[architecture]
            )
    return samples


def build_results_markdown(
    results: list[RunResult],
    aggregates: list[AggregateResult],
    deltas: list[PairwiseDelta],
    warnings: list[str],
    samples: dict[str, str] | None,
) -> str:
    seeds = sorted({row.seed for row in results})
    evidence = "one-seed screening" if len(seeds) == 1 else f"{len(seeds)}-seed preliminary"
    corpus_hashes = sorted({row.corpus_sha256 for row in results})
    validation_seeds = sorted({row.validation_seed for row in results})
    training_tokens = sorted({row.training_tokens for row in results})
    validation_tokens = sorted({row.validation_tokens for row in results})

    lines = [
        "# Preliminary Architecture Comparison",
        "",
        f"This is a {evidence} experiment for a compact character-level language model. "
        "It compares one shared latent update, four repeated shared updates, and four "
        "learned branches with four updates. Runs use the same corpus, split, fixed "
        "validation windows, optimizer settings, and training-step budget.",
        "",
        "Lower validation bits per character (BPC) is better. The runs are matched by "
        "data and optimizer steps, not by wall-clock compute.",
        "",
        "## Per-run results",
        "",
        run_table_markdown(results),
        "",
        "## Aggregate results",
        "",
        aggregate_table_markdown(aggregates),
        "",
        "## Paired architecture differences",
        "",
        delta_table_markdown(deltas),
        "",
        "Negative candidate-minus-baseline BPC favours the candidate.",
        "",
        delta_summary_markdown(deltas),
        "",
        "## Reproducibility metadata",
        "",
        f"- Training seeds: `{', '.join(map(str, seeds))}`",
        f"- Validation seed(s): `{', '.join(map(str, validation_seeds))}`",
        f"- Training token count(s): `{', '.join(map(str, training_tokens))}`",
        f"- Validation token count(s): `{', '.join(map(str, validation_tokens))}`",
        f"- Corpus SHA-256 value(s): `{', '.join(corpus_hashes)}`",
        "",
        "The saved branch variance and effective branch count are diagnostics from the "
        "training batch at the best checkpoint step. They are not validation-set averages "
        "or calibrated uncertainty estimates.",
        "",
        "## Interpretation constraints",
        "",
        "- The task is held-out next-character prediction, not a reasoning benchmark.",
        "- Learned branches are architectural components, not Monte Carlo samples.",
        "- Architectures are not compute matched; wall time is reported explicitly.",
        "- Tiny Shakespeare is small and stylistically narrow, so conclusions are preliminary.",
        "- Negative and near-zero findings should be reported rather than hidden.",
    ]

    if warnings:
        lines.extend(["", "## Consistency warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    if samples:
        lines.extend(
            [
                "",
                "## Deterministic qualitative samples",
                "",
                "Samples use greedy decoding (`temperature=0`) and are supplementary to BPC.",
            ]
        )
        for architecture, sample in samples.items():
            lines.extend(["", f"### {architecture}", "", "```text", sample, "```"])

    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FlowReasoning experiment checkpoints, verify matched settings, "
            "aggregate seeds, compute paired deltas, and write README-ready Markdown."
        )
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/application"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--generate-samples", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--sample-device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else runs_dir / "analysis"
    )
    repository_root = args.repository_root.expanduser().resolve()

    if args.sample_tokens < 1:
        raise ValueError("--sample-tokens must be positive")

    results, configs = discover_runs(runs_dir)
    warnings = validate_consistency(results, configs)
    aggregates = aggregate_runs(results)
    deltas = pairwise_deltas(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "per_run_results.csv", [asdict(row) for row in results])
    write_csv(
        output_dir / "aggregate_results.csv",
        [asdict(row) for row in aggregates],
    )
    write_csv(
        output_dir / "pairwise_deltas.csv",
        [asdict(row) for row in deltas],
    )

    samples = None
    if args.generate_samples:
        samples = generate_samples(
            results,
            repository_root=repository_root,
            prompt=args.prompt,
            tokens=args.sample_tokens,
            device=args.sample_device,
        )
        (output_dir / "samples.json").write_text(
            json.dumps(samples, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    markdown = build_results_markdown(results, aggregates, deltas, warnings, samples)
    (output_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")
    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "runs": [asdict(row) for row in results],
                "aggregates": [asdict(row) for row in aggregates],
                "pairwise_deltas": [asdict(row) for row in deltas],
                "warnings": warnings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nPer-run results\n===============")
    print(run_table_markdown(results))
    print("\nAggregate results\n=================")
    print(aggregate_table_markdown(aggregates))
    print("\nPaired architecture differences\n===============================")
    print(delta_table_markdown(deltas))

    if warnings:
        print("\nConsistency warnings\n====================")
        for warning in warnings:
            print(f"- {warning}")

    print(f"\nWrote analysis to: {output_dir}")
    if args.strict and warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
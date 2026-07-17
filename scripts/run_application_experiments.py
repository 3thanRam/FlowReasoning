from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    name: str
    seed: int
    flow_steps: int
    executor_mode: str
    num_paths: int


def experiments_for_mode(mode: str) -> list[Experiment]:
    seeds = {
        "screen": [11],
        "replicate": [22, 33],
        "all": [11, 22, 33],
    }[mode]

    experiments: list[Experiment] = []

    for seed in seeds:
        experiments.extend(
            [
                Experiment(
                    name="single-1",
                    seed=seed,
                    flow_steps=1,
                    executor_mode="single",
                    num_paths=1,
                ),
                Experiment(
                    name="single-4",
                    seed=seed,
                    flow_steps=4,
                    executor_mode="single",
                    num_paths=1,
                ),
                Experiment(
                    name="paths-4",
                    seed=seed,
                    flow_steps=4,
                    executor_mode="paths",
                    num_paths=4,
                ),
            ]
        )

    return experiments


def stream_process(
    command: Sequence[str],
    *,
    log_path: Path,
) -> int:
    """Run a command while writing output to both the console and a log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return process.wait()


def run_experiment(
    experiment: Experiment,
    *,
    corpus: Path,
    device: str,
    training_steps: int,
    batch_size: int,
    sequence_length: int,
) -> None:
    run_name = f"{experiment.name}-seed{experiment.seed}"

    run_directory = (
        REPOSITORY_ROOT
        / "runs"
        / "application"
        / run_name
    )

    log_path = (
        REPOSITORY_ROOT
        / "logs"
        / "application"
        / f"{run_name}.log"
    )

    run_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_directory / "flow_reasoning.pt"
    wall_time_path = run_directory / "wall_seconds.txt"

    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "main.py"),
        "train",
        "--data-path",
        str(corpus),
        "--device",
        device,
        "--seq-length",
        str(sequence_length),
        "--batch-size",
        str(batch_size),
        "--dim",
        "128",
        "--num-heads",
        "4",
        "--max-flow-steps",
        "16",
        "--training-steps",
        str(training_steps),
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.01",
        "--grad-clip",
        "1.0",
        "--dropout",
        "0.0",
        "--validation-fraction",
        "0.10",
        "--validation-seed",
        "10042",
        "--evaluation-interval",
        "250",
        "--evaluation-batches",
        "40",
        "--log-interval",
        "50",
        "--seed",
        str(experiment.seed),
        "--flow-steps",
        str(experiment.flow_steps),
        "--executor-mode",
        experiment.executor_mode,
        "--num-paths",
        str(experiment.num_paths),
        "--model-save-dir",
        str(checkpoint_path),
    ]

    print()
    print("=" * 60)
    print(f"Run:          {run_name}")
    print(f"Seed:         {experiment.seed}")
    print(f"Flow steps:   {experiment.flow_steps}")
    print(f"Mode:         {experiment.executor_mode}")
    print(f"Branches:     {experiment.num_paths}")
    print(f"Checkpoint:   {checkpoint_path}")
    print("=" * 60)

    started = time.perf_counter()

    return_code = stream_process(
        command,
        log_path=log_path,
    )

    elapsed = time.perf_counter() - started
    wall_time_path.write_text(
        f"{elapsed:.6f}\n",
        encoding="utf-8",
    )

    if return_code != 0:
        raise RuntimeError(
            f"Experiment {run_name} failed with exit code "
            f"{return_code}. See {log_path}."
        )

    print(
        f"Completed {run_name} in "
        f"{elapsed / 60:.1f} minutes.",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FlowReasoning application experiments "
            "with matched data and optimization settings."
        )
    )

    parser.add_argument(
        "mode",
        choices=("screen", "replicate", "all"),
        help=(
            "screen runs seed 11; replicate runs seeds 22 and 33; "
            "all runs seeds 11, 22, and 33."
        ),
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="UTF-8 text corpus used by every experiment.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--training-steps",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
    )

    return parser


def main() -> int:
    arguments = build_parser().parse_args()

    corpus = arguments.corpus.expanduser().resolve()

    if not corpus.is_file():
        raise FileNotFoundError(
            f"Corpus does not exist: {corpus}"
        )

    for experiment in experiments_for_mode(arguments.mode):
        run_experiment(
            experiment,
            corpus=corpus,
            device=arguments.device,
            training_steps=arguments.training_steps,
            batch_size=arguments.batch_size,
            sequence_length=arguments.seq_length,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
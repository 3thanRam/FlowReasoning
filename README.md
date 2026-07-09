# FlowReasoning

FlowReasoning is a small PyTorch prototype for language modelling with flow-like latent-space evolution.

The intended pipeline is:

```text
text -> tokens -> latent space -> evolved latent state -> output logits
```

The borrowed/incomplete pieces have been reorganised around `src/core.py`, which contains the latent dynamics engine. The project now includes a runnable character-level tokenizer, training loop, checkpoint format, and generation CLI.

## Structured reasoning loop

Use this loop when developing the project and reading experiment logs:

1. **Observation** — what is visible in the code, data, loss curve, or diagnostics?
2. **Assumptions and objectives** — what are we assuming, and what are we trying to make happen?
3. **Expectation vs reality** — what did we expect to happen, and what actually happened?
4. **Revision** — if expectation and reality differ, revise the assumption, objective, or experiment.

The training script prints logs in this style.

## Architecture

`FlowReasoningLM` in `src/model.py` implements:

1. Token ids are embedded with token and position embeddings.
2. The latent tensor `z0` is passed into `FlowEvolver`.
3. `FlowEvolver` applies repeated small latent updates using:
   - RMS normalisation
   - FFT-based sequence mixing
   - causal attention with rotary embeddings
   - diagonal unitary-style memory
   - a gated residual MLP
4. The evolved latent state is normalised and projected to next-token logits.

This is **normalising-flow-like** rather than a strict invertible normalising flow: the latent state evolves through controlled residual steps, but the current prototype does not compute exact inverse maps or log-determinants.

## Quick start

From the project root:

```bash
python main.py train --training-steps 20 --seq-length 64 --batch-size 8 --dim 64 --num-heads 4 --log-interval 5
python main.py generate --checkpoint checkpoints/flow_reasoning.pt --prompt "Observation:" --max-new-tokens 200
```

Train on your own text file:

```bash
python main.py train --data-path path/to/text.txt --training-steps 1000 --seq-length 128 --batch-size 16
```

Try multiple latent reasoning paths:

```bash
python main.py train --executor-mode paths --num-paths 4 --training-steps 200
```

## Files

- `config.py` — dataclass configuration and validation.
- `main.py` — command-line training and generation.
- `src/core.py` — latent flow/evolver components.
- `src/model.py` — text model wrapper around the latent flow.
- `src/data_loading.py` — character tokenizer and batch creation.
- `src/train.py` — training, checkpointing, and checkpoint loading.
- `src/utils.py` — seed, parameter count, diagnostics, and formatting helpers.

## Next revisions

Useful next experiments:

- Compare `--executor-mode single` against `--executor-mode paths --num-paths 4`.
- Add a validation split and plot loss curves.
- Add a true invertibility/log-det objective if exact normalising-flow training becomes the goal.
- Replace the character tokenizer with a BPE tokenizer once the latent flow is stable.

# FlowReasoning

An experimental character-level language model that treats depth as an iterative evolution of a latent state.

Most language models pass activations through a stack of distinct transformer blocks. FlowReasoning instead reuses one operator for several small updates:

```text
tokens -> embeddings -> z₀ -> z₁ -> ... -> zₜ -> next-token logits
```

The project is deliberately small enough to inspect and run on a CPU. It is a research prototype, not a claim that latent flow updates outperform a standard
transformer. Its purpose is to make the architecture easy to experiment with and to expose diagnostics for the iterative computation.

>**Scope:** The project name refers to iterative latent computation. The current task is next-character prediction and does not constitute a reasoning benchmark.

## What is implemented

Each update combines:

- causal self-attention with rotary position embeddings;
- an FFT-based causal sequence mixer;
- a diagonal complex-valued feature rotation;
- a gated SwiGLU residual branch; and
- a learned step size with bounded updates.

The same operator is reused at every flow step. In `paths` mode, several learned latent branches are initialized with distinct branch embeddings, evolved through the shared operator, and merged using learned sequence-level weights. These branches are architectural components rather than independent Monte Carlo samples.

> **Terminology:** "flow" describes the repeated, controlled evolution of the latent state. The model is not an invertible normalizing flow and does not compute a Jacobian determinant.

## Current status

The repository currently demonstrates that a shared latent operator can be applied repeatedly and that several learned latent branches can be aggregated while exposing branch-level diagnostics.

The included corpus is a smoke-test fixture. No conclusion about language-model
quality, reasoning ability, or the value of multiple branches is supported
without a held-out validation split and controlled baselines.

The immediate experimental question is:

> At a comparable training budget, do repeated shared latent updates or learned parallel branches improve held-out next-character prediction over a one-step shared-operator model?

## Quick start

FlowReasoning requires Python 3.10+ and PyTorch 2.1+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Short CPU smoke test
python main.py train \
  --training-steps 20 \
  --seq-length 64 \
  --batch-size 8 \
  --dim 64 \
  --num-heads 4

python main.py generate \
  --checkpoint checkpoints/flow_reasoning.pt \
  --prompt "The latent state"
```

The built-in corpus only exists to verify the training pipeline. For a meaningful
experiment, supply a UTF-8 text file:

```bash
python main.py train \
  --data-path data/corpus.txt \
  --training-steps 1000 \
  --seq-length 128 \
  --batch-size 16
```

To compare a single trajectory with four learned trajectories:

```bash
python main.py train --executor-mode paths --num-paths 4
```

## Architecture

```mermaid
flowchart LR
    A[Token and position embeddings] --> B[Latent state z₀]
    B --> C[Shared flow operator]
    C -->|bounded residual update| C
    C --> D[Final RMSNorm]
    D --> E[Tied vocabulary projection]
    E --> F[Next-token logits]
```

The main implementation lives in [`src/core.py`](src/core.py). `FlowEvolver`
owns the recurrent update, memory state, and optional path aggregation.
`FlowReasoningLM` in [`src/model.py`](src/model.py) provides the language-model
interface and autoregressive generation.

## Diagnostics

Training reports cross-entropy loss, perplexity, elapsed time, and latent-update
statistics. Path mode additionally records:

- normalized weight for each trajectory;
- variance between latent trajectories; and
- effective sample size, which indicates whether aggregation uses several paths
  or collapses onto one.

Checkpoints contain the model configuration, tokenizer vocabulary, parameters,
final loss, and the latest diagnostics. A checkpoint is therefore self-contained
for inference; reproducing training also requires the original corpus.

Branch variance and inverse weight concentration describe internal branch diversity. They are not calibrated uncertainty estimates.

## Development

Install the development extras and run the test suite:

```bash
pip install -e '.[dev]'
pytest
```

The tests cover tokenization, batching, both execution modes, configuration
validation, and checkpoint round-tripping.

## Current limitations

- Tokenization is character-level, so results are not comparable with modern
  subword language models.
- Generation recomputes the full context and does not use a KV cache.
- The included corpus is a smoke-test fixture, not a training dataset.
- There is no benchmark against a parameter-matched transformer yet.

These are the next useful directions: add a validation split, compare against a
small transformer baseline, and measure how path diversity changes during
training.

## License

[MIT](LICENSE)

# Preliminary Architecture Comparison

This is a 3-seed preliminary experiment for a compact character-level language model. It compares one shared latent update, four repeated shared updates, and four learned branches with four updates. Runs use the same corpus, split, fixed validation windows, optimizer settings, and training-step budget.

Lower validation bits per character (BPC) is better. The runs are matched by data and optimizer steps, not by wall-clock compute.

## Per-run results

| Run | Seed | Total instantiated parameters | Best step | Validation BPC | Validation loss | Wall time | Branch variance | Effective branches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-1-seed11 | 11 | 397,252 | 3000 | 3.1123 | 2.1573 | 0.2 min | - | - |
| single-1-seed22 | 22 | 397,252 | 3000 | 2.9545 | 2.0479 | 0.2 min | - | - |
| single-1-seed33 | 33 | 397,252 | 3000 | 2.9584 | 2.0506 | 0.2 min | - | - |
| single-4-seed11 | 11 | 397,252 | 3000 | 2.5021 | 1.7343 | 0.5 min | - | - |
| single-4-seed22 | 22 | 397,252 | 3000 | 2.4811 | 1.7198 | 0.5 min | - | - |
| single-4-seed33 | 33 | 397,252 | 3000 | 2.5191 | 1.7461 | 0.5 min | - | - |
| paths-4-seed11 | 11 | 397,252 | 3000 | 2.6926 | 1.8663 | 0.9 min | 0.84992 | 3.06 |
| paths-4-seed22 | 22 | 397,252 | 3000 | 2.6317 | 1.8242 | 0.9 min | 0.86397 | 3.09 |
| paths-4-seed33 | 33 | 397,252 | 3000 | 2.6869 | 1.8624 | 0.9 min | 0.87175 | 3.17 |

## Aggregate results

| Architecture | Flow steps | Branches | Seeds | Validation BPC | Validation loss | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| single-1 | 1 | 1 | 3 | 3.0084 +/- 0.0900 | 2.0853 +/- 0.0624 | 0.2 +/- 0.0 min |
| single-4 | 4 | 1 | 3 | 2.5008 +/- 0.0190 | 1.7334 +/- 0.0132 | 0.5 +/- 0.0 min |
| paths-4 | 4 | 4 | 3 | 2.6704 +/- 0.0336 | 1.8510 +/- 0.0233 | 0.9 +/- 0.0 min |

## Paired architecture differences

| Seed | Comparison | Candidate - baseline BPC | Candidate - baseline loss | Wall-time ratio |
|---:|---|---:|---:|---:|
| 11 | single-4 vs single-1 | -0.610201 | -0.422959 | 2.38 |
| 11 | paths-4 vs single-4 | +0.190453 | +0.132012 | 1.71 |
| 22 | single-4 vs single-1 | -0.473376 | -0.328119 | 2.37 |
| 22 | paths-4 vs single-4 | +0.150595 | +0.104385 | 1.79 |
| 33 | single-4 vs single-1 | -0.439245 | -0.304462 | 2.36 |
| 33 | paths-4 vs single-4 | +0.167751 | +0.116276 | 1.73 |

Negative candidate-minus-baseline BPC favours the candidate.

- **Repeated updates** (`single-4` minus `single-1`): validation BPC -0.507607 +/- 0.090472; validation loss -0.351847 +/- 0.062711; mean wall-time ratio 2.37x. Lower BPC is better, so this favours the candidate.
- **Learned branches** (`paths-4` minus `single-4`): validation BPC +0.169600 +/- 0.019993; validation loss +0.117558 +/- 0.013858; mean wall-time ratio 1.74x. Lower BPC is better, so this favours the baseline.

## Reproducibility metadata

- Training seeds: `11, 22, 33`
- Validation seed(s): `10042`
- Training token count(s): `1003854`
- Validation token count(s): `111540`
- Corpus SHA-256 value(s): `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`

The saved branch variance and effective branch count are diagnostics from the training batch at the best checkpoint step. They are not validation-set averages or calibrated uncertainty estimates.

## Interpretation constraints

- The task is held-out next-character prediction, not a reasoning benchmark.
- Learned branches are architectural components, not Monte Carlo samples.
- Architectures are not compute matched; wall time is reported explicitly.
- Tiny Shakespeare is small and stylistically narrow, so conclusions are preliminary.
- Negative and near-zero findings should be reported rather than hidden.

## Deterministic qualitative samples

Samples use greedy decoding (`temperature=0`) and are supplementary to BPC.

### single-1

```text
ROMEO:
And the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the hear the heard,
And the the heare he the the the hee the the the the the the the the the the the the the the the the the the the the the the the the the th
```

### single-4

```text
ROMEO:
I will may the sent the sent the sent the stries
And the sent the sent the sent the sent the sent.

DUKE OF YORK:
And the with the will be soul be read the sent.

KING RICHARD III:
I will the sentle the shall be reatine.

Disher:
And the with the shall be not be standerengreet,
And the shall be sta
```

### paths-4

```text
ROMEO:
It the shall the shall the shall the shall
The shall the shall the shall the shall the shall
The shall the shall the shall the shall the shall the shall
The shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall the shall
The shall the shall the sh
```

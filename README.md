# causal-certificate

[![CI](https://github.com/Akhilesh-Gogikar/causal-certificate/actions/workflows/ci.yml/badge.svg)](https://github.com/Akhilesh-Gogikar/causal-certificate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/causal-certificate.svg)](https://pypi.org/project/causal-certificate/)
[![Python versions](https://img.shields.io/pypi/pyversions/causal-certificate.svg)](https://pypi.org/project/causal-certificate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, numeric **strict-causality certificate** for PyTorch sequence models.
One forward pass + `T−1` vector-Jacobian products tells you whether any output at
position `t` depends on an input at position `s > t` — the silent bug that
manufactures phantom autoregressive results.

```python
from causal_certificate import certify, assert_strictly_causal

report = certify(my_mixer, x)         # x: (B, T, D) float; my_mixer: x -> y
print(report.summary())
# CausalCertificate(T=128, exhaustive cuts)
#   temporal   : leak=0.000e+00  frac=0.000e+00  -> STRICTLY CAUSAL
#   cross-batch: leak=0.000e+00  frac=0.000e+00  -> BATCH-INDEPENDENT

assert_strictly_causal(my_mixer, x)   # drop into a pytest
```

## Why

A "blockwise causal" Walsh–Hadamard token mixer once produced a **7.21× lower BPB**
than a matched transformer — a number that reached a provisional patent application
before it was found to be a within-block future-token leak. Standard sanity checks
passed because they probed *across-block* causality; the violation lived *inside*
blocks. This tool is the check that would have caught it in CI. (Case study &
full write-up: `papers/causality_leaks/`.)

## What it catches (three leak classes)

| Class | Example | Caught by |
|---|---|---|
| Temporal (position) leak | block-WHT / FFT / butterfly mixing; off-by-one causal masks; KV/RoPE drift | exhaustive-cut temporal certificate |
| Pooled/block readout | a block statistic broadcast back to every position | temporal certificate |
| Batch/sequence-statistic coupling | batchnorm-style couplings across the batch | **cross-batch** certificate (per-example probes are blind to it) |

Genuinely-causal ops certify at **exactly 0.0** (structural autograd zeros) — in
fp32 as well as fp64 — so there is no per-model threshold tuning. Validated on
external attention/conv models it never saw: causal MHA and causal conv → 0.0;
an injected off-by-one mask and a batchnorm coupling → flagged.

## Modes

- `certify(fn, x, cuts="all")` — the **certificate** (exhaustive cuts, complete).
- `certify(fn, x, cuts="rand", K=8)` — a cheap **always-on training monitor**;
  a single-pair leak is caught with probability `1 − (1 − 1/(T−1))^K`.
- `batch_check=True` (default when `B>1`) — adds the cross-batch certificate.

## Scope & honest attribution

This is a *numeric* certificate on a given architecture/config (generic inputs and
random cotangents), not a symbolic proof; detection is almost-sure, not worst-case
adversarial; it assumes equal input/output sequence length. The **method is not
novel** — it packages known probes: Karpathy's 2019 backprop-from-`t` temporal
check, the per-cut VJP gradient energy of *Effective Context in Neural Speech
Models* (arXiv:2505.22487), and Krokotsch's 2020 batch-independence unit test. The
contribution is the **packaging**: exhaustive-cut completeness + the cross-batch
extension, as a single drop-in certificate for sequence-model CI.

## Install & test

```bash
pip install -e .            # editable install (src layout)
pytest                      # or: python tests/test_external_models.py
```
The generalization test certifies external causal attention / conv at exactly 0.0,
and fires on an injected off-by-one mask and a batch-statistic coupling.

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders a
"Cite this repository" button from it). Once the repository is archived on Zenodo,
add the DOI badge here and cite the versioned DOI. Zenodo/citation metadata for the
archive lives in [`.zenodo.json`](.zenodo.json).

<!-- After the first Zenodo release, paste the badge Zenodo shows for this repo, e.g.:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

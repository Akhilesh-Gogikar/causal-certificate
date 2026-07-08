# causal-certificate

[![CI](https://github.com/Akhilesh-Gogikar/causal-certificate/actions/workflows/ci.yml/badge.svg)](https://github.com/Akhilesh-Gogikar/causal-certificate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/causal-certificate.svg)](https://pypi.org/project/causal-certificate/)
[![Python versions](https://img.shields.io/pypi/pyversions/causal-certificate.svg)](https://pypi.org/project/causal-certificate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1287471007.svg)](https://doi.org/10.5281/zenodo.21151512)

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
full write-up: [`paper/`](paper/).)

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
- `certify_by_perturbation(fn, x)` — the **elementary black-box variant**: perturb a
  future input, check that earlier outputs don't move. No autograd needed (works on
  non-differentiable models); `1 + (T−1)` forward passes. Same verdicts as the VJP
  certificate, and it's the general form of the paper's "Test A" leak detector.

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

## Paper

The companion negative-results paper documenting the case study lives in
[`paper/`](paper/):

- [`paper/main.pdf`](paper/main.pdf) — compiled paper
- [`paper/main.tex`](paper/main.tex) — LaTeX source
- [`paper/arxiv_submission.tar.gz`](paper/arxiv_submission.tar.gz) — arXiv-ready tarball

It forensically traces how a within-block causality leak in a blockwise
Walsh--Hadamard mixer (AKI/SKC) manufactured a 7.2× bits-per-byte "win",
why standard sanity checks missed it, and presents the perturbation-based
strict-causality certificate implemented here as the paper's central original
contribution.

## Citation

Archived on Zenodo. Cite the **concept DOI** (always resolves to the latest version):

> **DOI: [10.5281/zenodo.21151512](https://doi.org/10.5281/zenodo.21151512)**  ·  v0.1.0: `10.5281/zenodo.21151513`

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders a
"Cite this repository" button from it) and [`.zenodo.json`](.zenodo.json).

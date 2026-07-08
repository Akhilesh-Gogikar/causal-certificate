# WHT In-Block Causality Leak — Forensic Findings (regenerated)

**Function under test:** `src/akilm/v2/train/submodules/model_blocks.py::causal_wht_blockwise`
(extracted verbatim and exec'd with `HAS_TRITON=False`; the Triton path
`kernels/submodules/fwht.py` computes the same dense per-block `H @ x`).

**Harness:** `confirm_wht_unsalvageable.py` · torch 2.2.2, CPU, seed-pinned · output `report.json`.

## Test A — future perturbation moves earlier outputs

Perturb in-block position t′=40; measure max |Δ output| at positions t < 40.

| variant | max backward Δ |
|---|---|
| repo `causal_wht_blockwise` | **0.125** |
| lower-triangular-masked block operator (only strict-causal fix) | **0.0** |

The transform is causal *between* 64-position blocks but dense *within* each block:
output at position t mixes inputs at t′ > t.

## Test B — i.i.d. oracle: the leak is exploitable by training

Next-token model on i.i.d. uniform tokens, V=16 ⇒ strict-causal floor = log2(16) = **4.0 BPB**.
Mixer mirrors the real SKC layer shape (WHT → per-position diagonal → WHT, cf. the
`s_wht`/`synth_wht` call sites). `H·diag(s)·H` is a dyadic (XOR) convolution, which contains
the permutation t ↔ t⊕1 — a literal read of the *next* token at every even position.

| mixer | val BPB |
|---|---|
| leaky repo WHT | **2.0721** (≈ predicted 2.0: 0 bits at even positions, 4 bits at odd) |
| tril-masked WHT | 4.0021 |
| no WHT (diag only) | 4.0029 |

## Verdict

- Any below-floor result from a model containing this mixer (incl. the provisional-patent
  Table 1 headline, V1 Associative-SKC BPB 0.2455 vs 1.7712, "7.2× lower") is explained by
  the leak, not by the architecture.
- The only strictly-causal fix — masking the future (upper-triangular) entries of the block
  matrix — destroys the Hadamard structure and with it the entire claimed advantage
  (`docs/contribution_fwht_causality.md`; matched clean re-runs were ~+0.09–0.24 BPB *worse*,
  see `docs/architecture/POST_MORTEM_WHY_NON_SOTA.md`).
- Note: this regenerated harness reports the exploit at BPB 2.07 (single spectral-diag layer,
  half the positions readable). Deeper stacks of the same primitive can reach ≈0 BPB; the
  earlier (lost) harness recorded 0.0000. Either number is far below the 4.0 floor — the
  qualitative verdict is identical and now reproducible in-repo.

Reproduce: `python3 experiments/wht_causality/confirm_wht_unsalvageable.py` (~35 s CPU).

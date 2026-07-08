#!/usr/bin/env python3
"""Forensic confirmation that `causal_wht_blockwise` leaks future in-block information.

The function under test is NOT re-implemented here. Its source is extracted verbatim
from src/akilm/v2/train/submodules/model_blocks.py and exec'd with HAS_TRITON=False,
so the CPU reference path (the same math the Triton kernel computes) is what is tested.

Test A (perturbation probe): perturb a *future* position t' inside a block and measure
the change at an *earlier* output position t < t'. Any nonzero change is a causality
violation. Repeated with a strictly-causal (lower-triangular-masked) version of the
same block operator, which must show zero backward propagation.

Test B (i.i.d. oracle test): train a tiny next-token model over i.i.d. uniform tokens
(V=16, so the strict-causal BPB floor is log2(16) = 4.0 bits). A mixer that can see
the future reads the label directly and drives BPB toward 0. Strictly-causal variants
must stay pinned at ~4.0. Run: python3 confirm_wht_unsalvageable.py
"""
from __future__ import annotations

import json
import math
import pathlib
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODEL_BLOCKS = REPO / "src/akilm/v2/train/submodules/model_blocks.py"

torch.manual_seed(0)


def load_function_under_test():
    """Extract causal_wht_blockwise verbatim from the repo source and exec it."""
    src = MODEL_BLOCKS.read_text()
    m = re.search(r"^def causal_wht_blockwise\(.*?(?=^\S)", src, re.M | re.S)
    if m is None:
        raise RuntimeError(f"causal_wht_blockwise not found in {MODEL_BLOCKS}")
    ns = {"torch": torch, "F": F, "math": math, "HAS_TRITON": False}
    exec(compile(m.group(0), str(MODEL_BLOCKS), "exec"), ns)
    return ns["causal_wht_blockwise"], m.group(0)


CAUSAL_WHT, FUT_SOURCE = load_function_under_test()
BLOCK = 64


def block_matrix(block_size: int = BLOCK) -> torch.Tensor:
    """The linear operator M (positions x positions) applied within each block."""
    # Feed the identity along the position axis, one feature channel per basis
    # vector: x[0, t, j] = delta(t, j). The WHT mixes positions independently per
    # channel, so out[0, t, j] = M[t, j] = response at position t to input at j.
    eye = torch.eye(block_size).unsqueeze(0)  # (1, T, D=T)
    return CAUSAL_WHT(eye)[0]


def strict_causal_variant(x: torch.Tensor) -> torch.Tensor:
    """The only strictly-causal fix: mask future (upper-triangular) entries of M."""
    B, T, D = x.shape
    pad = (BLOCK - T % BLOCK) % BLOCK
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    M = block_matrix(BLOCK)                      # (T_out, T_in)
    M = torch.tril(M)                            # kill t' > t
    xb = x.view(B, -1, BLOCK, D)
    yb = torch.einsum("st,bnto->bnso", M, xb)
    return yb.reshape(B, -1, D)[:, :T, :]


def test_a() -> dict:
    B, T, D = 1, BLOCK, 8
    t_perturb, t_read = 40, 10  # future -> earlier
    x = torch.randn(B, T, D)
    x2 = x.clone()
    x2[0, t_perturb] += 1.0
    results = {}
    for name, fn in (("leaky_repo_fn", CAUSAL_WHT), ("tril_fix", strict_causal_variant)):
        y, y2 = fn(x), fn(x2)
        delta_earlier = (y2[0, :t_perturb] - y[0, :t_perturb]).abs().max().item()
        results[name] = round(delta_earlier, 6)
    results["violation"] = results["leaky_repo_fn"] > 0
    return results


class TinyNextToken(nn.Module):
    """Embed -> WHT -> per-position diagonal -> WHT -> head.

    This mirrors the real SKC layer shape (WHT into the spectral domain,
    elementwise ops, WHT back out — see s_wht/synth_wht call sites in
    model_blocks.py). H diag(s) H is a dyadic (XOR) convolution; with
    s = WHT(delta_1) it computes y[t] = x[t XOR 1], i.e. reads the NEXT
    token at every even position. A strictly-causal mixer cannot express
    this, so its BPB floor on i.i.d. tokens stays at log2(V)."""

    def __init__(self, mixer: str, vocab: int = 16, d: int = 32, T: int = BLOCK):
        super().__init__()
        self.mixer = mixer
        self.emb = nn.Embedding(vocab, d)
        self.spec_gain = nn.Parameter(torch.ones(T, 1))
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.emb(tokens)
        if self.mixer == "leaky_wht":
            y = CAUSAL_WHT(self.spec_gain * CAUSAL_WHT(h, BLOCK), BLOCK)
        elif self.mixer == "tril_wht":
            y = strict_causal_variant(self.spec_gain * strict_causal_variant(h))
        elif self.mixer == "no_wht":
            y = self.spec_gain * h
        else:
            raise ValueError(self.mixer)
        return self.head(h + y)


def test_b(steps: int = 1500, batch: int = 64, T: int = BLOCK, vocab: int = 16) -> dict:
    out = {}
    for mixer in ("leaky_wht", "tril_wht", "no_wht"):
        torch.manual_seed(1234)
        model = TinyNextToken(mixer, vocab=vocab)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(steps):
            toks = torch.randint(0, vocab, (batch, T + 1))
            logits = model(toks[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, vocab), toks[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            toks = torch.randint(0, vocab, (4096, T + 1))
            logits = model(toks[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, vocab), toks[:, 1:].reshape(-1))
        out[mixer] = round(ce.item() / math.log(2), 4)  # bits per token
    out["floor_log2V"] = round(math.log2(vocab), 4)
    return out


if __name__ == "__main__":
    report = {
        "function_under_test": f"{MODEL_BLOCKS.relative_to(REPO)}::causal_wht_blockwise",
        "torch": torch.__version__,
        "test_a_future_perturbation_moves_earlier_output": test_a(),
        "test_b_iid_oracle_bpb": test_b(),
    }
    print(json.dumps(report, indent=2))
    (HERE / "report.json").write_text(json.dumps(report, indent=2))

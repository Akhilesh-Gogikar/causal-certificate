"""
Generalization de-risk for causal_certificate: does it work on models it never
saw, WITHOUT false-positiving on genuinely-causal ops under fp noise?

Runs standalone (`python -m causal_certificate.tests.test_external_models`) and
under pytest. External fixtures = hand-built multi-head attention and a causal
conv — nothing from the AkiLM WHT harness.
"""
import math
import pathlib
import sys
import torch
import torch.nn.functional as F

# run from source without installing: add the src/ dir to the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from causal_certificate import certify  # noqa: E402

DT = torch.float64  # certificate is dtype-agnostic; fp32 case tested separately below


def _mha(x, Wqkv, nh, mask):
    """Standard multi-head attention with an explicit boolean block-mask (True=blocked)."""
    B, T, D = x.shape
    dh = D // nh
    q, k, v = (x @ Wqkv).chunk(3, dim=-1)
    q = q.view(B, T, nh, dh).transpose(1, 2)
    k = k.view(B, T, nh, dh).transpose(1, 2)
    v = v.view(B, T, nh, dh).transpose(1, 2)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    logits = logits.masked_fill(mask, float("-inf"))
    o = torch.softmax(logits, dim=-1) @ v
    return o.transpose(1, 2).reshape(B, T, D)


def _make(D=32, nh=4, dtype=DT, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, 3 * D, dtype=dtype, generator=g) / math.sqrt(D), nh


def causal_conv(x, kernel):  # left-padded depthwise conv: strictly causal by construction
    B, T, D = x.shape
    k = kernel.shape[0]
    xp = F.pad(x.transpose(1, 2), (k - 1, 0))
    w = kernel.view(1, 1, k).expand(D, 1, k)
    return F.conv1d(xp, w, groups=D).transpose(1, 2)


def run():
    B, T, D = 3, 24, 32
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, D, dtype=DT, generator=g)
    Wqkv, nh = _make(D)
    tri1 = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # block j>i (causal)
    tri2 = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=2)   # block j>i+1 (off-by-one leak)
    nomask = torch.zeros(T, T, dtype=torch.bool)                        # bidirectional

    checks = []

    def check(name, rep, want_causal):
        fired = not rep.is_strictly_causal
        ok = (fired == (not want_causal))
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} frac={rep.temporal_fraction:.3e} "
              f"-> {'clean' if rep.is_strictly_causal else 'LEAK'}")

    print("External-model generalization (fp64):")
    check("causal MHA (-inf mask)", certify(lambda z: _mha(z, Wqkv, nh, tri1), x), True)
    check("causal depthwise conv", certify(lambda z: causal_conv(z, torch.randn(4, dtype=DT, generator=g)), x), True)
    check("off-by-one MHA (leaks 1 future tok)", certify(lambda z: _mha(z, Wqkv, nh, tri2), x), False)
    check("bidirectional MHA (no mask)", certify(lambda z: _mha(z, Wqkv, nh, nomask), x), False)

    # kill-criterion (a): finite mask + fp32 must NOT false-positive on a genuinely-causal model
    print("False-positive stress (fp32, finite -1e9 mask):")
    x32 = x.float()
    Wqkv32 = Wqkv.float()

    def mha_finite(z):
        B, T, D = z.shape
        dh = D // nh
        q, k, v = (z @ Wqkv32).chunk(3, dim=-1)
        q = q.view(B, T, nh, dh).transpose(1, 2)
        k = k.view(B, T, nh, dh).transpose(1, 2)
        v = v.view(B, T, nh, dh).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        logits = logits + tri1.to(logits.dtype) * (-1e9)     # finite additive mask
        return (torch.softmax(logits, -1) @ v).transpose(1, 2).reshape(B, T, D)

    rep32 = certify(mha_finite, x32, threshold=1e-9)
    print(f"  fp32 finite-mask causal MHA: frac={rep32.temporal_fraction:.3e} "
          f"leak={rep32.temporal_leak:.3e} -> {'clean' if rep32.is_strictly_causal else 'FALSE POSITIVE'}")
    checks.append(("fp32 finite-mask no false-positive", rep32.is_strictly_causal))

    # cross-batch: batchnorm-style coupling — clean temporally, leaks across the batch
    print("Cross-batch (batch-statistic coupling):")
    rep_bn = certify(lambda z: z - z.mean(dim=0, keepdim=True), x, threshold=1e-9)
    ok_bn = rep_bn.is_strictly_causal and (not rep_bn.is_batch_independent)
    checks.append(("batchnorm: temporal-clean + batch-LEAK", ok_bn))
    print(f"  [{'PASS' if ok_bn else 'FAIL'}] batch-mean: temporal frac={rep_bn.temporal_fraction:.2e} "
          f"(clean), cross-batch frac={rep_bn.crossbatch_fraction:.3f} (leak)")

    print()
    go = all(ok for _, ok in checks)
    print(f"GENERALIZATION: {'GO — works on unseen models, no fp false-positives.' if go else 'NO-GO'}")
    return go


def test_external_models():
    assert run()


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

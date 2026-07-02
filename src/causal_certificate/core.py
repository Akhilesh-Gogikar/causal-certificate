"""
Causal-Certificate — a numeric certificate of strict causality (and cross-example
independence) for PyTorch sequence models, via vector-Jacobian products.

Generalized from experiments/wht_causality/clc_certificate.py. Torch-only.

Strict (position) causality of a mixer y = f(x):   dy_t/dx_s = 0  for all s > t.
We certify it without materializing the O(T^2) Jacobian:

  * TEMPORAL certificate: for a cut tau, probe the outputs at positions t < tau
    with a random Gaussian cotangent and measure the gradient energy that lands
    on inputs at positions s >= tau. Zero (a.s. in the cotangent) iff no
    future->past dependence crosses that cut. Sweeping every cut tau in 1..T-1
    is COMPLETE: any leaky pair (t < s) is separated by the cut tau = s.
    Cost: 1 forward + (T-1) VJPs. Structurally-causal ops certify at EXACTLY 0.

  * CROSS-BATCH certificate: probe example 0's outputs, measure gradient energy
    on inputs of other examples. Catches batch-statistic couplings (batchnorm-
    style) that per-example probes are structurally blind to.

  * random-K mode: K random cuts as a cheap always-on training monitor; a
    single-pair leak is caught with prob 1 - (1 - 1/(T-1))^K.

This is a numeric certificate on a given architecture/config (generic inputs and
cotangents), not a symbolic proof. Prior probes it packages: Karpathy's 2019
temporal backprop check, the per-cut VJP energy of arXiv:2505.22487, and
Krokotsch's 2020 batch-independence test. The contribution is the packaging:
exhaustive-cut completeness + cross-batch, as a drop-in certificate.
"""

from dataclasses import dataclass, field
import torch


def _axis_slice(t, dim, lo=None, hi=None):
    idx = [slice(None)] * t.dim()
    idx[dim] = slice(lo, hi)
    return tuple(idx)


@dataclass
class LeakReport:
    T: int
    temporal_leak: float
    temporal_total: float
    crossbatch_leak: float = 0.0
    crossbatch_total: float = 0.0
    cuts_probed: int = 0
    exhaustive: bool = True
    batch_checked: bool = False
    threshold: float = 1e-9
    extra: dict = field(default_factory=dict)

    @property
    def temporal_fraction(self):
        return self.temporal_leak / self.temporal_total if self.temporal_total > 0 else 0.0

    @property
    def crossbatch_fraction(self):
        return self.crossbatch_leak / self.crossbatch_total if self.crossbatch_total > 0 else 0.0

    @property
    def is_strictly_causal(self):
        return self.temporal_fraction <= self.threshold

    @property
    def is_batch_independent(self):
        return (not self.batch_checked) or (self.crossbatch_fraction <= self.threshold)

    @property
    def ok(self):
        return self.is_strictly_causal and self.is_batch_independent

    def summary(self):
        mode = "exhaustive" if self.exhaustive else f"random-K={self.cuts_probed}"
        lines = [
            f"CausalCertificate(T={self.T}, {mode} cuts)",
            f"  temporal   : leak={self.temporal_leak:.3e}  frac={self.temporal_fraction:.3e}  "
            f"-> {'STRICTLY CAUSAL' if self.is_strictly_causal else 'LEAK'}",
        ]
        if self.batch_checked:
            lines.append(
                f"  cross-batch: leak={self.crossbatch_leak:.3e}  frac={self.crossbatch_fraction:.3e}  "
                f"-> {'BATCH-INDEPENDENT' if self.is_batch_independent else 'BATCH LEAK'}")
        if not self.exhaustive:
            lines.append("  (random-K is a monitor; use cuts='all' for a certificate)")
        return "\n".join(lines)


def certify(fn, x, *, cuts="all", K=8, batch_check=True, seq_dim=1, batch_dim=0,
            threshold=1e-9, generator=None):
    """Certify strict causality of `fn` at input `x`.

    fn : callable / nn.Module mapping a float tensor x -> y. Certify at a
         differentiable interface (embeddings / hidden states), not token ids.
         y must share the sequence axis length with x along `seq_dim`.
    x  : float tensor, shape (batch, T, ...) by default (batch_dim=0, seq_dim=1).
    cuts : "all" (exhaustive certificate) or "rand" (K random-cut monitor).
    Returns a LeakReport. `report.is_strictly_causal` is True iff no output at
    t<tau ever depends on an input at s>=tau (a.s.); clean ops report EXACTLY 0.
    """
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    if not torch.is_tensor(y):
        raise TypeError("fn must return a single tensor; wrap multi-output models.")
    if y.shape[seq_dim] != x.shape[seq_dim]:
        raise ValueError(
            f"certify assumes equal in/out sequence length along seq_dim={seq_dim}; "
            f"got x:{x.shape[seq_dim]} y:{y.shape[seq_dim]} (length-changing mixers unsupported).")
    T = y.shape[seq_dim]
    if T < 2:
        raise ValueError("need T >= 2 to probe a temporal cut.")

    taus = (list(range(1, T)) if cuts == "all"
            else [int(torch.randint(1, T, (1,), generator=generator)) for _ in range(K)])

    def randn_like(t):
        return torch.randn(t.shape, dtype=t.dtype, generator=generator)

    t_leak = t_tot = 0.0
    for tau in taus:
        v = randn_like(y)
        v[_axis_slice(v, seq_dim, tau, None)] = 0.0            # probe outputs at t < tau
        (g,) = torch.autograd.grad((y * v).sum(), x, retain_graph=True)
        t_leak += g[_axis_slice(g, seq_dim, tau, None)].pow(2).sum().item()   # energy on s >= tau
        t_tot += g.pow(2).sum().item()

    xb_leak = xb_tot = 0.0
    do_batch = batch_check and x.shape[batch_dim] > 1
    if do_batch:
        for _ in range(K):
            v = randn_like(y)
            keep = torch.zeros_like(v)
            keep[_axis_slice(keep, batch_dim, 0, 1)] = 1.0     # probe example 0 only
            (g,) = torch.autograd.grad((y * (v * keep)).sum(), x, retain_graph=True)
            g_all = g.pow(2).sum().item()
            g_self = g[_axis_slice(g, batch_dim, 0, 1)].pow(2).sum().item()
            xb_leak += g_all - g_self                          # energy on examples != 0
            xb_tot += g_all

    return LeakReport(
        T=T, temporal_leak=t_leak, temporal_total=t_tot,
        crossbatch_leak=xb_leak, crossbatch_total=xb_tot,
        cuts_probed=len(taus), exhaustive=(cuts == "all"),
        batch_checked=do_batch, threshold=threshold,
    )


def assert_strictly_causal(fn, x, *, threshold=1e-9, **kw):
    """pytest-friendly: raises AssertionError with the leak report if fn leaks."""
    rep = certify(fn, x, threshold=threshold, **kw)
    assert rep.ok, "\n" + rep.summary()
    return rep

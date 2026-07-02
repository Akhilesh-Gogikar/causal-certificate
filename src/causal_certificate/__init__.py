"""Causal-Certificate: a numeric strict-causality certificate for PyTorch sequence models.

    from causal_certificate import certify, assert_strictly_causal
    report = certify(my_mixer, x)          # x: (B, T, D) float
    print(report.summary())                # is_strictly_causal / is_batch_independent
    assert_strictly_causal(my_mixer, x)    # pytest one-liner
"""
from .core import certify, assert_strictly_causal, LeakReport

__all__ = ["certify", "assert_strictly_causal", "LeakReport"]
__version__ = "0.1.0"

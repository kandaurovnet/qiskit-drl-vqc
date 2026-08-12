#!/usr/bin/env python3
"""Contract check for build_q_network(). Run: python test_interface.py

The quantum team's network must pass this before it will work in the training
loop. If this passes, the drop-in is safe; nothing else in cartpole_dqn.py
needs to change.
"""

import sys

import torch

from cartpole_dqn import build_q_network, normalize_obs
import numpy as np


def check_network(net, name):
    for batch in (1, 128):
        x = torch.rand(batch, 4) * 2 - 1  # normalized obs live in [-1, 1]
        y = net(x)
        assert y.shape == (batch, 2), f"{name}: expected ({batch}, 2), got {tuple(y.shape)}"
        assert torch.isfinite(y).all(), f"{name}: produced NaN/inf"

    # Training loop calls .backward() through this; params must receive grads.
    y = net(torch.rand(4, 4) * 2 - 1)
    y.sum().backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert grads, f"{name}: no trainable parameters"
    assert any(g is not None and torch.isfinite(g).all() for g in grads), \
        f"{name}: no finite gradients reached parameters"
    print(f"  {name}: OK ({sum(p.numel() for p in net.parameters())} params)")


def check_normalize():
    # Velocities are unbounded in CartPole; normalize_obs must still clip to [-1, 1].
    extreme = np.array([100.0, -100.0, 100.0, -100.0], dtype=np.float32)
    out = normalize_obs(extreme)
    assert out.shape == (4,), f"expected shape (4,), got {out.shape}"
    assert np.all(np.abs(out) <= 1.0 + 1e-6), f"not bounded to [-1,1]: {out}"
    assert out.dtype == np.float32, f"expected float32, got {out.dtype}"
    print("  normalize_obs: OK (clips unbounded velocities to [-1, 1])")


if __name__ == "__main__":
    print("Checking observation normalization...")
    check_normalize()

    print("Checking Q-network interface...")
    check_network(build_q_network("classical"), "classical")

    try:
        check_network(build_q_network("quantum"), "quantum")
    except NotImplementedError:
        print("  quantum: not implemented yet (expected until the QNN lands)")
    except Exception as e:
        print(f"  quantum: FAILED -- {e}")
        sys.exit(1)

    print("All contract checks passed.")

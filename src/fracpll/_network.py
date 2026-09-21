"""State-space and modal form of the passive loop filter (internal).

The filter of `fracpll.filters` -- a shunt capacitor C0 at the
control node with series R_k-C_k branches to ground -- has capacitor
voltages x = [v_c, v_1, ..., v_n] as its natural state:

    C0 dv_c/dt = i(t) - sum_k (v_c - v_k)/R_k
    C_k dv_k/dt = (v_c - v_k)/R_k.

Written as C dx/dt = -G x + e0 i with C = diag(C0, C_1, ...) and G the
(symmetric, positive semi-definite) conductance Laplacian of the star
network, the symmetric transform y = C^(1/2) x diagonalises it
exactly: S = C^(-1/2) G C^(-1/2) = Q diag(mu) Q^T with real decay
rates mu_j >= 0.  The network is connected, so exactly one rate is
zero: the total charge sum_k C_k v_k is conserved when no current
flows.  That mode is set to zero EXACTLY (its eigenvector is known in
closed form, C^(1/2) 1 / ||.||), not taken from the numerical
eigen-solver, so the integrating behaviour of the loop is exact.

With the pump current held constant over an interval (true between
the edges of an event-driven simulation), each mode then has a closed
form, and so do the control voltage and its time integral -- which is
what makes the edge-level simulator of `fracpll.eventsim` exact for a
linear VCO rather than time-stepped.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = []


class FilterModes:
    """Modal closed forms of the shunt-C + series-RC loop filter."""

    def __init__(self, c_shunt, branches=()):
        c0 = float(c_shunt)
        if not (np.isfinite(c0) and c0 > 0.0):
            raise ValueError("c_shunt must be finite and positive")
        caps = [c0]
        cond = []
        for k, (r, c) in enumerate(branches):
            r, c = float(r), float(c)
            if not (np.isfinite(r) and r > 0.0):
                raise ValueError(
                    f"branch {k}: R must be finite and > 0 (an R = 0 "
                    "branch is the same node: add its C to c_shunt)")
            if not (np.isfinite(c) and c > 0.0):
                raise ValueError(f"branch {k}: C must be finite and > 0")
            caps.append(c)
            cond.append(1.0 / r)
        n = len(caps)
        g = np.zeros((n, n))
        for k, gk in enumerate(cond, start=1):
            g[0, 0] += gk
            g[k, k] += gk
            g[0, k] -= gk
            g[k, 0] -= gk
        cap = np.array(caps)
        s_half = 1.0 / np.sqrt(cap)
        s = s_half[:, None] * g * s_half[None, :]
        mu, q = np.linalg.eigh(s)
        # the exact zero (charge-conservation) mode
        z = np.sqrt(cap) / np.linalg.norm(np.sqrt(cap))
        i0 = int(np.argmin(np.abs(mu)))
        mu[i0] = 0.0
        # re-orthogonalise the others against the exact zero mode
        q[:, i0] = z
        for j in range(n):
            if j == i0:
                continue
            v = q[:, j] - (q[:, j] @ z) * z
            q[:, j] = v / np.linalg.norm(v)
        self.caps = cap
        self.g = g
        self.mu = [float(m) for m in mu]
        self.q = q
        # input: current into node 0 -> dy0/dt += i / sqrt(C0)
        self.beta = [float(q[0, j] / math.sqrt(c0)) for j in range(n)]
        # output: v_c = x0 = y0 / sqrt(C0)
        self.gamma = [float(q[0, j] / math.sqrt(c0)) for j in range(n)]
        self.n = n
        self.zero_mode = i0

    # -- conversions ------------------------------------------------
    def modal_from_voltages(self, x):
        y = np.sqrt(self.caps) * np.asarray(x, dtype=float)
        return [float(v) for v in self.q.T @ y]

    def voltages_from_modal(self, w):
        y = self.q @ np.asarray(w, dtype=float)
        return y / np.sqrt(self.caps)

    def state_matrices(self):
        """(A, b, c) of dx/dt = A x + b i, v_c = c x."""
        cinv = 1.0 / self.caps
        a = -cinv[:, None] * self.g
        b = np.zeros(self.n)
        b[0] = 1.0 / self.caps[0]
        c = np.zeros(self.n)
        c[0] = 1.0
        return a, b, c

    # -- closed forms under constant current i over [0, t] ----------
    def advance(self, w, i, t):
        out = []
        for mu, wj, bj in zip(self.mu, w, self.beta):
            if mu == 0.0:
                out.append(wj + bj * i * t)
            else:
                e = math.exp(-mu * t)
                out.append(wj * e + bj * i * (-math.expm1(-mu * t)) / mu)
        return out

    def vc(self, w):
        return sum(g * wj for g, wj in zip(self.gamma, w))

    def vc_at(self, w, i, t):
        s = 0.0
        for mu, wj, bj, g in zip(self.mu, w, self.beta, self.gamma):
            if mu == 0.0:
                s += g * (wj + bj * i * t)
            else:
                s += g * (wj * math.exp(-mu * t)
                          + bj * i * (-math.expm1(-mu * t)) / mu)
        return s

    def vc_integral(self, w, i, t):
        """int_0^t v_c(s) ds, closed form."""
        s = 0.0
        for mu, wj, bj, g in zip(self.mu, w, self.beta, self.gamma):
            if mu == 0.0:
                s += g * (wj * t + 0.5 * bj * i * t * t)
            else:
                x = mu * t
                one_m_e = -math.expm1(-x)          # 1 - e^-x
                if x < 1e-4:
                    # (x - (1 - e^-x)) / mu^2 = t^2 (1/2 - x/6 + x^2/24)
                    rem = t * t * (0.5 - x / 6.0 + x * x / 24.0)
                else:
                    rem = (x - one_m_e) / (mu * mu)
                s += g * (wj * one_m_e / mu + bj * i * rem)
        return s

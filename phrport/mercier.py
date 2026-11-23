"""Mercier stability metric from VMEX's traceable DMerc profile (vmex.core.stability.d_merc_state).

    x_j  = s_j DMerc(s_j)                 on the full-mesh surfaces with s_min <= s_j <= s_max
    merc = -tau log sum_j exp(-x_j / tau)  (smooth min of x; a lower bound on min x, by at most tau log N)

Positive DMerc is Mercier-stable (VMEC convention), so the row phr.mercier asks merc >= phr.mercier_margin.
The s weight takes out the ~1/s growth of DMerc toward the axis, so the mid and outer surfaces are not
swamped by the near-axis values. VMEC's axis, first near-axis surface and edge carry noisy DMerc and are
always excluded (j in [2, ns - 2]). jvp-able through the implicit tangent like the QI rows.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from vmex.core.stability import d_merc_state


def _surface_mask(ns, s_min, s_max):
    s = np.linspace(0.0, 1.0, ns)
    j = np.arange(ns)
    keep = (s >= s_min - 1e-12) & (s <= s_max + 1e-12) & (j >= 2) & (j <= ns - 2)
    return s, np.nonzero(keep)[0]


def mercier_metric(state, rt, *, s_min=0.1, s_max=0.9, tau=0.1):
    """-> dict(soft_min, min, unstable_frac, s, dmerc) of x = s DMerc on s in [s_min, s_max]."""
    d = d_merc_state(state, rt)
    s, idx = _surface_mask(int(d.shape[0]), float(s_min), float(s_max))
    x = jnp.asarray(s[idx]) * d[idx]
    t = float(tau)
    soft = -t * jax.scipy.special.logsumexp(-x / t)
    return dict(soft_min=soft, min=jnp.min(x), unstable_frac=jnp.mean((x < 0).astype(x.dtype)),
                s=s[idx], dmerc=d[idx])


def mercier_metric_from_wout(wout, *, s_min=0.1, s_max=0.9, tau=0.1):
    """Same metric from a wout's DMerc table (numpy; for reporting and calibration)."""
    d = np.asarray(wout["DMerc"][:] if hasattr(wout, "__getitem__") and not hasattr(wout, "DMerc")
                   else wout.DMerc, dtype=float)
    s, idx = _surface_mask(len(d), float(s_min), float(s_max))
    x = s[idx] * d[idx]
    t = float(tau)
    soft = -t * (np.log(np.sum(np.exp(-(x - x.min()) / t))) - x.min() / t)
    return dict(soft_min=float(soft), min=float(x.min()), unstable_frac=float(np.mean(x < 0)),
                s=s[idx], dmerc=d[idx])

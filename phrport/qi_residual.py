"""Continuous Goodman-style constructed-QI residual (level-set form).

Goodman et al., JPP 89, 905890504 (2023) build a perfectly QI field B_QI from the
current |B| on each surface (squash, stretch, shuffle) and penalize the L2 distance
(Eq. 15).  VMEX's port (``vmex.core.optimize.quasi_isodynamic_residual``) splits each
field line at its grid argmin, so the residual jumps whenever the argmin moves to a
neighbouring grid point.  This version builds the same target without any index
selection, so the residual is a continuous function of the Boozer spectrum:

* |B| along each line is treated as the piecewise-linear interpolant of the samples.
* **stretch**: each line is rescaled to [0, 1] by its own min/max (Eq. 6); the
  constructed field spans the global [B_min, B_max] on every line.
* For each trapping level l the sublevel set {b_line < l} is measured exactly,
  segment by segment: its length delta(alpha, l) and its centroid c(alpha, l).  On a
  single-well line this is the squashed well's bounce distance and midpoint; with
  secondary wells it is a continuous relaxation of the squash (the measure instead
  of the convex hull).
* **shuffle**: every line gets the alpha-mean bounce distance (optionally Goodman's
  Eq. 7 weights), placed symmetrically about its own midpoint (Delta phi_1 = -Delta
  phi_2, the paper's choice), clipped to the field period.
* B_QI on the sampling grid is the layer-cake integral over levels of "x outside the
  shuffled well", with bounce points linear in the level inside each band.  There is
  no nesting/sorting step, so B_QI is continuous in the bounce points.  The residual
  is (b - B_QI) on the grid, normalized as in Eq. 15.

For an exactly QI field (every line has the same bounce distances and its maxima at
the period ends) the residual vanishes up to the level discretization.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import jax
import jax.numpy as jnp

from vmex.core.omnigenity import boozer_spectrum_state

Array = Any

__all__ = ["smooth_constructed_qi_residual", "SmoothConstructedQI", "constructed_maximum_j"]


def _bmag_lines(bmnc_b, bmns_b, xm_b, xn_b, iota_b, nfp, nphi, nalpha, phimin):
    """|B| on (surface, alpha, phi) over one field period, endpoints included."""
    dtype = bmnc_b.dtype
    phi0 = jnp.asarray(float(phimin), dtype=dtype)
    phi = phi0 + jnp.linspace(0.0, 2.0 * np.pi / nfp, nphi, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * np.pi, nalpha, endpoint=False, dtype=dtype)
    theta = alpha[None, :, None] + iota_b[:, None, None] * phi[None, None, :]
    ang = theta[..., None] * xm_b - phi[None, None, :, None] * xn_b
    b = jnp.sum(bmnc_b[:, None, None, :] * jnp.cos(ang), axis=-1)
    if bmns_b is not None:
        b = b + jnp.sum(bmns_b[:, None, None, :] * jnp.sin(ang), axis=-1)
    return phi, alpha, b


def _sublevel_length_centroid(b, x, levels):
    """Exact length and centroid of {x : b(x) < l} for the piecewise-linear b(x).

    b: (..., n) samples at the uniform nodes x (n,); levels: (L,).
    Returns (length, centroid), each (..., L).
    """
    h = x[1] - x[0]
    b0, b1 = b[..., :-1, None], b[..., 1:, None]
    lo, hi = jnp.minimum(b0, b1), jnp.maximum(b0, b1)
    span = jnp.maximum(hi - lo, jnp.finfo(b.dtype).eps)
    frac = jnp.clip((levels - lo) / span, 0.0, 1.0)              # (..., n-1, L)
    seg = h * frac
    left = x[:-1, None]
    # the sub-segment sits at the low end of the segment
    cen = jnp.where(b0 <= b1, left + 0.5 * seg, left + h - 0.5 * seg)
    length = jnp.sum(seg, axis=-2)
    centroid = jnp.sum(seg * cen, axis=-2) / jnp.maximum(length, jnp.finfo(b.dtype).tiny)
    return length, centroid


def smooth_constructed_qi_residual(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    bmns_b=None,
    weights: Iterable[float] | None = None,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
    phimin: float = 0.0,
    alpha_weights: bool = False,
    alpha_weight_floor: float = 1.0e-4,
    soft: float = 2.0e-4,
) -> dict[str, Array]:
    """Continuous constructed-QI residual (module docstring). Returns residuals1d/total."""
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    bmns_b = None if bmns_b is None else jnp.asarray(bmns_b, dtype=jnp.float64)
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    nsurf = bmnc_b.shape[0]
    dtype = bmnc_b.dtype
    w_surf = (jnp.ones((nsurf,), dtype) if weights is None
              else jnp.asarray(np.asarray(weights, float), dtype))

    phi, alpha, b = _bmag_lines(bmnc_b, bmns_b, xm_b, xn_b, iota_b, int(nfp),
                                int(nphi), int(nalpha), phimin)
    x = (phi - phi[0])                                            # [0, period]
    period = x[-1]
    tiny = jnp.finfo(dtype).tiny

    gmin = jnp.min(b, axis=(1, 2), keepdims=True)
    gmax = jnp.max(b, axis=(1, 2), keepdims=True)
    bnorm = (b - gmin) / jnp.maximum(gmax - gmin, tiny)           # global [0, 1]

    lmin = jnp.min(bnorm, axis=-1, keepdims=True)                 # stretch (Eq. 6)
    lmax = jnp.max(bnorm, axis=-1, keepdims=True)
    bline = (bnorm - lmin) / jnp.maximum(lmax - lmin, tiny)

    # levels clustered at the bottom and top of the well, where phi(l) ~ sqrt(l)
    k = jnp.arange(1, int(n_bounce) + 1, dtype=dtype)
    levels = 0.5 * (1.0 - jnp.cos(np.pi * k / (int(n_bounce) + 1)))
    delta, cen = _sublevel_length_centroid(bline, x, levels)      # (nsurf, nalpha, L)

    if alpha_weights:                                             # Goodman Eq. 7
        dev = jnp.mean((bnorm - bline) ** 2, axis=-1)
        wa = 1.0 / (dev + alpha_weight_floor)
        wa = wa / jnp.sum(wa, axis=1, keepdims=True)
    else:
        wa = jnp.full((nsurf, int(nalpha)), 1.0 / int(nalpha), dtype)
    dbar = jnp.sum(wa[..., None] * delta, axis=1, keepdims=True)  # (nsurf, 1, L)

    half = 0.5 * dbar
    xl = jnp.clip(cen - half, 0.0, period)                        # shuffled bounce points
    xr = jnp.clip(cen + half, 0.0, period)
    # bounce-point curves over the full level range: both meet at the lowest level's
    # midpoint for l = 0 and reach the period ends at l = 1 (B_max there, JPP Eq. 12)
    zero = jnp.zeros_like(levels[:1]); one = jnp.ones_like(levels[:1])
    lev_full = jnp.concatenate([zero, levels, one])
    c0 = 0.5 * (xl[..., :1] + xr[..., :1])
    xl = jnp.concatenate([c0, xl, jnp.zeros_like(c0)], axis=-1)
    xr = jnp.concatenate([c0, xr, jnp.full_like(c0, period)], axis=-1)

    # B_QI(x) = |{l : x outside [xl(l), xr(l)]}| (layer cake), with xl, xr linear in l
    # inside each level band.  No sorting or nesting step, so the result is continuous
    # in the bounce points even where shuffled points are not nested.
    dl = jnp.diff(lev_full)                                       # (L+1,)
    xq = x[None, None, :, None]

    def band_measure(curve, above):
        a, bb = curve[..., None, :-1], curve[..., None, 1:]
        span = jnp.maximum(jnp.abs(bb - a), jnp.finfo(dtype).eps)
        if above:      # fraction of the band where curve(l) > x
            fr = jnp.clip((jnp.maximum(a, bb) - xq) / span, 0.0, 1.0)
        else:          # fraction of the band where curve(l) < x
            fr = jnp.clip((xq - jnp.minimum(a, bb)) / span, 0.0, 1.0)
        return jnp.sum(fr * dl, axis=-1)

    bqi = band_measure(xl, True) + band_measure(xr, False)
    # B_QI = B_max at the period ends (JPP Eq. 12); clipped top levels pile up there.
    bqi = bqi.at[..., 0].set(1.0).at[..., -1].set(1.0)

    tsoft = float(soft) * jnp.mean(jnp.abs(b), axis=(1, 2))
    npts = int(nalpha) * int(nphi)
    res = (bnorm - bqi) * jnp.sqrt(w_surf)[:, None, None] / jnp.sqrt(jnp.asarray(npts, dtype))
    r1 = jnp.ravel(res)
    return {"residuals1d": r1, "total": jnp.sum(r1 * r1), "bnorm": bnorm, "bqi": bqi,
            "delta": delta, "dbar": dbar, "phi": phi, "alpha": alpha, "levels": levels,
            "per_surface": jnp.sum(res * res, axis=(1, 2)), "period": period,
            "bmin_s": gmin[:, 0, 0], "bmax_s": gmax[:, 0, 0],
            # smooth (log-sum-exp) extrema at temperature soft*<B>, for constraints built on them
            "bmin_soft": -tsoft * jax.scipy.special.logsumexp(-b.reshape(nsurf, -1) / tsoft[:, None], axis=1),
            "bmax_soft": tsoft * jax.scipy.special.logsumexp(b.reshape(nsurf, -1) / tsoft[:, None], axis=1)}


def _action_kernel(bnode, bstar):
    """H(B) with dH/dB = sqrt(1 - B/B*)/B, H = 0 for B >= B* (smooth there: H ~ u^3)."""
    u = jnp.sqrt(jnp.maximum(1.0 - bnode / bstar, 1.0e-300))
    return 2.0 * u - 2.0 * jnp.arctanh(u)


def constructed_maximum_j(qi_out, *, iota_b, G_b, I_b, s_b, target=-0.06,
                          depths=(0.1, 0.3, 0.5, 0.7, 0.9), tau=0.005):
    """Max-J on the continuous constructed-QI field (PRX Energy 3, 023010, Sec. II.2 analogue).

    The constructed field of surface s is exactly omnigenous: every line has the mean bounce
    distance dbar(l) at normalized level l, and B = B_min(s) + (B_max(s) - B_min(s)) l.  With dbar
    piecewise linear in l, the second invariant at physical pitch 1/B*,

        J(s, B*) = |G + iota I| * int_well sqrt(1 - B/B*) / B dphi
                 = |G + iota I| * sum_k ddbar_k/dB_k [H(B_k+1) - H(B_k)],

    is a closed form (``_action_kernel``), smooth in the Boozer spectrum.  For each adjacent surface
    pair, B* runs over ``depths`` of the pair's common trapping range; the relative slope
    (1/J) dJ/ds is compared with ``target`` (max-J: slope <= target < 0).  The constraint value is a
    log-sum-exp (temperature ``tau``) of slope - target over all pairs and pitches, an upper bound
    of the max within tau*log(n).  Returns dict(g, slope, fraction, action, bstar).
    """
    period = qi_out["period"]
    lev = qi_out["levels"]
    dtype = lev.dtype
    lev_full = jnp.concatenate([jnp.zeros((1,), dtype), lev, jnp.ones((1,), dtype)])
    db = qi_out["dbar"][:, 0, :]                                         # (nsurf, L)
    ns = db.shape[0]
    dfull = jnp.concatenate([jnp.zeros((ns, 1), dtype), db, jnp.full((ns, 1), period, dtype)], axis=1)
    # smooth extrema: hard ones put kinks in the constraint, which stall TRF at large penalties
    bmin, bmax = qi_out["bmin_soft"], qi_out["bmax_soft"]
    bnode = bmin[:, None] + (bmax - bmin)[:, None] * lev_full[None, :]   # (nsurf, L+2)
    slope = jnp.diff(dfull, axis=1) / jnp.maximum(jnp.diff(bnode, axis=1), jnp.finfo(dtype).tiny)
    cfac = jnp.abs(jnp.asarray(G_b) + jnp.asarray(iota_b) * jnp.asarray(I_b))
    dep = jnp.asarray(np.asarray(depths, float), dtype)

    def action(i, bstar):                                                 # (P,) on surface i
        H = _action_kernel(bnode[i][None, :], bstar[:, None])
        return cfac[i] * jnp.sum(slope[i][None, :] * jnp.diff(H, axis=1), axis=1)

    rows, acts, bst = [], [], []
    s_b = jnp.asarray(s_b, dtype)
    for i in range(ns - 1):
        tp = 1.0e-3 * (bmin[i] + bmin[i + 1])
        lo = tp * jnp.logaddexp(bmin[i] / tp, bmin[i + 1] / tp)
        hi = -tp * jnp.logaddexp(-bmax[i] / tp, -bmax[i + 1] / tp)
        bstar = hi - dep * (hi - lo)
        j0, j1 = action(i, bstar), action(i + 1, bstar)
        rel = (j1 - j0) / ((s_b[i + 1] - s_b[i]) * 0.5 * (jnp.abs(j0) + jnp.abs(j1)))
        rows.append(rel); acts.append(jnp.stack([j0, j1])); bst.append(bstar)
    rel = jnp.stack(rows)                                                 # (pairs, P)
    z = (rel - target) / tau
    g = tau * jax.scipy.special.logsumexp(z)
    return dict(g=g, slope=rel, fraction=jnp.mean(rel < 0.0), action=jnp.stack(acts),
                bstar=jnp.stack(bst), max_violation=jnp.max(rel - target))


def _sqrt_linear_segments(x, h):
    """int sqrt(max(x, 0)) dphi over a uniform grid (step h), x linear in each segment: exact for piecewise-linear x,
    C1 in x (no sqrt kink at the bounce points). x: (..., n) -> (...,)."""
    x0, x1 = x[..., :-1], x[..., 1:]
    p0 = jnp.maximum(x0, 0.0); p1 = jnp.maximum(x1, 0.0)
    a0 = jnp.where(p0 > 0.0, p0, 1.0) ** 1.5 * (p0 > 0.0)
    a1 = jnp.where(p1 > 0.0, p1, 1.0) ** 1.5 * (p1 > 0.0)
    dx = x1 - x0
    flat = jnp.abs(dx) < 1.0e-12
    xm = jnp.maximum(0.5 * (x0 + x1), 0.0)
    seg = jnp.where(flat, h * jnp.where(xm > 0.0, xm, 1.0) ** 0.5 * (xm > 0.0),
                    (2.0 / 3.0) * h * (a1 - a0) / jnp.where(flat, 1.0, dx))
    return jnp.sum(seg, axis=-1)


def real_field_j_closure(*, bmnc_b, xm_b, xn_b, iota_b, G_b, I_b, nfp, bmns_b=None, fracs=(0.03, 0.1, 0.2),
                         target=-0.02, nphi=401, nalpha=24, soft=1.0e-4, tau=0.005, tau_alpha=0.002, order="max",
                         enclose_inner=None, measure="range", tau_sep=None, width=1.0e-3, row_weights=None):
    """Closure of the J contours of deeply trapped particles on the real field (not the constructed one).

    J(s, alpha) = |G + iota I| int sqrt(1 - B/B*) / B dphi over one field period of each line, at fixed B* (a particle
    keeps B* = E/mu as it drifts). B* runs over fracs of the band trapped on every line of every surface given
    (0 = the bottom of the shallowest well). For each adjacent surface pair (s_k, s_k+1) and pitch, the max-J separation
    d = (max_alpha J(s_k+1) - min_alpha J(s_k)) / <J>_pair is negative when every J contour started on s_k stays inside
    s_k+1 (closed contours, max-J ordering); d > 0 means the J ranges overlap and contours can open. With order="either"
    each (pitch, pair) takes the smaller (log-sum-exp at tau) of that and the min-J separation
    (max_alpha J(s_k) - min_alpha J(s_k+1)) / <J>_pair, so either ordering counts as closed.
    With enclose_inner = n, the rows are instead the first n surfaces (birth surfaces), each paired with every outer
    surface: e(k) = soft-min over c > k of the separation of J(s_k) from J(s_c) (in the chosen ordering), i.e. the
    J range of s_k is enclosed if ANY outer surface has its whole range beyond it. The constraint value
    is a log-sum-exp (temperature tau) of d - target over pairs and pitches. Extrema over alpha and phi are smooth
    (log-sum-exp at tau_alpha relative to <J>, and soft * <B>). Returns dict(g, d, J, bstar).

    measure="starts" (needs enclose_inner) scores each start (s_k, alpha_j) on its own, as closure.py does: its
    separation e(k, j) is the soft-min (temperature tau_sep, default tau) over outer surfaces c > k, and over both
    orderings when order="either", of (max_alpha J(s_c) - J(s_k, alpha_j)) / <J>_pair and
    (J(s_k, alpha_j) - min_alpha J(s_c)) / <J>_pair; e < 0 means some outer surface has its whole J range on one side
    of the start. Per pitch, d_p is the mean over starts (row_weights over k) of softplus(e / width), the mean opening
    of the open starts in units of width (O(1) like the other constraint rows; about ln 2 per start at e = 0, so
    d_p ~ 0.1 is a few per cent of starts open). target may then be one value per pitch, and
    g = tau * logsumexp((d - target) / tau).
    """
    dtype = jnp.asarray(bmnc_b).dtype
    phi, alpha, b = _bmag_lines(jnp.asarray(bmnc_b), None if bmns_b is None else jnp.asarray(bmns_b),
                                jnp.asarray(xm_b, dtype), jnp.asarray(xn_b, dtype), jnp.asarray(iota_b, dtype),
                                int(nfp), int(nphi), int(nalpha), 0.0)
    ts = float(soft) * jnp.mean(jnp.abs(b))
    lmin = -ts * jax.scipy.special.logsumexp(-b / ts, axis=-1)                    # line minima (nsurf, nalpha)
    lmax = ts * jax.scipy.special.logsumexp(b / ts, axis=-1)
    lo = ts * jax.scipy.special.logsumexp(lmin / ts)                              # trapped on every line
    hi = -ts * jax.scipy.special.logsumexp(-lmax / ts)
    fr = jnp.asarray(np.asarray(fracs, float), dtype)
    bstar = lo + fr * (hi - lo)                                                   # (P,)
    h = phi[1] - phi[0]
    x = 1.0 - b[None] / bstar[:, None, None, None]                                # (P, nsurf, nalpha, nphi)
    # 1/B varies little across a deep well: the exact sqrt integral at 1/B*, times the sqrt(x)-weighted mean of B*/B
    # (continuous as nodes enter or leave the well, since the weight vanishes at the bounce points)
    pw = jnp.where(x > 0.0, x, 1.0) ** 0.5 * (x > 0.0)
    corr = jnp.sum(pw * bstar[:, None, None, None] / b[None], axis=-1) / jnp.maximum(jnp.sum(pw, axis=-1), 1.0e-30)
    cfac = jnp.abs(jnp.asarray(G_b) + jnp.asarray(iota_b) * jnp.asarray(I_b))
    J = cfac[None, :, None] * _sqrt_linear_segments(x, h) * corr / bstar[:, None, None]   # (P, nsurf, nalpha)
    jm = jnp.mean(J, axis=-1)                                                     # (P, nsurf)
    ta = float(tau_alpha) * jm[..., None]
    jmax = ta[..., 0] * jax.scipy.special.logsumexp(J / ta, axis=-1)
    jmin = -ta[..., 0] * jax.scipy.special.logsumexp(-J / ta, axis=-1)
    if order not in ("max", "either"):
        raise ValueError(f"order must be 'max' or 'either', got {order!r}")

    def sep(i, o):                                                                # separation of s_i from s_o > s_i
        jp = 0.5 * (jm[:, o] + jm[:, i])
        dmax = (jmax[:, o] - jmin[:, i]) / jp                                     # max-J ordering
        if order == "max":
            return dmax
        dmin = (jmax[:, i] - jmin[:, o]) / jp                                     # min-J ordering
        return -float(tau) * jnp.logaddexp(-dmax / float(tau), -dmin / float(tau))

    ns_ = J.shape[1]
    if measure == "starts":
        if enclose_inner is None:
            raise ValueError("measure='starts' needs enclose_inner")
        ts_ = float(tau if tau_sep is None else tau_sep)
        ni = int(enclose_inner)
        rows = []
        for k in range(ni):
            seps = []
            for c in range(k + 1, ns_):
                jp = 0.5 * (jm[:, c] + jm[:, k])[:, None]                         # (P, 1)
                seps.append((jmax[:, c, None] - J[:, k]) / jp)                    # max-J side (P, nalpha)
                if order == "either":
                    seps.append((J[:, k] - jmin[:, c, None]) / jp)                # min-J side
            e = jnp.stack(seps, axis=0)                                            # (sides*outer, P, nalpha)
            rows.append(-ts_ * jax.scipy.special.logsumexp(-e / ts_, axis=0))
        e = jnp.stack(rows, axis=1)                                                # (P, ni, nalpha)
        op = jax.nn.softplus(e / float(width))
        rw = jnp.ones((ni,), dtype) if row_weights is None else jnp.asarray(np.asarray(row_weights, float), dtype)
        d = jnp.sum(jnp.mean(op, axis=-1) * rw[None], axis=1) / jnp.sum(rw)      # (P,)
        tg = jnp.broadcast_to(jnp.asarray(np.asarray(target, float), dtype), d.shape)
        g = float(tau) * jax.scipy.special.logsumexp((d - tg) / float(tau))
        return dict(g=g, d=d, e=e, J=J, bstar=bstar, max_violation=jnp.max(d - tg),
                    open_fraction=jnp.mean(e > 0.0, axis=(1, 2)))
    if enclose_inner is None:
        d = jnp.stack([sep(k, k + 1) for k in range(ns_ - 1)], axis=1)           # (P, pairs)
    else:
        rows = []
        for k in range(int(enclose_inner)):
            e = jnp.stack([sep(k, c) for c in range(k + 1, ns_)], axis=1)        # (P, outer)
            rows.append(-float(tau) * jax.scipy.special.logsumexp(-e / float(tau), axis=1))
        d = jnp.stack(rows, axis=1)                                               # (P, inner)
    g = float(tau) * jax.scipy.special.logsumexp((d - float(target)) / float(tau))
    return dict(g=g, d=d, J=J, bstar=bstar, max_violation=jnp.max(d - float(target)))


class SmoothConstructedQI:
    """Least-squares term (state, runtime) -> continuous constructed-QI residual vector."""

    name = "smooth_constructed_qi"

    def __init__(self, surfaces, *, weights=None, mboz=16, nboz=16, oversample=2, **opts):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.weights = None if weights is None else np.asarray(weights, float)
        self.mboz, self.nboz, self.oversample = int(mboz), int(nboz), int(oversample)
        self.opts = opts

    def compute_state(self, state, rt):
        booz = boozer_spectrum_state(state, rt, surfaces=self.surfaces, mboz=self.mboz,
                                     nboz=self.nboz, oversample=self.oversample)
        out = smooth_constructed_qi_residual(
            bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"), xm_b=booz["xm_b"],
            xn_b=booz["xn_b"], iota_b=booz["iota_b"], nfp=booz["nfp"],
            weights=self.weights, **self.opts)
        out.update(booz)
        return out

    def residuals_state(self, state, rt):
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state, rt):
        return self.compute_state(state, rt)["total"]

    def J(self, eq):
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J


def bmin_flatness_rows(bmnc_b, xm_b, xn_b, iota_b, nfp, *, bmns_b=None, nphi=101, nalpha=21, soft=2.0e-3):
    """Well-bottom flatness rows (opt-in, 28 Oct): for each surface and field line alpha, the soft minimum of |B|
    along one field period minus its alpha-mean, in units of the surface's |B| range (B_max - B_min), / sqrt(nalpha).

    QI needs B_min (and B_max) independent of alpha; the constructed-QI residual weights the well-bottom offset
    weakly, and an alpha-dependent B_min makes the second invariant J of deeply trapped particles vary across field
    lines, opening their J contours.  Soft minimum = log-sum-exp at temperature soft * <B> (smooth, jvp-able).
    -> (nsurf * nalpha,) rows; their sum of squares is the alpha-variance of B_min / (B range)^2 summed over surfaces.
    """
    phi, alpha, b = _bmag_lines(bmnc_b, bmns_b, xm_b, xn_b, iota_b, int(nfp), int(nphi), int(nalpha), 0.0)
    t = float(soft) * jnp.mean(jnp.abs(b), axis=(1, 2), keepdims=True)               # (nsurf, 1, 1)
    lmin = -t[..., 0] * jax.scipy.special.logsumexp(-b / t, axis=-1)                 # (nsurf, nalpha)
    rng = jnp.max(b, axis=(1, 2)) - jnp.min(b, axis=(1, 2))                          # (nsurf,)
    dev = (lmin - jnp.mean(lmin, axis=1, keepdims=True)) / rng[:, None]
    return jnp.ravel(dev) / jnp.sqrt(jnp.asarray(int(nalpha), dev.dtype))


def bmax_spread(bmnc_b, xm_b, xn_b, iota_b, nfp, *, bmns_b=None, nphi=101, nalpha=21, soft=2.0e-3, tau_alpha=2.0e-4,
                which="max"):
    """qh_funk (5 Nov): field-line spread of B_max on each surface, (max_alpha Bmax_alpha - min_alpha Bmax_alpha) / B00.

    Bmax_alpha is the soft maximum of |B| along field line alpha over one field period (log-sum-exp at temperature
    soft * <B>, whose bias is common to all lines and cancels in the spread); the extrema over alpha are log-sum-exp
    at temperature tau_alpha * B00 (overestimate the spread by <~ 2 tau_alpha log(nalpha)). This is the variation that
    the constructed QI target removes by stretching every well, and that drives the bounce-averaged radial drift of
    the trapped particles.  -> (nsurf,) spreads.
    which="min" (qh_funk 6 Nov): the same spread of the soft minimum B_min along each line (well-bottom spread).
    """
    phi, alpha, b = _bmag_lines(bmnc_b, bmns_b, xm_b, xn_b, iota_b, int(nfp), int(nphi), int(nalpha), 0.0)
    t = float(soft) * jnp.mean(jnp.abs(b), axis=(1, 2), keepdims=True)               # (nsurf, 1, 1)
    sg = -1.0 if which == "min" else 1.0
    lmax = sg * t[..., 0] * jax.scipy.special.logsumexp(sg * b / t, axis=-1)         # (nsurf, nalpha)
    b00 = jnp.sum(jnp.where((jnp.asarray(xm_b) == 0) & (jnp.asarray(xn_b) == 0), bmnc_b, 0.0), axis=1)   # (nsurf,)
    ta = float(tau_alpha) * b00[:, None]
    hi = ta[:, 0] * jax.scipy.special.logsumexp(lmax / ta, axis=1)
    lo = -ta[:, 0] * jax.scipy.special.logsumexp(-lmax / ta, axis=1)
    return (hi - lo) / b00


# ---------------------------------------------------------------------------
# QSS physics block for a QI problem profile (phrport.equilibrium.PHYSICS == "qi")
# ---------------------------------------------------------------------------
def bmin_line_mean(bmnc_b, xm_b, xn_b, iota_b, nfp, *, bmns_b=None, nphi=101, nalpha=21, soft=2.0e-3):
    """qh_funk (8 Nov): field-line average of the soft minimum of |B| along each line over one field period, and B00.
    -> ((nsurf,) <B_min>, (nsurf,) B00). The soft-min bias (soft * <B>) is nearly common to neighbouring surfaces."""
    phi, alpha, b = _bmag_lines(bmnc_b, bmns_b, xm_b, xn_b, iota_b, int(nfp), int(nphi), int(nalpha), 0.0)
    t = float(soft) * jnp.mean(jnp.abs(b), axis=(1, 2), keepdims=True)
    lmin = -t[..., 0] * jax.scipy.special.logsumexp(-b / t, axis=-1)                  # (nsurf, nalpha)
    b00 = jnp.sum(jnp.where((jnp.asarray(xm_b) == 0) & (jnp.asarray(xn_b) == 0), bmnc_b, 0.0), axis=1)
    return jnp.mean(lmin, axis=1), b00


def mirror_ratio_from_boozer(bmnc_b, xm_b, xn_b, nfp, *, bmns_b=None, ntheta=64, nphi=64, soft=0.0):
    """Goodman's mirror ratio (B_max - B_min) / (B_max + B_min) on each given surface.

    |B| from the Boozer harmonics on a (theta_B, phi_B) grid over one field period.
    soft = 0: hard max/min (continuous, kinks where the arg-extremum moves).  soft > 0: log-sum-exp
    max/min at temperature soft * <B> (smooth; overestimates the ratio by <~ soft*log(n)/2).
    """
    th = jnp.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    ph = jnp.linspace(0.0, 2.0 * np.pi / nfp, nphi, endpoint=False)
    ang = th[:, None, None] * xm_b - ph[None, :, None] * xn_b
    b = jnp.einsum("sm,tpm->stp", bmnc_b, jnp.cos(ang))
    if bmns_b is not None:
        b = b + jnp.einsum("sm,tpm->stp", bmns_b, jnp.sin(ang))
    if soft > 0.0:
        t = soft * jnp.mean(jnp.abs(b), axis=(1, 2), keepdims=True)
        bf = b.reshape(b.shape[0], -1); tf = t.reshape(-1, 1)
        bmax = tf[:, 0] * jax.scipy.special.logsumexp(bf / tf, axis=1)
        bmin = -tf[:, 0] * jax.scipy.special.logsumexp(-bf / tf, axis=1)
    else:
        bmax = jnp.max(b, axis=(1, 2))
        bmin = jnp.min(b, axis=(1, 2))
    return (bmax - bmin) / (bmax + bmin)


def bmin_uniformity(*, bmnc_b, bmns_b=None, xm_b, xn_b, iota_b, nfp, nalpha=24, nphi=64, tau=0.002, weight=1.0):
    """Field-line B_min uniformity rows for deeply trapped particles: soft-min of B/<B_min> along each of nalpha field
    lines within one field period, per surface; rows = weight * (B_min(alpha) - <B_min>) / <B_min>, shape (ns * nalpha,)."""
    al = jnp.linspace(0.0, 2.0 * jnp.pi, int(nalpha), endpoint=False)
    ph = jnp.linspace(0.0, 2.0 * jnp.pi / int(nfp), int(nphi), endpoint=False)
    xm, xn = jnp.asarray(xm_b, jnp.float64), jnp.asarray(xn_b, jnp.float64)
    rows = []
    for k in range(bmnc_b.shape[0]):
        ang = xm[None, None, :] * (al[:, None, None] + iota_b[k] * ph[None, :, None]) - xn[None, None, :] * ph[None, :, None]
        b = jnp.sum(bmnc_b[k][None, None, :] * jnp.cos(ang), axis=2)
        if bmns_b is not None:
            b = b + jnp.sum(bmns_b[k][None, None, :] * jnp.sin(ang), axis=2)
        b0 = jnp.mean(b)
        bmin = -float(tau) * jax.scipy.special.logsumexp(-b / b0 / float(tau), axis=1)
        rows.append(float(weight) * (bmin - jnp.mean(bmin)) / jnp.mean(bmin))
    return jnp.concatenate(rows)


def real_field_djds(*, bmnc_b, bmns_b=None, xm_b, xn_b, iota_b, G_b, I_b, nfp, s_b, depths=(0.05, 0.1), nalpha=24, nphi=256,
                    tau_v=1.0e-3, tau_g=0.02, weight=1.0):
    """Real-field max-J rows for shallow wells: J(s, alpha) = (G + iota I) int sqrt(1 - B/B*)/B dphi over one field period at fixed
    B* = B_min + k (B_max - B_min) (extrema of the middle surface); rows = weight * softplus(dlnJ/ds), one per
    depth, surface interval and field line, so only dJ/ds > 0 is penalized. Shape (len(depths) * (ns - 1) * nalpha,)."""
    al = jnp.linspace(0.0, 2.0 * jnp.pi, int(nalpha), endpoint=False)
    ph = jnp.linspace(0.0, 2.0 * jnp.pi / int(nfp), int(nphi), endpoint=False)
    xm, xn = jnp.asarray(xm_b, jnp.float64), jnp.asarray(xn_b, jnp.float64)
    ns = bmnc_b.shape[0]
    bs = []
    for k in range(ns):
        ang = xm[None, None, :] * (al[:, None, None] + iota_b[k] * ph[None, :, None]) - xn[None, None, :] * ph[None, :, None]
        b = jnp.sum(bmnc_b[k][None, None, :] * jnp.cos(ang), axis=2)
        if bmns_b is not None:
            b = b + jnp.sum(bmns_b[k][None, None, :] * jnp.sin(ang), axis=2)
        bs.append(b)
    B = jnp.stack(bs)                    # (surface, alpha, phi)
    mid = B[ns // 2]
    bmin, bmax = jnp.min(mid), jnp.max(mid)
    dphi = 2.0 * jnp.pi / int(nfp) / int(nphi)
    ds = jnp.asarray(s_b)[1:] - jnp.asarray(s_b)[:-1]
    rows = []
    for d in depths:
        Bt = bmin + float(d) * (bmax - bmin)
        x = 1.0 - B / Bt
        v = float(tau_v) * jnp.logaddexp(0.0, x / float(tau_v))              # smooth max(0, 1 - B/B*)
        J = jnp.sqrt(v + 1.0e-12) / B
        J = jnp.sum(J, axis=2) * dphi * jnp.abs(jnp.asarray(G_b) + jnp.asarray(iota_b) * jnp.asarray(I_b))[:, None]
        Jm = jnp.mean(J)
        g = (J[1:] - J[:-1]) / ds[:, None] / Jm                               # dlnJ/ds per interval and field line
        rows.append(float(weight) * float(tau_g) * jnp.logaddexp(0.0, g / float(tau_g)))
    return jnp.concatenate([r.reshape(-1) for r in rows])


def real_field_j_spread(*, bmnc_b, bmns_b=None, xm_b, xn_b, iota_b, G_b, I_b, nfp, depths=(0.05, 0.1), nalpha=24, nphi=256,
                        tau_v=1.0e-3, weight=1.0):
    """Real-field J-closure rows for shallow wells: J(s, alpha) as in real_field_djds, at fixed B* (extrema of the middle surface);
    rows = weight * (J - <J>_alpha) / <J>_alpha / sqrt(nalpha), so the sum of squares on one surface and depth is weight^2 (std_alpha J / <J>_alpha)^2.
    Shape (len(depths) * ns * nalpha,)."""
    al = jnp.linspace(0.0, 2.0 * jnp.pi, int(nalpha), endpoint=False)
    ph = jnp.linspace(0.0, 2.0 * jnp.pi / int(nfp), int(nphi), endpoint=False)
    xm, xn = jnp.asarray(xm_b, jnp.float64), jnp.asarray(xn_b, jnp.float64)
    ns = bmnc_b.shape[0]
    bs = []
    for k in range(ns):
        ang = xm[None, None, :] * (al[:, None, None] + iota_b[k] * ph[None, :, None]) - xn[None, None, :] * ph[None, :, None]
        b = jnp.sum(bmnc_b[k][None, None, :] * jnp.cos(ang), axis=2)
        if bmns_b is not None:
            b = b + jnp.sum(bmns_b[k][None, None, :] * jnp.sin(ang), axis=2)
        bs.append(b)
    B = jnp.stack(bs)                    # (surface, alpha, phi)
    mid = B[ns // 2]
    bmin, bmax = jnp.min(mid), jnp.max(mid)
    dphi = 2.0 * jnp.pi / int(nfp) / int(nphi)
    rows = []
    for d in depths:
        Bt = bmin + float(d) * (bmax - bmin)
        v = float(tau_v) * jnp.logaddexp(0.0, (1.0 - B / Bt) / float(tau_v))  # smooth max(0, 1 - B/B*)
        J = jnp.sum(jnp.sqrt(v + 1.0e-12) / B, axis=2) * dphi * jnp.abs(jnp.asarray(G_b) + jnp.asarray(iota_b) * jnp.asarray(I_b))[:, None]
        Jm = jnp.mean(J, axis=1, keepdims=True)
        rows.append(float(weight) * (J - Jm) / Jm / jnp.sqrt(float(int(nalpha))))
    return jnp.concatenate([r.reshape(-1) for r in rows])


def qi_physics(state, rt, *, surfaces, mirror_s, mboz, nboz, oversample=2, mirror_soft=0.0, maxj_target=None,
               maxj_depths=(0.1, 0.3, 0.5, 0.7, 0.9), maxj_tau=0.005, mirror_all=False, mirror_all_tau=5e-4,
               mirror_tau=None,
               bmin_weight=0.0, bmin_surfaces=(), bmin_soft=2.0e-3, bmaxsp_target=None, bmaxsp_soft=2.0e-3,
               bmaxsp_tau_alpha=2.0e-4, bmaxsp_tau=2.0e-4, bmaxsp_smax=1.0, bmaxsp_which="max", bminslope_target=None,
               jreal_surfaces=None, jreal_fracs=(0.03, 0.1, 0.2), jreal_target=-0.02, jreal_nphi=401,
               jreal_nalpha=24, jreal_tau=0.005, jreal_order="max", jreal_enclose_inner=None, jreal_measure="range",
               jreal_tau_sep=None, jreal_tau_alpha=0.002, jreal_width=1.0e-3, jreal_row_weights=None,
               bminu_surfaces=None, bminu_nalpha=24, bminu_nphi=64, bminu_tau=0.002, bminu_weight=1.0,
               djds_surfaces=None, djds_depths=(0.05, 0.1), djds_nalpha=24, djds_nphi=256, djds_tau_g=0.02, djds_weight=1.0,
               jspread_depths=(0.05, 0.1), jspread_weight=None, **opts):
    """-> dict(qs=QI residual vector, qs_total, mirror) from a converged (state, runtime).

    One traceable Boozer transform on ``surfaces + [mirror_s]``; jvp-able, so it is
    pushed through the exact implicit tangent exactly like the QS rows.
    mirror_all: ``mirror`` is the largest mirror ratio over ``surfaces (+ bmin_surfaces) + [mirror_s]`` (log-sum-exp
    at temperature mirror_tau, alias mirror_all_tau, which overestimates it by <= tau log(n)) instead of the ratio at
    mirror_s, so a mirror_max row caps every surface; ``mirror_profile`` holds the per-surface ratios.
    bmin_weight > 0 (opt-in): append sqrt(bmin_weight) x bmin_flatness_rows on ``surfaces + bmin_surfaces`` to the
    QI residual vector ``qs`` (so they enter the objective); ``qi_core_total`` keeps the pure constructed-QI total.
    bmin_weight = 0 (default): bit-for-bit the previous behaviour.
    bmaxsp_target (opt-in): constraint value ``bmaxsp`` = log-sum-exp at temperature bmaxsp_tau over the
    constructed-QI surfaces with s <= bmaxsp_smax of (bmax_spread - bmaxsp_target) <= 0; ``bmaxsp_profile`` holds the
    spreads on all constructed-QI surfaces. bmaxsp_which="min": the row caps the B_min spread instead; "both": both
    spreads in one row. bminslope_target: also cap d<B_min>/ds / B00 between consecutive kept surfaces (well-bottom
    radial gradient), in the same row.
    jreal_surfaces (opt-in): real_field_j_closure on those surfaces, folded into the max-J row (log-sum-exp at
    maxj_tau with the constructed max-J value, or alone without it).
    bminu_surfaces (opt-in): bmin_uniformity rows (field-line B_min uniformity) on those surfaces, appended to ``qs``.
    djds_surfaces (opt-in): real_field_djds rows (shallow-well dJ/ds >= 0) on those surfaces, appended to ``qs``;
    with jspread_weight also real_field_j_spread rows on the same surfaces.
    """
    qsurf = np.atleast_1d(np.asarray(surfaces, float))
    extra = np.atleast_1d(np.asarray(bmin_surfaces, float)) if bmin_weight > 0.0 else np.zeros(0)
    jsurf = np.zeros((0,)) if jreal_surfaces is None else np.atleast_1d(np.asarray(jreal_surfaces, float))
    usurf = np.zeros((0,)) if bminu_surfaces is None else np.atleast_1d(np.asarray(bminu_surfaces, float))
    dsurf = np.zeros((0,)) if djds_surfaces is None else np.atleast_1d(np.asarray(djds_surfaces, float))
    # surface layout: constructed-QI, B_min flatness extras, real-field J, B_min uniformity, shallow-well dJ/ds, mirror
    surf = np.concatenate([qsurf, extra, jsurf, usurf, dsurf, [float(mirror_s)]])
    booz = boozer_spectrum_state(state, rt, surfaces=surf, mboz=int(mboz), nboz=int(nboz),
                                 oversample=int(oversample))
    bmnc = jnp.asarray(booz["bmnc_b"])
    bmns = booz.get("bmns_b")
    bmns = None if bmns is None else jnp.asarray(bmns)
    nq = len(qsurf)                                      # constructed-QI surfaces
    nb = nq + len(extra)                                 # + well-bottom extra surfaces
    nj = len(jsurf)                                      # real-field J surfaces
    nu, nd = len(usurf), len(dsurf)
    out = smooth_constructed_qi_residual(
        bmnc_b=bmnc[:nq], bmns_b=None if bmns is None else bmns[:nq], xm_b=booz["xm_b"],
        xn_b=booz["xn_b"], iota_b=jnp.asarray(booz["iota_b"])[:nq], nfp=int(booz["nfp"]), **opts)
    if mirror_all:
        keep = np.r_[np.arange(nb), len(surf) - 1]         # QI (+ well-bottom) surfaces and the mirror surface
        prof = mirror_ratio_from_boozer(bmnc[keep], jnp.asarray(booz["xm_b"], jnp.float64),
                                        jnp.asarray(booz["xn_b"], jnp.float64), int(booz["nfp"]),
                                        bmns_b=None if bmns is None else bmns[keep], soft=float(mirror_soft))
        tau = float(mirror_all_tau if mirror_tau is None else mirror_tau)
        mirror = tau * jax.scipy.special.logsumexp(prof / tau) if tau > 0.0 else jnp.max(prof)
    else:
        prof = None
        mirror = mirror_ratio_from_boozer(bmnc[-1:], jnp.asarray(booz["xm_b"], jnp.float64),
                                          jnp.asarray(booz["xn_b"], jnp.float64), int(booz["nfp"]),
                                          bmns_b=None if bmns is None else bmns[-1:], soft=float(mirror_soft))[0]
    res = dict(qs=out["residuals1d"], qs_total=out["total"], mirror=mirror)
    if bmin_weight > 0.0:
        rows = jnp.sqrt(float(bmin_weight)) * bmin_flatness_rows(
            bmnc[:nb], jnp.asarray(booz["xm_b"], jnp.float64), jnp.asarray(booz["xn_b"], jnp.float64),
            jnp.asarray(booz["iota_b"])[:nb], int(booz["nfp"]), bmns_b=None if bmns is None else bmns[:nb],
            nphi=int(opts.get("nphi", 151)), nalpha=int(opts.get("nalpha", 31)), soft=float(bmin_soft))
        res.update(qs=jnp.concatenate([out["residuals1d"], rows]), qs_total=out["total"] + jnp.sum(rows * rows),
                   qi_core_total=out["total"], bmin_total=jnp.sum(rows * rows) / float(bmin_weight))
    if prof is not None:
        res["mirror_profile"] = prof
    if bmaxsp_target is not None:   # field-line spread of B_max (bmax_spread), one constraint scalar
        spf = lambda w: bmax_spread(bmnc[:nq], jnp.asarray(booz["xm_b"], jnp.float64),
                                    jnp.asarray(booz["xn_b"], jnp.float64), jnp.asarray(booz["iota_b"])[:nq],
                                    int(booz["nfp"]), bmns_b=None if bmns is None else bmns[:nq],
                                    nphi=int(opts.get("nphi", 101)), nalpha=int(opts.get("nalpha", 21)),
                                    soft=float(bmaxsp_soft), tau_alpha=float(bmaxsp_tau_alpha), which=w)
        tb = float(bmaxsp_tau)
        keep = np.nonzero(np.asarray(surfaces, float) <= float(bmaxsp_smax) + 1e-12)[0]
        if str(bmaxsp_which) == "both":   # one row over the B_max and the B_min spreads together
            spx, spn = spf("max"), spf("min")
            v = jnp.concatenate([spx[keep], spn[keep]]) - float(bmaxsp_target)
            res.update(bmaxsp_profile=spx, bminsp_profile=spn)
        else:
            sp = spf(str(bmaxsp_which))
            v = sp[keep] - float(bmaxsp_target)
            res.update(bmaxsp_profile=sp)
        if bminslope_target is not None and len(keep) > 1:   # well-bottom radial gradient in the same row
            bm, b00 = bmin_line_mean(bmnc[:nq], jnp.asarray(booz["xm_b"], jnp.float64),
                                     jnp.asarray(booz["xn_b"], jnp.float64), jnp.asarray(booz["iota_b"])[:nq],
                                     int(booz["nfp"]), bmns_b=None if bmns is None else bmns[:nq],
                                     nphi=int(opts.get("nphi", 101)), nalpha=int(opts.get("nalpha", 21)),
                                     soft=float(bmaxsp_soft))
            sk = jnp.asarray(booz["s_b"])[:nq][keep]
            slope = jnp.diff(bm[keep]) / jnp.diff(sk) / (0.5 * (b00[keep][1:] + b00[keep][:-1]))
            v = jnp.concatenate([v, slope - float(bminslope_target)])
            res.update(bminslope_profile=slope)
        res.update(bmaxsp=tb * jax.scipy.special.logsumexp(v / tb))
    if nu:                          # B_min uniformity rows appended to the QI residual vector
        bs = slice(nb + nj, nb + nj + nu)
        rb = bmin_uniformity(bmnc_b=bmnc[bs], bmns_b=None if bmns is None else bmns[bs], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
                             iota_b=jnp.asarray(booz["iota_b"])[bs], nfp=int(booz["nfp"]), nalpha=int(bminu_nalpha),
                             nphi=int(bminu_nphi), tau=float(bminu_tau), weight=float(bminu_weight))
        res.update(qs=jnp.concatenate([res["qs"], rb]), qs_total=res["qs_total"] + jnp.sum(rb * rb),
                   qi_only=out["total"], bmin_pen=jnp.sum(rb * rb))
    if nd:                          # real-field dJ/ds rows for shallow wells, appended to the QI residual vector
        ds_ = slice(nb + nj + nu, nb + nj + nu + nd)
        rd = real_field_djds(bmnc_b=bmnc[ds_], bmns_b=None if bmns is None else bmns[ds_], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
                             iota_b=jnp.asarray(booz["iota_b"])[ds_], G_b=jnp.asarray(booz["G_b"])[ds_], I_b=jnp.asarray(booz["I_b"])[ds_],
                             nfp=int(booz["nfp"]), s_b=jnp.asarray(dsurf), depths=tuple(djds_depths), nalpha=int(djds_nalpha),
                             nphi=int(djds_nphi), tau_g=float(djds_tau_g), weight=float(djds_weight))
        res.update(qs=jnp.concatenate([res["qs"], rd]), qs_total=res["qs_total"] + jnp.sum(rd * rd), djds_pen=jnp.sum(rd * rd))
        if jspread_weight is not None:   # real-field J spread over alpha at the same surfaces (J-contour closure)
            rs = real_field_j_spread(bmnc_b=bmnc[ds_], bmns_b=None if bmns is None else bmns[ds_], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
                                     iota_b=jnp.asarray(booz["iota_b"])[ds_], G_b=jnp.asarray(booz["G_b"])[ds_], I_b=jnp.asarray(booz["I_b"])[ds_],
                                     nfp=int(booz["nfp"]), depths=tuple(jspread_depths), nalpha=int(djds_nalpha), nphi=int(djds_nphi),
                                     weight=float(jspread_weight))
            res.update(qs=jnp.concatenate([res["qs"], rs]), qs_total=res["qs_total"] + jnp.sum(rs * rs), jspread_pen=jnp.sum(rs * rs))
    if maxj_target is not None:     # max-J constraint on the constructed field (constructed_maximum_j)
        s_b = jnp.asarray(booz["s_b"])[:nq]
        mj = constructed_maximum_j(out, iota_b=jnp.asarray(booz["iota_b"])[:nq],
                                   G_b=jnp.asarray(booz["G_b"])[:nq], I_b=jnp.asarray(booz["I_b"])[:nq],
                                   s_b=s_b, target=float(maxj_target), depths=maxj_depths, tau=float(maxj_tau))
        res.update(maxj=mj["g"], maxj_fraction=mj["fraction"], maxj_slope=mj["slope"])
    if nj:                          # J-contour closure on the real field, folded into the max-J row
        js = slice(nb, nb + nj)
        jr = real_field_j_closure(bmnc_b=bmnc[js], bmns_b=None if bmns is None else bmns[js], xm_b=booz["xm_b"],
                                  xn_b=booz["xn_b"], iota_b=jnp.asarray(booz["iota_b"])[js],
                                  G_b=jnp.asarray(booz["G_b"])[js], I_b=jnp.asarray(booz["I_b"])[js],
                                  nfp=int(booz["nfp"]), fracs=jreal_fracs,
                                  target=(float(jreal_target) if np.ndim(jreal_target) == 0 else jreal_target),
                                  nphi=int(jreal_nphi), nalpha=int(jreal_nalpha), tau=float(jreal_tau),
                                  order=str(jreal_order), enclose_inner=jreal_enclose_inner,
                                  measure=str(jreal_measure), tau_sep=jreal_tau_sep,
                                  tau_alpha=float(jreal_tau_alpha), width=float(jreal_width),
                                  row_weights=jreal_row_weights)
        res.update(maxj_constructed=res.get("maxj"), jreal=jr["g"], jreal_d=jr["d"])
        res["maxj"] = (jr["g"] if "maxj" not in res else
                       float(maxj_tau) * jnp.logaddexp(res["maxj"] / float(maxj_tau), jr["g"] / float(maxj_tau)))
    return res

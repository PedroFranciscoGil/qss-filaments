"""QI physics building blocks on a synthetic Boozer spectrum (pure JAX, no VMEX solve)."""
import jax.numpy as jnp
import numpy as np
import pytest

from phrport import qi_residual as Q
from phrport.coils import lp_curvature


def _spectrum(ns=3, nfp=5, eps=0.1, ripple=0.0):
    # |B| = 1 + eps cos(nfp phi) [+ ripple cos(theta - nfp phi)] in Boozer modes (m, n) = (0,0), (0,nfp), (1,nfp)
    xm = np.array([0, 0, 1]); xn = np.array([0, nfp, nfp])
    b = np.tile(np.array([1.0, eps, ripple]), (ns, 1))
    return jnp.asarray(b), xm, xn


def test_mirror_ratio():
    b, xm, xn = _spectrum(ns=1, eps=0.1)
    m = Q.mirror_ratio_from_boozer(b, jnp.asarray(xm, float), jnp.asarray(xn, float), 5)
    assert float(m[0]) == pytest.approx(0.1, rel=1e-6)          # (1.1 - 0.9) / (1.1 + 0.9)


def test_bmax_spread_zero_for_omnigenous_field_and_positive_with_ripple():
    b, xm, xn = _spectrum(eps=0.1)
    iota = jnp.full(3, 0.9)
    sp0 = Q.bmax_spread(b, xm, xn, iota, 5, nphi=101, nalpha=11)
    b2, _, _ = _spectrum(eps=0.1, ripple=0.02)
    sp1 = Q.bmax_spread(b2, xm, xn, iota, 5, nphi=101, nalpha=11)
    assert np.all(np.abs(np.asarray(sp0)) < 2e-3)
    assert np.all(np.asarray(sp1) > np.asarray(sp0))


def test_real_field_j_closure_is_finite_and_differentiable():
    b, xm, xn = _spectrum(ns=4, eps=0.1, ripple=0.01)
    kw = dict(xm_b=xm, xn_b=xn, iota_b=jnp.full(4, 0.9), G_b=jnp.ones(4), I_b=jnp.zeros(4), nfp=5, nphi=101,
              nalpha=8)
    out = Q.real_field_j_closure(bmnc_b=b, **kw)
    assert np.isfinite(float(out["g"]))
    import jax
    g = jax.grad(lambda bb: Q.real_field_j_closure(bmnc_b=bb, **kw)["g"])(b)
    assert np.all(np.isfinite(np.asarray(g)))


def test_lp_curvature_circle():
    R, n = 2.0, 256
    t = jnp.arange(n) / n * 2 * jnp.pi
    gd = jnp.stack([-R * jnp.sin(t), R * jnp.cos(t), 0 * t], -1)[None] * 2 * jnp.pi    # d/dt, t in [0, 1)
    gdd = jnp.stack([-R * jnp.cos(t), -R * jnp.sin(t), 0 * t], -1)[None] * (2 * jnp.pi) ** 2
    v = float(lp_curvature(gd, gdd, 2.0, 0.0)[0])
    assert v == pytest.approx(0.5 * (1 / R) ** 2 * 2 * np.pi * R, rel=1e-10)
    assert float(lp_curvature(gd, gdd, 2.0, 1.0)[0]) == 0.0      # kappa = 0.5 below the threshold

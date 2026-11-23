"""Plasma's own magnetic field on the plasma boundary (finite beta), for the coil problem.

Virtual casing (vmex.core.virtual_casing + virtual_casing_jax, the accurate on-surface singular
quadrature) on a uniform (phi, theta) grid of one field period of the LCFS of a VMEX state, stored as
cylindrical components (B_R, B_phi, B_Z) -- smooth and periodic in theta and in phi over a field
period -- and evaluated anywhere on the boundary by trigonometric (FFT) interpolation.  The whole
chain (state, params) -> grid field -> interpolated field is JAX-traceable once the virtual-casing
precision plan is fixed, so the field term of the envelope gradient is one reverse pass.

The grid's angles are the boundary's own (VMEX keeps the input parametrization at s = 1), i.e. the
(theta, phi) of XCoil's surface tables: B_plasma(theta_q, phi_q) needs no geometry matching.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


def _trig_eval(F, kphi, kth, nfp, TH, PH):
    """Real trigonometric interpolant sum_k F_k e^{i(kphi nfp phi + kth theta)} at points (TH, PH)."""
    ph = jnp.exp(1j * nfp * PH[:, None] * kphi[None, :])           # (N, nphi)
    th = jnp.exp(1j * TH[:, None] * kth[None, :])                   # (N, ntheta)
    return jnp.real(jnp.einsum("pk,ckl,pl->pc", ph, F, th))         # (N, ncomp)


class PlasmaField:
    """B_plasma on the LCFS of a VMEX state (virtual casing), interpolable to any (theta, phi)."""

    def __init__(self, nfp: int, n: int = 32, digits: int = 4, chunk_size: int = 1024):
        self.nfp, self.n, self.digits, self.chunk = int(nfp), int(n), int(digits), int(chunk_size)
        self.plan = None

    def surface_data(self, inp, state, runtime=None):
        from vmex.core import virtual_casing as vc
        return vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=self.n, ntheta=self.n)

    def make_plan(self, sd):
        """Fix the virtual-casing precision plan on a concrete surface (once; keeps the map smooth).

        virtual_casing_jax's quadrature-size selection runs its self-test unchunked (hard-coded source
        chunk 1024, all singular corrections in one vmap: 14 GB at 7 digits, 19 GB at 8); it is run here
        with this object's source chunk and the singular corrections scanned 8 targets at a time (same
        result, bounded memory)."""
        from vmex.core import virtual_casing as vc
        import virtual_casing_jax.virtual_casing as vcj
        orig = vcj.laplace_dx_u_eval_singular

        def chunked(*args, **kw):
            kw["chunk_size"] = min(int(kw.get("chunk_size", self.chunk)), self.chunk)
            if not kw.get("target_chunk_size"):
                kw["target_chunk_size"] = 8          # singular corrections in a scan over targets
            return orig(*args, **kw)

        vcj.laplace_dx_u_eval_singular = chunked
        try:
            self.plan = vc.plan_vc_precision(sd, digits=self.digits, chunk_size=self.chunk)
        finally:
            vcj.laplace_dx_u_eval_singular = orig
        return self.plan

    def grid_cyl(self, sd):
        """(3, nphi, ntheta) cylindrical (B_R, B_phi, B_Z) of the plasma field on sd's grid."""
        from vmex.core import virtual_casing as vc
        if self.plan is None:
            self.make_plan(sd)
        Bp = vc.plasma_field_on_boundary(sd, digits=self.digits, chunk_size=self.chunk, precision=self.plan)
        phi = jnp.asarray(sd.phi)[:, None]
        c, s = jnp.cos(phi), jnp.sin(phi)
        return jnp.stack([Bp[0] * c + Bp[1] * s, -Bp[0] * s + Bp[1] * c, Bp[2]], axis=0)

    def interp_cart(self, Bcyl, TH, PH):
        """Cartesian B_plasma (N, 3) at boundary angles (TH, PH) from the grid field Bcyl."""
        nphi, nth = Bcyl.shape[1], Bcyl.shape[2]
        F = jnp.fft.fft2(Bcyl, axes=(1, 2)) / (nphi * nth)
        kphi = jnp.asarray(np.fft.fftfreq(nphi) * nphi)
        kth = jnp.asarray(np.fft.fftfreq(nth) * nth)
        TH, PH = jnp.asarray(TH), jnp.asarray(PH)
        cyl = _trig_eval(F, kphi, kth, self.nfp, TH, PH)            # (N, 3): B_R, B_phi, B_Z
        c, s = jnp.cos(PH), jnp.sin(PH)
        return jnp.stack([cyl[:, 0] * c - cyl[:, 1] * s, cyl[:, 0] * s + cyl[:, 1] * c, cyl[:, 2]], axis=-1)

    def at(self, inp, state, TH, PH, runtime=None):
        """Cartesian B_plasma (N, 3) at (TH, PH) for a VMEX state (traceable in state / runtime)."""
        return self.interp_cart(self.grid_cyl(self.surface_data(inp, state, runtime)), TH, PH)

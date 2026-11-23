"""Exact (implicit-function) Jacobian of the PHR equilibrium rows.

Opt-in replacement for the forward finite-difference equilibrium block of
:func:`phrport.phr.least_squares_qss`: d(QS residual vector, aspect, mean iota,
elongation)/d(free boundary DOFs) of a continuation step from ONE cold VMEX
solve (identical to :func:`phrport.equilibrium.evaluate`) plus one
block-factored multi-RHS implicit tangent solve, pushed through JAX copies of
the metrics. Mode 4 (80 DOFs): ~3.5 s warm vs ~48 s for PHR's 81-point FD over
an 8-worker :class:`phrport.vmex_pool.VmexPool`.

Pieces
------
* :func:`traceable_metrics` -- JAX re-implementation of the wout path used by
  ``evaluate``: the ``wout_from_state`` field lane on the Nyquist-extended grid,
  the VMEC2000 ``bsubvmnc`` (bcovar IEQUI=1 correction + jxbforce filter, see
  :func:`phrport.equilibrium.vmec2000_bsubvmnc`), the ``wrout.f`` Nyquist
  analysis, then simsopt/VMEX's ``QuasisymmetryRatioResidual`` (same 63x64 grid,
  same half-grid interpolation/extrapolation in s), ``aspect`` (aspectratio.f),
  ``mean(iotas[1:])`` (ncurr = 1: chips/phips) and the axis/LCFS coefficients.
  Values equal ``evaluate``'s to ~1e-14.
* :func:`elongation_jax` -- PHR's ``MaxElongationMetric`` in JAX: the 50x50
  fsolve roots are recomputed by vectorized Newton and differentiated by the
  implicit function theorem (one un-stopped Newton step at the converged root),
  the ellipse fit by bisection + one Newton step, ``ellipe`` by AGM. Value
  equals the vendored scipy one to ~1e-14; gradient equals FD of the scipy
  metric to < 1e-7.
* :class:`ExactEquilibriumJacobian` -- cold solve, then VMEX's raw-force
  block-tridiagonal factorization (``implicit._raw_block_system``) solving all
  free-DOF tangents at once, then ``jax.jvp`` (vmapped) through the metrics.
* :class:`GaugeContinuedMap` -- the smooth map whose derivative this is EXACTLY
  (FD of it agrees to <= 5e-8); :func:`phr_rows_jacobian` /
  :class:`PHRExactJacobian` -- assembly in phr.py's row order/scaling.
* Opt-in state reuse (roadmap S-c): ``ExactEquilibriumJacobian(...)(boundary,
  state=...)`` takes the converged state of an earlier solve of the same
  boundary (``equilibrium.evaluate(..., return_state=True)`` or
  ``VmexPool(return_state=True)``; checked by VMEX-input fingerprint) and skips
  the cold re-solve and the numpy metrics: the Jacobian is bitwise the same;
  warm time per call -0.9/-1.7/-2.1/-1.8 s at modes 1-4 (-44 to -77%,
  the developer tests). ``PHRExactJacobian(state_lookup=...)``
  is the runner-side hook.

The m=1 gauge (the essential caveat)
------------------------------------
VMEC/VMEX stop evolving one m=1 combination (``Z_sin(1,+n) - Z_sin(1,-n)`` in
the m=1-constrained representation; residue.f90 zeroes its force once
fsqz < 1e-6). It is an angle gauge in the continuum, but each cold solve
freezes it wherever its iteration path happened to be (in the PHR runs: the
first ~40 iterations of the ns=50 stage). The cold map's derivative therefore
contains a path-dependent gauge drift that no implicit derivative can
reproduce (verified: implicit tangent + the FD-measured gauge drift reproduces
the cold FD to 1e-4..1e-3). This module differentiates a gauge-continued map
instead (``gauge="sqrt"``: the gauge coordinate follows the boundary with the
sqrt(s) profile of the initial guess; ``"frozen"``: VMEX's convention). Against
FD of ``evaluate`` (scale200 checkpoints) the "sqrt" Jacobian agrees to ~5% in
the QS block and <1% in mean iota at modes 1-2 and with large-step (1e-3) FD at
mode 4, where FD at PHR's step 1e-5 is itself dominated by solver jumps.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
import time
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from . import equilibrium as E

__all__ = [
    "StepContext", "step_context", "traceable_metrics", "elongation_jax", "GAUGES",
    "ExactEquilibriumJacobian", "ExactJacResult", "phr_rows_jacobian", "PHRExactJacobian",
    "GaugeContinuedMap",
]


# ---------------------------------------------------------------------------
# static per-step context
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class StepContext:
    step: int
    M: int
    inp: object               # VmecInput template (seed boundary, PHR settings)
    res: object               # final-stage Resolution (ns = 50)
    modes: object
    trig_nyq: object
    nyq_modes: object
    xm: np.ndarray
    xn: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    signgs: int
    s_full: np.ndarray
    sqrts: np.ndarray
    names: tuple
    consts: dict


def _pshalf_from_s(s_full):
    s_arr = np.asarray(s_full, dtype=float)
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    return np.sqrt(np.maximum(np.concatenate([sh[:1], sh], axis=0), 0.0))


@lru_cache(maxsize=8)
def step_context(step: int) -> StepContext:
    from vmex.core import nyquist as NQ
    from vmex.core.fourier import Resolution, mode_table, trig_tables
    from vmex.core.setup import boundary_from_input, radial_grids
    from vmex.core.solver import resolution_from_input
    from vmex.core.transforms import physical_to_internal_scale

    step = int(step)
    inp = E.vmec_input(E.seed_boundary(), step)
    ns = int(np.max(np.asarray(inp.ns_array)))
    res = resolution_from_input(inp, ns=ns)
    mpol, ntor, nfp = int(res.mpol), int(res.ntor), int(res.nfp)
    assert not bool(res.lasym)
    modes = mode_table(mpol, ntor)
    mnyq_grid = max(res.ntheta1 // 2, mpol - 1)
    nnyq_grid = max(int(res.nzeta) // 2, ntor)
    trig = trig_tables(Resolution(mpol=mnyq_grid + 1, ntor=nnyq_grid, ntheta=int(res.ntheta),
                                  nzeta=int(res.nzeta), nfp=nfp, lasym=False, ns=ns))
    grids = radial_grids(ns)
    s = np.asarray(grids.s_full, dtype=float)
    bnd = boundary_from_input(inp, modes=modes, trig=trig, lconm1=True)
    signgs = int(bnd.signgs)
    mnyq, nnyq = NQ.nyquist_limits(trig)
    nyq_modes = mode_table(max(mnyq, mpol - 1) + 1, max(nnyq, ntor))
    xm_nyq = np.asarray(nyq_modes.m, dtype=float)
    xn_nyq = np.asarray(nyq_modes.n, dtype=float) * nfp
    nzeta_vmec = int(inp.nzeta) or (1 if ntor == 0 else int(res.nzeta))
    if nzeta_vmec // 2 != int(np.max(xn_nyq)) // nfp:
        raise NotImplementedError("Nyquist column expansion not needed at PHR settings")

    nt2 = int(trig.ntheta2)
    c = {}
    # wrout.f cosine analysis (nyquist.wrout_cos_coeffs)
    cosmui_w, sinmui_w = NQ._wrout_theta_tables(trig)
    cosnv_w, sinnv_w = NQ._wrout_zeta_tables(trig)
    m_idx = np.asarray(nyq_modes.m, dtype=int)
    n_idx = np.asarray(nyq_modes.n, dtype=int)
    c["w_cosmui"], c["w_sinmui"], c["w_cosnv"], c["w_sinnv"] = cosmui_w, sinmui_w, cosnv_w, sinnv_w
    c["w_m"], c["w_n"] = m_idx, np.abs(n_idx)
    c["w_sgn"] = np.where(n_idx < 0, -1.0, 1.0)
    c["w_dmult"] = NQ._wrout_dmult(nyq_modes, trig)
    c["nt2"] = nt2
    # jxbforce.f symmetric filter (nyquist.filter_bsubuv_symmetric)
    mmax, nmax = mpol - 1, ntor
    cosmui_a, sinmui_a = NQ._analysis_theta_tables(trig)
    c["f_cosmui"] = cosmui_a[:, : mmax + 1]
    c["f_sinmui"] = sinmui_a[:, : mmax + 1]
    c["f_cosmu"] = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    c["f_sinmu"] = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    c["f_cosnv"] = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    c["f_sinnv"] = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]
    dmult = np.ones((mmax + 1, nmax + 1))
    if 0 < mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if 0 < nnyq <= nmax:
        dmult[:, nnyq] *= 0.5
    c["f_dmult"] = dmult
    c["wint"] = np.asarray(trig.wint, dtype=float)
    c["pshalf"] = _pshalf_from_s(s)
    # QS grid (optimize.QuasisymmetryRatioResidual.compute)
    ntheta_q, nphi_q = E.QS_NTHETA, E.QS_NPHI
    theta1d = np.linspace(0.0, 2.0 * np.pi, ntheta_q, endpoint=False)
    phi1d = np.linspace(0.0, 2.0 * np.pi / nfp, nphi_q, endpoint=False)
    c["q_dtheta"] = float(theta1d[1] - theta1d[0])
    c["q_dphi"] = float(phi1d[1] - phi1d[0])
    ang = theta1d[:, None, None] * xm_nyq[None, None, :] - phi1d[None, :, None] * xn_nyq[None, None, :]
    c["q_cos"], c["q_sin"] = np.cos(ang), np.sin(ang)
    surfaces = np.asarray(E.QS_SURFACES, dtype=float)
    s_full_q = np.linspace(0.0, 1.0, ns)
    s_half = 0.5 * (s_full_q[:-1] + s_full_q[1:])
    hi = np.clip(np.searchsorted(s_half, surfaces, side="left"), 1, s_half.shape[0] - 1)
    lo = hi - 1
    x0, x1 = s_half[lo], s_half[hi]
    c["q_lo"], c["q_hi"] = lo, hi
    c["q_t"] = (surfaces - x0) / np.where(x1 != x0, x1 - x0, 1.0)
    mode_scale = 1.0 / physical_to_internal_scale(modes, trig)
    c["mode_scale"] = mode_scale

    return StepContext(step=step, M=E.vmec_resolution(step), inp=inp, res=res, modes=modes,
                       trig_nyq=trig, nyq_modes=nyq_modes,
                       xm=np.asarray(modes.m, dtype=float), xn=np.asarray(modes.n, dtype=float) * nfp,
                       xm_nyq=xm_nyq, xn_nyq=xn_nyq, signgs=signgs, s_full=s,
                       sqrts=np.asarray(grids.sqrts, dtype=float),
                       names=tuple(E.free_dof_names(step)), consts=c)


# ---------------------------------------------------------------------------
# traceable wout path
# ---------------------------------------------------------------------------
def _wrout_cos(f, c):
    import jax.numpy as jnp
    f = f[:, : c["nt2"], :]
    ftc = jnp.einsum("sik,im->smk", f, c["w_cosmui"])
    fts = jnp.einsum("sik,im->smk", f, c["w_sinmui"])
    cz = jnp.einsum("smk,kn->smn", ftc, c["w_cosnv"])
    sz = jnp.einsum("smk,kn->smn", fts, c["w_sinnv"])
    coeff = cz[:, c["w_m"], c["w_n"]] + c["w_sgn"][None, :] * sz[:, c["w_m"], c["w_n"]]
    return coeff * c["w_dmult"][None, :]


def _filter_sym(f, c):
    import jax.numpy as jnp
    f = f[:, : c["nt2"], :]
    ftc = jnp.einsum("sik,im->smk", f, c["f_cosmui"])
    fts = jnp.einsum("sik,im->smk", f, c["f_sinmui"])
    c1 = jnp.einsum("smk,kn->smn", ftc, c["f_cosnv"]) * c["f_dmult"][None]
    c2 = jnp.einsum("smk,kn->smn", fts, c["f_sinnv"]) * c["f_dmult"][None]
    tc = jnp.einsum("smn,im->sin", c1, c["f_cosmu"])
    ts = jnp.einsum("smn,im->sin", c2, c["f_sinmu"])
    return jnp.einsum("sin,kn->sik", tc, c["f_cosnv"]) + jnp.einsum("sin,kn->sik", ts, c["f_sinnv"])


def _bsubv_vmec2000(geometry, jacobian, metrics, fields, s, phipf, c):
    """jnp port of nyquist.bsubv_lambda_full_mesh + apply_bsubv_equif_correction (VMEX, MIT License,
    Copyright (c) 2026 UW Plasma; see THIRD_PARTY_NOTICES.md)."""
    import jax.numpy as jnp
    ns = int(s.shape[0])
    sqrt_g = jacobian.sqrt_g
    lamscale = jnp.asarray(fields.lamscale)
    lu_even = lamscale * geometry.dlambda_dtheta_even + phipf[:, None, None]
    lu_odd = lamscale * geometry.dlambda_dtheta_odd
    nz = sqrt_g != 0.0
    phipog = jnp.where(nz, 1.0 / jnp.where(nz, sqrt_g, 1.0), 0.0)
    phipog = phipog.at[0].set(0.0)
    pshalf = jnp.asarray(c["pshalf"])[:, None, None]
    lvv = phipog * metrics.gvv
    lvv_sh = lvv * pshalf
    cross = metrics.guv * fields.bsupu
    full_in = 0.5 * ((lvv[:-1] + lvv[1:]) * lu_even[:-1] + (lvv_sh[:-1] + lvv_sh[1:]) * lu_odd[:-1]
                     + cross[:-1] + cross[1:])
    full_last = 0.5 * (lvv[-1] * lu_even[-1] + lvv_sh[-1] * lu_odd[-1] + cross[-1])
    bsubv_full = jnp.concatenate([full_in, full_last[None]], axis=0)
    bh = fields.bsubv
    avg = jnp.concatenate([0.5 * (bh[:-1] + bh[1:]), 0.5 * bh[-1:]], axis=0)
    bdamp = (2.0 * 0.05 * (1.0 - np.asarray(s, float)))[:, None, None]
    bsubv_e = bdamp * bsubv_full + (1.0 - bdamp) * avg
    # IEQUI=1 backward recurrence + average-preserving shift
    wint = c["wint"]
    fpsi = jnp.sum(bh * wint[None], axis=(1, 2))
    rows = [None] * ns
    rows[ns - 1] = bh[ns - 1]
    for js in range(ns - 2, 0, -1):
        rows[js] = 2.0 * bsubv_e[js] - rows[js + 1]
    rows[0] = bh[0]
    out = jnp.stack(rows, axis=0)
    shift = fpsi - jnp.sum(out * wint[None], axis=(1, 2))
    shift = shift.at[0].set(0.0)
    return out + shift[:, None, None]


def _interp(values, c):
    """optimize._interp_half_grid(values[1:], surfaces, s_half) with static indices."""
    v = values[1:]
    t = c["q_t"].reshape((-1,) + (1,) * (v.ndim - 1))
    return v[c["q_lo"]] + t * (v[c["q_hi"]] - v[c["q_lo"]])


def traceable_metrics(state, params, ctx: StepContext, cfg, merc=True):
    """JAX map (state, params) -> PHR wout-path metrics (traceable, jvp-able).

    Returns a dict: qs (44352,), aspect, mean_iota, iota_axis, iota_edge,
    rmnc_edge, zmns_edge (main modes, wout normalization), raxis_cc, zaxis_cs.
    """
    import jax.numpy as jnp
    from vmex.core import implicit as imp
    from vmex.core.fields import magnetic_fields, metric_elements
    from vmex.core.geometry import apply_lambda_axis_closure, half_mesh_jacobian, real_space_geometry
    from vmex.core.residuals import m1_constrained_to_physical

    c = ctx.consts
    res = ctx.res
    rt = imp.runtime_from_params(params, cfg)
    setup = rt.setup
    s = jnp.asarray(ctx.s_full)
    R_cos_p, Z_sin_p, R_sin_p, Z_cos_p = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=ctx.modes, lthreed=bool(res.lthreed), lasym=False, lconm1=True)
    lambda_sin = apply_lambda_axis_closure(state.L_sin, modes=ctx.modes, ntor=int(res.ntor))
    geometry = real_space_geometry(R_cos=R_cos_p, R_sin=R_sin_p, Z_cos=Z_cos_p, Z_sin=Z_sin_p,
                                   lambda_cos=state.L_cos, lambda_sin=lambda_sin,
                                   modes=ctx.modes, trig=ctx.trig_nyq, s=s)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(geometry=geometry, jacobian=jacobian, metrics=metrics, trig=ctx.trig_nyq,
                             s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
                             signgs=ctx.signgs, gamma=float(ctx.inp.gamma), mass=setup.mass,
                             ncurr=int(ctx.inp.ncurr), enclosed_current=setup.icurv)
    sqrtg = jacobian.sqrt_g
    bmag = jnp.sqrt(2.0 * jnp.abs(fields.total_pressure - fields.pressure[:, None, None]))
    bsubv_c = _bsubv_vmec2000(geometry, jacobian, metrics, fields, ctx.s_full, setup.phipf, c)

    def tab(f):
        return _wrout_cos(f, c).at[0].set(0.0)

    gmnc = tab(sqrtg)
    bmnc = tab(bmag)
    bsupumnc = tab(fields.bsupu)
    bsupvmnc = tab(fields.bsupv)
    bsubumnc = tab(_filter_sym(fields.bsubu, c))
    bsubvmnc = tab(_filter_sym(bsubv_c, c))
    i00 = 0
    buco = bsubumnc[:, i00].at[0].set(0.0)
    bvco = bsubvmnc[:, i00].at[0].set(0.0)
    # iotas (ncurr = 1: add_fluxes.f90), axis slot zeroed as in wout_from_state
    phips = jnp.asarray(setup.phips).at[0].set(0.0)
    if int(ctx.inp.ncurr) == 1:
        safe = jnp.where(phips != 0.0, phips, 1.0)
        iotas = jnp.where(phips != 0.0, fields.chips / safe, 0.0)
    else:
        iotas = jnp.asarray(setup.iotas)
    iotas = iotas.at[0].set(0.0)
    iotaf0 = 1.5 * iotas[1] - 0.5 * iotas[2]
    iotaf_edge = 1.5 * iotas[-1] - 0.5 * iotas[-2]
    phipf_out = 2.0 * np.pi * ctx.signgs * jnp.asarray(setup.phipf)
    phi_edge = jnp.sum(phipf_out[1:] * (s[1:] - s[:-1]))
    # aspect (aspectratio.f on the internal grid)
    se = ctx.sqrts[-1]
    rb = geometry.R_even[-1] + se * geometry.R_odd[-1]
    zub = geometry.dZ_dtheta_even[-1] + se * geometry.dZ_dtheta_odd[-1]
    t1 = rb * zub * c["wint"]
    volume_p = 2.0 * np.pi * np.pi * jnp.abs(jnp.sum(rb * t1))
    area = 2.0 * np.pi * jnp.abs(jnp.sum(t1))
    aminor = jnp.sqrt(area / np.pi)
    rmajor = volume_p / (2.0 * np.pi * area)
    aspect = rmajor / aminor
    # QS (optimize.QuasisymmetryRatioResidual.compute, lasym = False)
    nfp = int(res.nfp)
    iota_h = _interp(iotas, c)
    G = _interp(bvco, c)
    I = _interp(buco, c)  # noqa: E741
    T = {k: _interp(v, c) for k, v in dict(gmnc=gmnc, bmnc=bmnc, bsubumnc=bsubumnc, bsubvmnc=bsubvmnc,
                                             bsupumnc=bsupumnc, bsupvmnc=bsupvmnc).items()}
    cs, sn = c["q_cos"], c["q_sin"]
    xm, xn = ctx.xm_nyq, ctx.xn_nyq

    def synth(t):
        return jnp.einsum("sm,tpm->stp", t, cs)

    modB = synth(T["bmnc"])
    dB_dtheta = jnp.einsum("sm,tpm,m->stp", T["bmnc"], -sn, xm)
    dB_dphi = jnp.einsum("sm,tpm,m->stp", T["bmnc"], sn, xn)
    sg = synth(T["gmnc"])
    bsubu = synth(T["bsubumnc"]); bsubv = synth(T["bsubvmnc"])
    bsupu = synth(T["bsupumnc"]); bsupv = synth(T["bsupvmnc"])
    d_psi_d_s = -phi_edge / (2.0 * np.pi)
    sg_safe = jnp.where(sg != 0.0, sg, 1.0)
    BdgB = bsupu * dB_dtheta + bsupv * dB_dphi
    BxgB = d_psi_d_s * (bsubu * dB_dphi - bsubv * dB_dtheta) / sg_safe
    tiny = np.finfo(np.float64).tiny
    sg_abs = jnp.maximum(jnp.abs(sg), tiny)
    modB_safe = jnp.maximum(jnp.abs(modB), tiny)
    dth, dph = c["q_dtheta"], c["q_dphi"]
    Vp = nfp * dth * dph * jnp.sum(sg_abs, axis=(1, 2))
    hm, hn = E.QS_HELICITY_M, E.QS_HELICITY_N * nfp
    pref = jnp.sqrt(nfp * dth * dph / Vp[:, None, None] * sg_abs)
    r3 = pref * (BxgB * (hn - iota_h[:, None, None] * hm)
                 - BdgB * (hm * G[:, None, None] + hn * I[:, None, None])) / modB_safe ** 3
    ms = c["mode_scale"]
    rmnc = R_cos_p * ms[None, :]
    zmns = Z_sin_p * ms[None, :]
    n_axis = int(res.ntor) + 1
    # vacuum well (simsopt Vmec.vacuum_well == equilibrium.vacuum_well_from_wout) on gmnc(0,0)
    dVds = 4.0 * np.pi * np.pi * jnp.abs(gmnc[1:, i00])
    dV0 = 1.5 * dVds[0] - 0.5 * dVds[1]
    dV1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]
    well = (dV0 - dV1) / dV0
    qs_vec = jnp.ravel(r3)
    extra = {}
    if E.PHYSICS == "qi":
        # QI profile: the physics rows are the continuous constructed-QI residual (one traceable
        # Boozer transform of the state), plus the mirror ratio as a constraint scalar
        from .qi_residual import qi_physics
        o = dict(E.QI_OPTIONS)
        qi = qi_physics(state, rt, surfaces=E.QI_SURFACES, mirror_s=E.QI_MIRROR_S, mboz=o.pop("mboz"),
                        nboz=o.pop("nboz"), oversample=o.pop("oversample"), **o)
        qs_vec = qi["qs"]
        extra["mirror"] = qi["mirror"]
        if "maxj" in qi:
            extra["maxj"] = qi["maxj"]
        if "bmaxsp" in qi:
            extra["bmaxsp"] = qi["bmaxsp"]
    if E.MERC_OPTIONS is not None and merc:     # Jacobian pass: pushed separately in column batches (core)
        from .mercier import mercier_metric
        extra["merc"] = mercier_metric(state, rt, **E.MERC_OPTIONS)["soft_min"]
    return dict(qs=qs_vec, aspect=aspect, mean_iota=jnp.mean(iotas[1:]), well=well, **extra,
                volume=volume_p, aminor=aminor,
                iota_axis=iotaf0, iota_edge=iotaf_edge,
                rmnc_edge=rmnc[-1], zmns_edge=zmns[-1],
                raxis_cc=rmnc[0, :n_axis], zaxis_cs=zmns[0, :n_axis],
                # diagnostics for table-level validation
                _tables=dict(gmnc=gmnc, bmnc=bmnc, bsubumnc=bsubumnc, bsubvmnc=bsubvmnc,
                             bsupumnc=bsupumnc, bsupvmnc=bsupvmnc, iotas=iotas, bvco=bvco,
                             buco=buco, rmnc=rmnc, zmns=zmns))


# ---------------------------------------------------------------------------
# elongation (MaxElongationMetric) in JAX with implicit root derivatives
# ---------------------------------------------------------------------------
def _ellipe_agm(m, n_iter=12):
    import jax.numpy as jnp
    a = jnp.ones_like(m)
    b = jnp.sqrt(1.0 - m)
    csum = 0.5 * m
    p = 0.5
    for _ in range(n_iter):
        cc = 0.5 * (a - b)
        a, b = 0.5 * (a + b), jnp.sqrt(a * b)
        p = 2.0 * p
        csum = csum + p * cc * cc
    K = np.pi / (2.0 * a)
    return K * (1.0 - csum)


def elongation_jax(rmnc_edge, zmns_edge, raxis_cc, zaxis_cs, xm, xn, nfp,
                   ntheta=E.ELONG_NTHETA, nphi=E.ELONG_NPHI, n_newton=30, n_bisect=80,
                   return_profile=False):
    """PHR ``MaxElongationMetric`` (vendored ``phrport.elongation``) as a JAX function.

    Same grids, same ``+xn*phi`` phase convention, same 'noise subtraction'.
    Roots are converged with stop_gradient Newton/bisection and finished by
    one differentiable Newton step (exact implicit derivative at the root).
    """
    import jax
    import jax.numpy as jnp
    sg = jax.lax.stop_gradient
    xm = np.asarray(xm, float); xn = np.asarray(xn, float)
    theta1D = np.linspace(0, 2 * np.pi, num=ntheta)
    phi1D = np.linspace(0, 2 * np.pi / nfp, num=nphi)
    nax = np.arange(np.size(raxis_cc)) * nfp
    sa, ca = np.sin(nax[:, None] * phi1D[None]), np.cos(nax[:, None] * phi1D[None])
    Rax = raxis_cc @ ca
    Zax = zaxis_cs @ sa
    Raxp = raxis_cc @ (-nax[:, None] * sa)
    Zaxp = zaxis_cs @ (nax[:, None] * ca)
    dl = jnp.sqrt(Rax * Rax + Raxp * Raxp + Zaxp * Zaxp)
    tR, tP, tZ = Raxp / dl, Rax / dl, Zaxp / dl
    tX = tR * np.cos(phi1D) - tP * np.sin(phi1D)
    tY = tR * np.sin(phi1D) + tP * np.cos(phi1D)
    T = jnp.stack([tX, tY, tZ], axis=-1)                      # (nphi, 3)
    P = jnp.stack([Rax * np.cos(phi1D), Rax * np.sin(phi1D), Zax], axis=-1)

    def bnd(theta, p):                                       # theta (nth,), p (nphi, nth)
        arg = xm[None, None, :] * theta[None, :, None] + xn[None, None, :] * p[..., None]
        rb = jnp.sum(rmnc_edge * jnp.cos(arg), axis=-1)
        zb = jnp.sum(zmns_edge * jnp.sin(arg), axis=-1)
        return jnp.stack([rb * jnp.cos(p), rb * jnp.sin(p), zb], axis=-1)

    def fdot(p):
        return jnp.sum(T[:, None, :] * (bnd(theta1D, p) - P[:, None, :]), axis=-1)

    def fdot_p(p):
        return jax.jvp(fdot, (p,), (jnp.ones_like(p),))

    p = jnp.broadcast_to(jnp.asarray(phi1D)[:, None], (nphi, ntheta))
    p = sg(p)
    for _ in range(n_newton):
        f, fp = fdot_p(p)
        p = sg(p - f / fp)
    f, fp = fdot_p(p)                 # differentiable finishing step
    p = p - f / sg(fp)
    sb = bnd(theta1D, p)                                      # (nphi, nth, 3)
    sb = sb - jnp.sum(sb * T[:, None, :], axis=-1, keepdims=True)
    d = sb - jnp.roll(sb, 1, axis=1)
    d2 = jnp.sum(d * d, axis=-1)
    pos_ = d2 > 0.0          # theta = 0 and 2*pi are the same point: zero-length segment
    perim = jnp.sum(jnp.where(pos_, jnp.sqrt(jnp.where(pos_, d2, 1.0)), 0.0), axis=1)
    cr = jnp.cross(sb, jnp.roll(sb, -1, axis=1))
    total = jnp.sum(cr, axis=1)                               # (nphi, 3)
    a, b, cc = sb[:, 0], sb[:, 1], sb[:, 2]
    one = jnp.ones_like(a[:, 0])

    def det3(r0, r1, r2):
        return jnp.linalg.det(jnp.stack([jnp.stack(r0, -1), jnp.stack(r1, -1), jnp.stack(r2, -1)], -2))
    x = det3((one, a[:, 1], a[:, 2]), (one, b[:, 1], b[:, 2]), (one, cc[:, 1], cc[:, 2]))
    y = det3((a[:, 0], one, a[:, 2]), (b[:, 0], one, b[:, 2]), (cc[:, 0], one, cc[:, 2]))
    z = det3((a[:, 0], a[:, 1], one), (b[:, 0], b[:, 1], one), (cc[:, 0], cc[:, 1], one))
    mag = jnp.sqrt(x * x + y * y + z * z)
    nrm = jnp.stack([x / mag, y / mag, z / mag], axis=-1)
    A = jnp.abs(jnp.sum(total * nrm, axis=-1) / 2.0)

    def presid(aa, perim_, A_):
        return perim_ - 4.0 * aa * _ellipe_agm(1.0 - (A_ / (np.pi * aa * aa)) ** 2)

    lo = sg(jnp.sqrt(A / np.pi)); hi = sg(perim / 4.0)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        fm = presid(mid, sg(perim), sg(A))
        lo = jnp.where(fm > 0, mid, lo)                       # perimeter too small -> a larger
        hi = jnp.where(fm > 0, hi, mid)
    a1 = sg(0.5 * (lo + hi))
    f, fa = jax.jvp(lambda aa: presid(aa, sg(perim), sg(A)), (a1,), (jnp.ones_like(a1),))
    a1 = sg(a1 - f / fa)
    f = presid(a1, perim, A)
    a1 = a1 - f / sg(fa)
    a2 = A / (np.pi * a1)
    el = jnp.maximum(a1, a2) / jnp.minimum(a1, a2)
    if return_profile:
        return el
    return jnp.max(el)


# ---------------------------------------------------------------------------
# exact Jacobian driver
# ---------------------------------------------------------------------------
@dataclass
class ExactJacResult:
    eq: object                     # EqResult (identical to equilibrium.evaluate at the same boundary)
    names: list                    # free DOF names (column order)
    jac: dict                      # 'qs' (44352, n); 'aspect', 'mean_iota', 'iota_axis', 'iota_edge',
                                   # 'elongation' (n,); 'rmnc_edge', 'zmns_edge', 'raxis_cc', 'zaxis_cs'
    values_jax: dict               # traceable-path values at the base point (diagnostics)
    report: dict                   # timings and the linear-solve residual
    # finite beta (opt-in, want_state_tangents): the converged state / params at the point and the
    # state / parameter tangents of every free DOF (leading axis = DOF), on the VMEX device
    state: object = None
    params: object = None
    state_tangents: object = None
    param_tangents: object = None


GAUGES = ("sqrt", "frozen")


class ExactEquilibriumJacobian:
    """Exact d(PHR equilibrium metrics)/d(free boundary DOFs) for continuation step ``step``.

    One call = one cold VMEX solve identical to :func:`phrport.equilibrium.evaluate`
    (same input, same solver options; the returned ``EqResult`` is that of
    ``evaluate``), then the implicit-function tangent of the converged state for
    every free DOF -- one raw block-tridiagonal factorization of VMEX's force
    Jacobian, all right-hand sides back-solved and refined -- pushed through
    :func:`traceable_metrics` / :func:`elongation_jax` by forward-mode AD.

    What exactly is differentiated (see :class:`GaugeContinuedMap`): the map
    whose evolved dofs solve ``F(x, p) = F(x0, p0)`` and whose NON-evolved m=1
    gauge coordinate (the anti-symmetric ``Z_sin(1,+n) - Z_sin(1,-n)``
    combination VMEC stops evolving once fsqz < 1e-6) follows ``gauge``:

    * ``"sqrt"`` (default): it moves with the boundary's own m=1 anti-combination
      with the sqrt(s) profile of the solver's initial guess;
    * ``"frozen"``: it is held (VMEX's ``implicit_state_tangent_multi_rhs``
      convention; produces a kink at the edge row for the m=1, n != 0 DOFs).

    A cold re-solve instead leaves that coordinate wherever its iteration path
    put it, so FD of ``evaluate`` contains a path-dependent gauge drift no
    implicit derivative can reproduce; at the PHR checkpoints the "sqrt" model
    is the closer one (see the developer tests).
    """

    def __init__(self, step: int, *, gauge: str = "sqrt"):
        import jax
        import jax.numpy as jnp
        from vmex.core import implicit as imp
        if gauge not in GAUGES:
            raise ValueError(f"gauge must be one of {GAUGES}, got {gauge!r}")
        self.jax, self.imp = jax, imp
        self.step = int(step)
        self.gauge = gauge
        self.want_state_tangents = False      # finite beta: also return the state tangents
        self.ctx = ctx = step_context(self.step)
        self.cfg = cfg = imp.make_config(ctx.inp, multigrid=True)
        imp._template_runtime(cfg)          # prime host-side caches outside any trace
        imp._boundary_pack_tables(cfg)
        self.dof_mask = mask = imp._fixed_boundary_dof_mask(cfg)
        self.names = list(ctx.names)
        xm, xn, nfp = ctx.xm, ctx.xn, int(ctx.res.nfp)
        self._metrics = jax.jit(lambda st, prm: _strip(traceable_metrics(st, prm, ctx, cfg)))
        self._elong = jax.jit(lambda r, z, ra, za: elongation_jax(r, z, ra, za, xm, xn, nfp))
        self._elong_grad = jax.jit(jax.grad(
            lambda r, z, ra, za: elongation_jax(r, z, ra, za, xm, xn, nfp), argnums=(0, 1, 2, 3)))
        af = imp._active_state_fields(cfg)
        mn = int(np.shape(mask.R_cos)[1])
        probe_chunk = len(af) * mn
        project = imp._dof_projector(cfg, mask)
        edge = imp._edge_mask(cfg)

        def G(z, p_, fr):        # raw (scalxc-scaled, m=1 released) force on the evolved dofs
            rt_ = imp.runtime_from_params(p_, cfg)
            return project(imp._raw_force_state(imp._assemble(z, rt_, fr, project, edge), rt_))

        merc_split = E.MERC_OPTIONS is not None     # any physics

        def core(prm, x_star, batch, gauge_):
            frozen = jax.lax.stop_gradient(x_star)
            system = imp._raw_block_system(prm, cfg, frozen, mask, af, probe_chunk)
            z_star = system.project(x_star)
            gb = jax.vmap(lambda tp: jax.jvp(
                lambda p_: self.gauge_shift(p_, prm, frozen, gauge_), (prm,), (tp,))[1])(batch)

            def rhs(tp, g):
                v = jax.jvp(lambda p_, fr: G(z_star, p_, fr), (prm, frozen), (tp, g))[1]
                return jax.tree.map(jnp.negative, v)
            b = jax.vmap(rhs)(batch, gb)
            sol, _ = imp._raw_block_solve(system, b, cfg)

            def refine(args):
                x, bi = args
                for _ in range(2):
                    d = jax.tree.map(jnp.subtract, bi, system.operator(x))
                    x = jax.tree.map(jnp.add, x, imp._raw_block_apply(system, d))
                r = jax.tree.map(jnp.subtract, bi, system.operator(x))
                return x, imp._tree_norm(r) / jnp.maximum(imp._tree_norm(bi), 1e-300)
            sol, relres = jax.lax.map(refine, (sol, b))

            def tangent_and_push(dzi, tp, g):
                dx = jax.jvp(lambda z, p_, fr: imp._assemble(z, imp.runtime_from_params(p_, cfg),
                                                             fr, project, edge),
                             (z_star, prm, frozen), (project(dzi), tp, g))[1]
                return jax.jvp(lambda st, p_: _strip(traceable_metrics(st, p_, ctx, cfg, merc=not merc_split)),
                               (x_star, prm), (dx, tp))[1], dx
            dm, dx_all = jax.vmap(tangent_and_push)(sol, batch, gb)
            if merc_split:
                # the DMerc pushforward carries full Nyquist real-space grids per column: vmapping it over every
                # free DOF does not fit in memory, so it runs as a lax.map over column batches
                from .mercier import mercier_metric

                def merc_push(a):
                    return jax.jvp(lambda st, p_: mercier_metric(st, imp.runtime_from_params(p_, cfg),
                                                                 **E.MERC_OPTIONS)["soft_min"],
                                   (x_star, prm), a)[1]
                dm = dict(dm, merc=jax.lax.map(merc_push, (dx_all, batch), batch_size=E.MERC_JAC_BATCH))
            return dm, relres

        def core_st(prm, x_star, batch, gauge_):
            dm, relres = core(prm, x_star, batch, gauge_)
            frozen = jax.lax.stop_gradient(x_star)
            system = imp._raw_block_system(prm, cfg, frozen, mask, af, probe_chunk)
            z_star = system.project(x_star)
            gb = jax.vmap(lambda tp: jax.jvp(
                lambda p_: self.gauge_shift(p_, prm, frozen, gauge_), (prm,), (tp,))[1])(batch)

            def rhs(tp, g):
                v = jax.jvp(lambda p_, fr: G(z_star, p_, fr), (prm, frozen), (tp, g))[1]
                return jax.tree.map(jnp.negative, v)
            b = jax.vmap(rhs)(batch, gb)
            sol, _ = imp._raw_block_solve(system, b, cfg)

            def refine(args):
                x, bi = args
                for _ in range(2):
                    d = jax.tree.map(jnp.subtract, bi, system.operator(x))
                    x = jax.tree.map(jnp.add, x, imp._raw_block_apply(system, d))
                return x
            sol = jax.lax.map(refine, (sol, b))
            dxs = jax.vmap(lambda dzi, tp, g: jax.jvp(
                lambda z, p_, fr: imp._assemble(z, imp.runtime_from_params(p_, cfg), fr, project, edge),
                (z_star, prm, frozen), (project(dzi), tp, g))[1])(sol, batch, gb)
            return dm, relres, dxs
        self._core = {g: jax.jit(lambda prm, x, bt, _g=g: core(prm, x, bt, _g)) for g in GAUGES}
        self._core_st = {g: jax.jit(lambda prm, x, bt, _g=g: core_st(prm, x, bt, _g)) for g in GAUGES}

    # -- parameter tangents for the free DOFs (dense INDATA rbc/zbs layout)
    def tangent_batch(self, params):
        import jax.numpy as jnp
        M = self.ctx.M
        N = int(getattr(self.ctx.inp, "ntor", M))           # dense INDATA layout [n + ntor, m] (ntor may exceed mpol)
        n = len(self.names)
        rbc = np.zeros((n, 2 * N + 1, M)); zbs = np.zeros((n, 2 * N + 1, M))
        for j, name in enumerate(self.names):
            kind, m, nn = E.parse_name(name)
            if m < M and abs(nn) <= N:
                (rbc if kind == "rc" else zbs)[j, nn + N, m] = 1.0
        zeros = self.jax.tree.map(lambda a: jnp.zeros((n,) + jnp.shape(a)), params)
        return replace(zeros, rbc=jnp.asarray(rbc), zbs=jnp.asarray(zbs))

    def gauge_shift(self, params, params0, state, model=None):
        """Displacement of the non-evolved m=1 gauge coordinate at ``params`` (anchor ``params0``).

        'frozen': zero. 'sqrt': ``Z_sin[js, +n] - Z_sin[js, -n]`` (js = 1..ns-2)
        moves by ``sqrt(s_js) * d(edge anti-combination)``, the profile of the
        solver's initial guess (profil3d.f). Linear in ``params``.
        """
        import jax.numpy as jnp
        jax, imp, cfg = self.jax, self.imp, self.cfg
        model = self.gauge if model is None else model
        zeros = jax.tree.map(jnp.zeros_like, state)
        if model == "frozen":
            return zeros
        if model != "sqrt":
            raise ValueError(f"unknown gauge model {model!r}")
        pos, neg = imp._m1_pair_columns(cfg)
        prof = np.sqrt(np.clip(self.ctx.s_full, 0.0, None))
        prof[0] = 0.0
        prof[-1] = 0.0
        e = (imp.runtime_from_params(params, cfg).setup.boundary_Z_sin
             - imp.runtime_from_params(params0, cfg).setup.boundary_Z_sin)
        anti = 0.5 * (e[pos] - e[neg])
        Z = jnp.zeros(zeros.Z_sin.shape)
        Z = Z.at[:, pos].set(prof[:, None] * anti[None, :]).at[:, neg].set(-prof[:, None] * anti[None, :])
        return replace(zeros, Z_sin=Z)

    @staticmethod
    def _split_state(state, eq):
        """``state=`` may be a VmexStateArrays or an EqResult carrying one (-> also its eq)."""
        if state is None or isinstance(state, E.VmexStateArrays):
            return state, eq
        vs = getattr(state, "vmex_state", None)
        if vs is None:
            raise ValueError(f"state= must be a VmexStateArrays or an EqResult with vmex_state, "
                             f"got {type(state).__name__} without one")
        return vs, (state if eq is None else eq)

    def state_matches(self, boundary, state) -> bool:
        """True iff ``state`` (VmexStateArrays or EqResult carrying one) solves ``boundary`` at this step."""
        try:
            vs, _ = self._split_state(state, None)
        except ValueError:
            return False
        return vs is not None and vs.matches(E.vmec_input(boundary, self.step))

    def __call__(self, boundary, *, gauge=None, with_elongation=True, state=None, eq=None):
        """Cold solve (== ``equilibrium.evaluate(boundary, step)``) + exact Jacobian.

        Returns ``None`` when the cold solve fails (callers fall back to FD).

        Opt-in reuse (roadmap S-c): ``state`` = the converged state of THIS
        boundary/step from an earlier solve -- a
        :class:`phrport.equilibrium.VmexStateArrays`, or an ``EqResult`` from
        ``evaluate(..., return_state=True)`` / ``VmexPool(return_state=True)``
        carrying one -- skips the cold re-solve. The state must belong to the
        identical VMEX input (``input_fingerprint`` of ``vmec_input(boundary,
        step)``), else ``ValueError``. ``eq`` (default: the EqResult passed as
        ``state``, if any) is returned as ``result.eq``; without it the metrics
        are recomputed from the state with ``evaluate``'s numpy code. Since the
        solve is deterministic (pool == serial, bitwise) the Jacobian equals
        the cold-path one bitwise.
        """
        from .equilibrium import vmex_device_ctx
        vs, eq = self._split_state(state, eq)
        if eq is not None and vs is None:
            raise ValueError("eq= is only used together with state=")
        with vmex_device_ctx():
            return self._call_impl(boundary, gauge=gauge, with_elongation=with_elongation,
                                   vstate=vs, eq=eq)

    def _equilibrium_from_state(self, inp, vs):
        """Minimal VMEX Equilibrium stand-in (inp, state, wout) for the numpy metric path."""
        from types import SimpleNamespace
        from vmex.core.wout import wout_from_state
        st = vs.spectral_state()
        fsqr, fsqz, fsql = vs.fsq
        wout = wout_from_state(inp=inp, state=st, fsqr=fsqr, fsqz=fsqz, fsql=fsql,
                               niter=int(vs.iterations), converged=bool(vs.converged))
        return SimpleNamespace(inp=inp, state=st, wout=wout)

    def _call_impl(self, boundary, *, gauge=None, with_elongation=True, vstate=None, eq=None):
        jax, imp = self.jax, self.imp
        gauge = self.gauge if gauge is None else gauge
        rep = {}
        t0 = time.perf_counter()
        inp = E.vmec_input(boundary, self.step)
        if vstate is not None:
            if not vstate.matches(inp):
                raise ValueError("state= does not belong to this boundary/step "
                                 "(VMEX input fingerprint differs)")
            if not vstate.converged:
                return None
            state = vstate.spectral_state()
            rep["state_reused"] = True
            rep["t_solve"] = 0.0
            rep["t_state_load"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            if eq is not None:
                if not bool(getattr(eq, "converged", False)):
                    return None
                eqres = eq
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        m = E.evaluate_wout(E.vmec2000_wout(self._equilibrium_from_state(inp, vstate)))
                        if E.PHYSICS == "qi":
                            st_ = vstate.spectral_state()
                            m.update(E.qi_metrics(st_, E._runtime_for(inp, st_)))
                    except Exception:  # noqa: BLE001
                        return None
                eqres = E._with_size(E.EqResult(True, m["qs_residuals"], m["qs_total"], m["aspect"], m["mean_iota"],
                                   m["iota_axis"], m["iota_edge"], m["elongation"], m["well"], "",
                                   dict(t_solve=0.0, iterations=int(vstate.iterations),
                                        state_reused=True)), m)
            rep["t_metrics_numpy"] = time.perf_counter() - t1
        else:
            from vmex import optimize as opt
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    eqv = opt.solve_equilibrium(inp, **E.VMEX_SOLVE_KWARGS)
                except Exception:  # noqa: BLE001  (VmecJacobianError, ...) == evaluate's failure
                    return None
            rep["t_solve"] = time.perf_counter() - t0
            if not bool(eqv.result.converged):
                return None
            t1 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    m = E.evaluate_wout(E.vmec2000_wout(eqv))
                    if E.PHYSICS == "qi":
                        m.update(E.qi_metrics(eqv.state, eqv.runtime))
                except Exception:  # noqa: BLE001
                    return None
            eqres = E._with_size(E.EqResult(True, m["qs_residuals"], m["qs_total"], m["aspect"], m["mean_iota"],
                               m["iota_axis"], m["iota_edge"], m["elongation"], m["well"], "",
                               dict(t_solve=rep["t_solve"], iterations=int(eqv.result.iterations))), m)
            rep["t_metrics_numpy"] = time.perf_counter() - t1
            state = eqv.state
        params = imp.params_from_input(inp)
        t3 = time.perf_counter()
        tb = self.tangent_batch(params)
        dxs = None
        if self.want_state_tangents:
            dm, relres, dxs = self._core_st[gauge](params, state, tb)
        else:
            dm, relres = self._core[gauge](params, state, tb)
        dm = jax.device_get(dm)
        rep["t_tangent_push"] = time.perf_counter() - t3
        rep["linear_relres_max"] = float(np.max(np.asarray(relres)))
        jac = {k: np.asarray(v, float).T for k, v in dm.items()}     # (rows, n) / (n,)
        vals = jax.device_get(self._metrics(state, params))
        if with_elongation:
            t5 = time.perf_counter()
            el_args = (vals["rmnc_edge"], vals["zmns_edge"], vals["raxis_cc"], vals["zaxis_cs"])
            vals["elongation"] = float(self._elong(*el_args))
            g = self._elong_grad(*el_args)
            jac["elongation"] = sum(np.asarray(gk) @ np.asarray(dm[k]).T for gk, k in
                                    zip(g, ("rmnc_edge", "zmns_edge", "raxis_cc", "zaxis_cs")))
            rep["t_elongation"] = time.perf_counter() - t5
        if float(E._PROB.get("phr", "qs_max", 0.0)) > 0.0 and "qs" in jac:   # opt-in QS-ceiling row (phr.qs_max)
            jac["qs_total"] = 2.0 * (np.asarray(eqres.qs_residuals, float).ravel() @ np.asarray(jac["qs"], float))
        rep["t_total"] = time.perf_counter() - t0
        out = ExactJacResult(eq=eqres, names=list(self.names), jac=jac,
                             values_jax={k: np.asarray(v) for k, v in vals.items()}, report=rep)
        if dxs is not None:
            out.state, out.params, out.state_tangents, out.param_tangents = state, params, dxs, tb
        return out


def _strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def phr_rows_jacobian(ex: ExactJacResult, lam, rho, cfg):
    """Equilibrium block of the PHR least-squares Jacobian, rows of ``ALSubproblem.residuals``.

    Rows (same order/scaling as the FD block of ``phr.least_squares_qss``): the
    44352 QS residuals (weight 1), then for each ``eq_constraint_labels(cfg)``
    entry ``sqrt(rho_i/2) * d max(0, g_i + lam_i/rho_i)`` = ``sqrt(rho_i/2) * dg_i``
    where the hinge is active (> 0) and 0 where it is not.
    """
    from .phr import eq_constraint_labels, eq_constraint_values, exact_constraint_grads
    labels = eq_constraint_labels(cfg)
    g = eq_constraint_values(ex.eq, cfg)
    lam = np.asarray(lam, float); rho = np.asarray(rho, float)
    dg = exact_constraint_grads(ex.jac, cfg)
    rows = [np.asarray(ex.jac["qs"], float)]
    for i, lab in enumerate(labels):
        h = float(g[i]) + float(lam[i] / rho[i])
        w = math.sqrt(float(rho[i]) / 2.0)
        rows.append((w * dg[lab] if h > 0.0 else np.zeros_like(dg[lab]))[None, :])
    return np.vstack(rows)


class PHRExactJacobian:
    """Adapter for ``phr.least_squares_qss(eq_jac=...)``: x -> equilibrium Jacobian block.

    ``boundary_fn(x)`` builds the full boundary of the current step (the runner's
    ``_boundary_with``); ``sub`` is the current ``ALSubproblem`` (its ``lam``/``rho``).
    Returns ``None`` (-> FD fallback) when the cold solve fails or anything raises.
    One :class:`ExactEquilibriumJacobian` (one XLA compilation) per step is cached.

    Opt-in state reuse (roadmap S-c; default off = the cold re-solve, as before):
    the converged state at ``x`` -- ``state=`` on the call, or
    ``state_lookup(step, x)`` (e.g. the runner's f-eval cache holding EqResults
    from ``VmexPool(return_state=True)``) -- replaces the cold re-solve. A state
    that is missing (``None``) or does not match the boundary's VMEX input
    fingerprint is ignored (cold re-solve; counted in ``stats``), never a
    reason for the FD fallback.
    """

    def __init__(self, gauge: str = "sqrt", log=None, state_lookup=None):
        self.gauge = gauge
        self.log = log or (lambda *a, **k: None)
        self.state_lookup = state_lookup
        self._by_step = {}
        self.stats = dict(calls=0, fallbacks=0, seconds=0.0)
        self.want_state_tangents = False      # finite beta (set by the runner)
        if state_lookup is not None:
            self.stats.update(state_reused=0, state_missing=0, state_mismatch=0)
        self.last = None

    def get(self, step: int) -> ExactEquilibriumJacobian:
        if step not in self._by_step:
            self._by_step[step] = ExactEquilibriumJacobian(step, gauge=self.gauge)
        self._by_step[step].want_state_tangents = bool(self.want_state_tangents)
        return self._by_step[step]

    def _reusable_state(self, ej, x, step, boundary, state):
        if state is None and self.state_lookup is not None:
            try:
                state = self.state_lookup(step, np.asarray(x, dtype=np.float64))
            except Exception as e:  # noqa: BLE001
                self.log(f"  [exact-jac] state lookup raised {type(e).__name__}: {e}; cold re-solve")
                state = None
        if state is None or (not isinstance(state, E.VmexStateArrays)
                             and getattr(state, "vmex_state", None) is None):
            if self.state_lookup is not None or state is not None:
                self.stats["state_missing"] = self.stats.get("state_missing", 0) + 1
            return None
        if not ej.state_matches(boundary, state):
            self.stats["state_mismatch"] = self.stats.get("state_mismatch", 0) + 1
            self.log("  [exact-jac] reusable VMEX state does not match the Jacobian point; cold re-solve")
            return None
        self.stats["state_reused"] = self.stats.get("state_reused", 0) + 1
        return state

    def __call__(self, x, *, step, names, boundary_fn, sub, cfg, state=None):
        t0 = time.perf_counter()
        self.stats["calls"] += 1
        try:
            ej = self.get(step)
            if list(names) != ej.names:
                raise ValueError("free DOF order differs from equilibrium.free_dof_names(step)")
            boundary = boundary_fn(x)
            if state is None and self.state_lookup is None:
                ex = ej(boundary)                      # default: cold re-solve (unchanged)
            else:
                ex = ej(boundary, state=self._reusable_state(ej, x, step, boundary, state))
            if ex is None:
                raise RuntimeError("cold solve failed at the Jacobian point")
            J = phr_rows_jacobian(ex, sub.lam, sub.rho, cfg)
            if not np.all(np.isfinite(J)):
                raise FloatingPointError("non-finite exact Jacobian")
            self.last = ex
            return J
        except Exception as e:  # noqa: BLE001
            self.stats["fallbacks"] += 1
            self.log(f"  [exact-jac] {type(e).__name__}: {e} -> finite-difference fallback")
            return None
        finally:
            self.stats["seconds"] += time.perf_counter() - t0


# ---------------------------------------------------------------------------
# validation helper: the smooth map whose derivative the exact Jacobian is
# ---------------------------------------------------------------------------
class GaugeContinuedMap:
    """The map ``p -> metrics`` that :class:`ExactEquilibriumJacobian` differentiates EXACTLY.

    Anchored at the cold solve ``x0`` of ``boundary0`` (so its value at ``p0`` is
    exactly ``evaluate(boundary0)``): ``x(p)`` solves the raw VMEX force equations
    on the evolved dofs, ``F(x, p) = F(x0, p0)`` (the cold solve's ftol-level
    residual is carried as a constant offset), with the non-evolved m=1 gauge
    entries set to ``x0 + gauge_shift(p)``. Newton through the anchor's raw
    block-tridiagonal factorization. Used by the tests to validate the
    Jacobian against finite differences of a smooth map, independently of the
    cold solver's path-dependent gauge.
    """

    def __init__(self, ej: ExactEquilibriumJacobian, boundary0, *, gauge=None, tol=1e-14, maxit=30):
        import jax
        import jax.numpy as jnp
        from vmex import optimize as opt
        imp, cfg = ej.imp, ej.cfg
        self.ej, self.jax, self.imp = ej, jax, imp
        self.gauge = ej.gauge if gauge is None else gauge
        self.tol, self.maxit = tol, maxit
        self.boundary0 = dict(boundary0)
        inp = E.vmec_input(boundary0, ej.step)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eqv = opt.solve_equilibrium(inp, **E.VMEX_SOLVE_KWARGS)
        self.x0 = eqv.state
        self.p0 = imp.params_from_input(inp)
        mask = ej.dof_mask
        P = imp._dof_projector(cfg, mask)
        edge = imp._edge_mask(cfg)
        af = imp._active_state_fields(cfg)
        mn = int(np.shape(mask.R_cos)[1])
        frozen0 = jax.lax.stop_gradient(self.x0)
        self.system = imp._raw_block_system(self.p0, cfg, frozen0, mask, af, len(af) * mn)
        self.z0 = P(self.x0)

        def G(z, p_, fr):
            rt_ = imp.runtime_from_params(p_, cfg)
            return P(imp._raw_force_state(imp._assemble(z, rt_, fr, P, edge), rt_))
        self.F0 = jax.jit(G)(self.z0, self.p0, frozen0)
        F0, system = self.F0, self.system

        def frozen_of(p_):
            return jax.tree.map(jnp.add, frozen0, ej.gauge_shift(p_, self.p0, frozen0, self.gauge))

        def step(z, p_):
            fr = frozen_of(p_)
            r = jax.tree.map(jnp.subtract, G(z, p_, fr), F0)
            dz = imp._raw_block_apply(system, jax.tree.map(jnp.negative, r))
            return jax.tree.map(jnp.add, z, dz), imp._tree_norm(r)
        self._step = jax.jit(step)
        self._assemble = jax.jit(lambda z, p_: imp._assemble(z, imp.runtime_from_params(p_, cfg),
                                                              frozen_of(p_), P, edge))

    def params(self, x_free):
        inp = E.vmec_input(E.boundary_from_free_dofs(x_free, self.ej.step, base=self.boundary0),
                           self.ej.step)
        return self.imp.params_from_input(inp)

    def state(self, x_free):
        p_ = self.params(x_free)
        z = self.z0
        hist = []
        for _ in range(self.maxit):
            z_new, rn = self._step(z, p_)
            hist.append(float(rn))
            if hist[-1] <= self.tol or (len(hist) > 2 and hist[-1] >= 0.5 * hist[-2]):
                break
            z = z_new
        return self._assemble(z, p_), p_, hist

    def values(self, x_free, *, elongation=True):
        x, p_, hist = self.state(x_free)
        v = self.jax.device_get(self.ej._metrics(x, p_))
        out = {k: np.asarray(v[k]) for k in ("qs", "aspect", "mean_iota", "iota_axis", "iota_edge", "well",
                                             "volume", "aminor", "mirror", "maxj", "bmaxsp",
                                             "merc") if k in v}
        if elongation:
            out["elongation"] = float(self.ej._elong(v["rmnc_edge"], v["zmns_edge"], v["raxis_cc"],
                                                     v["zaxis_cs"]))
        out["newton"] = hist
        return out

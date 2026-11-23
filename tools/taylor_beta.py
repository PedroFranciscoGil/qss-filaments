"""Finite-beta Taylor test of the full envelope gradient of the coil subproblem (paper figure).

    J*(s) = min_c J_f(c, s; B_plasma(s)),   B_plasma(s) = virtual casing on the equilibrium of s
    dJ*/ds = [explicit envelope term at fixed B_plasma] + <dJ_f/dB_plasma, dB_plasma/ds>

The equilibrium follows the GaugeContinuedMap (the smooth map the exact Jacobian differentiates),
continued in epsilon (Newton from the previous epsilon's state) so larger steps stay on the branch;
the field term is one reverse pass through virtual casing + Fourier interpolation (shared by all
directions), contracted with each direction's continued-map state tangent.

  PHRPORT_PROBLEM=examples/qa_finite_beta.json python -u tools/taylor_beta.py [--ndir 3]
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from phrport import equilibrium as E
from phrport import exact_jacobian as X
from phrport.coils import CoilSolver, CoilConfig, parse_name
from phrport.plasma_field import PlasmaField
from vmex.core import implicit as imp


def tdot(a, b):
    return float(sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))))


class RefreshedContinuation:
    """The GaugeContinuedMap's equations F(z, p) = F0 on the evolved dofs, gauge entries x0 + gauge_shift(p),
    solved by Newton with the block factorization REFRESHED at the current iterate (the map itself is the
    one the exact Jacobian differentiates; only the linear solver changes), so epsilon can go far beyond
    the anchor-factorization range."""

    def __init__(self, cm, ej):
        imp, cfg = ej.imp, ej.cfg
        self.cm, self.ej, self.imp, self.cfg = cm, ej, imp, cfg
        mask = ej.dof_mask
        self.P = P = imp._dof_projector(cfg, mask)
        self.edge = edge = imp._edge_mask(cfg)
        self.af = af = imp._active_state_fields(cfg)
        self.mask = mask
        self.chunk = len(af) * int(np.shape(mask.R_cos)[1])
        frozen0 = jax.lax.stop_gradient(cm.x0)

        def frozen_of(p_):
            return jax.tree.map(jnp.add, frozen0, ej.gauge_shift(p_, cm.p0, frozen0, cm.gauge))

        def G(z, p_):
            fr = frozen_of(p_)
            rt_ = imp.runtime_from_params(p_, cfg)
            return P(imp._raw_force_state(imp._assemble(z, rt_, fr, P, edge), rt_))
        self._G = jax.jit(G)
        self._assemble = jax.jit(lambda z, p_: imp._assemble(z, imp.runtime_from_params(p_, cfg),
                                                             frozen_of(p_), P, edge))

    def solve(self, xf, z_start, x_start, p_start, tol=1e-14, maxit=12):
        imp, cfg = self.imp, self.cfg
        p_ = self.cm.params(xf)
        system = imp._raw_block_system(p_start, cfg, jax.lax.stop_gradient(x_start), self.mask, self.af, self.chunk)
        z, hist = z_start, []
        for it in range(maxit):
            r = jax.tree.map(jnp.subtract, self._G(z, p_), self.cm.F0)
            rn = float(imp._tree_norm(r))
            hist.append(rn)
            if rn <= tol or (len(hist) > 2 and rn >= 0.5 * hist[-2]):
                break
            z = jax.tree.map(jnp.add, z, imp._raw_block_apply(system, jax.tree.map(jnp.negative, r)))
            if it == 1:      # refresh once at the first Newton iterate (quadratic convergence afterwards)
                system = imp._raw_block_system(p_, cfg, jax.lax.stop_gradient(self._assemble(z, p_)),
                                               self.mask, self.af, self.chunk)
        return self._assemble(z, p_), p_, z, hist


def cm_state_from(cm, xf, z_start, tol=1e-14, maxit=40):
    """GaugeContinuedMap state at xf, Newton started from z_start (epsilon continuation)."""
    p_ = cm.params(xf)
    z, hist = z_start, []
    for _ in range(maxit):
        z_new, rn = cm._step(z, p_)
        hist.append(float(rn))
        if hist[-1] <= tol or (len(hist) > 2 and hist[-1] >= 0.5 * hist[-2]):
            break
        z = z_new
    return cm._assemble(z, p_), p_, z, hist


def fit_slope(eps, t, lo, hi):
    eps, t = np.asarray(eps), np.asarray(t)
    m = (eps >= lo) & (eps <= hi) & (t > 0)
    return float(np.polyfit(np.log(eps[m]), np.log(t[m]), 1)[0]) if m.sum() >= 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=6)
    ap.add_argument("--ndir", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=256, help="virtual-casing chunk size (memory)")
    ap.add_argument("--coil-continuation", action="store_true", help="polish each epsilon from the previous one")
    ap.add_argument("--no-m1", action="store_true", help="directions without m = 1 boundary modes (gauge-model free)")
    ap.add_argument("--hot-restart", action="store_true",
                    help="equilibria along epsilon by VMEX hot restarts from the previous state (no gauge continuation)")
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--fit", type=float, nargs=2, default=[1e-5, 3e-3])
    ap.add_argument("--out-dir", default="runs/analysis/taylor_beta")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--vmec-ftol", type=float, default=1e-15, help="hot-restart VMEX ftol (J noise floor ~ sqrt(ftol))")
    ap.add_argument("--vacuum", action="store_true", help="vacuum design: no plasma field, no field term")
    ap.add_argument("--zero-control", action="store_true",
                    help="J0 from one more polish of the base with the same budget as every perturbed point "
                         "(use without --coil-continuation): a polish drift then cancels in dJ")
    ap.add_argument("--fp64", action="store_true", help="XCoil in full FP64 (default: mixed FP32 kernels)")
    ap.add_argument("--spec-power", type=float, default=1.0, help="direction weights 1/(1+m^2+n^2)^p")
    ap.add_argument("--ftol-scan", type=float, nargs="*", default=None,
                    help="noise-floor diagnostic: J at epsilon = 0 versus the hot-restart VMEX ftol, then exit")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "taylor_beta.json")
    if a.replot:
        R = json.load(open(out))
        figure(R, a)
        return
    warnings.simplefilter("ignore")
    t0 = time.time()
    with E.vmex_device_ctx():
        ej = X.ExactEquilibriumJacobian(4)
        b0 = E.seed_boundary()
        cm = X.GaugeContinuedMap(ej, b0)
        inp, cfg = ej.ctx.inp, ej.cfg
        pf = None if a.vacuum else PlasmaField(E.NFP, 32, a.digits, chunk_size=a.chunk)
        if pf is not None:
            pf.make_plan(pf.surface_data(inp, cm.x0, imp.runtime_from_params(cm.p0, cfg)))
    names = E.free_dof_names(4)
    xf0 = E.free_dofs_from_boundary(b0, 4)
    with E.vmex_device_ctx():
        rc = RefreshedContinuation(cm, ej)
    cur = {}

    def provider(boundary, TH, PH):
        with E.vmex_device_ctx():
            return np.asarray(pf.at(inp, cur["state"], np.asarray(TH), np.asarray(PH),
                                    imp.runtime_from_params(cur["params"], cfg)))

    cs = CoilSolver(CoilConfig(precision="fp64") if a.fp64 else CoilConfig())
    if not a.vacuum:
        cs.set_plasma(provider)
    cur["state"], cur["params"] = cm.x0, cm.p0
    x_base = cm.x0
    if a.hot_restart:
        # the base must lie on the same equilibrium path as the perturbed points:
        # re-solve epsilon = 0 with the same hot-restart settings used along the sweep
        import dataclasses
        from vmex import optimize as opt
        with E.vmex_device_ctx():
            inp_0 = dataclasses.replace(E.vmec_input(b0, 4), ns_array=np.array([int(np.max(np.asarray(inp.ns_array)))]),
                                        ftol_array=np.array([a.vmec_ftol]), niter_array=np.array([100000]))
            x_base = opt.solve_equilibrium(inp_0, initial_state=cm.x0, **E.VMEX_SOLVE_KWARGS).state
        cur["state"] = x_base
    r0 = cs.final_coil_stage(b0)
    r0 = cs.polish(b0, r0, k_max=400, ftol=1e-13, viol_tol=1e-10)
    # repolish the base until it no longer moves: the perturbed points are polished from it, and a base
    # that still descends shows up as an epsilon-independent drift in dJ
    for it in range(int(os.environ.get("TAYLOR_REPOLISH", "200"))):
        r1 = cs.polish(b0, r0, k_max=int(os.environ.get("TAYLOR_POLISH_KMAX", "2000")), ftol=1e-13, viol_tol=1e-10)
        dJb = float(r1.c0_local) - float(r0.c0_local)
        r0 = r1
        print(f"  base repolish {it + 1}: dJ = {dJb:+.3e}", flush=True)
        if abs(dJb) <= 1e-13:
            break
    J0 = float(r0.c0_local)
    if not a.vacuum:
        sd0 = pf.surface_data(inp, x_base, imp.runtime_from_params(cm.p0, cfg))
        _g = np.asarray(sd0.gamma); _b = np.asarray(sd0.B_total)
        _g = _g.reshape(-1, 3) if _g.shape[-1] == 3 else np.moveaxis(_g, 0, -1).reshape(-1, 3)
        _b = _b.reshape(-1, 3) if _b.shape[-1] == 3 else np.moveaxis(_b, 0, -1).reshape(-1, 3)
        cosm, ok = cs.plasma_orientation(r0.x, r0.currents, _g, _b)
        print(f"  coil field vs VMEX field: mean cos {cosm:+.4f}; plasma-field sign {cs.plasma_sign:+.0f} -> "
              f"{'consistent' if ok else 'INCONSISTENT'}", flush=True)
        if not ok:
            raise SystemExit("plasma-field sign inconsistent with the coils' field orientation")
    if a.zero_control:
        J0 = float(cs.polish(b0, r0, k_max=400, ftol=1e-13, viol_tol=1e-10).c0_local)
        print(f"  zero control: J0 = {J0:.10e} (drift {J0 - float(r0.c0_local):+.3e})", flush=True)
    g_exp = cs.envelope_grad(b0, r0, names)
    cot = None if a.vacuum else jnp.asarray(cs.bp_cotangent(b0, r0))
    TH, PH = cs.P.flux_TH, cs.P.flux_PH
    el = {k: float(v) for k, v in (r0.info.get("elasticity") or {}).items()}
    print(f"base: J*={J0:.10e} certified={r0.certified} |g_explicit|={np.linalg.norm(g_exp):.3e} elasticities {el} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    if a.ftol_scan:
        import dataclasses
        from vmex import optimize as opt
        nsf = int(np.max(np.asarray(inp.ns_array)))
        for tag, x_start in (("from multigrid state", cm.x0), ("from hot-restart base", x_base)):
            for ft in a.ftol_scan:
                with E.vmex_device_ctx():
                    inp_f = dataclasses.replace(E.vmec_input(b0, 4), ns_array=np.array([nsf]),
                                                ftol_array=np.array([ft]), niter_array=np.array([100000]))
                    eqf = opt.solve_equilibrium(inp_f, initial_state=x_start, **E.VMEX_SOLVE_KWARGS)
                cur["state"] = eqf.state
                rf = cs.polish(b0, r0, k_max=400, ftol=1e-13, viol_tol=1e-10)
                res = eqf.result
                print(f"  ftol={ft:.0e} {tag}: J-J0={float(rf.c0_local) - J0:+.3e}  fsqr={float(res.fsqr):.1e} "
                      f"fsqz={float(res.fsqz):.1e} fsql={float(res.fsql):.1e} iters={getattr(res, 'iterations', getattr(res, 'niter', '?'))}",
                      flush=True)
        return
    with E.vmex_device_ctx():
      if not a.vacuum:
        def Fb(state, params):
            return jnp.sum(cot * pf.at(inp, state, TH, PH, imp.runtime_from_params(params, cfg)))
        t = time.time(); _, vjp = jax.vjp(Fb, x_base, cm.p0); vs, vp = vjp(jnp.ones(()))
        jax.block_until_ready(vp.rbc); print(f"field-term reverse pass: {time.time() - t:.0f}s", flush=True)
    rng = np.random.default_rng(3)
    mn = [parse_name(n)[1:] for n in names]
    wspec = np.array([(1 + m * m + n * n) ** -a.spec_power for m, n in mn])
    dirs = []
    for j in range(a.ndir):
        d = rng.standard_normal(len(names)) * wspec
        if a.no_m1:
            d *= np.array([0.0 if m == 1 else 1.0 for m, n in mn])
        d /= np.linalg.norm(d)
        h = 1e-6
        g_field_d = 0.0
        if not a.vacuum:
            with E.vmex_device_ctx():
                sp, pp, _, _ = cm_state_from(cm, xf0 + h * d, cm.z0)
                sm, pm, _, _ = cm_state_from(cm, xf0 - h * d, cm.z0)
            ds = jax.tree.map(lambda x_, y_: (x_ - y_) / (2 * h), sp, sm)
            dp = jax.tree.map(lambda x_, y_: (x_ - y_) / (2 * h), pp, pm)
            g_field_d = tdot(vs, ds) + tdot(vp, dp)
        g_exp_d = float(g_exp @ d)
        gd = g_exp_d + g_field_d
        print(f"dir {j}: g.d = explicit {g_exp_d:+.6e} + field {g_field_d:+.6e} = {gd:+.6e}", flush=True)
        rows = []
        z, xs, ps = cm.z0, x_base, cm.p0
        r_prev = r0
        for e in sorted(a.eps):
            xf = xf0 + e * d
            with E.vmex_device_ctx():
                if a.hot_restart:
                    import dataclasses
                    from vmex import optimize as opt
                    inp_e = dataclasses.replace(E.vmec_input(E.boundary_from_free_dofs(xf, 4, base=b0), 4),
                                                ns_array=np.array([int(np.max(np.asarray(inp.ns_array)))]),
                                                ftol_array=np.array([a.vmec_ftol]), niter_array=np.array([100000]))
                    eqh = opt.solve_equilibrium(inp_e, initial_state=xs, **E.VMEX_SOLVE_KWARGS)
                    st, pr = eqh.state, cm.params(xf)
                    hist = [float(eqh.result.fsqr + eqh.result.fsqz + eqh.result.fsql)]
                else:
                    st, pr, z, hist = rc.solve(xf, z, xs, ps)
            xs, ps = st, pr
            cur["state"], cur["params"] = st, pr
            b = E.boundary_from_free_dofs(xf, 4, base=b0)
            t = time.time()
            r = cs.polish(b, r_prev if a.coil_continuation else r0, k_max=400, ftol=1e-13, viol_tol=1e-10)
            r_prev = r
            dJ = float(r.c0_local) - J0
            ce = (r.info or {}).get("cert") or {}
            rows.append(dict(eps=e, dJ=dJ, T1=abs(dJ), T2=abs(dJ - e * gd), T2_explicit=abs(dJ - e * g_exp_d),
                             D_fd=dJ / e, newton=hist[-1], wall=time.time() - t,
                             max_kappa=ce.get("max_kappa"), elasticity={k: float(v) for k, v in
                                                                      ((r.info or {}).get("elasticity") or {}).items()}))
            print(f"  eps={e:.0e}: dJ={dJ:+.6e}  T2={rows[-1]['T2']:.3e}  T2(explicit only)={rows[-1]['T2_explicit']:.3e}"
                  f"  D_fd/(g.d)={dJ / e / gd:.5f}  (Newton res {hist[-1]:.1e}) kappa_max={rows[-1]['max_kappa']} "
                  f"elast={rows[-1]['elasticity']}", flush=True)
        dirs.append(dict(gd=gd, g_explicit_d=g_exp_d, g_field_d=g_field_d, rows=rows))
        json.dump(dict(J0=J0, vacuum=bool(a.vacuum), digits=a.digits, fp64=bool(a.fp64), vmec_ftol=a.vmec_ftol, elasticity=el, dirs=dirs), open(out, "w"), indent=1)
    figure(json.load(open(out)), a)


def figure(R, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = a.fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    eps_all = []
    vac = bool(R.get("vacuum", False))
    for j, D in enumerate(R["dirs"]):
        c = f"C{j}"
        eps = np.array([r["eps"] for r in D["rows"]]); eps_all = eps
        T2 = np.array([r["T2"] for r in D["rows"]]); T2e = np.array([r["T2_explicit"] for r in D["rows"]])
        s2, s2e = fit_slope(eps, T2, lo, hi), fit_slope(eps, T2e, lo, hi)
        ax1.loglog(eps, T2, "o-", color=c, ms=5, lw=1.5,
                   label=f"direction {j + 1}" + ("" if vac else ", full gradient") + f" (slope {s2:.2f})")
        if not vac:
          ax1.loglog(eps, T2e, "s--", color=c, ms=4, lw=0.9, alpha=0.6,
                   label=f"direction {j + 1}, without field term (slope {s2e:.2f})")
        ax2.semilogx(eps, [r["D_fd"] / D["gd"] for r in D["rows"]], "o-", color=c, ms=5, lw=1.5,
                     label=f"direction {j + 1}" if vac else
                     f"direction {j + 1} (field term {100 * D['g_field_d'] / D['gd']:.0f}% of $g\\cdot d$)")
    em = np.sqrt(lo * hi); ee = np.array([lo, hi])
    T2m = np.exp(np.mean([np.interp(np.log(em), np.log(eps_all), np.log([r["T2"] for r in D["rows"]]))
                          for D in R["dirs"]]))
    ax1.loglog(ee, 2.0 * T2m * (ee / em) ** 2, "k--", lw=1.1, label="slope 2")
    ax1.loglog(ee, 2.0 * T2m * (ee / em), color="0.45", ls=":", lw=1.3, label="slope 1")
    ax1.axvspan(lo, hi, color="g", alpha=0.06)
    ax1.set_ylabel(r"$T_2 = |\Delta J_f^* - \epsilon\, g\cdot d|$")
    ax1.set_title((r"Taylor remainder, vacuum" if vac else r"Taylor remainder, finite $\beta$")
                  + " (slopes fitted in the shaded window)", fontsize=10)
    ax1.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax2.axhline(1.0, color="k", lw=1.0)
    ax2.set_ylim(0.85, 1.15)
    ax2.set_ylabel(r"$D_{FD} / (g\cdot d)$,  $D_{FD} = \Delta J_f^* / \epsilon$")
    ax2.set_title("Finite difference / envelope gradient" + ("" if vac else " (with field term)"), fontsize=10)
    ax2.legend(frameon=False, fontsize=7.5, loc="lower left")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both"); ax.set_xlabel(r"perturbation size $\epsilon$")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "taylor_beta.png"), dpi=200)
    print("wrote", os.path.join(a.out_dir, "taylor_beta.png"))


if __name__ == "__main__":
    main()

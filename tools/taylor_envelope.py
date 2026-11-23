"""Taylor test of the envelope-theorem gradient of the coil subproblem (XCoil, current setup).

    Jf*(s) = min_x c0_local(x; s)  s.t.  PHR coil limits (length 4 m, curvature, MSC, coil-coil,
                                         coil-surface), free currents, fixed total current,
    dJf*/ds = d/ds [ c0(x*, s) + c0_ref lam*^T g(x*, s) ]   (CoilSolver.envelope_grad: coils x* and
                                                           multipliers lam* frozen at the optimum).

At a base surface s0 (a front point) and random unit directions d in the free-boundary-DOF space:
    T1(eps) = |Jf*(s0 + eps d) - Jf*(s0)|                    -> slope 1
    T2(eps) = |Jf*(s0 + eps d) - Jf*(s0) - eps g.d|          -> slope 2 iff g is the gradient
    D_FD(eps) = (Jf*(s0 + eps d) - Jf*(s0)) / eps            -> g.d
Every Jf* is a fresh XCoil solve on the perturbed surface: the AL is warm-started from the base
optimum (same basin) and polished to a tight stall tolerance with near-exact constraint satisfaction.

    \\
    python -u tools/taylor_envelope.py [--label sweepTS5:k=8] [--ndir 3]
"""
import argparse
import json
import os
import time

import numpy as np

from phrport import equilibrium as E
from phrport.coils import CoilSolver, CoilConfig, parse_name


def fit_slope(eps, t, lo, hi):
    m = (eps >= lo) & (eps <= hi) & (t > 0)
    return float(np.polyfit(np.log(eps[m]), np.log(t[m]), 1)[0]) if m.sum() >= 2 else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collect", default="runs/analysis/frontM5/front_collect.json")
    p.add_argument("--label", default="sweepTS5:k=8")
    p.add_argument("--step", type=int, default=4, help="continuation step whose free DOFs span d")
    p.add_argument("--ndir", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spectral", type=float, default=0.0, metavar="P",
                   help="direction weights (1 + m^2 + n^2)^(-P/2) on the Gaussian draw (0 = isotropic), "
                        "i.e. a perturbation with the decaying spectrum of a smooth boundary")
    p.add_argument("--eps", type=float, nargs="+", default=list(np.logspace(-7, -1.5, 12)))
    p.add_argument("--ftol", type=float, default=1e-13)
    p.add_argument("--viol-tol", type=float, default=1e-10)
    p.add_argument("--k-max", type=int, default=400)
    p.add_argument("--fit", type=float, nargs=2, default=[1e-5, 3e-3], help="slope-fit eps window (T2)")
    p.add_argument("--fit-t1", type=float, nargs=2, default=None, help="slope-fit window for T1 (default: --fit)")
    p.add_argument("--out-dir", default="runs/analysis/taylor")
    p.add_argument("--replot", action="store_true", help="redraw the figure from out-dir/taylor_envelope.json")
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.replot:
        R = json.load(open(os.path.join(a.out_dir, "taylor_envelope.json")))
        eps = np.array([r["eps"] for r in R["dirs"][0]["rows"]])
        for D in R["dirs"]:
            T1 = np.array([r["T1"] for r in D["rows"]]); T2 = np.array([r["T2"] for r in D["rows"]])
            D["slope_T1"] = fit_slope(eps, T1, *(a.fit_t1 or a.fit))
            D["slope_T2"] = fit_slope(eps, T2, *a.fit)
            print(f"slopes in [{a.fit[0]:g}, {a.fit[1]:g}]: T1 {D['slope_T1']:.2f}, T2 {D['slope_T2']:.2f}")
        a.label = R["label"]
        figure(a, eps, R["dirs"])
        return

    pts = {q["label"]: q for q in json.load(open(a.collect))["points"]}
    base = dict(pts[a.label]["boundary"])
    names = list(E.free_dof_names(a.step))
    for n in names:
        base.setdefault(n, 0.0)
    cs = CoilSolver(CoilConfig())
    cs.set_ordering_axis(boundary=base)          # ordering rows: the axis of the base surface, fixed (as in a step)
    t0 = time.time()
    r0 = cs.final_coil_stage(base)
    r0 = cs.polish(base, r0, k_max=a.k_max, ftol=a.ftol, viol_tol=a.viol_tol)
    J0 = float(r0.c0_local)
    g = cs.envelope_grad(base, r0, names)
    ce = r0.info.get("cert") or {}
    print(f"base {a.label}: Jf*={J0:.10e} ok={r0.ok} certified={r0.certified} v={ce.get('v')} "
          f"|grad|={np.linalg.norm(g):.3e} [{time.time() - t0:.0f}s]", flush=True)
    active = {k: float(v) for k, v in (r0.info.get("elasticity") or {}).items()}
    print(f"binding-limit elasticities at the base: {active}", flush=True)

    rng = np.random.default_rng(a.seed)
    eps = np.array(sorted(a.eps))
    dirs = []
    for j in range(a.ndir):
        d = rng.standard_normal(len(names))
        if a.spectral:
            mn = [parse_name(n)[1:] for n in names]
            d *= np.array([(1.0 + m * m + n * n) ** (-0.5 * a.spectral) for m, n in mn])
        d /= np.linalg.norm(d)
        gd = float(g @ d)
        rows = []
        for e in eps:
            b = dict(base)
            for n, dv in zip(names, d):
                b[n] = base[n] + e * dv
            t1 = time.time()
            r = cs.polish(b, r0, k_max=a.k_max, ftol=a.ftol, viol_tol=a.viol_tol)
            J = float(r.c0_local)
            dJ = J - J0
            rows.append(dict(eps=float(e), J=J, dJ=dJ, T1=abs(dJ), T2=abs(dJ - e * gd), D_fd=dJ / e,
                             ok=bool(r.ok), wall_s=time.time() - t1))
            print(f"  dir {j} eps={e:.2e}  dJ={dJ:+.6e}  T2={abs(dJ - e * gd):.3e}  D_fd={dJ / e:+.6e} "
                  f"(g.d={gd:+.6e})  ok={r.ok}", flush=True)
        T1 = np.array([r["T1"] for r in rows]); T2 = np.array([r["T2"] for r in rows])
        s1, s2 = fit_slope(eps, T1, *(a.fit_t1 or a.fit)), fit_slope(eps, T2, *a.fit)
        print(f"  dir {j}: slope T1 = {s1:.2f}, slope T2 = {s2:.2f}", flush=True)
        dirs.append(dict(gd=gd, slope_T1=s1, slope_T2=s2, rows=rows))
        json.dump(dict(label=a.label, step=a.step, spectral=a.spectral, J0=J0, grad_norm=float(np.linalg.norm(g)),
                       elasticity=active, fit_window=a.fit, ftol=a.ftol, viol_tol=a.viol_tol, dirs=dirs),
                  open(os.path.join(a.out_dir, "taylor_envelope.json"), "w"), indent=1)
    figure(a, eps, dirs)


def figure(a, eps, dirs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    lo, hi = a.fit
    floor = float(np.median([r["T2"] for D in dirs for r in D["rows"] if r["eps"] <= 1e-6]))
    for j, D in enumerate(dirs):
        c = f"C{j}"
        ax1.loglog(eps, [r["T2"] for r in D["rows"]], "o-", color=c, ms=5, lw=1.5,
                   label=f"direction {j + 1} (slope {D['slope_T2']:.2f})")
        ax2.semilogx(eps, [r["D_fd"] / D["gd"] for r in D["rows"]], "o-", color=c, ms=5, lw=1.5,
                     label=f"direction {j + 1} ($g\\cdot d$ = {D['gd']:+.2e})")
    em = np.sqrt(lo * hi)
    T2m = np.exp(np.mean([np.interp(np.log(em), np.log(eps), np.log([r["T2"] for r in D["rows"]]))
                          for D in dirs]))
    ee = np.array([lo, hi])
    ax1.loglog(ee, 1.6 * T2m * (ee / em) ** 2, "k--", lw=1.1, label="slope 2 (correct gradient)")
    ax1.loglog(ee, 1.6 * T2m * (ee / em), color="0.45", ls=":", lw=1.3, label="slope 1 (wrong gradient)")
    ax1.axhline(floor, color="0.6", lw=0.8, ls="-.")
    ax1.text(eps[0], floor * 1.6, f"solver noise floor ~{floor:.0e}", fontsize=7.5, color="0.4")
    ax1.axvspan(lo, hi, color="g", alpha=0.06)
    ax1.set_ylabel(r"Taylor remainder $T_2 = |\Delta J_f^* - \epsilon\, g\cdot d|$")
    ax1.set_title("Taylor remainder (slope fitted in the shaded window)", fontsize=10)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax2.axhline(1.0, color="k", lw=1.0)
    ax2.set_ylim(0.0, 2.0)
    ax2.set_ylabel(r"$D_{FD} / (g\cdot d)$,  $D_{FD} = \Delta J_f^* / \epsilon$")
    ax2.set_title("Finite difference / envelope gradient", fontsize=10)
    ax2.legend(frameon=False, fontsize=7.5, loc="lower left")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"perturbation size $\epsilon$")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "taylor_envelope.png"), dpi=200)


if __name__ == "__main__":
    main()

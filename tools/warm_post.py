"""Publication post-processing of one coil set on one target boundary (no simsopt needed).

  PHRPORT_PROBLEM=examples/qa_vacuum.json python -u tools/warm_post.py \\
      COILS_JSON OUTDIR --label two-stage [--boundary BOUNDARY_JSON] [--no-poincare]

COILS_JSON is a phrport coil export (export_coils: solver_x, base currents, boundary).  Writes OUTDIR/post.json:
  * coil properties of the certified set (lengths, max curvature, MSC, coil-coil / coil-surface distances,
    currents) and the field error <|B.n|>/<|B|> (ratio of area integrals) plus max |B.n|/|B| on converged
    unfolded grids (CoilSolver.evaluate_coils / flux_metrics_unfolded);
  * an mgrid of the coil field (VMEX write_mgrid; XCoil Biot-Savart, one group, mode S, EXTCUR = 1);
  * PHIEDGE = the coil field's toroidal flux through the target phi = 0 cross-section;
  * a Poincare plot (field lines traced with the same Biot-Savart, RK4 in phi).

The free-boundary equilibrium of the coils is computed with DESC only (tools/desc_free_boundary.py); there is no
VMEX free-boundary step.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from phrport import equilibrium as E
from phrport import problem as PROB
from phrport.coils import CoilSolver, CoilConfig, parse_name



def surface_rz(boundary, theta, phi, nfp):
    R = np.zeros(np.broadcast(theta, phi).shape); Z = np.zeros_like(R)
    for name, v in boundary.items():
        kind, m, n = parse_name(name)
        if v == 0.0:
            continue
        ang = m * theta - n * nfp * phi
        if kind == "rc":
            R = R + v * np.cos(ang)
        else:
            Z = Z + v * np.sin(ang)
    return R, Z


def cyl_points(R, phi, Z):
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1)


def field_cyl(cs, xc, I, R, phi, Z):
    X = cyl_points(R, phi, Z).reshape(-1, 3)
    B = cs._field_chunked(xc, I, X).reshape(np.shape(R) + (3,))
    c, s = np.cos(phi), np.sin(phi)
    return B[..., 0] * c + B[..., 1] * s, -B[..., 0] * s + B[..., 1] * c, B[..., 2]


def toroidal_flux(cs, xc, I, boundary, nfp, nrho=200, ntheta=400):
    """Flux of B_phi through the target phi = 0 cross-section (star-shaped w.r.t. its centroid)."""
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    Rb, Zb = surface_rz(boundary, th, 0.0, nfp)
    Rc, Zc = Rb.mean(), Zb.mean()
    dRb = np.gradient(Rb, th, edge_order=2); dZb = np.gradient(Zb, th, edge_order=2)
    rho = (np.arange(nrho) + 0.5) / nrho
    RR = Rc + rho[:, None] * (Rb - Rc)[None]; ZZ = Zc + rho[:, None] * (Zb - Zc)[None]
    area = (Rb - Rc) * dZb - (Zb - Zc) * dRb   # signed: a section that is not star-shaped about its centroid would otherwise count some area twice
    jac = rho[:, None] * (np.sign(area.sum()) * area)[None]
    _, Bp, _ = field_cyl(cs, xc, I, RR, np.zeros_like(RR), ZZ)
    return float(np.sum(Bp * jac) * (1.0 / nrho) * (2 * np.pi / ntheta))


def write_mgrid(cs, xc, I, boundary, nfp, path, nr=96, nz=96, kp=48, margin_frac=0.12):
    from vmex.core.mgrid import MgridData, write_mgrid as _w
    th = np.linspace(0, 2 * np.pi, 256, endpoint=False); ph = np.linspace(0, 2 * np.pi / nfp, 64, endpoint=False)
    R, Z = surface_rz(boundary, th[:, None], ph[None], nfp)
    m = margin_frac * float(boundary["rc(0,0)"])
    rmin, rmax, zmin, zmax = R.min() - m, R.max() + m, Z.min() - m, Z.max() + m
    rr = np.linspace(rmin, rmax, nr); zz = np.linspace(zmin, zmax, nz); pp = np.arange(kp) * 2 * np.pi / nfp / kp
    P, ZZ, RR = np.meshgrid(pp, zz, rr, indexing="ij")
    br, bp, bz = field_cyl(cs, xc, I, RR, P, ZZ)
    data = MgridData(rmin=float(rmin), rmax=float(rmax), zmin=float(zmin), zmax=float(zmax), ir=nr, jz=nz, kp=kp,
                     nfp=int(nfp), nextcur=1, mgrid_mode="S", coil_groups=("phrport",), raw_coil_cur=(1.0,),
                     br=br[None], bp=bp[None], bz=bz[None])
    _w(path, data)
    return dict(path=str(path), rmin=float(rmin), rmax=float(rmax), zmin=float(zmin), zmax=float(zmax), nr=nr, nz=nz, kp=kp)


def poincare(cs, xc, I, boundary, nfp, png, title, nlines=16, nper=200, steps=48):
    """Field lines from the outboard phi = 0 midplane, RK4 in phi; punctures at phi = 0, pi/(2 nfp), pi/nfp."""
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = np.linspace(0, 2 * np.pi, 400)
    Rb0, Zb0 = surface_rz(boundary, th, 0.0, nfp)
    i_out = np.argmax(Rb0)
    r_axis_guess = 0.5 * (Rb0.max() + Rb0.min())
    starts_R = np.linspace(r_axis_guess, Rb0.max() + 0.05 * (Rb0.max() - Rb0.min()), nlines)
    z0 = Zb0[i_out] * 0.0
    P = cs.P
    xcj, Ij = jnp.asarray(xc), jnp.asarray(I)

    def rhs(y, phi):
        R, Z = y[:, 0], y[:, 1]
        X = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], axis=-1)
        B = P._B64(xcj, Ij, X)
        c, s = jnp.cos(phi), jnp.sin(phi)
        BR = B[:, 0] * c + B[:, 1] * s; Bp = -B[:, 0] * s + B[:, 1] * c; BZ = B[:, 2]
        return jnp.stack([R * BR / Bp, R * BZ / Bp], axis=-1)

    h = 2 * np.pi / nfp / steps

    @jax.jit
    def period(y, phi0):
        def body(carry, i):
            y = carry
            phi = phi0 + i * h
            k1 = rhs(y, phi); k2 = rhs(y + 0.5 * h * k1, phi + 0.5 * h)
            k3 = rhs(y + 0.5 * h * k2, phi + 0.5 * h); k4 = rhs(y + h * k3, phi + h)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return y, y
        y_end, ys = jax.lax.scan(body, y, jnp.arange(steps))
        return y_end, ys

    y = jnp.asarray(np.stack([starts_R, np.full(nlines, z0)], axis=-1))
    hits = {0: [], steps // 4: [], steps // 2: []}
    for k in range(nper):
        y, ys = period(y, 0.0)          # field periods are identical: restart phi at 0
        ys = np.asarray(ys)
        hits[0].append(np.asarray(y))
        hits[steps // 4].append(ys[steps // 4 - 1]); hits[steps // 2].append(ys[steps // 2 - 1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (j, pts), frac in zip(axes, hits.items(), (0.0, 0.25, 0.5)):
        pts = np.array(pts)                         # (nper, nlines, 2)
        for l in range(nlines):
            ok = np.isfinite(pts[:, l]).all(axis=-1)
            ax.scatter(pts[ok, l, 0], pts[ok, l, 1], s=0.6, color=plt.cm.viridis(l / max(1, nlines - 1)), rasterized=True)
        Rb, Zb = surface_rz(boundary, th, frac * 2 * np.pi / nfp, nfp)
        ax.plot(Rb, Zb, "k-", lw=1.2, label="target boundary")
        ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
        ax.set_title(f"{title}  $\\phi$ = {frac:.2f} $\\cdot 2\\pi/N_{{fp}}$")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=180); plt.close(fig)
    return str(png)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("coils"); ap.add_argument("out"); ap.add_argument("--label", default="")
    ap.add_argument("--boundary", default=None, help="boundary JSON (default: the one stored in the coil export)")
    ap.add_argument("--no-poincare", action="store_true")
    ap.add_argument("--fb-kp", type=int, default=48, help="toroidal planes of the mgrid")
    ap.add_argument("--final-stage", action="store_true",
                    help="replace the given coils by the R1 final coil stage on the boundary (exported to OUTDIR)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cj = json.load(open(a.coils))
    boundary = json.load(open(a.boundary)) if a.boundary else cj["boundary"]
    cs = CoilSolver(CoilConfig())
    from phrport.phr import PHRConfig
    pcfg = PHRConfig()
    if pcfg.finite_beta:                    # all B.n/B metrics with B_coil + B_plasma (virtual casing)
        from phrport.finite_beta import standalone_coupler
        cs.set_plasma(standalone_coupler(boundary, 4, n=pcfg.fb_n, digits_value=pcfg.fb_digits_value,
                                         log=print).provider)
    nc = cs.P.nc
    xc = np.asarray(cj["solver_x"], float)[:nc]
    I = np.asarray(cj["base_currents_A"], float)
    nfp = int(E.NFP)
    if a.final_stage:
        from phrport.phr import export_coils
        t = time.time()
        r = cs.final_coil_stage(boundary)
        export_coils(str(out / "final_coils"), cs, r, boundary, source=f"warm_post final stage ({a.label})")
        xc = np.asarray(r.x, float)[:nc]; I = np.asarray(r.currents, float)
        print(f"final coil stage: Jf={r.c0_local:.6e} certified={r.certified} [{time.time() - t:.0f}s]", flush=True)
    post = dict(label=a.label, problem=PROB.NAME, coils=str(Path(a.coils).resolve()))
    ev = cs.evaluate_coils(boundary, xc, I, certify=True)
    fu = cs.flux_metrics_unfolded(boundary, xc, I)
    post["coil_properties"] = dict(lengths=ev["cert"]["L"], max_curvature=ev["cert"]["max_kappa"],
                                   msc=ev["cert"]["msc"], min_coil_coil=ev["cert"]["d_cc"],
                                   min_coil_surface=ev["cert"]["d_cs"], certified_violation=ev["cert"]["v"],
                                   currents_A=I.tolist(), total_length=float(np.sum(ev["cert"]["L"])))
    post["flux"] = dict(c0_local=ev["c0_local"], unfolded={k: fu["value"][k] for k in ("c0", "bn_ratio_int", "mean_area", "max_bn")},
                        converged=fu.get("converged"), grid=fu.get("value_grid"))
    t = time.time(); post["phiedge"] = phi = toroidal_flux(cs, xc, I, boundary, nfp); post["t_phiedge"] = time.time() - t
    t = time.time(); post["mgrid"] = write_mgrid(cs, xc, I, boundary, nfp, out / "mgrid.nc", kp=a.fb_kp); post["t_mgrid"] = time.time() - t
    with open(out / "boundary.json", "w") as fh:
        json.dump(boundary, fh, indent=1)
    (out / "post.json").write_text(json.dumps(post, indent=1, default=float))
    print("POST", json.dumps(dict(flux=post["flux"]["unfolded"], phiedge=phi, coil=post["coil_properties"]), default=float), flush=True)
    if not a.no_poincare:
        t = time.time(); post["poincare"] = poincare(cs, xc, I, boundary, nfp, out / "poincare.png", a.label); post["t_poincare"] = time.time() - t
    (out / "post.json").write_text(json.dumps(post, indent=1, default=float))
    print("DONE", out / "post.json", flush=True)


if __name__ == "__main__":
    main()

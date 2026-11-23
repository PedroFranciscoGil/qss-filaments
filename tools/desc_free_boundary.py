"""DESC free-boundary verification of a fixed-boundary target with its filament coils.

Optional tool (needs DESC: pip install -e ".[desc]"). The paper's free-boundary protocol: every free-boundary
number is evaluated on this DESC equilibrium (M = N = 16, --fb-ftol 1e-8, --maxiter 150, --keep-best).

python tools/desc_free_boundary.py --wout W --coils C --tag T --out DIR [--beta] [--M 16] [--maxiter 100]
                  [--cont 8,12,16] [--vmex-fb wout_fb.nc] [--lam auto|1|<float>] [--psi target|coil] [--device gpu|cpu]

C is a phrport-coils-v1 JSON (qss-run final_coils.json, qss-coils output); W is the run's fixed-boundary wout
(wout_optimized.nc). At finite beta pass --beta (and --exact-current <deck> to use the deck's current profile).
Check the wout's Rmajor_p sign first: a left-handed (negative Jacobian) wout must be flipped with
desc.compat.ensure_positive_jacobian before alpha tracing.

1. Load the fixed-boundary target (VMEC wout, current profile) and re-solve it in DESC at
   fixed boundary with L = M = N = --M (the free-boundary initial guess).
2. Build the coils (phrport-coils-v1 json, simsopt CurveXYZFourier convention) as a DESC CoilSet.
   Coil currents are scaled by lambda (beta: lambda_G = I_G / I_coils, as in the paper's VMEX
   validation; vacuum: 1). Psi is the target's (beta) or the coil flux through the target's
   phi = 0 cross-section (vacuum), as in the paper.
3. Free boundary against the coils (VacuumBoundaryError in vacuum, BoundaryError with virtual
   casing at finite beta), force balance as a constraint (proximal LSQ). Direct at --M first; if
   that does not converge, mode continuation over --cont, each warm-started from the previous.
4. Diagnostics -> DIR/summary.json, DIR/eq_fb.h5, DIR/wout_desc_fb.nc.
"""
import argparse, json, os, sys, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--wout", required=True); p.add_argument("--coils", required=True)
p.add_argument("--tag", required=True); p.add_argument("--out", required=True)
p.add_argument("--beta", action="store_true"); p.add_argument("--M", type=int, default=16)
p.add_argument("--maxiter", type=int, default=100); p.add_argument("--fb-ftol", type=float, default=1e-6)
p.add_argument("--cont", default="8,12,16"); p.add_argument("--vmex-fb", default=None)
p.add_argument("--lam", default="auto"); p.add_argument("--psi", default="auto")
p.add_argument("--force-tol", type=float, default=1e-3, help="converged if <|F|>/<|grad p_mag|> below")
p.add_argument("--bn-tol", type=float, default=1e-3, help="converged if mean |B.n|/|B| on LCFS below")
p.add_argument("--skip-direct", action="store_true")
p.add_argument("--keep-best", action="store_true", help="report the lowest-force fb stage, not the last")
p.add_argument("--sign", default="auto", help="coil current sign (auto: B_coil . B_eq > 0)")
p.add_argument("--jc", type=int, default=100, help="Jacobian chunk size (columns)")
p.add_argument("--exact-current", default=None, help="VMEC deck: use its power-series AC (I'(s)) and CURTOR exactly")
p.add_argument("--zero-current", action="store_true", help="set the current profile identically to zero")
p.add_argument("--pres", type=float, default=None, help="set p = PRES (1 - s) Pa")
p.add_argument("--device", default="gpu", choices=("gpu", "cpu"), help="DESC device (default gpu)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "log.txt"), "a")
def log(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

from desc import set_device
set_device(a.device)
import jax; jax.config.update("jax_enable_x64", True)
from desc.vmec import VMECIO
from desc.coils import FourierXYZCoil, CoilSet
from desc.grid import LinearGrid, Grid
from desc.objectives import (ObjectiveFunction, ForceBalance, FixPressure, FixCurrent, FixPsi,
                             BoundaryError, VacuumBoundaryError)
from netCDF4 import Dataset
mu0 = 4e-7 * np.pi
t0 = time.time()
log(f"== {a.tag} beta={a.beta} M={a.M} jax {jax.__version__} devices {jax.devices()}")

# ---------- coils (simsopt CurveXYZFourier: dofs per dim [c0, s1, c1, s2, c2, ...], t in [0,1))
cd = json.load(open(a.coils))
order, nfp, sym = cd["order"], cd["nfp"], cd["stellsym"]
modes = np.arange(-order, order + 1)
def desc_coeffs(d):
    c = np.zeros(2 * order + 1); c[order] = d[0]
    for n in range(1, order + 1):
        c[order - n] = d[2 * n - 1]  # sin -> negative n
        c[order + n] = d[2 * n]      # cos
    return c
base = []
for dofs, I in zip(cd["base_curve_dofs"], cd["base_currents_A"]):
    d = np.asarray(dofs); k = 2 * order + 1
    base.append(FourierXYZCoil(current=I, X_n=desc_coeffs(d[:k]), Y_n=desc_coeffs(d[k:2 * k]),
                               Z_n=desc_coeffs(d[2 * k:]), modes=modes))
coils = CoilSet(*base, NFP=nfp, sym=sym)

# independent check of the convention: explicit simsopt-style copies, numpy Biot-Savart
def simsopt_points(d, nq=400):
    t = np.linspace(0, 1, nq, endpoint=False); k = 2 * order + 1; out = []
    for j in range(3):
        c = d[j * k:(j + 1) * k]; v = np.full_like(t, c[0])
        for n in range(1, order + 1):
            v += c[2 * n - 1] * np.sin(2 * np.pi * n * t) + c[2 * n] * np.cos(2 * np.pi * n * t)
        out.append(v)
    return np.stack(out, 1)
def bs_numpy(pts, xyz_list, cur_list):
    B = np.zeros_like(pts)
    for X, I in zip(xyz_list, cur_list):
        dl = np.roll(X, -1, 0) - X; mid = X + 0.5 * dl
        r = pts[:, None, :] - mid[None]; rn = np.linalg.norm(r, axis=2)
        B += 1e-7 * I * np.sum(np.cross(dl[None], r) / rn[..., None] ** 3, axis=1)
    return B
xyzs, curs = [], []
for dofs, I in zip(cd["base_curve_dofs"], cd["base_currents_A"]):
    X = simsopt_points(np.asarray(dofs))
    for kk in range(nfp):
        ph = 2 * np.pi * kk / nfp; Rm = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
        xyzs.append(X @ Rm.T); curs.append(I)
        if sym:
            Xf = X * np.array([1, -1, -1]); xyzs.append(Xf @ Rm.T); curs.append(-I)
pts = np.array([[1.0, 0.05, 0.02], [0.95, 0.3, -0.04], [-0.2, 1.02, 0.01]])
Bn_ref = bs_numpy(pts, xyzs, curs)
Bn_desc = np.asarray(coils.compute_magnetic_field(pts, basis="xyz", source_grid=LinearGrid(N=400)))
log("coil convention check |dB|/|B| =", float(np.max(np.linalg.norm(Bn_ref - Bn_desc, axis=1) / np.linalg.norm(Bn_ref, axis=1))))

# ---------- target
from desc.profiles import PowerSeriesProfile
if a.wout.endswith(".nc"):
    eq_t = VMECIO.load(a.wout, profile="current")
    wt = Dataset(a.wout)
    log("target wout: NFP", eq_t.NFP, "Psi", float(eq_t.Psi), "ns", wt["ns"][:].item(), "mpol", wt["mpol"][:].item(), "ntor", wt["ntor"][:].item())
    if a.pres is not None:
        eq_t.pressure = PowerSeriesProfile(params=[a.pres, -a.pres], modes=[0, 2])
    if a.zero_current:
        eq_t.current = PowerSeriesProfile(params=[0.0], modes=[0])
    if a.exact_current:
        import re
        txt = open(a.exact_current).read()
        def nl(key):
            m = re.search(r"^\s*" + key + r"\s*=\s*([^\n]*(?:\n\s*[-+0-9.Ee][^=\n]*)*)", txt, re.M | re.I)
            return np.array([float(v) for v in re.split(r"[\s,]+", m.group(1).strip()) if v])
        ac = nl("AC"); curtor = float(nl("CURTOR")[0])
        k = np.arange(len(ac))
        cint = ac / (k + 1)                           # int_0^s I' = sum ac_k s^(k+1)/(k+1)
        coef = curtor * cint / np.sum(cint)           # I(s) = CURTOR * int_0^s I' / int_0^1 I'
        cmodes = 2 * (k + 1)                          # s^(k+1) = rho^(2k+2)
        prof = PowerSeriesProfile(params=np.concatenate([[0.0], coef]), modes=np.concatenate([[0], cmodes]))
        rr = np.linspace(0, 1, 200); ss = rr ** 2
        ref = curtor * np.array([np.polyval(np.concatenate([cint[::-1], [0.0]]), x) for x in ss]) / np.sum(cint)
        got = np.asarray(prof(rr))
        log(f"exact current: AC {len(ac)} terms, CURTOR {curtor:.4f}; max rel err vs deck {np.max(np.abs(got - ref)) / abs(curtor):.2e}; I(rho=1) {float(got[-1]):.4f}")
        eq_t.current = prof
    eq = eq_t.copy()
    eq.change_resolution(L=a.M, M=a.M, N=a.M, L_grid=2 * a.M, M_grid=2 * a.M, N_grid=2 * a.M)
    eq.solve(objective="force", ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300, verbose=1)
else:  # VMEC input deck (two-stage LP QA targets): build and solve from scratch with DESC continuation
    import inspect
    from desc.input_reader import InputReader
    from desc.equilibrium import Equilibrium
    from desc.continuation import solve_continuation_automatic
    d0 = InputReader.parse_vmec_inputs(a.wout)[-1]
    keys = set(inspect.signature(Equilibrium.__init__).parameters)
    kw = {k: v for k, v in d0.items() if k in keys and k not in ("L", "M", "N", "L_grid", "M_grid", "N_grid")}
    if a.pres is not None:
        kw["pressure"] = np.array([[0, a.pres], [2, -a.pres]])
    eq_t = Equilibrium(L=a.M, M=a.M, N=a.M, L_grid=2 * a.M, M_grid=2 * a.M, N_grid=2 * a.M, **kw)
    log("target deck: NFP", eq_t.NFP, "Psi", float(eq_t.Psi))
    eq = solve_continuation_automatic(eq_t.copy(), objective="force", ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300, verbose=1)[-1]
eq_fix = eq.copy()
eq_fix.save(os.path.join(a.out, "eq_fixed.h5"))
try:
    VMECIO.save(eq_fix, os.path.join(a.out, "wout_desc_fixed.nc"), surfs=101)
except Exception as ex:
    log("fixed wout save failed:", repr(ex)[:300])
log("fixed-boundary iota(0)", float(np.abs(eq_fix.compute("iota", grid=LinearGrid(rho=np.array([0.0]), M=2, N=2, NFP=eq_fix.NFP))["iota"][0])))
log("target <beta>_vol", float(eq_fix.compute("<beta>_vol")["<beta>_vol"]), "net current at edge [A]",
    float(eq_fix.current(np.array([1.0]))[0]) if eq_fix.current is not None else None)

def force_err(e):
    f = e.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol", "<|grad(p)|>_vol"])
    fb = float(f["<|F|>_vol"] / f["<|grad(|B|^2)|/2mu0>_vol"])
    gp = float(f["<|grad(p)|>_vol"])
    return fb if gp == 0 else min(fb, float(f["<|F|>_vol"]) / gp)
log("fixed-boundary DESC force error", force_err(eq_fix), f"t={time.time()-t0:.0f}s")

# ---------- current scale and flux
src_grid = LinearGrid(N=max(64, 16 * order))
I_tot = 2 * nfp * np.sum(np.abs(cd["base_currents_A"])) if sym else nfp * np.sum(np.abs(cd["base_currents_A"]))
gG = LinearGrid(rho=np.array([1.0]), M=2 * a.M, N=2 * a.M, NFP=eq.NFP)
G = float(np.mean(eq_fix.compute("G", grid=gG)["G"]))
I_G = 2 * np.pi * abs(G) / mu0
lamG = I_G / I_tot
lam = lamG if (a.lam == "auto" and a.beta) else (1.0 if a.lam == "auto" else float(a.lam))
# coil flux through the target's phi = 0 cross-section
xr, wr = np.polynomial.legendre.leggauss(48); rho = 0.5 * (xr + 1); wr = 0.5 * wr
nth = 128; th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
R_, T_ = np.meshgrid(rho, th, indexing="ij")
gx = Grid(np.stack([R_.ravel(), T_.ravel(), np.zeros(R_.size)], 1), sort=False)
d = eq_fix.compute(["e_rho", "e_theta", "R", "phi", "Z", "B"], grid=gx)
pos = np.stack([d["R"], d["phi"], d["Z"]], 1)
Bc = np.asarray(coils.compute_magnetic_field(pos, basis="rpz", source_grid=src_grid))
dA = np.cross(np.asarray(d["e_rho"]), np.asarray(d["e_theta"]))
w2 = (wr[:, None] * np.full(nth, 2 * np.pi / nth)[None]).ravel()
Psi_coil = float(np.sum(np.sum(Bc * dA, 1) * w2))
sgn = np.sign(np.sum(np.sum(Bc * np.asarray(d["B"]), 1)))
log(f"I_coils {I_tot:.6e} A, I_G {I_G:.6e} A, lambda_G {lamG:.5f}; coil flux at phi=0 {Psi_coil:.6e} (lam=1), target Psi {float(eq_fix.Psi):.6e}; B_coil.B_eq sign {sgn}")
cur_scale = lam * ((1.0 if sgn > 0 else -1.0) if a.sign == "auto" else float(a.sign))
if a.sign != "auto" and float(a.sign) * sgn < 0: log("WARNING: given coil sign opposes B_coil . B_eq orientation")
coils = CoilSet(*[FourierXYZCoil(current=c.current * cur_scale, X_n=c.X_n, Y_n=c.Y_n, Z_n=c.Z_n, modes=modes) for c in base], NFP=nfp, sym=sym)
psi_rule = a.psi if a.psi != "auto" else ("target" if a.beta else "coil")
if psi_rule == "coil":
    eq_fix.Psi = abs(Psi_coil) * lam
log(f"using lambda {lam:.5f} (sign {np.sign(cur_scale):+.0f}), Psi rule {psi_rule}: Psi {float(eq_fix.Psi):.6e}")

# ---------- free boundary
def bn_err(e):
    g = LinearGrid(rho=np.array([1.0]), M=2 * e.M, N=2 * e.N, NFP=e.NFP, sym=False)
    obj = (BoundaryError if a.beta else VacuumBoundaryError)(e, coils, field_fixed=True, eval_grid=g, field_grid=src_grid)
    obj.build(verbose=0)
    f = np.asarray(obj.compute(*obj.xs(e)))
    return float(np.sqrt(np.mean(f ** 2)))
def mean_bn(e):
    g = LinearGrid(rho=np.array([1.0]), M=64, N=64, NFP=e.NFP)
    dd = e.compute(["R", "phi", "Z", "n_rho", "|e_theta x e_zeta|"], grid=g)
    pos = np.stack([dd["R"], dd["phi"], dd["Z"]], 1)
    B = np.asarray(coils.compute_magnetic_field(pos, basis="rpz", source_grid=src_grid))
    w = np.asarray(dd["|e_theta x e_zeta|"])
    Bn = np.abs(np.sum(B * np.asarray(dd["n_rho"]), 1)); Bm = np.linalg.norm(B, axis=1)
    return float(np.sum(Bn * w) / np.sum(Bm * w)), float(np.max(Bn / Bm))
log("coil <|B.n|>/<|B|> on target (vacuum field only), max:", mean_bn(eq_fix))

def fb_solve(e, M, maxiter):
    e.change_resolution(L=M, M=M, N=M, L_grid=2 * M, M_grid=2 * M, N_grid=2 * M)
    g = LinearGrid(rho=np.array([1.0]), M=2 * M, N=2 * M, NFP=e.NFP, sym=False)
    if a.beta:
        obj = BoundaryError(e, coils, field_fixed=True, eval_grid=g, field_grid=src_grid, jac_chunk_size=a.jc,
                            bs_chunk_size=2000, B_plasma_chunk_size=2000)
    else:
        obj = VacuumBoundaryError(e, coils, field_fixed=True, eval_grid=g, field_grid=src_grid, jac_chunk_size=a.jc,
                                  bs_chunk_size=2000)
    cons = (ForceBalance(e, jac_chunk_size=a.jc), FixPressure(e), FixCurrent(e), FixPsi(e))
    t = time.time()
    e, res = e.optimize(ObjectiveFunction(obj), cons, optimizer="proximal-lsq-exact", ftol=a.fb_ftol,
                        xtol=1e-10, gtol=1e-10, maxiter=maxiter, verbose=2,
                        options={"solve_options": {"ftol": 1e-10, "xtol": 1e-12, "gtol": 1e-12, "maxiter": 100}},
                        copy=True)
    e.save(os.path.join(a.out, f"eq_fb_stage_M{M}_{len(hist)}.h5"))
    fe, be = force_err(e), bn_err(e)
    log(f"  FB M={M}: force {fe:.3e}, boundary residual rms {be:.3e}, nit {res.get('nit')}, {time.time()-t:.0f}s")
    return e, fe, be

hist = []; eqfb = None
if not a.skip_direct:
    log("direct free boundary at M =", a.M)
    try:
        eqfb, fe, be = fb_solve(eq_fix.copy(), a.M, a.maxiter); hist.append(("direct", a.M, fe, be))
    except Exception as ex:
        log("direct failed:", repr(ex)[:400]); fe = be = np.inf
ok = eqfb is not None and fe < a.force_tol and be < a.bn_tol
if not ok:
    log("continuation over", a.cont)
    e = eq_fix.copy()
    for M in [int(x) for x in a.cont.split(",")]:
        e2 = e.copy(); e2.change_resolution(L=M, M=M, N=M, L_grid=2 * M, M_grid=2 * M, N_grid=2 * M)
        if M == int(a.cont.split(",")[0]):
            e2.solve(objective="force", ftol=1e-10, maxiter=200, verbose=0)
        e, fe, be = fb_solve(e2, M, a.maxiter); hist.append(("cont", M, fe, be))
    if a.keep_best and eqfb is not None and hist[0][0] == "direct" and hist[0][2] < fe:
        log("keep-best: continuation worse than direct, keeping direct"); fe, be = hist[0][2], hist[0][3]
    else:
        eqfb = e
    ok = fe < a.force_tol and be < a.bn_tol

# ---------- diagnostics
def lcfs_rz_wout(path, phis, nth=2048):
    w = Dataset(path); xm = w["xm"][:]; xn = w["xn"][:]
    rmn = w["rmnc"][-1, :]; zmn = w["zmns"][-1, :]
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    out = []
    for ph in phis:
        ang = xm[None] * th[:, None] - xn[None] * ph
        out.append(np.stack([np.cos(ang) @ rmn, np.sin(ang) @ zmn], 1))
    return out
def lcfs_rz_desc(e, phis, nth=256):
    out = []
    for ph in phis:
        g = LinearGrid(rho=np.array([1.0]), theta=nth, zeta=np.array([ph]), NFP=1)
        dd = e.compute(["R", "Z"], grid=g); out.append(np.stack([dd["R"], dd["Z"]], 1))
    return out
phis = np.linspace(0, 2 * np.pi / eq.NFP, 32, endpoint=False)
def dev(c_desc, c_ref):
    ds = []
    for A, B in zip(c_desc, c_ref):
        ds.append(np.min(np.linalg.norm(A[:, None] - B[None], axis=2), 1))
    ds = np.concatenate(ds); return float(ds.mean()), float(ds.max())
fbc = lcfs_rz_desc(eqfb, phis)
dev_t = dev(fbc, lcfs_rz_desc(eq_fix, phis, nth=2048))
dev_v = dev(fbc, lcfs_rz_wout(a.vmex_fb, phis)) if a.vmex_fb else None
rr = np.linspace(0, 1, 101)
iota = np.array([float(np.mean(eqfb.compute("iota", grid=LinearGrid(rho=np.array([max(r, 1e-4)]), M=2, N=2, NFP=eqfb.NFP))["iota"])) for r in rr])
s = rr ** 2
summ = dict(tag=a.tag, beta=a.beta, M=a.M, converged=bool(ok), history=hist, lambda_used=lam, lambda_G=lamG,
            coil_sign=float(np.sign(cur_scale)), psi_rule=psi_rule, Psi=float(eqfb.Psi), Psi_coil_lam1=Psi_coil,
            force_err_fixed=force_err(eq_fix), force_err_fb=force_err(eqfb), boundary_resid_rms=bn_err(eqfb),
            lcfs_dev_vs_target_mean_max_m=dev_t, lcfs_dev_vs_vmexfb_mean_max_m=dev_v,
            iota_axis=float(abs(iota[0])), iota_edge=float(abs(iota[-1])), iota_mean_s=float(abs(np.trapezoid(iota, s))),
            aspect=float(eqfb.compute("R0/a")["R0/a"]), aspect_target=float(eq_fix.compute("R0/a")["R0/a"]),
            beta_vol=float(eqfb.compute("<beta>_vol")["<beta>_vol"]), wall_s=time.time() - t0)
log(json.dumps(summ, indent=1))
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
eqfb.save(os.path.join(a.out, "eq_fb.h5")); eq_fix.save(os.path.join(a.out, "eq_fixed.h5"))
try:
    VMECIO.save(eqfb, os.path.join(a.out, "wout_desc_fb.nc"), surfs=101)
except Exception as ex:
    log("wout save failed:", repr(ex)[:300])
log("DONE", a.tag)

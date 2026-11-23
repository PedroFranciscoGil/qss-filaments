"""Seed decks: write a VMEC input deck (the seed of a run, profile key equilibrium.input_file) from a tokamak
shape, a rotating ellipse, a boundary JSON, a wout file or a checkpoint (command ``qss-seed``; docs/seeding.md).

The deck keeps everything but the boundary (and NFP / PHIEDGE when given) from a template deck: pressure and
current profiles, PHIEDGE, radial ladder. Boundary dictionaries use the simsopt names ``rc(m,n)`` / ``zs(m,n)``
(stellarator symmetric; n in units of nfp, the VMEC convention RBC(n,m)).

Tokamak (axisymmetric) seeds: an exactly axisymmetric vacuum boundary has iota = 0 and a zero gradient in the
non-axisymmetric directions (a saddle), so a small symmetry break is always added:
  * "qa": 1e-3 on (m, n) = (1, 1) and (2, 1)                    (LP QA tokamak start of the paper);
  * "qi": a 2e-3 rotating ellipse tall at phi = 0, (1, 1) rc = -2e-3, zs = +2e-3, and a 2e-3 outward helical
          axis shift, (0, 1) rc = +2e-3, zs = -2e-3             (the QI tokamak start of the paper; the signs pick
          the branch that reached the low-QI designs);
  * "none", or explicit --perturb m,n,rc,zs terms.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re

import numpy as np

_NAME = re.compile(r"^\s*(rc|zs)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEMPLATE = os.path.join(_REPO, "inputs", "input.LandremanPaul2021_QA")

BREAKS = {
    "qa": [(1, 1, 1e-3, 1e-3), (2, 1, 1e-3, 1e-3)],
    "qi": [(1, 1, -2e-3, 2e-3), (0, 1, 2e-3, -2e-3)],
    "none": [],
}


def tokamak_boundary(R0=1.0, aspect=6.0, kappa=1.0, delta=0.0, brk="qa", perturb=()):
    """Axisymmetric boundary R = R0 + a cos(t + d sin t), Z = kappa a sin t (Fourier-fitted for delta != 0) with
    minor radius a = R0 / aspect (area-equivalent for kappa != 1: a sqrt(kappa) = R0 / aspect), plus the break."""
    a = R0 / aspect / math.sqrt(kappa)
    b = {"rc(0,0)": R0}
    if delta == 0.0:
        b["rc(1,0)"] = a
        b["zs(1,0)"] = kappa * a
    else:
        t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        R = R0 + a * np.cos(t + math.asin(max(-0.99, min(0.99, delta))) * np.sin(t))
        Z = kappa * a * np.sin(t)
        for m in range(0, 7):
            rc = 2 * np.mean(R * np.cos(m * t)) if m else np.mean(R)
            zs = 2 * np.mean(Z * np.sin(m * t))
            if abs(rc) > 1e-9:
                b[f"rc({m},0)"] = float(rc)
            if m and abs(zs) > 1e-9:
                b[f"zs({m},0)"] = float(zs)
    for m, n, rc, zs in list(BREAKS[brk]) + list(perturb):
        b[f"rc({m},{n})"] = b.get(f"rc({m},{n})", 0.0) + float(rc)
        b[f"zs({m},{n})"] = b.get(f"zs({m},{n})", 0.0) + float(zs)
    return b


def rotating_ellipse_boundary(R0=1.0, aspect=6.0, amp=0.05, axis_shift=0.0):
    """Classic rotating-ellipse stellarator seed: a circle plus an (m, n) = (1, 1) ellipse of relative amplitude
    amp (rc = -amp a, zs = +amp a: tall at phi = 0) and an optional (0, 1) helical axis shift."""
    a = R0 / aspect
    b = {"rc(0,0)": R0, "rc(1,0)": a, "zs(1,0)": a, "rc(1,1)": -amp * a, "zs(1,1)": amp * a}
    if axis_shift:
        b["rc(0,1)"], b["zs(0,1)"] = axis_shift, -axis_shift
    return b


def boundary_from_wout(path):
    """Last-surface boundary of a VMEC / VMEX wout (stellarator-symmetric part) and its nfp."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        nfp = int(ds["nfp"][()])
        xm = np.asarray(ds["xm"][:], int)
        xn = np.asarray(ds["xn"][:], int) // nfp
        rmnc = np.asarray(ds["rmnc"][:])[-1]
        zmns = np.asarray(ds["zmns"][:])[-1]
    b = {}
    for m, n, r, z in zip(xm, xn, rmnc, zmns):
        # wout uses cos(m theta - n nfp phi); the deck uses the same sign convention for RBC(n, m)
        if abs(r) > 0:
            b[f"rc({m},{n})"] = float(r)
        if m or n:
            if abs(z) > 0:
                b[f"zs({m},{n})"] = float(z)
    return b, nfp


def boundary_from_json(path):
    """A boundary dict {'rc(m,n)': v, ...}, or a file holding one under 'boundary' (coil exports), or a checkpoint
    (needs the profile: use boundary_from_checkpoint)."""
    d = json.load(open(path))
    if "boundary" in d and isinstance(d["boundary"], dict):
        d = d["boundary"]
    if "surface_dofs" in d:
        raise ValueError(f"{path} is a checkpoint: use 'qss-seed from-checkpoint' (it needs the run's profile)")
    return {k: float(v) for k, v in d.items() if _NAME.match(k)}


def boundary_from_checkpoint(path):
    """Full boundary of a checkpoint (file or run folder) under the current profile (PHRPORT_PROBLEM)."""
    from phrport import equilibrium as E
    cp = os.path.join(path, "al1_checkpoint_final.json") if os.path.isdir(path) else path
    if os.path.isdir(path) and not os.path.exists(cp):
        cp = os.path.join(path, "al1_checkpoint_latest.json")
    ck = json.load(open(cp))
    b = dict(E.seed_boundary())
    names = ck.get("surface_dof_names") or E.free_dof_names(int(ck["step"]))
    b.update({k: float(v) for k, v in zip(names, ck["surface_dofs"])})
    return b


def boundary_from_deck(path):
    """Boundary dict of a VMEC input deck."""
    from vmex import VmecInput
    inp = VmecInput.from_file(path)
    nt = int(inp.ntor)
    out = {}
    for arr, t in ((np.asarray(inp.rbc), "rc"), (np.asarray(inp.zbs), "zs")):
        for i, j in zip(*np.nonzero(np.abs(arr) > 0)):
            out[f"{t}({j},{i - nt})"] = float(arr[i, j])
    return out


def write_deck(boundary, out, template=None, nfp=None, phiedge=None, comment="", pressure=None):
    """Write a VMEC input deck: the template with this boundary (all other boundary modes removed).
    pressure p0 > 0: p(s) = p0 (1 - s) [Pa], current-free (CURTOR 0), a generic finite-beta deck."""
    from vmex import VmecInput
    inp = VmecInput.from_file(template or DEFAULT_TEMPLATE)
    modes = [(t, int(m), int(n)) for t, m, n in (_NAME.match(k).groups() for k in boundary)]
    mpol = max(int(inp.mpol), max(m for _, m, _ in modes) + 1)
    ntor = max(int(inp.ntor), max(abs(n) for _, _, n in modes))
    inp = inp.change_resolution(mpol=mpol, ntor=ntor)
    rbc = np.zeros_like(np.asarray(inp.rbc))
    zbs = np.zeros_like(np.asarray(inp.zbs))
    for (t, m, n), v in zip(modes, boundary.values()):
        (rbc if t == "rc" else zbs)[n + ntor, m] = float(v)
    rax = np.zeros_like(np.asarray(inp.raxis_c))
    rax[0] = float(boundary.get("rc(0,0)", 1.0))
    kw = dict(rbc=rbc, zbs=zbs, raxis_c=rax, zaxis_s=np.zeros_like(np.asarray(inp.zaxis_s)))
    if nfp is not None:
        kw["nfp"] = int(nfp)
    if phiedge is not None:
        kw["phiedge"] = float(phiedge)
    if pressure is not None:
        am = np.zeros_like(np.asarray(inp.am, dtype=float))
        am[0], am[1] = float(pressure), -float(pressure)
        kw.update(am=am, pres_scale=1.0, pmass_type="power_series", curtor=0.0,
                  ac=np.zeros_like(np.asarray(inp.ac, dtype=float)))
    inp = dataclasses.replace(inp, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    inp.to_indata(out)
    if comment:
        txt = open(out).read()
        open(out, "w").write("".join(f"! {ln}\n" for ln in comment.splitlines()) + txt)
    return out


def _perturb(s):
    m, n, rc, zs = s.split(",")
    return int(m), int(n), float(rc), float(zs)


def build_parser():
    ap = argparse.ArgumentParser(prog="qss-seed", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="deck to write (e.g. inputs/input.my_seed)")
    common.add_argument("--template", default=None, help=f"deck whose profiles / settings are kept "
                                                         f"(default {os.path.relpath(DEFAULT_TEMPLATE, _REPO)})")
    common.add_argument("--nfp", type=int, default=None, help="field periods (default: the template's)")
    common.add_argument("--phiedge", type=float, default=None, help="toroidal flux at the edge [Wb] (default: template)")
    common.add_argument("--pressure", type=float, default=None, metavar="P0",
                        help="finite beta: p(s) = P0 (1 - s) [Pa], current-free (default: the template's profiles)")
    t = sub.add_parser("tokamak", parents=[common], help="axisymmetric (circular / elliptic / D-shaped) seed")
    t.add_argument("--R0", type=float, default=1.0)
    t.add_argument("--aspect", type=float, default=6.0)
    t.add_argument("--kappa", type=float, default=1.0, help="elongation")
    t.add_argument("--delta", type=float, default=0.0, help="triangularity")
    t.add_argument("--break", dest="brk", choices=sorted(BREAKS), default="qa",
                   help="symmetry break: qa (1e-3 on (1,1),(2,1)), qi (re-signed rotating ellipse + axis shift), none")
    t.add_argument("--perturb", type=_perturb, action="append", default=[], metavar="m,n,rc,zs",
                   help="extra boundary terms (repeatable)")
    e = sub.add_parser("rotating-ellipse", parents=[common], help="circle + (1,1) rotating ellipse")
    e.add_argument("--R0", type=float, default=1.0)
    e.add_argument("--aspect", type=float, default=6.0)
    e.add_argument("--amp", type=float, default=0.05, help="ellipse amplitude relative to a")
    e.add_argument("--axis-shift", type=float, default=0.0, help="(0,1) helical axis shift [m]")
    j = sub.add_parser("from-boundary", parents=[common], help="boundary JSON {'rc(m,n)': ...} (or a coil export)")
    j.add_argument("boundary")
    sub.add_parser("deck", parents=[common], help="the template's own boundary (e.g. to change its pressure or nfp)")
    w = sub.add_parser("from-wout", parents=[common], help="last surface of a wout_*.nc")
    w.add_argument("wout")
    c = sub.add_parser("from-checkpoint", parents=[common],
                       help="boundary of a checkpoint / run folder (set --profile to the run's profile)")
    c.add_argument("checkpoint")
    c.add_argument("--profile", default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    nfp = a.nfp
    if a.kind == "tokamak":
        b = tokamak_boundary(a.R0, a.aspect, a.kappa, a.delta, a.brk, a.perturb)
        note = (f"tokamak seed: R0 {a.R0}, aspect {a.aspect}, kappa {a.kappa}, delta {a.delta}, break {a.brk}"
                + (f", perturb {a.perturb}" if a.perturb else ""))
    elif a.kind == "rotating-ellipse":
        b = rotating_ellipse_boundary(a.R0, a.aspect, a.amp, a.axis_shift)
        note = f"rotating-ellipse seed: R0 {a.R0}, aspect {a.aspect}, amp {a.amp}, axis shift {a.axis_shift}"
    elif a.kind == "deck":
        b, note = boundary_from_deck(a.template or DEFAULT_TEMPLATE), f"boundary of {a.template or 'the template'}"
    elif a.kind == "from-boundary":
        b, note = boundary_from_json(a.boundary), f"boundary of {a.boundary}"
    elif a.kind == "from-wout":
        b, nfp_w = boundary_from_wout(a.wout)
        nfp = nfp if nfp is not None else nfp_w
        note = f"last surface of {a.wout}"
    else:
        if a.profile:
            os.environ["PHRPORT_PROBLEM"] = os.path.abspath(a.profile)
        b, note = boundary_from_checkpoint(a.checkpoint), f"boundary of {a.checkpoint}"
    if a.pressure is not None:
        note += f"; pressure p = {a.pressure:g} (1 - s) Pa, current-free"
    out = write_deck(b, a.out, a.template, nfp, a.phiedge, comment=f"written by qss-seed: {note}",
                     pressure=a.pressure)
    print(f"wrote {out} ({len(b)} boundary terms{', nfp ' + str(nfp) if nfp else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
